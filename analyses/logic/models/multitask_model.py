"""Multi-task Transformer for joint ASO toxicity prediction.

Shared encoder with factored HELM embeddings and RoPE, attention pooling,
and independent task heads. Adapted from archive/05_oligoai2/model.py.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Vocabulary ──────────────────────────────────────────────────────────

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 4, "5meC": 5}
SUGAR_TO_IDX = {"MOE": 0, "DNA": 1, "LNA": 2, "cEt": 3, "fR": 4, "OMe": 5, "RNA": 6}
BACKBONE_TO_IDX = {"PS": 0, "PO": 1, "PAD": 2}
N_BASES, N_SUGARS, N_BACKBONES = len(BASE_TO_IDX), len(SUGAR_TO_IDX), len(BACKBONE_TO_IDX)

TASK_NAMES = ["inhibition", "mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro"]
TASK_ID = {name: i for i, name in enumerate(TASK_NAMES)}
N_TASKS = len(TASK_NAMES)


# ── Config ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MultiTaskConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 384
    dropout: float = 0.1
    max_len: int = 30
    d_head_hidden: int = 64
    n_head_layers: int = 2


# ── Factored Embedding ──────────────────────────────────────────────────

class FactoredEmbedding(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.base_emb = nn.Embedding(N_BASES, cfg.d_model)
        self.sugar_emb = nn.Embedding(N_SUGARS, cfg.d_model)
        self.backbone_emb = nn.Embedding(N_BACKBONES, cfg.d_model)

    def forward(self, base_idx, sugar_idx, backbone_idx):
        return self.base_emb(base_idx) + self.sugar_emb(sugar_idx) + self.backbone_emb(backbone_idx)


# ── RoPE ────────────────────────────────────────────────────────────────

def precompute_rope(d_head: int, max_len: int, device: torch.device):
    theta = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# ── Multi-Head Self-Attention with RoPE ─────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        cos, sin = precompute_rope(self.d_head, cfg.max_len, torch.device("cpu"))
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x, mask):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, S, -1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, S, D))


# ── Transformer ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff(self.ln2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.embed = FactoredEmbedding(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)

    def forward(self, base_idx, sugar_idx, backbone_idx, mask):
        x = self.dropout(self.embed(base_idx, sugar_idx, backbone_idx))
        for block in self.blocks:
            x = block(x, mask)
        return self.final_ln(x)


# ── Attention Pooling ───────────────────────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, h, mask):
        q = self.query.expand(h.shape[0], -1, -1)
        out, _ = self.attn(q, h, h, key_padding_mask=~mask)
        return out.squeeze(1)


# ── Multi-Task Model ────────────────────────────────────────────────────

class MultiTaskTransformer(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(cfg)
        self.pool = AttentionPool(cfg.d_model, cfg.n_heads)

        # Independent task heads
        self.heads = nn.ModuleDict()
        for name in TASK_NAMES:
            layers = [nn.Linear(cfg.d_model, cfg.d_head_hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            for _ in range(cfg.n_head_layers - 2):
                layers += [nn.Linear(cfg.d_head_hidden, cfg.d_head_hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            layers.append(nn.Linear(cfg.d_head_hidden, 1))
            self.heads[name] = nn.Sequential(*layers)

        # Homoscedastic uncertainty weighting
        self.log_var = nn.Parameter(torch.zeros(N_TASKS))

    def forward(self, base_idx, sugar_idx, backbone_idx, mask, task_name: str):
        h = self.encoder(base_idx, sugar_idx, backbone_idx, mask)
        z = self.pool(h, mask)
        return self.heads[task_name](z).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
