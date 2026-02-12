"""OligoAI2 model architecture.

Transformer encoder with factored embeddings and RoPE, perceiver bottleneck
for fixed-size latent, covariate encoders, and FiLM-gated multi-task heads.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

from .data import N_BASES, N_SUGARS, N_BACKBONES, N_TASKS, TASK_NAMES


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 384
    dropout: float = 0.1
    max_len: int = 40
    n_latents: int = 4
    d_latent: int = 64
    d_cov: int = 32
    d_head_hidden: int = 64
    n_bases: int = N_BASES
    n_sugars: int = N_SUGARS
    n_backbones: int = N_BACKBONES


# ── Factored Embedding ──────────────────────────────────────────────────

class FactoredEmbedding(nn.Module):
    """Additive embedding: e_i = W_base[b_i] + W_sugar[s_i] + W_backbone[k_i]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.base_emb = nn.Embedding(cfg.n_bases, cfg.d_model)
        self.sugar_emb = nn.Embedding(cfg.n_sugars, cfg.d_model)
        self.backbone_emb = nn.Embedding(cfg.n_backbones, cfg.d_model)

    def forward(self, base_idx, sugar_idx, backbone_idx):
        """
        Args: all (batch, seq_len) LongTensors
        Returns: (batch, seq_len, d_model)
        """
        return self.base_emb(base_idx) + self.sugar_emb(sugar_idx) + self.backbone_emb(backbone_idx)


# ── RoPE Helpers ────────────────────────────────────────────────────────

def precompute_rope(d_head: int, max_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embedding."""
    theta = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, theta)  # (max_len, d_head/2)
    cos_table = freqs.cos()  # (max_len, d_head/2)
    sin_table = freqs.sin()
    return cos_table, sin_table


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to x of shape (batch, heads, seq_len, d_head)."""
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ── Multi-Head Self-Attention with RoPE ─────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # Precompute RoPE tables
        cos, sin = precompute_rope(self.d_head, cfg.max_len, torch.device("cpu"))
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) bool, True = valid
        Returns: (batch, seq_len, d_model)
        """
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Attention mask: (B, 1, 1, S) - False means masked out
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        # Use F.scaled_dot_product_attention (PyTorch 2.0+)
        # It expects attn_mask as a bool where True = attend
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask.expand(-1, self.n_heads, S, -1),
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # (B, H, S, d_head)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


# ── Transformer Block (Pre-norm) ────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


# ── Transformer Encoder ────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = FactoredEmbedding(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)

    def forward(self, base_idx, sugar_idx, backbone_idx, mask):
        """
        Returns: (batch, seq_len, d_model) encoder hidden states.
        """
        x = self.embed(base_idx, sugar_idx, backbone_idx)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.final_ln(x)


# ── Perceiver Bottleneck ────────────────────────────────────────────────

class PerceiverBottleneck(nn.Module):
    """Cross-attend K learned latents into encoder output, project to d_latent."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(cfg.n_latents, cfg.d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(cfg.n_latents * cfg.d_model, cfg.d_latent)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, d_model) encoder output
            mask: (batch, seq_len) bool, True = valid
        Returns: (batch, d_latent)
        """
        B = h.shape[0]
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)

        # nn.MultiheadAttention uses key_padding_mask where True = ignore
        key_padding_mask = ~mask  # invert: True means padding

        z, _ = self.cross_attn(queries, h, h, key_padding_mask=key_padding_mask)  # (B, K, d_model)
        z = z.reshape(B, -1)  # (B, K * d_model)
        return F.gelu(self.proj(z))  # (B, d_latent)


# ── Covariate Encoder ──────────────────────────────────────────────────

class CovariateEncoder(nn.Module):
    """Encode continuous + categorical covariates to a fixed-size vector."""

    def __init__(self, n_continuous: int, categorical_sizes: list[int], d_out: int, d_emb: int = 8):
        super().__init__()
        self.n_continuous = n_continuous
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(size, d_emb) for size in categorical_sizes
        ])
        total_in = n_continuous + len(categorical_sizes) * d_emb
        self.mlp = nn.Sequential(
            nn.Linear(total_in, d_out * 2),
            nn.GELU(),
            nn.Linear(d_out * 2, d_out),
        )

    def forward(self, continuous: torch.Tensor, categoricals: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            continuous: (batch, n_continuous)
            categoricals: list of (batch,) LongTensors
        Returns: (batch, d_out)
        """
        parts = [continuous]
        for emb, cat in zip(self.cat_embeddings, categoricals):
            parts.append(emb(cat))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


# ── FiLM Head ──────────────────────────────────────────────────────────

class FiLMHead(nn.Module):
    """FiLM-gated regression head shared across all tasks."""

    def __init__(self, cfg: ModelConfig, n_tasks: int = N_TASKS):
        super().__init__()
        self.task_emb = nn.Embedding(n_tasks, cfg.d_latent)
        self.gate = nn.Linear(cfg.d_latent + cfg.d_cov, cfg.d_latent)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_latent, cfg.d_head_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_head_hidden, 1),
        )

    def forward(self, z: torch.Tensor, task_id: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, d_latent) - ASO representation
            task_id: (batch,) LongTensor
            cov: (batch, d_cov) - covariate encoding
        Returns: (batch,) predictions
        """
        t = self.task_emb(task_id)  # (batch, d_latent)
        gamma = torch.sigmoid(self.gate(torch.cat([t, cov], dim=-1)))  # (batch, d_latent)
        return self.mlp(z * gamma).squeeze(-1)  # (batch,)


# ── Top-Level Model ────────────────────────────────────────────────────

class OligoAI(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        n_transfection: int = 3,
        n_hepatic_admin: int = 2,
        n_neuro_admin: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(cfg)
        self.bottleneck = PerceiverBottleneck(cfg)

        # Covariate encoders
        self.cov_invitro = CovariateEncoder(
            n_continuous=2,  # log10_dose, treatment_hrs
            categorical_sizes=[n_transfection],
            d_out=cfg.d_cov,
        )
        self.cov_hepatic = CovariateEncoder(
            n_continuous=3,  # dose_mgkg, num_doses, dosing_days
            categorical_sizes=[max(n_hepatic_admin, 1)],
            d_out=cfg.d_cov,
        )
        self.cov_neuro = CovariateEncoder(
            n_continuous=2,  # dose_ug, latency_hrs_log
            categorical_sizes=[max(n_neuro_admin, 1)],
            d_out=cfg.d_cov,
        )

        self.head = FiLMHead(cfg)

        # Homoscedastic uncertainty weighting: log(sigma^2) per task
        self.log_var = nn.Parameter(torch.zeros(N_TASKS))

    def encode(self, batch: dict) -> torch.Tensor:
        """Encode ASO HELM to latent vector. Returns (batch, d_latent)."""
        h = self.encoder(
            batch["base_idx"], batch["sugar_idx"],
            batch["backbone_idx"], batch["mask"],
        )
        return self.bottleneck(h, batch["mask"])

    def forward_invitro(self, batch: dict) -> torch.Tensor:
        """Forward pass for in vitro data. Returns (batch,) predictions."""
        z = self.encode(batch)
        cov = self.cov_invitro(batch["continuous"], [batch["transfection_idx"]])
        task_id = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)  # TASK_INHIBITION = 0
        return self.head(z, task_id, cov)

    def forward_invivo(self, batch: dict) -> torch.Tensor:
        """Forward pass for in vivo data. Routes to appropriate covariate encoder.

        batch["continuous"] layout: [dose_mgkg, num_doses, dosing_days, dose_ug, latency_hrs_log]
        Hepatic tasks (1-6) use [:3], neuro task (7) uses [3:].
        """
        z = self.encode(batch)
        B = z.shape[0]
        cov = torch.zeros(B, self.cfg.d_cov, device=z.device)

        is_hepatic = batch["task_id"] <= 6
        if is_hepatic.any():
            idx = is_hepatic.nonzero(as_tuple=True)[0]
            cov[idx] = self.cov_hepatic(
                batch["continuous"][idx, :3],
                [batch["admin_idx"][idx]],
            )

        is_neuro = ~is_hepatic
        if is_neuro.any():
            idx = is_neuro.nonzero(as_tuple=True)[0]
            cov[idx] = self.cov_neuro(
                batch["continuous"][idx, 3:],
                [batch["admin_idx"][idx]],
            )

        return self.head(z, batch["task_id"], cov)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss Functions ──────────────────────────────────────────────────────

def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Concordance Correlation Coefficient loss: 1 - CCC.

    CCC = 2 * cov(pred, target) / (var(pred) + var(target) + (mean(pred) - mean(target))^2)
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var()
    target_var = target.var()
    cov = ((pred - pred_mean) * (target - target_mean)).mean()
    numerator = 2.0 * cov
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2 + eps
    ccc = numerator / denominator
    return 1.0 - ccc


def multitask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_ids: torch.Tensor,
    log_var: torch.Tensor,
    min_samples: int = 8,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Multi-task CCC loss with homoscedastic uncertainty weighting.

    L = sum_t [ 1/(2*sigma_t^2) * L_t + log(sigma_t) ]

    Returns (total_loss, per_task_losses dict).
    """
    total_loss = torch.tensor(0.0, device=pred.device)
    per_task = {}
    n_tasks_used = 0

    for tid in task_ids.unique():
        tid_int = tid.item()
        mask = task_ids == tid
        n = mask.sum().item()
        if n < min_samples:
            continue

        task_pred = pred[mask]
        task_target = target[mask]
        loss_t = ccc_loss(task_pred, task_target)
        per_task[tid_int] = loss_t.item()

        # Uncertainty weighting: 1/(2*sigma^2) * L_t + log(sigma)
        # log_var = log(sigma^2) => sigma^2 = exp(log_var), log(sigma) = log_var/2
        precision = torch.exp(-log_var[tid_int])
        total_loss = total_loss + 0.5 * precision * loss_t + 0.5 * log_var[tid_int]
        n_tasks_used += 1

    if n_tasks_used > 0:
        total_loss = total_loss / n_tasks_used

    return total_loss, per_task


def median_spearman(
    pred: torch.Tensor,
    target: torch.Tensor,
    groups: list[str],
    min_samples: int = 3,
) -> float:
    """Compute median Spearman rho across groups.

    Groups predictions by target gene/patent, computes Spearman per group
    (min_samples threshold), returns median.
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    group_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        if g is not None:
            group_to_indices.setdefault(g, []).append(i)

    rhos = []
    for indices in group_to_indices.values():
        if len(indices) < min_samples:
            continue
        r, _ = sp_stats.spearmanr(pred_np[indices], target_np[indices])
        if not (math.isnan(r) or math.isinf(r)):
            rhos.append(r)

    return float(np.median(rhos)) if rhos else float("nan")
