"""
OligoAI-tox: interpretable 1D CNN for ASO toxicity prediction.

Multi-scale CNN with position-weighted pooling that learns dinucleotide and
trinucleotide composition patterns end-to-end.

Architecture:
  Conv1D(8, 8, k=2) || Conv1D(8, 8, k=3) → ReLU
  Position-weighted pooling: sum, d5-weighted sum, d3-weighted sum
    → 3 pools × 8 filters × 2 branches = 48 features
  Concat covariates → Linear(48+n_cov, 32) → ReLU → Dropout(0.25) → Linear(32, 1)

Position weights (computed at runtime, no learnable params):
  d5[i] = i / (L-1)    — distance from 5' end (0→1)
  d3[i] = 1 - d5[i]    — distance from 3' end (1→0)

This captures where in the gapmer a motif fires: the same dinucleotide at the
5' wing vs 3' wing can have different toxicity effects.

~2,100 parameters total. Uses MPS (Apple Silicon) when available.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from analyses.utils.helm import Helm

# ── Constants ────────────────────────────────────────────────────

BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
SUGAR_IDX = {"DNA": 0, "MOE": 1, "cEt": 2}
N_BASE = len(BASE_IDX)      # 4
N_SUGAR = len(SUGAR_IDX)    # 3
N_BACKBONE = 1              # PS=1, PO=0
CHANNEL_LABELS_8 = ["A", "C", "G", "T", "DNA", "MOE", "cEt", "PS"]
CHANNEL_LABELS_9 = CHANNEL_LABELS_8 + ["pos"]


# ── Encoding ─────────────────────────────────────────────────────

def encode_helm(helm_str: str, max_len: int = 21, add_position: bool = False) -> np.ndarray | None:
    """Encode a HELM string into a (max_len, C) float32 array.

    Channels:
      0-3:  base one-hot (A, C, G, T)
      4-6:  sugar one-hot (DNA, MOE, cEt)
      7:    backbone PS flag (assigned to 5' nucleotide of each linkage)
      [8]:  position channel = i / (L-1) for real positions, 0 for padding

    Returns None if the HELM string is unparseable.
    """
    parsed = Helm.parse(helm_str)
    if parsed is None:
        return None

    n_channels = 9 if add_position else 8
    arr = np.zeros((max_len, n_channels), dtype=np.float32)
    length = min(parsed.length, max_len)

    for i in range(length):
        base = parsed.bases[i]
        sugar = parsed.sugars[i]

        if base in BASE_IDX:
            arr[i, BASE_IDX[base]] = 1.0
        if sugar in SUGAR_IDX:
            arr[i, N_BASE + SUGAR_IDX[sugar]] = 1.0

        # Backbone: PS flag assigned to 5' nucleotide
        if i < len(parsed.backbones) and parsed.backbones[i] == "PS":
            arr[i, N_BASE + N_SUGAR] = 1.0

    if add_position and length > 1:
        for i in range(length):
            arr[i, 8] = i / (length - 1)

    return arr


def encode_batch(helms, max_len: int = 21, add_position: bool = False):
    """Encode a list/Series of HELM strings, returning (arrays, valid_mask).

    Returns:
        encoded: np.ndarray of shape (N, max_len, C) for valid entries
        valid_mask: boolean array of length len(helms)
    """
    results = []
    valid = []
    for h in helms:
        arr = encode_helm(h, max_len=max_len, add_position=add_position)
        if arr is not None:
            results.append(arr)
            valid.append(True)
        else:
            valid.append(False)
    valid_mask = np.array(valid)
    if results:
        encoded = np.stack(results, axis=0)
    else:
        n_channels = 9 if add_position else 8
        encoded = np.zeros((0, max_len, n_channels), dtype=np.float32)
    return encoded, valid_mask


# ── Model ────────────────────────────────────────────────────────

class OligoAITox(nn.Module):
    """Multi-scale 1D CNN with position-weighted pooling.

    For each conv filter activation, computes 3 pooled values:
      sum(a)       — standard global sum (motif count)
      sum(a × d5)  — 3'-biased sum (motif near 3' weighs more)
      sum(a × d3)  — 5'-biased sum (motif near 5' weighs more)

    This triples features per filter with zero extra learnable params.
    k=2 filters detect dinucleotides, k=3 captures trinucleotide context.
    ~2,100 params total.
    """

    def __init__(self, in_channels: int = 8, n_filters: int = 8,
                 n_hidden: int = 32, n_cov: int = 4, dropout: float = 0.25):
        super().__init__()
        self.conv1_k2 = nn.Conv1d(in_channels, n_filters, kernel_size=2, padding=0)
        self.conv1_k3 = nn.Conv1d(in_channels, n_filters, kernel_size=3, padding=0)
        # 3 pools × n_filters × 2 branches
        self.hidden = nn.Linear(6 * n_filters + n_cov, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(n_hidden, 1)
        self.n_filters = n_filters

    @staticmethod
    def _pos_pool(act: torch.Tensor, L_orig: int) -> torch.Tensor:
        """Position-weighted pooling over conv activations.

        Args:
            act: (batch, F, L') conv output after ReLU
            L_orig: original sequence length (before conv)

        Returns:
            (batch, 3F) — [sum, d5-weighted sum, d3-weighted sum]
        """
        L_out = act.shape[2]
        pos = torch.arange(L_out, device=act.device, dtype=act.dtype)
        d5 = pos / max(L_orig - 1, 1)   # 0 at 5', 1 at 3'
        d3 = 1.0 - d5                    # 1 at 5', 0 at 3'
        s = act.sum(2)                                        # (B, F)
        s_d5 = (act * d5.unsqueeze(0).unsqueeze(0)).sum(2)    # (B, F)
        s_d3 = (act * d3.unsqueeze(0).unsqueeze(0)).sum(2)    # (B, F)
        return torch.cat([s, s_d5, s_d3], dim=1)              # (B, 3F)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract position-pooled filter activations.

        Args:
            x: (batch, L, C) sequence tensor

        Returns:
            (batch, 6 * n_filters) position-pooled features
        """
        h = x.transpose(1, 2)  # (batch, C, L)
        h2 = torch.relu(self.conv1_k2(h))  # (batch, F, L-1)
        h3 = torch.relu(self.conv1_k3(h))  # (batch, F, L-2)
        L = x.shape[1]
        p2 = self._pos_pool(h2, L)  # (batch, 3F)
        p3 = self._pos_pool(h3, L)  # (batch, 3F)
        return torch.cat([p2, p3], dim=1)  # (batch, 6F)

    def forward(self, x: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, C) sequence tensor
            covariates: (batch, n_cov) dosing covariates

        Returns:
            (batch,) probability of high toxicity
        """
        h = self._features(x)
        h = torch.cat([h, covariates], dim=1)
        h = self.dropout(torch.relu(self.hidden(h)))
        return torch.sigmoid(self.output(h)).squeeze(-1)

    def logits(self, x: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        """Raw logits (for BCEWithLogitsLoss during training)."""
        h = self._features(x)
        h = torch.cat([h, covariates], dim=1)
        h = self.dropout(torch.relu(self.hidden(h)))
        return self.output(h).squeeze(-1)


# ── Training ─────────────────────────────────────────────────────

def _optimal_threshold(y_true, y_proba):
    """Find threshold maximising Youden's J."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def train_and_evaluate(
    helms,
    y: np.ndarray,
    groups: np.ndarray,
    covariates: np.ndarray,
    n_splits: int = 5,
    max_len: int = 21,
    lr: float = 2e-3,
    max_epochs: int = 300,
    patience: int = 30,
) -> dict | None:
    """Train OligoAI-tox with GroupKFold CV.

    Args:
        helms: array-like of HELM strings
        y: binary labels (1=high toxicity, 0=low)
        groups: group labels for GroupKFold
        covariates: (N, n_cov) dosing covariates
        n_splits: number of CV folds
        max_len: max sequence length for encoding

    Returns:
        dict with accuracy, sensitivity, specificity, auc, predictions,
        confusion, filters, hidden/output weights — or None if insufficient data.
    """
    y = np.asarray(y, dtype=np.float32)
    groups = np.asarray(groups)
    covariates = np.asarray(covariates, dtype=np.float32)

    n_high = int(y.sum())
    n_low = int(len(y) - n_high)
    if n_high < 10 or n_low < 10:
        return None

    # Encode sequences
    encoded, valid_mask = encode_batch(helms, max_len=max_len)
    if valid_mask.sum() < 50:
        return None

    # Filter to valid (encoded already contains only valid entries)
    y_valid = y[valid_mask]
    groups_valid = groups[valid_mask]
    cov_valid = covariates[valid_mask]

    n_groups = len(np.unique(groups_valid))
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        return None

    in_channels = 8
    n_cov = cov_valid.shape[1]

    # Device selection: MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pos_weight = torch.tensor([n_low / max(n_high, 1)], dtype=torch.float32, device=device)

    all_preds = np.full(len(y_valid), np.nan, dtype=np.float64)
    all_filters_k2 = []
    all_filters_k3 = []
    all_hidden_weights = []
    all_output_weights = []

    gkf = GroupKFold(n_splits=actual_splits)

    for train_idx, test_idx in gkf.split(encoded, y_valid, groups_valid):
        X_train = torch.from_numpy(encoded[train_idx]).to(device)
        X_test = torch.from_numpy(encoded[test_idx]).to(device)
        y_train = torch.from_numpy(y_valid[train_idx]).to(device)
        cov_train = torch.from_numpy(cov_valid[train_idx]).to(device)
        cov_test = torch.from_numpy(cov_valid[test_idx]).to(device)

        model = OligoAITox(n_cov=n_cov).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Train with early stopping
        best_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()

            logits = model.logits(X_train, cov_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                wait = 0
                if device.type == "mps":
                    torch.mps.synchronize()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.cpu()
            model.load_state_dict(best_state)

        # Predict on test fold (on CPU for numpy conversion)
        model.cpu()
        model.eval()
        with torch.no_grad():
            proba = model(X_test.cpu(), cov_test.cpu()).numpy()
        all_preds[test_idx] = proba

        # Save weights for interpretability
        all_filters_k2.append(model.conv1_k2.weight.detach().numpy().copy())
        all_filters_k3.append(model.conv1_k3.weight.detach().numpy().copy())
        all_hidden_weights.append(model.hidden.weight.detach().numpy().copy())
        all_output_weights.append(model.output.weight.detach().numpy().squeeze(0).copy())

    # Aggregate metrics
    try:
        auc = roc_auc_score(y_valid, all_preds)
    except ValueError:
        auc = np.nan

    try:
        threshold = _optimal_threshold(y_valid, all_preds)
    except ValueError:
        threshold = 0.5

    pred_labels = (all_preds > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, pred_labels).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    _, pval = fisher_exact([[tp, fn], [fp, tn]])

    # Average weights across folds for visualization
    avg_filters_k2 = np.mean(all_filters_k2, axis=0)   # (8, C, 2)
    avg_filters_k3 = np.mean(all_filters_k3, axis=0)   # (8, C, 3)
    avg_hidden = np.mean(all_hidden_weights, axis=0)     # (32, 48+n_cov)
    avg_output = np.mean(all_output_weights, axis=0)     # (32,)

    return {
        "n": int(len(y_valid)),
        "n_high": int(y_valid.sum()),
        "n_low": int(len(y_valid) - y_valid.sum()),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "threshold": float(threshold),
        "auc": float(auc),
        "p_value": float(pval),
        "predictions": all_preds,
        "labels": y_valid,  # filtered to valid HELM parses
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "filters_k2": avg_filters_k2,       # (8, C, 2)
        "filters_k3": avg_filters_k3,       # (8, C, 3)
        "hidden_weights": avg_hidden,        # (32, 48+n_cov)
        "output_weights": avg_output,        # (32,)
        "n_groups": n_groups,
    }
