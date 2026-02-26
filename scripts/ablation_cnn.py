"""Quick ablation: test 3 CNN variants against base on ALT + FOB."""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from analyses.logic.models.oligoai_tox import encode_batch

# ── Device ────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Models ────────────────────────────────────────────────────────

class Base(nn.Module):
    """Current model: k=2+k=3, sum-pool, hidden layer. ~1.3K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        self.hidden = nn.Linear(32 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h)).sum(2)
        h3 = torch.relu(self.k3(h)).sum(2)
        h = torch.cat([h2, h3, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


class K234(nn.Module):
    """Adds k=4 branch for tetranucleotide patterns. ~2.1K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        self.k4 = nn.Conv1d(C, 16, 4)
        self.hidden = nn.Linear(48 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h)).sum(2)
        h3 = torch.relu(self.k3(h)).sum(2)
        h4 = torch.relu(self.k4(h)).sum(2)
        h = torch.cat([h2, h3, h4, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


class Region(nn.Module):
    """Sum-pool separately for DNA gap vs MOE wing. ~1.8K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        # 16*2 regions * 2 branches = 64
        self.hidden = nn.Linear(64 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def _region_pool(self, act, x_orig, k):
        """Pool conv activations separately by chemistry region.

        act: (batch, 16, L')  conv output
        x_orig: (batch, L, C) original input
        k: kernel size
        """
        center = k // 2
        L_out = act.shape[2]
        # Chemistry at center of each conv window
        dna = x_orig[:, center:center + L_out, 4].unsqueeze(1)   # (B, 1, L')
        moe = x_orig[:, center:center + L_out, 5].unsqueeze(1)
        h_dna = (act * dna).sum(2)  # (B, 16)
        h_moe = (act * moe).sum(2)
        return torch.cat([h_dna, h_moe], 1)  # (B, 32)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h))
        h3 = torch.relu(self.k3(h))
        r2 = self._region_pool(h2, x, 2)
        r3 = self._region_pool(h3, x, 3)
        h = torch.cat([r2, r3, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


class Dual(nn.Module):
    """Sum-pool + max-pool (counts + presence). ~1.8K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        # 16*2 pools * 2 branches = 64
        self.hidden = nn.Linear(64 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h))
        h3 = torch.relu(self.k3(h))
        s2 = h2.sum(2)
        s3 = h3.sum(2)
        m2 = h2.max(2).values
        m3 = h3.max(2).values
        h = torch.cat([s2, s3, m2, m3, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


class RegionDual(nn.Module):
    """Region-aware sum + max pooling. ~2.3K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        # 16 * 2 regions * 2 pools * 2 branches = 128
        self.hidden = nn.Linear(128 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def _region_dual_pool(self, act, x_orig, k):
        center = k // 2
        L_out = act.shape[2]
        dna = x_orig[:, center:center + L_out, 4].unsqueeze(1)
        moe = x_orig[:, center:center + L_out, 5].unsqueeze(1)
        # Sum-pool per region
        s_dna = (act * dna).sum(2)
        s_moe = (act * moe).sum(2)
        # Max-pool per region (use -inf for masked positions)
        act_dna = act * dna + (1 - dna) * (-1e9)
        act_moe = act * moe + (1 - moe) * (-1e9)
        m_dna = act_dna.max(2).values.clamp(min=0)
        m_moe = act_moe.max(2).values.clamp(min=0)
        return torch.cat([s_dna, s_moe, m_dna, m_moe], 1)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h))
        h3 = torch.relu(self.k3(h))
        r2 = self._region_dual_pool(h2, x, 2)
        r3 = self._region_dual_pool(h3, x, 3)
        h = torch.cat([r2, r3, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


class PosPool(nn.Module):
    """Position-weighted pooling: sum, d5-sum, d3-sum. ~2.3K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        # 3 pools × 16 filters × 2 branches = 96
        self.hidden = nn.Linear(96 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def _pos_pool(self, act, L_orig):
        """Position-weighted pooling over conv activations.

        act: (B, F, L') conv output
        L_orig: original sequence length (before conv)
        """
        L_out = act.shape[2]
        # Position of center of each conv window relative to full sequence
        pos = torch.arange(L_out, device=act.device).float()
        d5 = pos / max(L_orig - 1, 1)   # 0 at 5', 1 at 3'
        d3 = 1.0 - d5                    # 1 at 5', 0 at 3'
        s = act.sum(2)                                            # (B, F)
        s_d5 = (act * d5.unsqueeze(0).unsqueeze(0)).sum(2)        # (B, F)
        s_d3 = (act * d3.unsqueeze(0).unsqueeze(0)).sum(2)        # (B, F)
        return torch.cat([s, s_d5, s_d3], 1)                     # (B, 3F)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h))
        h3 = torch.relu(self.k3(h))
        L = x.shape[1]
        p2 = self._pos_pool(h2, L)  # (B, 48)
        p3 = self._pos_pool(h3, L)  # (B, 48)
        h = torch.cat([p2, p3, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


class PosPoolDual(nn.Module):
    """Position-weighted pooling + max pool. ~2.8K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        # (3 pos-pools + 1 max-pool) × 16 filters × 2 branches = 128
        self.hidden = nn.Linear(128 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def _pos_pool_dual(self, act, L_orig):
        L_out = act.shape[2]
        pos = torch.arange(L_out, device=act.device).float()
        d5 = pos / max(L_orig - 1, 1)
        d3 = 1.0 - d5
        s = act.sum(2)
        s_d5 = (act * d5.unsqueeze(0).unsqueeze(0)).sum(2)
        s_d3 = (act * d3.unsqueeze(0).unsqueeze(0)).sum(2)
        m = act.max(2).values
        return torch.cat([s, s_d5, s_d3, m], 1)  # (B, 4F)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h))
        h3 = torch.relu(self.k3(h))
        L = x.shape[1]
        p2 = self._pos_pool_dual(h2, L)  # (B, 64)
        p3 = self._pos_pool_dual(h3, L)  # (B, 64)
        h = torch.cat([p2, p3, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


class PosPoolRegion(nn.Module):
    """Position-weighted pooling × region (DNA/MOE). ~3.7K params."""
    def __init__(self, C=8, n_cov=4):
        super().__init__()
        self.k2 = nn.Conv1d(C, 16, 2)
        self.k3 = nn.Conv1d(C, 16, 3)
        # 3 pos-pools × 2 regions × 16 filters × 2 branches = 192
        self.hidden = nn.Linear(192 + n_cov, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)

    def _pos_region_pool(self, act, x_orig, k, L_orig):
        center = k // 2
        L_out = act.shape[2]
        dna = x_orig[:, center:center + L_out, 4].unsqueeze(1)  # (B, 1, L')
        moe = x_orig[:, center:center + L_out, 5].unsqueeze(1)
        pos = torch.arange(L_out, device=act.device).float()
        d5 = pos / max(L_orig - 1, 1)
        d3 = 1.0 - d5
        d5 = d5.unsqueeze(0).unsqueeze(0)  # (1, 1, L')
        d3 = d3.unsqueeze(0).unsqueeze(0)
        # Sum per region
        s_dna = (act * dna).sum(2)
        s_moe = (act * moe).sum(2)
        # d5-weighted sum per region
        d5_dna = (act * dna * d5).sum(2)
        d5_moe = (act * moe * d5).sum(2)
        # d3-weighted sum per region
        d3_dna = (act * dna * d3).sum(2)
        d3_moe = (act * moe * d3).sum(2)
        return torch.cat([s_dna, s_moe, d5_dna, d5_moe, d3_dna, d3_moe], 1)  # (B, 6F)

    def forward(self, x, cov):
        h = x.transpose(1, 2)
        h2 = torch.relu(self.k2(h))
        h3 = torch.relu(self.k3(h))
        L = x.shape[1]
        r2 = self._pos_region_pool(h2, x, 2, L)  # (B, 96)
        r3 = self._pos_region_pool(h3, x, 3, L)  # (B, 96)
        h = torch.cat([r2, r3, cov], 1)
        return self.drop(torch.relu(self.hidden(h)))

    def logits(self, x, cov):
        return self.out(self.forward(x, cov)).squeeze(-1)

    def predict(self, x, cov):
        return torch.sigmoid(self.logits(x, cov))


VARIANTS = {
    "base": Base, "k234": K234, "region": Region, "dual": Dual, "region+dual": RegionDual,
    "pospool": PosPool, "pospool+dual": PosPoolDual, "pospool+reg": PosPoolRegion,
}


# ── Training ──────────────────────────────────────────────────────

def _optimal_threshold(y, p):
    fpr, tpr, th = roc_curve(y, p)
    return float(th[np.argmax(tpr - fpr)])


def run_cv(model_cls, encoded, y, groups, cov, n_splits=5):
    n_cov = cov.shape[1]
    n_high, n_low = int(y.sum()), int(len(y) - y.sum())
    pw = torch.tensor([n_low / max(n_high, 1)], dtype=torch.float32, device=DEVICE)
    preds = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(encoded, y, groups):
        Xtr = torch.from_numpy(encoded[tr]).to(DEVICE)
        Xte = torch.from_numpy(encoded[te]).to(DEVICE)
        ytr = torch.from_numpy(y[tr]).to(DEVICE)
        ctr = torch.from_numpy(cov[tr]).to(DEVICE)
        cte = torch.from_numpy(cov[te]).to(DEVICE)

        model = model_cls(n_cov=n_cov).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)

        best_loss, wait, best_state = float("inf"), 0, None
        for _ in range(200):
            model.train()
            opt.zero_grad()
            loss = crit(model.logits(Xtr, ctr), ytr)
            loss.backward()
            opt.step()
            lv = loss.item()
            if lv < best_loss - 1e-4:
                best_loss, wait = lv, 0
                if DEVICE.type == "mps":
                    torch.mps.synchronize()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 20:
                    break

        if best_state:
            model.cpu()
            model.load_state_dict(best_state)
        model.cpu().eval()
        with torch.no_grad():
            preds[te] = model.predict(Xte.cpu(), cte.cpu()).numpy()

    auc = roc_auc_score(y, preds)
    th = _optimal_threshold(y, preds)
    pred_labels = (preds > th).astype(int)
    acc = (pred_labels == y).mean()
    return {"auc": auc, "acc": acc}


# ── Data loading ──────────────────────────────────────────────────

def load_alt():
    """Load hepatotox ALT data (same pipeline as hepatotox.py)."""
    from analyses.logic.models.hepatotox import (
        load_and_filter, assign_groups, build_covariates,
        LITERATURE_ULN, THRESHOLDS,
    )
    df = load_and_filter()
    groups = assign_groups(df)
    cov_df = build_covariates(df)

    uln = LITERATURE_ULN["ALT"]
    high_mult, low_mult = THRESHOLDS["ALT"]
    col = "mean_ALT"
    y_full = pd.Series(index=df.index, dtype=str)
    y_full[df[col] >= high_mult * uln] = "high"
    y_full[df[col] < low_mult * uln] = "low"
    mask = y_full.isin(["high", "low"]) & groups.notna()

    helms = df.loc[mask, "HELM Annotation"].values
    y = (y_full[mask] == "high").astype(int).values.astype(np.float32)
    g = groups[mask].values
    cov = cov_df.loc[mask].values.astype(np.float32)
    encoded, vm = encode_batch(helms)
    return encoded, y[vm], g[vm], cov[vm]


def load_fob():
    """Load neurotox FOB data (same pipeline as neurotox.py)."""
    from analyses.logic.models.neurotox import (
        load_and_filter, assign_groups, binary_labels,
    )
    df = load_and_filter()
    groups = assign_groups(df)
    y_labels = binary_labels(df)
    mask = y_labels.isin(["high", "low"]) & groups.notna()

    helms = df.loc[mask, "HELM Annotation"].values
    y = (y_labels[mask] == "high").astype(int).values.astype(np.float32)
    g = groups[mask].values
    cov = np.zeros((mask.sum(), 0), dtype=np.float32)
    encoded, vm = encode_batch(helms)
    return encoded, y[vm], g[vm], cov[vm]


def load_rat_alt():
    """Load rat hepatotox ALT data."""
    from analyses.logic.models.hepatotox import (
        load_and_filter_rat, assign_groups, build_covariates,
        RAT_LITERATURE_ULN, THRESHOLDS,
    )
    df = load_and_filter_rat()
    groups = assign_groups(df)
    cov_df = build_covariates(df)

    uln = RAT_LITERATURE_ULN["ALT"]
    high_mult, low_mult = THRESHOLDS["ALT"]
    col = "mean_ALT"
    y_full = pd.Series(index=df.index, dtype=str)
    y_full[df[col] >= high_mult * uln] = "high"
    y_full[df[col] < low_mult * uln] = "low"
    mask = y_full.isin(["high", "low"]) & groups.notna()

    helms = df.loc[mask, "HELM Annotation"].values
    y = (y_full[mask] == "high").astype(int).values.astype(np.float32)
    g = groups[mask].values
    cov = cov_df.loc[mask].values.astype(np.float32)
    encoded, vm = encode_batch(helms)
    return encoded, y[vm], g[vm], cov[vm]


def load_rat_fob():
    """Load rat neurotox FOB data."""
    from analyses.logic.models.neurotox import (
        load_and_filter_rat, assign_groups, binary_labels,
    )
    df = load_and_filter_rat()
    groups = assign_groups(df)
    y_labels = binary_labels(df)
    mask = y_labels.isin(["high", "low"]) & groups.notna()

    helms = df.loc[mask, "HELM Annotation"].values
    y = (y_labels[mask] == "high").astype(int).values.astype(np.float32)
    g = groups[mask].values
    cov = np.zeros((mask.sum(), 0), dtype=np.float32)
    encoded, vm = encode_batch(helms)
    return encoded, y[vm], g[vm], cov[vm]


# ── Main ──────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")

    datasets = {"ALT": load_alt, "FOB": load_fob, "rat_ALT": load_rat_alt, "rat_FOB": load_rat_fob}
    # Run dual + all pospool variants on rat too
    rat_only = {"dual": Dual, "pospool": PosPool, "pospool+dual": PosPoolDual, "pospool+reg": PosPoolRegion}
    results = []

    for ds_name, loader in datasets.items():
        print(f"Loading {ds_name}...")
        encoded, y, groups, cov = loader()
        print(f"  N={len(y)}, high={int(y.sum())}, low={int(len(y)-y.sum())}\n")

        run_variants = rat_only if ds_name.startswith("rat_") else VARIANTS
        for vname, vcls in run_variants.items():
            n_params = sum(p.numel() for p in vcls(n_cov=cov.shape[1]).parameters())
            t0 = time.time()
            r = run_cv(vcls, encoded, y, groups, cov)
            dt = time.time() - t0
            results.append({
                "endpoint": ds_name, "variant": vname,
                "params": n_params, "AUC": r["auc"], "acc": r["acc"],
                "time_s": dt,
            })
            print(f"  {vname:14s}  params={n_params:5d}  AUC={r['auc']:.3f}  acc={r['acc']:.3f}  ({dt:.0f}s)")
        print()

    # Summary table
    print("=" * 70)
    print(f"{'Endpoint':<10} {'Variant':<15} {'Params':>6} {'AUC':>7} {'Acc':>7} {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['endpoint']:<10} {r['variant']:<15} {r['params']:>6} {r['AUC']:>7.3f} {r['acc']:>7.3f} {r['time_s']:>5.0f}s")
    print("=" * 70)

    # RF baselines for reference
    print("\nRF baselines: ALT=0.771, FOB=0.895, rat_ALT=0.670, rat_FOB=0.877")


if __name__ == "__main__":
    main()
