"""Species-transfer ablation for hepatic and neuro toxicity.

For each organ, evaluate BOTH transfer directions:
    rat test:   rat-only (baseline) vs mouse-only (transfer)
    mouse test: mouse-only (baseline) vs rat-only (transfer)

Each direction uses GroupKFold on the test species' patents so baseline
and transfer are evaluated on identical test rows. Writes per-fold ρ
to JSON for plotting + paper text.

Finding: hepatotoxicity does not transfer across species in either
direction; neurotoxicity transfers near-symmetrically. This justifies
using independent per-endpoint tox models over pooled-species models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

_root = Path(__file__).resolve().parents[2]
_data_dir = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"

N_SPLITS = 5
SEED = 42
DEFAULT_RF = dict(n_estimators=500, max_depth=20, n_jobs=-1, random_state=SEED)

BASES = ["A", "C", "G", "T", "U", "5meC"]
SUGARS = ["MOE", "DNA", "LNA", "cEt", "fR", "OMe", "RNA"]
BACKBONES = ["PS", "PO"]
MAX_LEN = 30


def featurise(helms: Iterable[str]) -> np.ndarray:
    """Position-aware one-hot (base, sugar, backbone) + dinuc counts + length."""
    helms = list(helms)
    n = len(helms)
    n_base = MAX_LEN * len(BASES)
    n_sugar = MAX_LEN * len(SUGARS)
    n_bb = (MAX_LEN - 1) * len(BACKBONES)
    n_dinuc = len(BASES) ** 2
    n_feat = n_base + n_sugar + n_bb + n_dinuc + 1
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, h in enumerate(helms):
        p = Helm.parse(h)
        if p is None:
            continue
        L = min(p.length, MAX_LEN)
        if L < 2:
            continue
        for j in range(L):
            b = p.bases[j]
            if b in BASES:
                X[i, j * len(BASES) + BASES.index(b)] = 1
            s = p.sugars[j]
            if s in SUGARS:
                X[i, n_base + j * len(SUGARS) + SUGARS.index(s)] = 1
        for j in range(min(L - 1, MAX_LEN - 1)):
            bb = p.backbones[j]
            if bb in BACKBONES:
                X[i, n_base + n_sugar + j * len(BACKBONES) + BACKBONES.index(bb)] = 1
        dinuc_start = n_base + n_sugar + n_bb
        for j in range(L - 1):
            b1, b2 = p.bases[j], p.bases[j + 1]
            if b1 in BASES and b2 in BASES:
                X[i, dinuc_start + BASES.index(b1) * len(BASES) + BASES.index(b2)] += 1
        X[i, dinuc_start:dinuc_start + n_dinuc] /= max(L - 1, 1)
        X[i, -1] = L
    return X


def _hepatic(species: str) -> dict:
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == species) & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["y"] = df["ALT"].apply(mean_of_array)
    df = df[df["y"].notna() & df["USPTO ID"].notna()].reset_index(drop=True)
    dose_cols = ["dosage_mg_per_kg", "num_doses", "dosing_period_days"]
    dose_cov = np.column_stack([
        df[col].fillna(df[col].median()).values.astype(np.float32)
        for col in dose_cols
    ])
    return {
        "helms": df["HELM Annotation"].values,
        "y": df["y"].values.astype(float),
        "groups": df["USPTO ID"].apply(lambda x: f"patent_{x}").values,
        "covariates": dose_cov,
    }


def _neuro(species: str, dose: int, admin: str | None = None,
           latency: int | None = None) -> dict:
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    filt = (df["species"] == species) & (df["dosage_ug"] == dose) & df["HELM Annotation"].apply(Helm.valid_chemistry)
    if admin:
        filt = filt & (df["administration_method"] == admin)
    if latency:
        filt = filt & (df["latency_time_hours"] == latency)
    df = df[filt].copy()
    df["y"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["y"].notna() & df["USPTO ID"].notna()].reset_index(drop=True)
    return {
        "helms": df["HELM Annotation"].values,
        "y": df["y"].values.astype(float),
        "groups": df["USPTO ID"].apply(lambda x: f"patent_{x}").values,
    }


ORGANS = {
    "hepatic": (lambda: _hepatic("mouse"), lambda: _hepatic("rat"), True),
    "neuro": (lambda: _neuro("Mouse", 700, admin="ICV"),
              lambda: _neuro("Rat", 3000, latency=3), False),
}


def _build(helms, covariates=None):
    X = featurise(helms)
    if covariates is not None:
        X = np.hstack([X, covariates])
    return X


def _evaluate_direction(X_test_sp, y_test_sp, g_test_sp,
                        X_train_sp, y_train_sp, g_train_sp):
    """Baseline (self-species) + transfer (cross-species) on the SAME folds."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X_test_sp, y_test_sp, g_test_sp))
    baseline_rho, transfer_rho = [], []
    for tr_te, te in folds:
        Xtr_self, ytr_self = X_test_sp[tr_te], y_test_sp[tr_te]
        Xte, yte = X_test_sp[te], y_test_sp[te]
        te_groups = set(g_test_sp[te])
        keep = np.array([g not in te_groups for g in g_train_sp])
        Xtr_cross, ytr_cross = X_train_sp[keep], y_train_sp[keep]

        pb = RandomForestRegressor(**DEFAULT_RF).fit(Xtr_self, ytr_self).predict(Xte)
        pt = RandomForestRegressor(**DEFAULT_RF).fit(Xtr_cross, ytr_cross).predict(Xte)
        rb, _ = spearmanr(yte, pb); rt, _ = spearmanr(yte, pt)
        baseline_rho.append(float(rb) if np.isfinite(rb) else float("nan"))
        transfer_rho.append(float(rt) if np.isfinite(rt) else float("nan"))
    return baseline_rho, transfer_rho


def _summ(lst):
    arr = np.asarray(lst, dtype=float)
    return {
        "per_fold": list(lst),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)),
    }


def run_ablation() -> dict:
    out: dict = {}
    for organ, (m_loader, r_loader, has_dose) in ORGANS.items():
        m = m_loader()
        r = r_loader()
        Xm = _build(m["helms"], m.get("covariates") if has_dose else None)
        Xr = _build(r["helms"], r.get("covariates") if has_dose else None)
        ym, yr = m["y"], r["y"]
        gm, gr = m["groups"], r["groups"]

        rat_self, rat_cross = _evaluate_direction(Xr, yr, gr, Xm, ym, gm)
        mouse_self, mouse_cross = _evaluate_direction(Xm, ym, gm, Xr, yr, gr)

        organ_out = {
            "rat_test": {"rat_only": _summ(rat_self),
                         "mouse_only": _summ(rat_cross)},
            "mouse_test": {"mouse_only": _summ(mouse_self),
                           "rat_only": _summ(mouse_cross)},
        }
        out[organ] = organ_out

        print(f"\n{organ}:")
        print(f"  rat test:    baseline ρ={organ_out['rat_test']['rat_only']['mean']:.3f} ± {organ_out['rat_test']['rat_only']['std']:.3f}")
        print(f"               transfer ρ={organ_out['rat_test']['mouse_only']['mean']:.3f} ± {organ_out['rat_test']['mouse_only']['std']:.3f}")
        print(f"  mouse test:  baseline ρ={organ_out['mouse_test']['mouse_only']['mean']:.3f} ± {organ_out['mouse_test']['mouse_only']['std']:.3f}")
        print(f"               transfer ρ={organ_out['mouse_test']['rat_only']['mean']:.3f} ± {organ_out['mouse_test']['rat_only']['std']:.3f}")
    return out


def main():
    res = run_ablation()
    out_path = RESULTS_DIR / "species_transfer.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
