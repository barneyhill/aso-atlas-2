"""Hyperparameter search for Hagedorn RF baselines (fair comparison with tuned CNN)."""

import itertools
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from analyses.utils.models import MODELS, prepare_data


def load_alt():
    from analyses.logic.models.hepatotox import load_and_filter, assign_groups, build_covariates, LITERATURE_ULN, THRESHOLDS
    df = load_and_filter()
    groups = assign_groups(df)
    cov_df = build_covariates(df)
    uln = LITERATURE_ULN["ALT"]
    high_mult, low_mult = THRESHOLDS["ALT"]
    y_full = pd.Series(index=df.index, dtype=str)
    y_full[df["mean_ALT"] >= high_mult * uln] = "high"
    y_full[df["mean_ALT"] < low_mult * uln] = "low"
    mask = y_full.isin(["high", "low"]) & groups.notna()
    return df[mask], (y_full[mask] == "high").astype(int), groups[mask], cov_df.loc[mask]


def load_fob():
    from analyses.logic.models.neurotox import load_and_filter, assign_groups, binary_labels
    df = load_and_filter()
    groups = assign_groups(df)
    y_labels = binary_labels(df)
    mask = y_labels.isin(["high", "low"]) & groups.notna()
    return df[mask], (y_labels[mask] == "high").astype(int), groups[mask], None


def run_rf_cv(X, y, groups, *, n_estimators, max_depth, min_samples_leaf, max_features, n_splits=5):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    preds = np.full(len(y), np.nan)

    for tr, te in gkf.split(X, y, groups):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = rf.predict_proba(X.iloc[te])[:, 1]

    return roc_auc_score(y, preds)


def main():
    print("Loading data...")
    alt_df, alt_y, alt_groups, alt_cov = load_alt()
    fob_df, fob_y, fob_groups, fob_cov = load_fob()

    # Best RF feature set: Dinucleotide (128) for ALT, Position (480) for FOB
    # But also test the other feature sets
    feature_sets = {
        "dinuc": "dinucleotide",  # 128 features — best for ALT
        "pos":   "position",     # 480 features — best for FOB
    }

    grid = {
        "n_estimators":    [500, 1000, 2000],
        "max_depth":       [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features":    ["sqrt", "log2", 8, 16],
    }

    all_combos = list(itertools.product(*grid.values()))
    rng = np.random.default_rng(42)
    n_samples = min(80, len(all_combos))
    chosen = rng.choice(len(all_combos), size=n_samples, replace=False)
    configs = [dict(zip(grid.keys(), all_combos[i])) for i in chosen]

    print(f"Searching {n_samples} configs across {len(all_combos)} total\n")

    # Prepare feature matrices (concat covariates for ALT, matching production)
    datasets = {}
    for fs_name, model_key in feature_sets.items():
        alt_X = pd.concat([prepare_data(alt_df, model_key), alt_cov], axis=1)
        fob_X = prepare_data(fob_df, model_key)  # FOB has no covariates
        datasets[fs_name] = {
            "ALT": (alt_X, alt_y, alt_groups),
            "FOB": (fob_X, fob_y, fob_groups),
        }

    results = []
    for i, cfg in enumerate(configs):
        for fs_name in feature_sets:
            aucs = {}
            t0 = time.time()
            for ep_name, (X, y, g) in datasets[fs_name].items():
                mf = cfg["max_features"]
                if isinstance(mf, int):
                    mf = min(mf, X.shape[1])
                aucs[ep_name] = run_rf_cv(X, y, g, n_estimators=cfg["n_estimators"],
                                           max_depth=cfg["max_depth"],
                                           min_samples_leaf=cfg["min_samples_leaf"],
                                           max_features=mf)
            dt = time.time() - t0
            avg = np.mean(list(aucs.values()))
            results.append({**cfg, "features": fs_name, **aucs, "avg": avg, "time": dt})

            print(f"[{i+1:3d}/{n_samples}] {fs_name:5s} est={cfg['n_estimators']:4d} "
                  f"depth={str(cfg['max_depth']):>4s} leaf={cfg['min_samples_leaf']:2d} "
                  f"mf={str(cfg['max_features']):>4s}  "
                  f"ALT={aucs['ALT']:.3f} FOB={aucs['FOB']:.3f} avg={avg:.3f}  ({dt:.0f}s)")

    results.sort(key=lambda r: r["avg"], reverse=True)

    print(f"\n{'='*95}")
    print(f"Top 15 configs by avg AUC:")
    print(f"{'feat':>5} {'est':>4} {'depth':>5} {'leaf':>4} {'mf':>5} {'ALT':>7} {'FOB':>7} {'avg':>7}")
    print(f"{'-'*95}")
    for r in results[:15]:
        print(f"{r['features']:>5} {r['n_estimators']:4d} {str(r['max_depth']):>5s} "
              f"{r['min_samples_leaf']:4d} {str(r['max_features']):>5s} "
              f"{r['ALT']:7.3f} {r['FOB']:7.3f} {r['avg']:7.3f}")
    print(f"{'='*95}")

    # Also show best per-endpoint
    best_alt = max(results, key=lambda r: r["ALT"])
    best_fob = max(results, key=lambda r: r["FOB"])
    print(f"\nBest ALT: {best_alt['ALT']:.3f} (feat={best_alt['features']}, "
          f"est={best_alt['n_estimators']}, depth={best_alt['max_depth']}, "
          f"leaf={best_alt['min_samples_leaf']}, mf={best_alt['max_features']})")
    print(f"Best FOB: {best_fob['FOB']:.3f} (feat={best_fob['features']}, "
          f"est={best_fob['n_estimators']}, depth={best_fob['max_depth']}, "
          f"leaf={best_fob['min_samples_leaf']}, mf={best_fob['max_features']})")

    print(f"\nCurrent RF defaults: est=1000, depth=None, leaf=1, mf=min(8,n)")
    print(f"Current RF AUCs:     ALT=0.771 (dinuc), FOB=0.902 (pos)")
    print(f"Tuned CNN:           ALT=0.803, FOB=0.918")


if __name__ == "__main__":
    main()
