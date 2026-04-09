"""
OligoGym benchmark: train all OligoGym model architectures on the four
mouse/rat × hepatic/neuro toxicity datasets.

Regression task, GroupKFold by patent, reports Spearman / R² / RMSE.
"""

import json
import logging
import os
import time
import warnings
from pathlib import Path

# Force CPU for PyTorch Lightning models (MPS causes SIGABRT on Apple Silicon)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import torch  # noqa: E402
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from analyses.utils.oligogym_adapter import (
    DATASETS,
    FEATURIZER_CONFIGS,
    MODEL_CONFIGS,
    is_compatible,
)

_root = Path(__file__).resolve().parents[3]
RESULTS_DIR = _root / "data/results"

N_SPLITS = 5


def _featurize(feat_cls, feat_kwargs, x, flatten: bool):
    """Run an OligoGym featurizer and return a numpy array."""
    feat = feat_cls(**feat_kwargs)
    X = feat.fit_transform(x)
    if isinstance(X, pd.DataFrame):
        X = X.values
    if flatten and X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    return X.astype(np.float32)


def _fold_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return {"spearman": np.nan, "r2": np.nan, "rmse": np.nan}
    rho, _ = spearmanr(yt, yp)
    return {
        "spearman": float(rho),
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
    }


def _needs_flatten(model_name: str) -> bool:
    """Sklearn models need flattened 2D input; lightning models keep 3D."""
    return model_name not in {"CNN", "CausalCNN", "Transformer"}


def _is_lightning(model_name: str) -> bool:
    return model_name in {"CNN", "MLP", "CausalCNN", "Transformer"}


def run_benchmark():
    all_results = []

    for ds_name, loader in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print("=" * 60)
        data = loader()
        n = len(data["y"])
        print(f"  N={n}, y range=[{data['y'].min():.1f}, {data['y'].max():.1f}]")

        for feat_name, (feat_cls, feat_kwargs) in FEATURIZER_CONFIGS.items():
            for model_name, (model_cls, hp_list) in MODEL_CONFIGS.items():
                if not is_compatible(feat_name, model_name):
                    continue

                for hp_kwargs in hp_list:
                    hp_str = ", ".join(f"{k}={v}" for k, v in hp_kwargs.items()
                                       if k != "verbose")
                    print(f"  {model_name} x {feat_name} [{hp_str}]...", end=" ", flush=True)
                    t0 = time.time()

                    try:
                        flatten = _needs_flatten(model_name)
                        X = _featurize(feat_cls, feat_kwargs, data["x"], flatten=flatten)
                        y = data["y"]
                        groups = data["groups"]

                        n_groups = len(np.unique(groups))
                        actual_splits = min(N_SPLITS, n_groups)
                        if actual_splits < 2:
                            print("skip (< 2 groups)")
                            continue

                        gkf = GroupKFold(n_splits=actual_splits)
                        fold_results = []

                        for fold_i, (train_idx, test_idx) in enumerate(
                            gkf.split(X, y, groups)
                        ):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]

                            # Construct model — Lightning models need input_dim
                            init_kwargs = {"task": "regression", **hp_kwargs}
                            if _is_lightning(model_name):
                                init_kwargs["input_dim"] = X.shape[-1]
                                if model_name == "Transformer":
                                    init_kwargs["seq_len"] = X.shape[1]

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                model = model_cls(**init_kwargs)
                                fit_kwargs = {}
                                if _is_lightning(model_name):
                                    fit_kwargs = {
                                        "max_epochs": 100,
                                        "batch_size": 32,
                                        "verbose": False,
                                        "accelerator": "cpu",
                                        "enable_progress_bar": False,
                                    }
                                model.fit(X_train, y_train, **fit_kwargs)
                                preds = np.asarray(model.predict(X_test)).ravel()

                            fm = _fold_metrics(y_test, preds)
                            fold_results.append(fm)

                        elapsed = time.time() - t0
                        avg = {
                            k: float(np.nanmean([f[k] for f in fold_results]))
                            for k in ["spearman", "r2", "rmse"]
                        }
                        std = {
                            f"{k}_std": float(np.nanstd([f[k] for f in fold_results]))
                            for k in ["spearman", "r2", "rmse"]
                        }

                        print(f"Spearman={avg['spearman']:.3f} "
                              f"R2={avg['r2']:.3f} "
                              f"RMSE={avg['rmse']:.1f} "
                              f"({elapsed:.0f}s)")

                        all_results.append({
                            "dataset": ds_name,
                            "featurizer": feat_name,
                            "model": model_name,
                            "hyperparams": {k: v for k, v in hp_kwargs.items()
                                            if k != "verbose"},
                            **avg,
                            **std,
                            "n_samples": n,
                            "n_folds": actual_splits,
                            "fold_metrics": fold_results,
                            "elapsed_s": round(elapsed, 1),
                        })

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"FAILED ({type(e).__name__}: {e}) ({elapsed:.0f}s)")
                        all_results.append({
                            "dataset": ds_name,
                            "featurizer": feat_name,
                            "model": model_name,
                            "hyperparams": {k: v for k, v in hp_kwargs.items()
                                            if k != "verbose"},
                            "spearman": np.nan,
                            "r2": np.nan,
                            "rmse": np.nan,
                            "error": str(e),
                            "n_samples": n,
                        })

    return all_results


def select_best(results: list[dict]) -> list[dict]:
    """Select the best HP config per model × featurizer × dataset by Spearman."""
    df = pd.DataFrame(results)
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")

    best = []
    for (ds, model), grp in df.groupby(["dataset", "model"]):
        valid = grp.dropna(subset=["spearman"])
        if valid.empty:
            continue
        best_row = valid.loc[valid["spearman"].idxmax()]
        best.append(best_row.to_dict())

    return best


def main():
    print("OligoGym Benchmark")
    print("=" * 60)
    t0 = time.time()

    all_results = run_benchmark()

    # Select best per model × dataset
    best = select_best(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "oligogym_benchmark.json"

    # Clean NaN for JSON serialization
    def clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    payload = {
        "all_results": clean(all_results),
        "best_per_model": clean(best),
        "metadata": {
            "n_folds": N_SPLITS,
            "split_strategy": "GroupKFold_patent",
            "task": "regression",
            "datasets": list(DATASETS.keys()),
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
