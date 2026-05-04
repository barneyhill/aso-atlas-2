"""
OligoGym benchmark: train OligoGym model architectures on per-species
hepatic/neuro toxicity datasets.

Per-species training with HELM-level dedup. Hepatic models include
dosage covariates. Reports Spearman / R² / RMSE.
"""

import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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

MAX_WORKERS = int(os.environ.get("BENCH_WORKERS", min(16, os.cpu_count() or 4)))

from analyses.utils.oligogym_adapter import (
    DATASETS,
    FEATURIZER_CONFIGS,
    MODEL_CONFIGS,
    is_compatible,
)

_root = Path(__file__).resolve().parents[3]
RESULTS_DIR = _root / "data/results"

N_SPLITS = 5
_ds_filter = os.environ.get("ONLY_DATASETS", "").strip()
ONLY_DATASETS = set(_ds_filter.split(",")) if _ds_filter else None


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


def _is_lightning(model_name: str) -> bool:
    return model_name in {"MLP"}


def _run_one_job(job: dict) -> dict:
    """Train one (dataset, featurizer, model) combo with 5-fold CV.

    Designed to run in a worker process — imports are re-done at the
    top level so the function is pickle-safe.
    """
    import warnings
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold

    ds_name = job["ds_name"]
    feat_name = job["feat_name"]
    model_name = job["model_name"]
    hp_kwargs = job["hp_kwargs"]
    X = job["X"]
    y = job["y"]
    groups = job["groups"]
    conditions = job.get("conditions")
    n = len(y)

    t0 = time.time()
    try:
        n_groups = len(np.unique(groups))
        actual_splits = min(N_SPLITS, n_groups)
        if actual_splits < 2:
            return {"skip": True, "ds_name": ds_name, "feat_name": feat_name,
                    "model_name": model_name}

        from analyses.utils.oligogym_adapter import MODEL_CONFIGS
        model_cls = MODEL_CONFIGS[model_name][0]

        gkf = GroupKFold(n_splits=actual_splits)
        fold_results = []

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            init_kwargs = {"task": "regression", **hp_kwargs}
            if _is_lightning(model_name):
                init_kwargs["input_dim"] = X.shape[-1]

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
                        "logger": False,
                    }
                model.fit(X_train, y_train, **fit_kwargs)
                preds = np.asarray(model.predict(X_test)).ravel()

            fm = _fold_metrics(y_test, preds)
            fm["y_true"] = y_test.astype(float).tolist()
            fm["y_pred"] = preds.astype(float).tolist()
            fm["groups_test"] = (
                groups[test_idx].tolist()
                if hasattr(groups, "tolist")
                else list(groups[test_idx])
            )
            if conditions is not None:
                fm["conditions"] = {
                    k: v[test_idx].tolist()
                    for k, v in conditions.items()
                }
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

        return {
            "dataset": ds_name,
            "featurizer": feat_name,
            "model": model_name,
            "hyperparams": {k: v for k, v in hp_kwargs.items() if k != "verbose"},
            **avg,
            **std,
            "n_samples": n,
            "n_folds": actual_splits,
            "fold_metrics": fold_results,
            "elapsed_s": round(elapsed, 1),
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "dataset": ds_name,
            "featurizer": feat_name,
            "model": model_name,
            "hyperparams": {k: v for k, v in hp_kwargs.items() if k != "verbose"},
            "spearman": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "error": str(e),
            "n_samples": n,
        }


def run_benchmark():
    # Phase 1: load data and featurize (serial — fast, I/O-bound)
    jobs = []
    for ds_name, loader in DATASETS.items():
        if ONLY_DATASETS is not None and ds_name not in ONLY_DATASETS:
            print(f"\nSkipping {ds_name} (not in ONLY_DATASETS)")
            continue
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print("=" * 60)
        data = loader()
        n = len(data["y"])
        has_covariates = "covariates" in data
        conditions = data.get("conditions")
        print(f"  N={n}, y range=[{data['y'].min():.1f}, {data['y'].max():.1f}]"
              f"{', with covariates' if has_covariates else ''}")

        for feat_name, (feat_cls, feat_kwargs) in FEATURIZER_CONFIGS.items():
            X = _featurize(feat_cls, feat_kwargs, data["x"], flatten=True)
            if has_covariates:
                X = np.column_stack([X, data["covariates"]])

            for model_name, (model_cls, hp_list) in MODEL_CONFIGS.items():
                if not is_compatible(feat_name, model_name):
                    continue
                for hp_kwargs in hp_list:
                    jobs.append({
                        "ds_name": ds_name,
                        "feat_name": feat_name,
                        "model_name": model_name,
                        "hp_kwargs": hp_kwargs,
                        "X": X,
                        "y": data["y"],
                        "groups": data["groups"],
                        "conditions": conditions,
                    })

    # Phase 2: train all models in parallel
    print(f"\nDispatching {len(jobs)} jobs across {MAX_WORKERS} workers...")
    all_results = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one_job, job): job for job in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            if result.get("skip"):
                continue
            label = f"{result['dataset']} / {result['model']} / {result.get('featurizer', '?')}"
            if "error" in result:
                print(f"  FAILED {label}: {result['error']}")
            else:
                print(f"  {label}: Spearman={result['spearman']:.3f} "
                      f"({result.get('elapsed_s', 0):.0f}s)")
            all_results.append(result)

    print(f"\nAll jobs done in {time.time() - t_start:.0f}s")
    return all_results


def select_best(results: list[dict]) -> list[dict]:
    """Select the best HP config per model × dataset by Spearman."""
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
    print("OligoGym Benchmark (per-species, HELM dedup, dosage covariates)")
    print("=" * 60)
    t0 = time.time()

    all_results = run_benchmark()

    best = select_best(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "oligogym_benchmark.json"

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

    new_results = clean(all_results)
    new_best = clean(best)

    if ONLY_DATASETS is not None and out_path.exists():
        prev = json.loads(out_path.read_text())
        touched = {(r["dataset"], r["model"]) for r in new_results}
        kept = [r for r in prev.get("all_results", [])
                if (r.get("dataset"), r.get("model")) not in touched]
        new_results = kept + new_results
        touched_best = {(r["dataset"], r["model"]) for r in new_best}
        kept_best = [r for r in prev.get("best_per_model", [])
                     if (r.get("dataset"), r.get("model")) not in touched_best]
        new_best = kept_best + new_best

    payload = {
        "all_results": new_results,
        "best_per_model": new_best,
        "metadata": {
            "n_folds": N_SPLITS,
            "split_strategy": "GroupKFold_HELM",
            "task": "regression",
            "training": "per_species",
            "covariates": "dosage (hepatic only)",
            "datasets": list(DATASETS.keys()),
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
