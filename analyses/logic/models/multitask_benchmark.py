"""Multi-task Transformer benchmark: joint prediction of IV inhibition + ALT + FOB.

Shared encoder, 5 task heads (inhibition, mouse_ALT, rat_ALT, mouse_FOB, rat_FOB).
Round-robin batching for equal gradient updates per task.
Same GroupKFold by patent evaluation as single-task benchmark.
"""

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ["PYTHONUNBUFFERED"] = "1"

import torch  # noqa: E402

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

import numpy as np  # noqa: E402
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .multitask_data import (
    VIVO_LOADERS,
    encode_helms,
    gather,
    load_invitro,
)
from .multitask_model import TASK_ID, MultiTaskConfig, MultiTaskTransformer

_root = Path(__file__).resolve().parents[3]
RESULTS_DIR = _root / "data/results"

VIVO_ENDPOINTS = ["mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro"]
N_SPLITS = 5
SEED = 42

# Training hyperparameters
CFG = MultiTaskConfig()
LR = 3e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
GRAD_CLIP = 1.0


def _metrics(y_true, y_pred):
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


def _make_batches(encoded_tensors, sample_indices, y, batch_size, shuffle=True, rng=None):
    """Create list of (batch_tensors, batch_y) tuples."""
    n = len(sample_indices)
    if n == 0:
        return []
    if rng is None:
        rng = np.random.default_rng(SEED)
    order = rng.permutation(n) if shuffle else np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        enc_idx = sample_indices[idx]
        valid = enc_idx >= 0
        if valid.sum() == 0:
            continue
        idx_valid = idx[valid]
        enc_valid = enc_idx[valid]
        batch_t = gather(encoded_tensors, enc_valid)
        batch_y = torch.from_numpy(y[idx_valid])
        batches.append((batch_t, batch_y))
    return batches


def train_one_fold(
    vivo_data: dict[str, dict],
    iv_data: dict | None,
    vivo_train_idx: dict[str, np.ndarray],
    vivo_val_idx: dict[str, np.ndarray],
    encoded_tensors: dict[str, torch.Tensor],
    iv_sample_indices: np.ndarray | None,
    iv_train_mask: np.ndarray | None,
) -> tuple:
    """Train one fold of the multi-task model. Returns (model, val_metrics)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = MultiTaskTransformer(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    rng = np.random.default_rng(SEED)

    best_val_spearman = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()

        # Build batches for each task (round-robin)
        task_batches = {}
        for ep_name in VIVO_ENDPOINTS:
            train_si = vivo_data[ep_name]["_sample_idx"][vivo_train_idx[ep_name]]
            train_y = vivo_data[ep_name]["y"][vivo_train_idx[ep_name]]
            task_batches[ep_name] = _make_batches(
                encoded_tensors, train_si, train_y, BATCH_SIZE, shuffle=True, rng=rng,
            )

        if iv_data is not None and iv_sample_indices is not None and iv_train_mask is not None:
            iv_si = iv_sample_indices[iv_train_mask]
            iv_y = iv_data["y"][iv_train_mask]
            # Subsample IV to match vivo size (equal gradient updates)
            max_vivo_batches = max(len(b) for b in task_batches.values())
            n_iv_samples = max_vivo_batches * BATCH_SIZE
            if len(iv_si) > n_iv_samples:
                sub = rng.choice(len(iv_si), n_iv_samples, replace=False)
                iv_si_sub, iv_y_sub = iv_si[sub], iv_y[sub]
            else:
                iv_si_sub, iv_y_sub = iv_si, iv_y
            task_batches["inhibition"] = _make_batches(
                encoded_tensors, iv_si_sub, iv_y_sub, BATCH_SIZE, shuffle=True, rng=rng,
            )

        # Round-robin training
        all_task_names = list(task_batches.keys())
        max_batches = max(len(task_batches[t]) for t in all_task_names)
        task_iters = {t: iter(task_batches[t]) for t in all_task_names}

        epoch_losses = []
        for step in range(max_batches):
            total_loss = torch.tensor(0.0)
            n_tasks = 0

            for task_name in all_task_names:
                batch = next(task_iters[task_name], None)
                if batch is None:
                    # Cycle for smaller tasks
                    task_iters[task_name] = iter(task_batches[task_name])
                    batch = next(task_iters[task_name], None)
                    if batch is None:
                        continue

                batch_t, batch_y = batch
                pred = model(
                    batch_t["base_idx"], batch_t["sugar_idx"],
                    batch_t["backbone_idx"], batch_t["mask"],
                    task_name=task_name,
                )
                loss_t = F.mse_loss(pred, batch_y)

                # Uncertainty weighting
                tid = TASK_ID[task_name]
                precision = torch.exp(-model.log_var[tid])
                total_loss = total_loss + 0.5 * precision * loss_t + 0.5 * model.log_var[tid]
                n_tasks += 1

            if n_tasks > 0:
                total_loss = total_loss / n_tasks
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                epoch_losses.append(total_loss.item())

        scheduler.step()

        # Validate on vivo endpoints only
        model.eval()
        val_spearmans = []
        with torch.no_grad():
            for ep_name in VIVO_ENDPOINTS:
                val_si = vivo_data[ep_name]["_sample_idx"][vivo_val_idx[ep_name]]
                val_y = vivo_data[ep_name]["y"][vivo_val_idx[ep_name]]
                valid = val_si >= 0
                if valid.sum() < 5:
                    continue
                batch_t = gather(encoded_tensors, val_si[valid])
                pred = model(
                    batch_t["base_idx"], batch_t["sugar_idx"],
                    batch_t["backbone_idx"], batch_t["mask"],
                    task_name=ep_name,
                ).numpy()
                rho, _ = spearmanr(val_y[valid], pred)
                if np.isfinite(rho):
                    val_spearmans.append(rho)

        mean_val = np.mean(val_spearmans) if val_spearmans else -1.0

        if mean_val > best_val_spearman:
            best_val_spearman = mean_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_spearman


def run_multitask_benchmark():
    print("Multi-task Transformer Benchmark")
    print("=" * 60)
    t0 = time.time()

    # Load all data
    print("Loading data...")
    vivo_data = {name: loader() for name, loader in VIVO_LOADERS.items()}
    iv_data = load_invitro()

    for name, d in vivo_data.items():
        print(f"  {name}: N={len(d['y'])}, groups={len(np.unique(d['groups']))}")
    print(f"  inhibition (IV): N={len(iv_data['y'])}, groups={len(np.unique(iv_data['groups']))}")

    # Encode all HELMs at once
    print("Encoding HELMs...")
    all_helms = np.concatenate(
        [d["helms"] for d in vivo_data.values()] + [iv_data["helms"]]
    )
    encoded_tensors, helm_to_idx, _ = encode_helms(all_helms, max_len=CFG.max_len)
    print(f"  {len(all_helms)} total → {len(helm_to_idx)} unique valid")

    # Build per-dataset sample index arrays (into encoded_tensors)
    for name, d in vivo_data.items():
        d["_sample_idx"] = np.array([helm_to_idx.get(h, -1) for h in d["helms"]])
    iv_sample_indices = np.array([helm_to_idx.get(h, -1) for h in iv_data["helms"]])

    # Collect vivo test patents for IV leakage prevention
    vivo_patents_by_fold = {}  # fold_key -> set of patents in test

    # Run 5-fold CV per endpoint
    results = {ep: [] for ep in VIVO_ENDPOINTS}

    # Build folds for each vivo endpoint (same GroupKFold as single-task benchmark)
    endpoint_folds = {}
    for ep_name in VIVO_ENDPOINTS:
        d = vivo_data[ep_name]
        gkf = GroupKFold(n_splits=N_SPLITS)
        folds = list(gkf.split(d["y"], d["y"], d["groups"]))
        endpoint_folds[ep_name] = folds

    # Group endpoints by family for aligned fold evaluation
    families = {
        "hepatic": ["mouse_hepatic", "rat_hepatic"],
        "neuro": ["mouse_neuro", "rat_neuro"],
    }

    for family_name, family_endpoints in families.items():
        print(f"\n{'='*60}")
        print(f"Family: {family_name}")
        print("=" * 60)

        for fold_i in range(N_SPLITS):
            print(f"\n  Fold {fold_i+1}/{N_SPLITS}")

            # Train/test splits for this family's endpoints
            vivo_train_idx = {}
            vivo_val_idx = {}
            test_patents = set()

            for ep_name in family_endpoints:
                train_idx, test_idx = endpoint_folds[ep_name][fold_i]
                # Use 10% of train as validation
                n_train = len(train_idx)
                rng = np.random.default_rng(SEED + fold_i)
                perm = rng.permutation(n_train)
                n_val = max(1, n_train // 10)
                vivo_val_idx[ep_name] = train_idx[perm[:n_val]]
                vivo_train_idx[ep_name] = train_idx[perm[n_val:]]

                # Collect test patents for IV leakage prevention
                test_patents.update(vivo_data[ep_name]["groups"][test_idx])

            # Other family's endpoints: use ALL data as training
            other_family = [f for f_name, eps in families.items()
                           if f_name != family_name for f in eps]
            for ep_name in other_family:
                d = vivo_data[ep_name]
                all_idx = np.arange(len(d["y"]))
                vivo_train_idx[ep_name] = all_idx
                vivo_val_idx[ep_name] = np.array([], dtype=int)

            # IV data: exclude patents in vivo test fold (prevent leakage)
            iv_train_mask = np.array([g not in test_patents for g in iv_data["groups"]])
            n_iv_train = iv_train_mask.sum()
            print(f"    IV train: {n_iv_train}/{len(iv_data['y'])} "
                  f"({len(iv_data['y']) - n_iv_train} excluded for leakage prevention)")

            # Train
            model, val_spearman = train_one_fold(
                vivo_data=vivo_data,
                iv_data=iv_data,
                vivo_train_idx=vivo_train_idx,
                vivo_val_idx=vivo_val_idx,
                encoded_tensors=encoded_tensors,
                iv_sample_indices=iv_sample_indices,
                iv_train_mask=iv_train_mask,
            )
            print(f"    Val mean Spearman: {val_spearman:.3f}")

            # Evaluate on this family's test sets
            model.eval()
            with torch.no_grad():
                for ep_name in family_endpoints:
                    _, test_idx = endpoint_folds[ep_name][fold_i]
                    d = vivo_data[ep_name]
                    test_si = d["_sample_idx"][test_idx]
                    test_y = d["y"][test_idx]
                    valid = test_si >= 0
                    if valid.sum() < 5:
                        results[ep_name].append({"spearman": np.nan, "r2": np.nan, "rmse": np.nan})
                        continue
                    batch_t = gather(encoded_tensors, test_si[valid])
                    pred = model(
                        batch_t["base_idx"], batch_t["sugar_idx"],
                        batch_t["backbone_idx"], batch_t["mask"],
                        task_name=ep_name,
                    ).numpy()
                    m = _metrics(test_y[valid], pred)
                    results[ep_name].append(m)
                    print(f"    {ep_name}: Spearman={m['spearman']:.3f}")

    # Aggregate results
    print(f"\n{'='*60}")
    print("Multi-task Transformer Results (mean ± std across folds)")
    print("=" * 60)

    benchmark_rows = []
    for ep_name in VIVO_ENDPOINTS:
        folds = results[ep_name]
        spearmans = [f["spearman"] for f in folds if np.isfinite(f.get("spearman", np.nan))]
        r2s = [f["r2"] for f in folds if np.isfinite(f.get("r2", np.nan))]
        rmses = [f["rmse"] for f in folds if np.isfinite(f.get("rmse", np.nan))]

        avg_s = float(np.mean(spearmans)) if spearmans else np.nan
        std_s = float(np.std(spearmans)) if spearmans else np.nan
        avg_r2 = float(np.mean(r2s)) if r2s else np.nan
        avg_rmse = float(np.mean(rmses)) if rmses else np.nan

        print(f"  {ep_name}: Spearman={avg_s:.3f}±{std_s:.3f}, R²={avg_r2:.3f}, RMSE={avg_rmse:.1f}")

        benchmark_rows.append({
            "dataset": ep_name,
            "featurizer": "Factored",
            "model": "Multi-task Transformer",
            "hyperparams": {},
            "spearman": avg_s if np.isfinite(avg_s) else None,
            "spearman_std": std_s if np.isfinite(std_s) else None,
            "r2": avg_r2 if np.isfinite(avg_r2) else None,
            "rmse": avg_rmse if np.isfinite(avg_rmse) else None,
            "fold_metrics": folds,
            "n_folds": N_SPLITS,
        })

    # Append to oligogym_benchmark.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_path = RESULTS_DIR / "oligogym_benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
    else:
        bench = {"all_results": [], "best_per_model": [], "metadata": {}}

    # Remove old multi-task entries
    bench["all_results"] = [r for r in bench["all_results"]
                            if r.get("model") != "Multi-task Transformer"]
    bench["best_per_model"] = [r for r in bench["best_per_model"]
                               if r.get("model") != "Multi-task Transformer"]

    # Clean NaN for JSON
    def clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    bench["all_results"].extend(clean(benchmark_rows))
    bench["best_per_model"].extend(clean(benchmark_rows))

    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"\nAppended to {bench_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_multitask_benchmark()


if __name__ == "__main__":
    main()
