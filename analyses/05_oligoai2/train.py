"""Training loop for OligoAI2 multi-task ASO regression model.

3-phase training:
  Phase 1: Pre-train on in vitro (~249K samples) with CCC loss
  Phase 2: Train in vivo heads (~32K samples) with multi-task CCC
  Phase 3: Joint end-to-end fine-tuning (optional)
"""

import argparse
import math
import time

import warnings

import numpy as np
import torch
from scipy import stats
from torch.utils.data import Dataset

from .data import TASK_NAMES, N_TASKS, load_all, move_to, iter_batches
from .model import ModelConfig, OligoAI, ccc_loss, multitask_loss


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Phase 1: Pre-train on In Vitro ─────────────────────────────────────

def train_phase1(model, train_ds, val_ds, device, epochs=30, bs=512, lr=3e-4,
                 patience=5):
    """Phase 1: Pre-train encoder + bottleneck + in_vitro head on inhibition data."""
    n_train = math.ceil(len(train_ds) / bs)
    n_val = math.ceil(len(val_ds) / bs)
    print(f"\n{'='*60}")
    print(f"Phase 1: Pre-train on in vitro ({len(train_ds):,} samples, {n_train} batches)")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_ccc = -float("inf")
    best_state = None
    stale = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss_acc = torch.zeros(1, device=device)
        n_batches = 0

        for batch in iter_batches(train_ds, bs):
            pred = model.forward_invitro(batch)
            loss = ccc_loss(pred, batch["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_acc += loss.detach()
            n_batches += 1

        scheduler.step()
        t_train = time.time() - t0

        # Validation
        t_val = time.time()
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in iter_batches(val_ds, bs, shuffle=False):
                pred = model.forward_invitro(batch)
                val_preds.append(pred)
                val_targets.append(batch["target"])

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_ccc = 1.0 - ccc_loss(val_preds, val_targets).item()
        t_val = time.time() - t_val

        print(
            f"  Epoch {epoch+1:3d}/{epochs}: "
            f"train_loss={train_loss_acc.item()/n_batches:.4f}  "
            f"val_CCC={val_ccc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"(train={t_train:.1f}s val={t_val:.1f}s)"
        )

        if val_ccc > best_val_ccc:
            best_val_ccc = val_ccc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val CCC = {best_val_ccc:.4f})")

    return best_val_ccc


# ── Phase 2: Train In Vivo Heads ────────────────────────────────────────

def train_phase2(model, train_ds, val_ds, device, epochs=100, bs=128,
                 encoder_lr=3e-6, head_lr=3e-4, patience=15):
    """Phase 2: Train in vivo heads with frozen/slow encoder."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Train in vivo heads ({len(train_ds):,} samples, bs={bs})")
    print(f"{'='*60}")

    if len(train_ds) == 0:
        print("  No in vivo training data, skipping.")
        return 0.0

    # Differential LR: encoder slow, heads fast
    encoder_params = list(model.encoder.parameters()) + list(model.bottleneck.parameters())
    head_params = (
        list(model.cov_hepatic.parameters())
        + list(model.cov_neuro.parameters())
        + list(model.head.parameters())
        + [model.log_var]
    )
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss_acc = torch.zeros(1, device=device)
        n_batches = 0

        for batch in iter_batches(train_ds, bs):
            pred = model.forward_invivo(batch)
            loss, per_task = multitask_loss(pred, batch["target"], batch["task_id"], model.log_var)
            if not per_task:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_acc += loss.detach()
            n_batches += 1

        scheduler.step()
        t_train = time.time() - t0

        # Validation
        t_val = time.time()
        model.eval()
        val_preds, val_targets, val_tids = [], [], []
        with torch.no_grad():
            for batch in iter_batches(val_ds, bs, shuffle=False):
                pred = model.forward_invivo(batch)
                val_preds.append(pred)
                val_targets.append(batch["target"])
                val_tids.append(batch["task_id"])

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_tids = torch.cat(val_tids)
        val_loss, val_per_task = multitask_loss(
            val_preds, val_targets, val_tids, model.log_var.detach(),
        )
        t_val = time.time() - t_val

        task_str = "  ".join(f"{TASK_NAMES[t]}={l:.3f}" for t, l in sorted(val_per_task.items()))
        print(
            f"  Epoch {epoch+1:3d}/{epochs}: "
            f"train={train_loss_acc.item()/max(n_batches,1):.4f}  val={val_loss.item():.4f}  "
            f"[{task_str}]  (train={t_train:.1f}s val={t_val:.1f}s)"
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val loss = {best_val_loss:.4f})")

    return best_val_loss


# ── Phase 3: Joint Fine-tuning ──────────────────────────────────────────

def train_phase3(model, train_iv_ds, train_vivo_ds,
                 val_iv_ds, val_vivo_ds, device,
                 epochs=20, bs_iv=512, bs_vivo=128, lr=1e-5,
                 max_iv_batches=100, patience=7):
    """Phase 3: Joint end-to-end fine-tuning with very low LR."""
    print(f"\n{'='*60}")
    print(f"Phase 3: Joint fine-tuning ({epochs} epochs, lr={lr})")
    print(f"{'='*60}")

    if len(train_vivo_ds) == 0:
        print("  No in vivo data, skipping.")
        return

    n_iv = max_iv_batches or math.ceil(len(train_iv_ds) / bs_iv)
    print(f"  {n_iv} iv batches/epoch, {math.ceil(len(train_vivo_ds)/bs_vivo)} vivo batches/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_acc = torch.zeros(1, device=device)
        n_steps = 0

        iv_gen = iter_batches(train_iv_ds, bs_iv)
        vivo_gen = iter_batches(train_vivo_ds, bs_vivo)

        for i, batch_iv in enumerate(iv_gen):
            if i >= n_iv:
                break

            # In vitro step
            pred = model.forward_invitro(batch_iv)
            loss = ccc_loss(pred, batch_iv["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_acc += loss.detach()
            n_steps += 1

            # In vivo step (cycle when exhausted)
            batch_vivo = next(vivo_gen, None)
            if batch_vivo is None:
                vivo_gen = iter_batches(train_vivo_ds, bs_vivo)
                batch_vivo = next(vivo_gen)

            pred = model.forward_invivo(batch_vivo)
            loss, per_task = multitask_loss(pred, batch_vivo["target"], batch_vivo["task_id"], model.log_var)
            if per_task:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_acc += loss.detach()
                n_steps += 1

        t_train = time.time() - t0

        # Validation
        model.eval()
        with torch.no_grad():
            # In vitro val
            iv_preds, iv_targets = [], []
            for batch in iter_batches(val_iv_ds, bs_iv, shuffle=False):
                iv_preds.append(model.forward_invitro(batch))
                iv_targets.append(batch["target"])
            val_iv_ccc_loss = ccc_loss(torch.cat(iv_preds), torch.cat(iv_targets)).item()

            # In vivo val
            vivo_preds, vivo_targets, vivo_tids = [], [], []
            for batch in iter_batches(val_vivo_ds, bs_vivo, shuffle=False):
                vivo_preds.append(model.forward_invivo(batch))
                vivo_targets.append(batch["target"])
                vivo_tids.append(batch["task_id"])
            val_vivo_loss, val_per_task = multitask_loss(
                torch.cat(vivo_preds), torch.cat(vivo_targets),
                torch.cat(vivo_tids), model.log_var.detach(),
            )
            val_vivo_loss = val_vivo_loss.item()

        val_combined = val_iv_ccc_loss + val_vivo_loss
        task_str = "  ".join(f"{TASK_NAMES[t]}={l:.3f}" for t, l in sorted(val_per_task.items()))

        marker = ""
        if val_combined < best_val_loss:
            best_val_loss = val_combined
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
            marker = " *"
        else:
            stale += 1

        print(
            f"  Epoch {epoch+1:3d}/{epochs}: "
            f"train={loss_acc.item()/max(n_steps,1):.4f}  "
            f"val_iv={1-val_iv_ccc_loss:.4f}  val_vivo={val_vivo_loss:.4f}  "
            f"[{task_str}]  ({t_train:.1f}s){marker}"
        )

        if stale >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val combined = {best_val_loss:.4f})")


# ── Evaluation ──────────────────────────────────────────────────────────

def median_spearman(preds, targets, group_ids, min_n=3):
    """Spearman ρ per target-gene group, return median. Measures within-target ranking."""
    p, t, g = preds.cpu().numpy(), targets.cpu().numpy(), group_ids.cpu().numpy()
    rhos = []
    for gid in np.unique(g):
        if gid < 0:
            continue
        mask = g == gid
        if mask.sum() < min_n:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            r, _ = stats.spearmanr(p[mask], t[mask])
        if np.isfinite(r):
            rhos.append(r)
    if not rhos:
        return float("nan"), 0
    return float(np.median(rhos)), len(rhos)


def evaluate(model, test_ds, device, norm_stats, dataset_type="invivo"):
    """Evaluate model on test set, print per-task metrics."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Evaluation on {dataset_type} test set ({len(test_ds):,} samples)")
    print(f"{'='*60}")

    if len(test_ds) == 0:
        print("  No test data.")
        return {}

    model.eval()
    all_preds, all_targets, all_groups = [], [], []
    all_task_ids = [] if dataset_type == "invivo" else None
    _last_batches = []

    with torch.no_grad():
        for batch in iter_batches(test_ds, 512, shuffle=False):
            pred = model.forward_invitro(batch) if dataset_type == "invitro" else model.forward_invivo(batch)
            all_preds.append(pred)
            all_targets.append(batch["target"])
            all_groups.append(batch["group_id"])
            _last_batches.append(batch)
            if all_task_ids is not None:
                all_task_ids.append(batch["task_id"])

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_groups = torch.cat(all_groups)

    med_sp, n_groups = median_spearman(all_preds, all_targets, all_groups)

    if dataset_type == "invitro":
        all_sources = torch.cat([b["source"] for b in _last_batches])

        def _iv_metrics(p, t, g, label):
            n = len(p)
            if n < 3:
                return
            ccc_val = 1.0 - ccc_loss(p, t).item()
            pearson_r = float(torch.corrcoef(torch.stack([p, t]))[0, 1])
            mae = float((p - t).abs().mean())
            sp, ng = median_spearman(p, t, g)
            print(f"  {label:<16} CCC={ccc_val:.4f}  r={pearson_r:.4f}  MAE={mae:.4f}  MedSp={sp:.4f} ({ng} grp, {n:,} samples)")
            return {"ccc": ccc_val, "pearson_r": pearson_r, "mae": mae, "median_spearman": sp, "n": n}

        results = {}
        results["all"] = _iv_metrics(all_preds, all_targets, all_groups, "All")
        for src, label in [(0, "Inhibition"), (1, "Dose-response")]:
            mask = all_sources == src
            if mask.any():
                results[label.lower()] = _iv_metrics(all_preds[mask], all_targets[mask], all_groups[mask], label)
        print(f"  ({time.time()-t0:.1f}s)")
        return results

    # In vivo: per-task metrics
    all_task_ids = torch.cat(all_task_ids)
    results = {}

    print(f"\n  {'Task':<12} {'CCC':>8} {'Pearson':>8} {'MAE':>8} {'MedSp':>8} {'N':>6} {'Grp':>4}")
    print(f"  {'-'*58}")

    for tid in sorted(all_task_ids.unique().tolist()):
        mask = all_task_ids == tid
        n = mask.sum().item()
        if n < 3:
            continue

        p, t = all_preds[mask], all_targets[mask]
        ccc_val = 1.0 - ccc_loss(p, t).item()
        pearson_r = float(torch.corrcoef(torch.stack([p, t]))[0, 1]) if n > 2 else float("nan")
        mae = float((p - t).abs().mean())
        mae_denorm = mae * norm_stats.std[tid] if tid in norm_stats.mean else mae
        task_sp, task_ng = median_spearman(p, t, all_groups[mask])

        name = TASK_NAMES[tid] if tid < len(TASK_NAMES) else f"task_{tid}"
        print(f"  {name:<12} {ccc_val:>8.4f} {pearson_r:>8.4f} {mae:>8.4f} {task_sp:>8.4f} {n:>6d} {task_ng:>4d}")
        results[name] = {"ccc": ccc_val, "pearson_r": pearson_r, "mae": mae,
                         "mae_denorm": mae_denorm, "median_spearman": task_sp, "n": n}

    print(f"\n  Overall median Spearman={med_sp:.4f} ({n_groups} groups)")
    print(f"  ({time.time()-t0:.1f}s)")
    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OligoAI2 Training")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-epoch run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-phase3", action="store_true", help="Skip joint fine-tuning")
    args = parser.parse_args()

    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    p1_epochs = 2 if args.smoke_test else 30
    p2_epochs = 2 if args.smoke_test else 100
    p3_epochs = 2 if args.smoke_test else 20
    p3_max_iv = 170 if args.smoke_test else 100

    # Load data + pre-move to device (eliminates per-batch transfers)
    data = load_all(seed=args.seed)
    for v in data.values():
        if isinstance(v, Dataset):
            move_to(v, device)

    # Build model
    cfg = ModelConfig()
    model = OligoAI(
        cfg,
        n_transfection=data["invitro_info"]["n_transfection"],
        n_hepatic_admin=data["hepatic_info"]["n_admin"],
        n_neuro_admin=data["neuro_info"]["n_admin"],
    ).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Phase 1
    train_phase1(model, data["train_invitro"], data["val_invitro"], device,
                 epochs=p1_epochs)

    # Phase 2
    train_phase2(model, data["train_invivo"], data["val_invivo"], device,
                 epochs=p2_epochs)

    # Phase 3
    if not args.skip_phase3:
        train_phase3(model, data["train_invitro"], data["train_invivo"],
                     data["val_invitro"], data["val_invivo"], device,
                     epochs=p3_epochs, max_iv_batches=p3_max_iv)

    # Evaluate
    model.to(device)
    evaluate(model, data["test_invitro"], device, data["norm_stats"], dataset_type="invitro")
    evaluate(model, data["test_invivo"], device, data["norm_stats"], dataset_type="invivo")

    # Learned uncertainty weights
    print(f"\nLearned log_var (uncertainty weights):")
    sigmas = torch.exp(0.5 * model.log_var).detach().cpu()
    for tid in range(N_TASKS):
        print(f"  {TASK_NAMES[tid]}: sigma={sigmas[tid]:.4f}, log_var={model.log_var[tid].item():.4f}")

    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
