"""Training loop for OligoAI2 simplified multi-task ASO regression model.

Single joint training loop: each step samples one in vitro + one in vivo batch,
computes combined CCC loss, and takes one optimizer step.
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


# ── Joint Training Loop ────────────────────────────────────────────────

def train(model, train_iv, train_vivo, val_iv, val_vivo, device,
          epochs=80, bs_iv=512, bs_vivo=128, lr=3e-4, patience=12,
          warmup_epochs=10, encoder_lr=1e-5):
    """Joint training with IV-only warmup and differential encoder LR."""
    n_iv_batches = math.ceil(len(train_iv) / bs_iv)
    print(f"\n{'='*60}")
    print(f"Training: {len(train_iv):,} IV + {len(train_vivo):,} vivo samples")
    print(f"  {n_iv_batches} IV batches/epoch, lr={lr}, patience={patience}")
    print(f"  warmup={warmup_epochs} epochs (IV-only), then joint (encoder_lr={encoder_lr})")
    print(f"{'='*60}")

    has_vivo = len(train_vivo) > 0

    # Separate encoder params (protected LR during joint phase) from rest
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(epochs):
        # Transition: warmup → joint (drop encoder LR)
        if epoch == warmup_epochs and warmup_epochs > 0:
            scheduler.base_lrs[0] = encoder_lr
            optimizer.param_groups[0]["lr"] = encoder_lr
            print(f"\n  → Joint phase: encoder LR {lr} → {encoder_lr}")

        t0 = time.time()
        model.train()
        loss_acc = torch.zeros(1, device=device)
        n_steps = 0
        in_joint = epoch >= warmup_epochs

        iv_gen = iter_batches(train_iv, bs_iv)
        vivo_gen = iter_batches(train_vivo, bs_vivo) if has_vivo and in_joint else None

        for batch_iv in iv_gen:
            # In vitro loss
            iv_pred = model.forward_invitro(batch_iv)
            loss = ccc_loss(iv_pred, batch_iv["target"])

            # In vivo loss (skip during warmup)
            if vivo_gen is not None:
                batch_vivo = next(vivo_gen, None)
                if batch_vivo is None:
                    vivo_gen = iter_batches(train_vivo, bs_vivo)
                    batch_vivo = next(vivo_gen)
                vivo_pred = model.forward_invivo(batch_vivo)
                vivo_loss, per_task = multitask_loss(
                    vivo_pred, batch_vivo["target"], batch_vivo["task_id"], model.log_var,
                )
                if per_task:
                    loss = loss + vivo_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_acc += loss.detach()
            n_steps += 1

        scheduler.step()
        t_train = time.time() - t0

        # Validation
        val_loss, val_iv_ccc, val_vivo_loss, val_per_task = _validate(
            model, val_iv, val_vivo, bs_iv, bs_vivo,
        )
        t_total = time.time() - t0

        task_str = ""
        if val_per_task:
            task_str = "  [" + "  ".join(
                f"{TASK_NAMES[t]}={l:.3f}" for t, l in sorted(val_per_task.items())
            ) + "]"

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if device.type == "mps":
                torch.mps.synchronize()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            marker = " *"
        else:
            stale += 1

        phase = "WU" if not in_joint else "JT"
        print(
            f"  {phase} {epoch+1:3d}/{epochs}: "
            f"train={loss_acc.item()/max(n_steps,1):.4f}  "
            f"val_iv_CCC={1-val_iv_ccc:.4f}  val_vivo={val_vivo_loss:.4f}"
            f"{task_str}  "
            f"lr={optimizer.param_groups[1]['lr']:.2e}"
            f"/{optimizer.param_groups[0]['lr']:.2e}  "
            f"({t_total:.1f}s){marker}"
        )

        if stale >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"  Restored best model (val combined = {best_val_loss:.4f})")


def _validate(model, val_iv, val_vivo, bs_iv, bs_vivo):
    """Compute combined validation loss. Returns (combined, iv_ccc_loss, vivo_loss, per_task)."""
    model.eval()
    with torch.no_grad():
        # In vitro
        iv_preds, iv_targets = [], []
        for batch in iter_batches(val_iv, bs_iv, shuffle=False):
            iv_preds.append(model.forward_invitro(batch))
            iv_targets.append(batch["target"])
        val_iv_ccc_loss = ccc_loss(torch.cat(iv_preds), torch.cat(iv_targets)).item()

        # In vivo
        val_vivo_loss = 0.0
        val_per_task = {}
        if len(val_vivo) > 0:
            vivo_preds, vivo_targets, vivo_tids = [], [], []
            for batch in iter_batches(val_vivo, bs_vivo, shuffle=False):
                vivo_preds.append(model.forward_invivo(batch))
                vivo_targets.append(batch["target"])
                vivo_tids.append(batch["task_id"])
            vl, val_per_task = multitask_loss(
                torch.cat(vivo_preds), torch.cat(vivo_targets),
                torch.cat(vivo_tids), model.log_var.detach(),
            )
            val_vivo_loss = vl.item()

    return val_iv_ccc_loss + val_vivo_loss, val_iv_ccc_loss, val_vivo_loss, val_per_task


# ── Evaluation ──────────────────────────────────────────────────────────

def median_spearman(preds, targets, group_ids, min_n=3):
    """Spearman rho per target-gene group, return median. Measures within-target ranking."""
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
    parser = argparse.ArgumentParser(description="OligoAI2 Simplified Training")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-epoch run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    epochs = 2 if args.smoke_test else 80

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

    # Train
    warmup_epochs = 1 if args.smoke_test else 10
    train(model, data["train_invitro"], data["train_invivo"],
          data["val_invitro"], data["val_invivo"], device,
          epochs=epochs, warmup_epochs=warmup_epochs)

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
