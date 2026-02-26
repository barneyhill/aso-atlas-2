"""Hyperparameter search for OligoAI-tox PosPool model."""

import itertools
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from analyses.logic.models.oligoai_tox import OligoAITox, encode_batch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_cv(encoded, y, groups, cov, *, n_filters, n_hidden, dropout, lr, wd, n_splits=5):
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

        model = OligoAITox(n_filters=n_filters, n_hidden=n_hidden, dropout=dropout, n_cov=n_cov).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)

        best_loss, wait, best_state = float("inf"), 0, None
        for _ in range(300):
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
                if wait >= 30:
                    break

        if best_state:
            model.cpu()
            model.load_state_dict(best_state)
        model.cpu().eval()
        with torch.no_grad():
            preds[te] = model(Xte.cpu(), cte.cpu()).numpy()

    return roc_auc_score(y, preds)


def load_data():
    from scripts.ablation_cnn import load_alt, load_fob
    return {"ALT": load_alt(), "FOB": load_fob()}


def main():
    print(f"Device: {DEVICE}\n")
    datasets = load_data()

    # Grid axes
    grid = {
        "n_filters": [8, 12, 16, 24, 32],
        "n_hidden":  [8, 16, 32],
        "dropout":   [0.15, 0.25, 0.35, 0.45],
        "lr":        [5e-4, 1e-3, 2e-3],
        "wd":        [1e-5, 1e-4, 5e-4],
    }

    # Random subsample from grid to keep runtime reasonable
    all_combos = list(itertools.product(*grid.values()))
    rng = np.random.default_rng(42)
    n_samples = min(80, len(all_combos))
    chosen = rng.choice(len(all_combos), size=n_samples, replace=False)
    configs = [dict(zip(grid.keys(), all_combos[i])) for i in chosen]

    print(f"Searching {n_samples} configs across {len(all_combos)} total\n")

    results = []
    for i, cfg in enumerate(configs):
        n_params = sum(p.numel() for p in OligoAITox(
            n_filters=cfg["n_filters"], n_hidden=cfg["n_hidden"], n_cov=4
        ).parameters())

        aucs = {}
        t0 = time.time()
        for ds_name, (encoded, y, groups, cov) in datasets.items():
            aucs[ds_name] = run_cv(encoded, y, groups, cov, **cfg)
        dt = time.time() - t0

        avg_auc = np.mean(list(aucs.values()))
        results.append({**cfg, "params": n_params, **aucs, "avg": avg_auc, "time": dt})

        print(f"[{i+1:3d}/{n_samples}] F={cfg['n_filters']:2d} H={cfg['n_hidden']:2d} "
              f"d={cfg['dropout']:.2f} lr={cfg['lr']:.0e} wd={cfg['wd']:.0e}  "
              f"ALT={aucs['ALT']:.3f} FOB={aucs['FOB']:.3f} avg={avg_auc:.3f}  "
              f"({n_params} params, {dt:.0f}s)")

    # Sort by average AUC
    results.sort(key=lambda r: r["avg"], reverse=True)

    print(f"\n{'='*90}")
    print(f"Top 15 configs by avg AUC:")
    print(f"{'F':>3} {'H':>3} {'drop':>5} {'lr':>8} {'wd':>8} {'params':>6} {'ALT':>7} {'FOB':>7} {'avg':>7}")
    print(f"{'-'*90}")
    for r in results[:15]:
        print(f"{r['n_filters']:3d} {r['n_hidden']:3d} {r['dropout']:5.2f} {r['lr']:8.0e} {r['wd']:8.0e} "
              f"{r['params']:6d} {r['ALT']:7.3f} {r['FOB']:7.3f} {r['avg']:7.3f}")
    print(f"{'='*90}")

    # Current defaults for reference
    print(f"\nCurrent defaults: F=16 H=16 d=0.30 lr=1e-3 wd=1e-4")
    print(f"RF baselines:     ALT=0.771, FOB=0.895")


if __name__ == "__main__":
    main()
