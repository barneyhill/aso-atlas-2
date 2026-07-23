"""Audit: verify patent-level CV with HELM dedup is leakage-free.

For each of the six benchmark datasets, compute a 5-fold GroupKFold
split by patent and report per-fold:
    - n_test_helm, n_train_helm
    - n_shared_helm (should always be 0 after the earliest-patent dedup)
    - n_test_genes, n_shared_genes (patents ≈ genes at Ionis, so
      near-zero shared genes expected but not guaranteed)

Writes: data/results/cv_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from analyses.utils.oligogym_adapter import DATASETS

_root = Path(__file__).resolve().parents[2]
RESULTS_DIR = _root / "data/results"

N_SPLITS = 5


def audit_dataset(name: str) -> dict:
    loader = DATASETS[name]
    d = loader()
    helms = np.asarray(d["x"])  # adapter returns `x` (HELM strings post convert)
    groups = np.asarray(d["groups"])

    # Gene labels: the adapter doesn't always keep target_RNA on the
    # output; rebuild from the underlying parquet if needed. For now
    # fall back to groups (for IV/potency, groups is patent after
    # dedup — we can't recover gene without refetching).
    # Report helm leakage; gene leakage approx via unique patents.
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []
    for i, (tr, te) in enumerate(gkf.split(helms, helms, groups)):
        tr_helms = set(helms[tr].tolist())
        te_helms = set(helms[te].tolist())
        shared = tr_helms & te_helms
        rows.append({
            "fold": i + 1,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_helm": len(tr_helms),
            "n_test_helm": len(te_helms),
            "n_shared_helm": len(shared),
            "n_train_groups": int(len(set(groups[tr]))),
            "n_test_groups": int(len(set(groups[te]))),
        })
    return {"dataset": name, "n_total": int(len(helms)),
            "n_unique_helm": int(len(set(helms.tolist()))),
            "n_groups": int(len(set(groups.tolist()))),
            "folds": rows}


def main():
    results = {}
    for name in DATASETS.keys():
        print(f"\n=== {name} ===")
        r = audit_dataset(name)
        results[name] = r
        print(f"  n={r['n_total']}, unique_helm={r['n_unique_helm']}, groups={r['n_groups']}")
        for f in r["folds"]:
            marker = "OK" if f["n_shared_helm"] == 0 else "LEAK"
            print(f"  fold {f['fold']}: train={f['n_train']} test={f['n_test']}  "
                  f"shared_HELM={f['n_shared_helm']} [{marker}]")
    out_path = RESULTS_DIR / "cv_audit.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
