"""Reattach compound identifiers to the stored OligoGym benchmark predictions.

The benchmark saved each fold as bare ``y_true`` / ``y_pred`` arrays plus the patent
group, with no per-row compound label, so a mouse prediction cannot be matched to its
rat counterpart. That blocked the model-arm half of the conditional-enrichment
correction for the hepatic link, and forced the neuro link to substitute the Hagedorn
2022 formula for the actual model.

No retraining is needed to fix it. ``GroupKFold`` is deterministic and its split
depends only on the group array and the sample count, not on the featurizer or the
model, which is why every benchmark row for a given dataset shares one ``y_true`` per
fold. Re-deriving the dataset in loader order and replaying the split therefore
recovers exactly the rows that were held out, and the loaders already carry the HELM
string as ``x``.

Every recovery is checked against the stored ``y_true`` elementwise and raises rather
than returning anything on mismatch, so a silent reordering upstream cannot produce a
plausible but wrong mapping.

    uv run python -m analyses.logic.models.benchmark_ids
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from analyses.utils.oligogym_adapter import (
    load_in_vitro_inhibition,
    load_mouse_hepatic,
    load_mouse_neuro,
    load_potency,
    load_rat_hepatic,
    load_rat_neuro,
)

_root = Path(__file__).resolve().parents[3]
RESULTS_DIR = _root / "data/results"
BENCH_PATH = RESULTS_DIR / "oligogym_benchmark.json"

N_SPLITS = 5

LOADERS = {
    "mouse_hepatic": load_mouse_hepatic,
    "rat_hepatic": load_rat_hepatic,
    "mouse_neuro": load_mouse_neuro,
    "rat_neuro": load_rat_neuro,
    "in_vitro_inhibition": load_in_vitro_inhibition,
    "potency": load_potency,
}


def _stored_folds(bench: dict, dataset: str, model: str | None = None) -> list[dict]:
    """Fold metrics for one dataset. Any model will do for y_true; a named model for y_pred."""
    for r in bench.get("all_results", []):
        if r.get("dataset") != dataset or "fold_metrics" not in r:
            continue
        if model is not None and r.get("model") != model:
            continue
        if all("y_true" in fm for fm in r["fold_metrics"]):
            return r["fold_metrics"]
    raise LookupError(f"no stored folds for dataset={dataset!r} model={model!r}")


def fold_identifiers(dataset: str, bench: dict | None = None) -> list[np.ndarray]:
    """HELM strings for the held-out rows of each fold, in stored order.

    Raises if the replayed split does not reproduce the stored ``y_true`` exactly.
    """
    bench = bench if bench is not None else json.loads(BENCH_PATH.read_text())
    data = LOADERS[dataset]()
    y = np.asarray(data["y"], dtype=float)
    helms = np.asarray(data["x"], dtype=object)
    groups = np.asarray(data["groups"])

    stored = _stored_folds(bench, dataset)
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    out = []
    for i, (_train, test) in enumerate(gkf.split(np.zeros((len(y), 1)), y, groups)):
        ref = np.asarray(stored[i]["y_true"], dtype=float)
        got = y[test]
        if len(ref) != len(got) or not np.allclose(ref, got, equal_nan=True):
            raise ValueError(
                f"{dataset} fold {i}: replayed split does not match the stored y_true "
                f"({len(got)} vs {len(ref)} rows). The dataset has changed since the "
                f"benchmark was run; identifiers cannot be recovered safely."
            )
        out.append(helms[test])
    return out


def labelled_predictions(dataset: str, model: str, bench: dict | None = None) -> pd.DataFrame:
    """Held-out predictions for every row, labelled with the compound's HELM.

    Each compound appears in exactly one test fold, so concatenating the folds gives one
    out-of-fold prediction per row, which is how ``ef_table`` pools them for the published
    enrichment factors.
    """
    bench = bench if bench is not None else json.loads(BENCH_PATH.read_text())
    ids = fold_identifiers(dataset, bench)
    folds = _stored_folds(bench, dataset, model)
    frames = []
    for i, fm in enumerate(folds):
        frames.append(pd.DataFrame({
            "helm": ids[i],
            "y_true": np.asarray(fm["y_true"], dtype=float),
            "y_pred": np.asarray(fm["y_pred"], dtype=float),
            "fold": i,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    bench = json.loads(BENCH_PATH.read_text())
    print("Recovering compound identifiers for the stored benchmark folds")
    print("(no retraining; the benchmark file is never written to)\n")
    for ds in LOADERS:
        try:
            ids = fold_identifiers(ds, bench)
        except (LookupError, ValueError) as e:
            print(f"  {ds:<22} FAILED: {e}")
            continue
        n = sum(len(a) for a in ids)
        uniq = len({h for a in ids for h in a})
        print(f"  {ds:<22} ok  {len(ids)} folds, {n:,} rows, {uniq:,} unique HELMs")


if __name__ == "__main__":
    main()
