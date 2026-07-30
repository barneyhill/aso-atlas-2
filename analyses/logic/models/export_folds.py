"""Release the exact cross-validation splits used for the paper (rebuttal A2).

Emits the exact endpoint-specific 5-fold assignments used for every reported
benchmark, so reviewers get the paper partitions rather than a newly generated
global split.  OligoGym assignments are checked against the frozen test-group
membership saved in ``oligogym_benchmark.json``.  OligoAI assignments are read
from the five frozen training CSVs used for the submitted runs.

Writes one CSV per endpoint to  aso-atlas-2-release/folds/  (staged for the HF
release), an OligoAI fold map, and a README.

    uv run python -m analyses.logic.models.export_folds
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from analyses.utils.oligogym_adapter import DATASETS

_root = Path(__file__).resolve().parents[3]
OUT_DIR = _root / "aso-atlas-2-release" / "folds"
RESULTS_DIR = _root / "data" / "results"
OLIGOAI_DIR = _root / "data" / "oligostack" / "processed"
N_SPLITS = 5


def fold_assignment(name: str) -> pd.DataFrame:
    d = DATASETS[name]()
    helms = np.asarray(d["x"])
    groups = np.asarray(d["groups"])
    fold = np.empty(len(helms), dtype=int)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for i, (_, te) in enumerate(gkf.split(helms, helms, groups), start=1):
        fold[te] = i
    df = pd.DataFrame({"group": groups, "model_input_helm": helms, "fold": fold})
    # per (helm, group) the fold is constant; release the unique mapping
    return (df.drop_duplicates(subset=["model_input_helm"])
              .sort_values(["fold", "group"])
              .reset_index(drop=True))


def frozen_group_assignment(name: str) -> dict[str, int]:
    """Patent-group fold membership saved by the submitted OligoGym run."""
    benchmark = json.loads((RESULTS_DIR / "oligogym_benchmark.json").read_text())
    row = next(
        r for r in benchmark["all_results"]
        if r["dataset"] == name and r["model"] == "XGBoost"
    )
    assignment: dict[str, int] = {}
    for fold, metrics in enumerate(row["fold_metrics"], start=1):
        for group in metrics["groups_test"]:
            previous = assignment.setdefault(group, fold)
            if previous != fold:
                raise ValueError(f"{name}: group {group} occurs in folds {previous} and {fold}")
    return assignment


def validate_against_submitted_run(name: str, df: pd.DataFrame) -> None:
    generated = df.drop_duplicates("group").set_index("group")["fold"].to_dict()
    frozen = frozen_group_assignment(name)
    if generated != frozen:
        missing = sorted(set(frozen) - set(generated))
        extra = sorted(set(generated) - set(frozen))
        moved = sorted(g for g in set(generated) & set(frozen) if generated[g] != frozen[g])
        raise ValueError(
            f"{name}: regenerated folds do not match the submitted run "
            f"(missing={missing[:5]}, extra={extra[:5]}, moved={moved[:5]})"
        )


def oligoai_fold_assignment() -> tuple[pd.DataFrame, list[int]]:
    """Exact HELM-to-fold map from the five CSVs used to train OligoAI."""
    assignments = []
    test_rows = []
    for fold_idx in range(N_SPLITS):
        path = OLIGOAI_DIR / f"oligoai_train_fold{fold_idx}.csv.gz"
        df = pd.read_csv(path, usecols=["helm_annotation", "split"])
        test = df[df["split"].eq("test")]
        test_rows.append(len(test))
        assignments.append(
            test[["helm_annotation"]].drop_duplicates().assign(fold=fold_idx + 1)
        )
    out = pd.concat(assignments, ignore_index=True)
    conflicts = out.groupby("helm_annotation")["fold"].nunique()
    if not conflicts.eq(1).all():
        raise ValueError("OligoAI: at least one HELM appears in multiple test folds")
    return out.sort_values(["fold", "helm_annotation"]).reset_index(drop=True), test_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in DATASETS:
        df = fold_assignment(name)
        validate_against_submitted_run(name, df)
        out = OUT_DIR / f"{name}_folds.csv.gz"
        df.to_csv(out, index=False, compression="gzip")
        counts = df["fold"].value_counts().sort_index().to_dict()
        summary.append((name, len(df), df["group"].nunique(), counts))
        print(f"  {name}: {len(df)} unique HELMs, {df['group'].nunique()} patents -> {out.name} "
              f"(per-fold {list(counts.values())})")

    oligoai, oligoai_test_rows = oligoai_fold_assignment()
    oligoai_out = OUT_DIR / "oligoai_folds.csv.gz"
    oligoai.to_csv(oligoai_out, index=False, compression="gzip")
    print(f"  OligoAI: {len(oligoai):,} unique HELMs -> {oligoai_out.name} "
          f"(test rows per fold {oligoai_test_rows})")

    readme = OUT_DIR / "README.md"
    lines = [
        "# ASO Atlas 2.0 — paper cross-validation folds",
        "",
        "These are the exact endpoint-specific 5-fold assignments used for the submitted",
        "paper's benchmark results. They are the reproduction partitions requested by the",
        "reviewers; they are distinct from the release's optional global canonical split.",
        "",
        "The paper constructed each benchmark endpoint independently after its own filtering",
        "and aggregation, then applied patent-level `GroupKFold` with HELM-level deduplication.",
        "Consequently each endpoint has its own patent-to-fold assignment. The assignments",
        "below are validated against the test-group membership frozen in the submitted run's",
        "result file, rather than assumed from the shared grouping rule.",
        "",
        "The six OligoGym endpoint CSVs contain:",
        "- `group` — patent id (the CV group)",
        "- `model_input_helm` — OligoGym adapter representation used by the benchmark",
        "- `fold`  — test fold in {1..5}; for fold k, test = rows with fold==k, train = the rest",
        "",
        "`model_input_helm` is deliberately not described as the raw release HELM: OligoGym's",
        "adapter normalises that representation before modelling. Use the corresponding",
        "adapter in `analyses/utils/oligogym_adapter.py`, or join on its `x` output.",
        "",
        "`oligoai_folds.csv.gz` separately maps each full `helm_annotation` to its exact",
        "OligoAI test fold. It is derived from the five frozen training CSVs used by the",
        "submitted runs; OligoAI and OligoGym must not be assumed to share fold numbers.",
        "",
        "| OligoGym endpoint | Unique model inputs | Patents | Per-fold sizes |",
        "|---|--:|--:|---|",
    ]
    for name, n, g, counts in summary:
        lines.append(f"| {name} | {n} | {g} | {', '.join(str(v) for v in counts.values())} |")
    lines.extend([
        "",
        f"OligoAI: {len(oligoai):,} unique HELMs; test rows per fold: "
        f"{', '.join(str(n) for n in oligoai_test_rows)}.",
    ])
    readme.write_text("\n".join(lines) + "\n")
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
