"""Generate a typst table from OligoGym benchmark results."""

import json
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
RESULTS_PATH = _root / "data/results/oligogym_benchmark.json"
OUT_PATH = _root / "typst/data/oligogym_benchmark.typ"

# Display order for models
MODEL_ORDER = [
    "Linear",
    "Random Forest",
    "XGBoost",
    "KNN",
    "CatBoost",
    "MLP",
    "CNN",
    "CausalCNN",
    "Transformer",
]

DATASET_ORDER = ["mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro"]
DATASET_LABELS = {
    "mouse_hepatic": "Mouse ALT",
    "rat_hepatic": "Rat ALT",
    "mouse_neuro": "Mouse FOB",
    "rat_neuro": "Rat FOB",
}


def _fmt(val, std=None):
    """Format a metric value, bolding will be handled separately."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    s = f"{val:.3f}"
    if std is not None and not np.isnan(std):
        s += f" ({std:.3f})"
    return s


def main():
    data = json.loads(RESULTS_PATH.read_text())
    best = data["best_per_model"]

    # Build lookup: (dataset, model) -> {spearman, spearman_std, r2, rmse, featurizer}
    lookup = {}
    for row in best:
        key = (row["dataset"], row["model"])
        lookup[key] = row

    # Find best model per dataset (for bolding)
    best_per_ds = {}
    for ds in DATASET_ORDER:
        best_val = -np.inf
        for m in MODEL_ORDER:
            row = lookup.get((ds, m))
            if row and row.get("spearman") is not None:
                if row["spearman"] > best_val:
                    best_val = row["spearman"]
                    best_per_ds[ds] = m

    # Generate typst table
    lines = []
    lines.append("#table(")
    lines.append("  columns: (auto, 1fr, 1fr, 1fr, 1fr),")
    lines.append("  align: (left, center, center, center, center),")
    lines.append("  stroke: none,")
    lines.append("  inset: (x: 6pt, y: 4pt),")
    lines.append("")
    # Header
    lines.append("  table.hline(),")
    lines.append("  table.header(")
    lines.append("    [*Model*],")
    for ds in DATASET_ORDER:
        lines.append(f"    [*{DATASET_LABELS[ds]}*],")
    lines.append("  ),")
    lines.append("  table.hline(),")
    lines.append("")

    for model in MODEL_ORDER:
        cells = [f"  [{model}],"]
        for ds in DATASET_ORDER:
            row = lookup.get((ds, model))
            if row and row.get("spearman") is not None:
                val = row["spearman"]
                std = row.get("spearman_std")
                text = _fmt(val, std)
                if best_per_ds.get(ds) == model:
                    text = f"*{text}*"
                cells.append(f"  [{text}],")
            else:
                cells.append("  [---],")
        lines.append("\n".join(cells))
        lines.append("")

    lines.append("  table.hline(),")
    lines.append(")")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
