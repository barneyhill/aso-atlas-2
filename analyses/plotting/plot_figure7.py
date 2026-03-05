"""
Figure 7 (supplementary): Joint 128-feature dinucleotide importance boxplots.

Each 4×4 subplot shows one base pair, with box+strip for 4 models
(Mouse ALT, Rat ALT, Mouse bFOB, Rat mFOB).

Reads: data/results/hepatotox.json, data/results/neurotox.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
HEPATOTOX_PATH = _root / "data/results/hepatotox.json"
NEUROTOX_PATH = _root / "data/results/neurotox.json"
OUT_DIR = _root / "typst/plots/fig7"

BASES = ["A", "C", "G", "T"]
BASE_PAIRS = [f"{b1}{b2}" for b1 in BASES for b2 in BASES]

JOINT_SPECS = [
    ("Mouse ALT", "hepatotox", "predictions", "ALT"),
    ("Rat ALT", "hepatotox", "rat_predictions", "rat_ALT"),
    ("Mouse bFOB", "neurotox", "predictions", "FOB"),
    ("Rat mFOB", "neurotox", "rat_predictions", "rat_FOB"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def collect_joint_importances(data, predictions_key, model_key):
    """Extract per-base-pair importances from joint 128-feature model."""
    pred = data.get(predictions_key, {}).get(model_key)
    if pred is None or "fold_importances" not in pred:
        return None

    feature_names = pred["feature_names"]
    fold_importances = pred["fold_importances"]

    bp_indices = {}
    for i, name in enumerate(feature_names):
        parts = name.split("_")
        if len(parts) != 5:
            continue
        _, b1, _, _, b2 = parts
        bp = f"{b1}{b2}"
        bp_indices.setdefault(bp, [])
        bp_indices[bp].append(i)

    result = {}
    for bp in BASE_PAIRS:
        indices = bp_indices.get(bp, [])
        vals = []
        for fold_imp in fold_importances:
            for idx in indices:
                vals.append(fold_imp[idx])
        result[bp] = vals

    return result


def _draw_boxstrip(ax, data_lists, colors, positions=None):
    """Draw box + strip overlay."""
    if positions is None:
        positions = list(range(len(data_lists)))
    bplot = ax.boxplot(
        data_lists,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    for m_idx, (pos, vals, color) in enumerate(zip(positions, data_lists, colors)):
        jitter = np.random.default_rng(42 + m_idx).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(pos + jitter, vals, c=color, s=12, alpha=0.7, edgecolors="none", zorder=3)


def main():
    for path in [HEPATOTOX_PATH, NEUROTOX_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `just hagerdorn` first.")

    with open(HEPATOTOX_PATH) as f:
        hepatotox = json.load(f)
    with open(NEUROTOX_PATH) as f:
        neurotox = json.load(f)

    sources = {"hepatotox": hepatotox, "neurotox": neurotox}

    all_importances = {}
    for label, source_key, predictions_key, model_key in JOINT_SPECS:
        imp = collect_joint_importances(sources[source_key], predictions_key, model_key)
        if imp is None:
            print(f"  Warning: no joint importances for {label}, skipping")
            continue
        all_importances[label] = imp

    if not all_importances:
        print("  No joint importances found, skipping joint plot")
        return

    model_labels = [s[0] for s in JOINT_SPECS if s[0] in all_importances]

    fig, axes = plt.subplots(4, 4, figsize=(14, 12), dpi=300, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.15)

    for idx, bp in enumerate(BASE_PAIRS):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        bp_data = []
        bp_colors = []
        for m_idx, label in enumerate(model_labels):
            vals = all_importances[label].get(bp, [])
            if vals:
                bp_data.append(vals)
                bp_colors.append(COLORS[m_idx])

        if bp_data:
            _draw_boxstrip(ax, bp_data, bp_colors)

        ax.set_title(f"{bp[0]}-{bp[1]}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Importance", fontsize=9)
        if row == 3:
            ax.set_xticks(range(len(model_labels)))
            ax.set_xticklabels(
                [l.replace(" ", "\n") for l in model_labels], fontsize=7,
            )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[i],
                    markersize=8, label=label)
        for i, label in enumerate(model_labels)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=len(model_labels), fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig7-joint.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
