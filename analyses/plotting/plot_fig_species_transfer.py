"""Supplementary figure: cross-species toxicity transfer (both directions).

Two panels (hepatic, neuro). Each panel has two grouped bars:
    Rat test:   baseline (rat→rat)   vs transfer (mouse→rat)
    Mouse test: baseline (mouse→mouse) vs transfer (rat→mouse)

Reads: data/results/species_transfer.json
Writes: typst/plots/fig_species_transfer/species_transfer.{svg,png}
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
RESULTS = _root / "data/results/species_transfer.json"
OUT_DIR = _root / "typst/plots/fig_species_transfer"

COLOR_BASELINE = "#4878A8"
COLOR_TRANSFER = "#D4A574"


def main():
    data = json.loads(RESULTS.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)

    # 4 bars per panel: rat/baseline, rat/transfer, [gap], mouse/baseline, mouse/transfer
    positions = [0, 0.9, 2.2, 3.1]
    group_centers = [0.45, 2.65]
    group_labels = ["Rat test", "Mouse test"]
    bar_labels = ["baseline", "transfer", "baseline", "transfer"]
    colors = [COLOR_BASELINE, COLOR_TRANSFER, COLOR_BASELINE, COLOR_TRANSFER]

    rng = np.random.default_rng(0)

    panel_titles = {"hepatic": "ALT (IU/L)", "neuro": "FOB (neuro)"}
    for ax, organ in zip(axes, ["hepatic", "neuro"]):
        d = data[organ]
        values = [
            d["rat_test"]["rat_only"],       # baseline rat→rat
            d["rat_test"]["mouse_only"],     # transfer mouse→rat
            d["mouse_test"]["mouse_only"],   # baseline mouse→mouse
            d["mouse_test"]["rat_only"],     # transfer rat→mouse
        ]
        means = [v["mean"] for v in values]
        stds = [v["std"] for v in values]
        ax.bar(
            positions, means, yerr=stds, capsize=3,
            color=colors, edgecolor="black", linewidth=0.5, width=0.8,
        )
        for i, v in enumerate(values):
            pf = v["per_fold"]
            ax.scatter(
                np.full(len(pf), positions[i]) + rng.uniform(-0.12, 0.12, len(pf)),
                pf, color="black", s=10, zorder=3, alpha=0.6,
            )
        ax.axhline(0, color="black", lw=0.5)

        # x tick labels: individual bar labels on top row, group labels below
        ax.set_xticks(positions)
        ax.set_xticklabels(bar_labels, fontsize=8)
        # Add group labels below
        for cx, gl in zip(group_centers, group_labels):
            ax.text(cx, -0.25, gl, ha="center", va="top", fontsize=9,
                    transform=ax.get_xaxis_transform())

        ax.set_title(panel_titles[organ], fontsize=10)
        ax.set_ylim(-0.2, 0.9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Spearman ρ", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=COLOR_BASELINE, label="Baseline (self-species)"),
        Patch(color=COLOR_TRANSFER, label="Cross-species transfer"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    svg = OUT_DIR / "species_transfer.svg"
    png = OUT_DIR / "species_transfer.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Wrote {svg} and {png}")


if __name__ == "__main__":
    main()
