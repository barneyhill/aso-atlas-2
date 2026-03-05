"""
Supplementary figure: ALT vs AST scatter plot by species.

Reads: data/oligostack/processed/hepatictoxicity_processed.parquet
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
_data_dir = _root / "data/oligostack/processed"
OUT_DIR = _root / "typst/plots/supp_alt_ast"

SPECIES_STYLE = {
    "mouse": {"color": "#4878A8", "label": "Mouse", "zorder": 2},
    "rat":   {"color": "#D97706", "label": "Rat",   "zorder": 3},
    "monkey": {"color": "#7B3F9E", "label": "Monkey", "zorder": 4},
}


def main():
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df["mean_AST"] = df["AST"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna() & df["mean_AST"].notna()]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    for species, style in SPECIES_STYLE.items():
        sub = df[df["species"] == species]
        if len(sub) == 0:
            continue
        label = f"{style['label']} (n={len(sub)})"
        ax.scatter(sub["mean_ALT"], sub["mean_AST"],
                   c=style["color"], label=label,
                   s=12, alpha=0.5, edgecolors="none",
                   zorder=style["zorder"])
        print(f"  {style['label']}: n={len(sub)}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean ALT (IU/L)", fontsize=10)
    ax.set_ylabel("Mean AST (IU/L)", fontsize=10)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    # Tick labels
    ticks = [10, 100, 1000, 10000]
    tick_labels = ["10", "100", "1,000", "10,000"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "alt_vs_ast.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")

    # ── ALT distribution histograms (overlaid by species) ──
    df_alt = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df_alt = df_alt[df_alt["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df_alt["mean_ALT"] = df_alt["ALT"].apply(mean_of_array)
    df_alt = df_alt[df_alt["mean_ALT"].notna()]

    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=300)

    bins = np.logspace(np.log10(5), np.log10(20000), 40)

    for species in ["mouse", "rat", "monkey"]:
        style = SPECIES_STYLE[species]
        sub = df_alt[df_alt["species"] == species]
        if len(sub) == 0:
            continue
        ax2.hist(sub["mean_ALT"], bins=bins, alpha=0.5,
                 color=style["color"], label=f"{style['label']} (n={len(sub)})",
                 edgecolor=style["color"], linewidth=0.8)
        print(f"  ALT dist — {style['label']}: n={len(sub)}")

    ax2.set_xscale("log")
    ax2.set_xlabel("Mean ALT (IU/L)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.legend(fontsize=9, frameon=True, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ticks = [10, 100, 1000, 10000]
    tick_labels = ["10", "100", "1,000", "10,000"]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels)

    out_svg2 = OUT_DIR / "alt_distribution.svg"
    fig2.savefig(out_svg2, format="svg", bbox_inches="tight")
    fig2.savefig(out_svg2.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out_svg2}")


if __name__ == "__main__":
    main()
