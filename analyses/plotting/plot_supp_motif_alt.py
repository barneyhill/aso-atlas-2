"""
Supplementary figure: Mouse ALT distributions by trinucleotide motif.

Three groups:
  1. All ASOs (mouse)
  2. ASOs containing TGC motif
  3. ASOs containing TCC motif

Mann-Whitney U tests compare group 1 (All) vs groups 2 and 3.
Dashed line at mouse ALT ULN (75 IU/L).

Reads: data/oligostack/processed/hepatictoxicity_processed.parquet
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
DATA_PATH = _root / "data/oligostack/processed/hepatictoxicity_processed.parquet"
OUT_DIR = _root / "typst/plots/supp_motif"

MOUSE_ALT_ULN = 75  # IU/L (Otto et al. 2016)


def _dna_sequence(helm_str: str) -> str | None:
    parsed = Helm.parse(helm_str)
    if parsed is None:
        return None
    return parsed.dna_sequence


def _format_p(p: float) -> str:
    if p < 1e-4:
        return f"p = {p:.1e}"
    if p < 0.001:
        return f"p = {p:.4f}"
    if p < 0.05:
        return f"p = {p:.3f}"
    return f"p = {p:.2f} (n.s.)"


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df[
        (df["species"] == "mouse")
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()

    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df.dropna(subset=["mean_ALT"])

    # Aggregate to one value per compound
    compound = df.groupby("Compound ID").agg(
        mean_ALT=("mean_ALT", "mean"),
        helm=("HELM Annotation", "first"),
    )
    compound["seq"] = compound["helm"].apply(_dna_sequence)
    compound = compound.dropna(subset=["seq"])

    compound["has_TGC"] = compound["seq"].str.contains("TGC")
    compound["has_TCC"] = compound["seq"].str.contains("TCC")

    all_alt = compound["mean_ALT"].values
    tgc_alt = compound.loc[compound["has_TGC"], "mean_ALT"].values
    tcc_alt = compound.loc[compound["has_TCC"], "mean_ALT"].values

    # Stats: ALL vs TGC, ALL vs TCC
    _, p_tgc = mannwhitneyu(all_alt, tgc_alt, alternative="two-sided")
    _, p_tcc = mannwhitneyu(all_alt, tcc_alt, alternative="two-sided")

    groups = [
        ("All ASOs", all_alt, "#888888"),
        ("TGC-containing", tgc_alt, "#d62728"),
        ("TCC-containing", tcc_alt, "#1f77b4"),
    ]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    positions = [0, 1, 2]
    for i, (label, vals, color) in enumerate(groups):
        bplot = ax.boxplot(
            [vals],
            positions=[positions[i]],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )
        bplot["boxes"][0].set_facecolor(color)
        bplot["boxes"][0].set_alpha(0.3)

        jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            positions[i] + jitter, vals,
            c=color, s=8, alpha=0.4, edgecolors="none", zorder=3,
        )

    # ULN dashed line
    ax.axhline(
        MOUSE_ALT_ULN, color="black", linestyle="--", linewidth=1, alpha=0.7,
        zorder=2,
    )
    ax.text(
        2.55, MOUSE_ALT_ULN, f"ULN = {MOUSE_ALT_ULN} IU/L",
        va="center", fontsize=8, color="black", alpha=0.7,
    )

    # Significance brackets
    y_max = max(np.percentile(v, 99) for _, v, _ in groups)
    bracket_y = y_max * 1.3

    for j, (p_val, x_right) in enumerate([(p_tgc, 1), (p_tcc, 2)]):
        y = bracket_y * (1 + 0.2 * j)
        ax.plot([0, 0, x_right, x_right], [y * 0.95, y, y, y * 0.95],
                lw=1, color="black")
        ax.text((0 + x_right) / 2, y * 1.02, _format_p(p_val),
                ha="center", va="bottom", fontsize=8)

    ax.set_yscale("log")
    ax.set_ylabel("Mean ALT (IU/L)", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"All\n(n={len(all_alt)})",
         f"TGC\n(n={len(tgc_alt)})",
         f"TCC\n(n={len(tcc_alt)})"],
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Mouse ALT by trinucleotide motif", fontsize=11, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "supp_motif_alt.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
