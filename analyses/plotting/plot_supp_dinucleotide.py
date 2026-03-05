"""
Supplementary figure: Dinucleotide–biomarker Spearman correlation heatmap.

4 rows (Mouse ALT, Rat ALT, Mouse bFOB, Rat mFOB) × 16 columns (AA..TT).
Values are Spearman ρ, masked where p ≥ 0.05/16 (Bonferroni-corrected).

Reads: data/oligostack/processed/{hepatictoxicity,neurotoxicity}_processed.parquet
"""

from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
_data_dir = _root / "data/oligostack/processed"
OUT_DIR = _root / "typst/plots/supp_dinucleotide"

BASES = ["A", "C", "G", "T"]
BASE_PAIRS = [f"{b1}{b2}" for b1 in BASES for b2 in BASES]


def _count_bp(helm_str):
    parsed = Helm.parse(helm_str)
    if parsed is None:
        return {bp: 0 for bp in BASE_PAIRS}
    counts = {bp: 0 for bp in BASE_PAIRS}
    for i in range(parsed.length - 1):
        bp = f"{parsed.bases[i]}{parsed.bases[i + 1]}"
        if bp in counts:
            counts[bp] += 1
    return counts


def _load_contexts():
    """Load data and return list of (label, bp_df, biomarker_series) tuples."""
    hep = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    hep = hep[hep["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    hep["mean_ALT"] = hep["ALT"].apply(mean_of_array)

    neuro = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    neuro = neuro[neuro["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    neuro["mean_FOB"] = neuro["FOB_score"].apply(mean_of_array)
    neuro = neuro[neuro["mean_FOB"].notna()]

    slices = [
        ("Mouse ALT", hep[hep["species"] == "mouse"], "mean_ALT"),
        ("Rat ALT", hep[hep["species"] == "rat"], "mean_ALT"),
        ("Mouse bFOB", neuro[
            (neuro["species"] == "Mouse") & (neuro["dosage_ug"] == 700)
            & (neuro["administration_method"] == "ICV")
        ], "mean_FOB"),
        ("Rat mFOB", neuro[
            (neuro["species"] == "Rat") & (neuro["dosage_ug"] == 3000)
            & (neuro["latency_time_hours"] == 3)
        ], "mean_FOB"),
    ]

    contexts = []
    for label, df, col in slices:
        bp_df = pd.DataFrame(
            df["HELM Annotation"].apply(_count_bp).tolist(), index=df.index,
        )
        biomarker = df[col]
        contexts.append((label, bp_df, biomarker))
        print(f"  {label}: n={len(df)}")
    return contexts


def main():
    print("Dinucleotide–biomarker Spearman correlations")
    contexts = _load_contexts()

    n_rows = len(contexts)
    n_cols = len(BASE_PAIRS)
    alpha = 0.05 / n_cols  # Bonferroni

    rho_matrix = np.full((n_rows, n_cols), np.nan)
    pval_matrix = np.full((n_rows, n_cols), np.nan)

    for i, (label, bp_df, biomarker) in enumerate(contexts):
        for j, bp in enumerate(BASE_PAIRS):
            valid = bp_df[bp].notna() & biomarker.notna()
            if valid.sum() < 20:
                continue
            rho, p = spearmanr(bp_df.loc[valid, bp], biomarker[valid])
            rho_matrix[i, j] = float(rho)
            pval_matrix[i, j] = float(p)

    sig_matrix = pval_matrix < alpha

    # Transpose: 16 rows (dinucleotides) × 4 cols (biomarkers)
    rho_matrix = rho_matrix.T
    pval_matrix = pval_matrix.T
    sig_matrix = pval_matrix < alpha
    n_rows, n_cols = rho_matrix.shape

    # Build display matrix: significant cells show ρ, others NaN
    display = np.where(sig_matrix, rho_matrix, np.nan)

    # Symmetric color range
    vmax = np.nanmax(np.abs(rho_matrix[~np.isnan(rho_matrix)]))
    vmax = np.ceil(vmax * 10) / 10  # round up to nearest 0.1

    fig, ax = plt.subplots(figsize=(3.2, 6), dpi=300)

    # Gray background for non-significant cells
    ax.imshow(
        np.ones((n_rows, n_cols)),
        cmap=matplotlib.colors.ListedColormap(["#e0e0e0"]),
        aspect="auto",
    )
    im = ax.imshow(display, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            if sig_matrix[i, j]:
                val = rho_matrix[i, j]
                color = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
            else:
                ax.text(j, i, "n.s.", ha="center", va="center",
                        fontsize=5.5, color="#888888")

    # Cell borders
    for i in range(-1, n_rows):
        ax.axhline(i + 0.5, color="black", linewidth=0.3, clip_on=False)
    for j in range(-1, n_cols):
        ax.axvline(j + 0.5, color="black", linewidth=0.3, clip_on=False)

    col_labels = ["ALT\n(Mouse)", "ALT\n(Rat)",
                   "bFOB\n(Mouse)", "mFOB\n(Rat)"]
    ax.xaxis.tick_top()
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(BASE_PAIRS, fontsize=8)
    ax.set_ylabel("Dinucleotide", fontsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "dinucleotide_associations.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
