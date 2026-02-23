"""
Figure 5: Cross-species concordance and mouse biomarker correlations.

(A) Cross-species Spearman ρ with 95% Fisher z CIs for ALT, AST, TBIL, FOB.
(B) FOB cross-species integer heatmap.
(C) Mouse inter-biomarker Spearman correlation matrix.

Reads: data/results/{hepatotox,neurotox}.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
HEPATOTOX_JSON = _root / "data/results/hepatotox.json"
NEUROTOX_JSON = _root / "data/results/neurotox.json"
OUT_DIR = _root / "typst/plots/fig5"


def _format_p(p: float) -> str:
    """Format p-value for display."""
    if p < 1e-100:
        return "p < 10\u207b\u00b9\u2070\u2070"
    if p < 0.001:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


def draw_cross_species_rho(ax, biomarkers):
    """Draw dot + 95% CI plot of cross-species Spearman ρ.

    biomarkers: list of (label, rho, ci_lo, ci_hi, pval, n) tuples.
    Only significant correlations (Bonferroni-corrected) are shown.
    """
    n_tests = len(biomarkers)
    alpha = 0.05 / n_tests

    labels, rhos, ci_los, ci_his, pvals, ns = zip(*biomarkers)
    y = np.arange(len(labels))

    for i in range(len(labels)):
        sig = pvals[i] < alpha
        color = "#4878A8" if sig else "#cccccc"
        ax.errorbar(rhos[i], y[i],
                    xerr=[[rhos[i] - ci_los[i]], [ci_his[i] - rhos[i]]],
                    fmt="o", color=color, ecolor=color, capsize=4,
                    markersize=6, elinewidth=1.5)
        # n annotation
        ax.text(ci_his[i] + 0.02, y[i], f"n={ns[i]}",
                va="center", fontsize=7, color="#666666")
        if not sig:
            ax.text(rhos[i], y[i] + 0.25, "n.s.", ha="center",
                    fontsize=7, color="#999999")

    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spearman \u03c1 (mouse vs rat)", fontsize=9)
    ax.set_xlim(-0.1, 1.0)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)


def draw_fob_heatmap(ax, mouse_vals, rat_vals, rho, pval, n):
    """Draw FOB integer-grid heatmap."""
    import pandas as pd

    mouse_int = np.round(mouse_vals).astype(int)
    rat_int = np.round(rat_vals).astype(int)

    paired = pd.DataFrame({"Mouse": mouse_int, "Rat": rat_int})
    count_matrix = paired.groupby(["Mouse", "Rat"]).size().unstack(fill_value=0)
    full_index = range(8)
    count_matrix = count_matrix.reindex(index=full_index, columns=full_index, fill_value=0)

    ax.imshow(count_matrix.values, origin="lower", cmap="Blues", aspect="equal")

    for i in range(8):
        for j in range(8):
            val = count_matrix.values[i, j]
            if val > 0:
                color = "white" if val > count_matrix.values.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    for i in range(-1, 8):
        ax.axhline(i + 0.5, color="black", linewidth=0.5)
        ax.axvline(i + 0.5, color="black", linewidth=0.5)

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("Mouse mean bFOB score", fontsize=9)
    ax.set_ylabel("Rat mean mFOB score", fontsize=9)


def draw_corr_matrix(ax, corr_data):
    """Draw inter-biomarker Spearman correlation matrix."""
    biomarkers = corr_data["biomarkers"]
    rho = np.array(corr_data["rho"])
    pvals = np.array(corr_data["p_values"])
    n = len(biomarkers)

    rho_display = np.where(np.isnan(rho), 0, rho)
    im = ax.imshow(rho_display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    n_pairs = n * (n - 1) // 2
    alpha = 0.05 / n_pairs

    for i in range(n):
        for j in range(n):
            if not np.isnan(rho[i, j]):
                val = rho[i, j]
                sig = (i == j) or pvals[i, j] < alpha
                color = "white" if abs(val) > 0.6 else ("black" if sig else "#999999")
                label = f"{val:.2f}" if sig else f"{val:.2f}\n(n.s.)"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(biomarkers, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(biomarkers, fontsize=8)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman \u03c1", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def main():
    for p in [HEPATOTOX_JSON, NEUROTOX_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `just hagerdorn` first.")

    with open(HEPATOTOX_JSON) as f:
        hep_data = json.load(f)
    with open(NEUROTOX_JSON) as f:
        neuro_data = json.load(f)

    hep_cs = hep_data["cross_species"]
    neuro_cs = neuro_data["cross_species"]
    corr_data = hep_data["mouse_biomarker_correlations"]

    # Collect cross-species data for panel A
    biomarkers = []
    for bm in ["ALT", "AST", "TBIL"]:
        cs = hep_cs[bm]
        biomarkers.append((
            bm, cs["spearman_rho"], cs["spearman_ci_lo"], cs["spearman_ci_hi"],
            cs["spearman_p"], cs["n_shared"],
        ))
    fob_cs = neuro_cs["FOB"]
    biomarkers.append((
        "bFOB vs mFOB", fob_cs["spearman_rho"], fob_cs["spearman_ci_lo"], fob_cs["spearman_ci_hi"],
        fob_cs["spearman_p"], fob_cs["n_shared"],
    ))

    # 1×3 layout
    fig = plt.figure(figsize=(14, 4), dpi=300)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.8, 1, 1.1],
                           wspace=0.4)

    # A: cross-species ρ
    ax_a = fig.add_subplot(gs[0, 0])
    draw_cross_species_rho(ax_a, biomarkers)
    ax_a.text(-0.15, 1.08, "A", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold")

    # B: FOB heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    draw_fob_heatmap(
        ax_b, np.array(fob_cs["mouse_values"]), np.array(fob_cs["rat_values"]),
        rho=fob_cs["spearman_rho"], pval=fob_cs["spearman_p"], n=fob_cs["n_shared"],
    )
    ax_b.text(-0.08, 1.08, "B", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold")

    # C: correlation matrix
    ax_c = fig.add_subplot(gs[0, 2])
    draw_corr_matrix(ax_c, corr_data)
    ax_c.text(-0.08, 1.08, "C", transform=ax_c.transAxes,
              fontsize=14, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig5.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
