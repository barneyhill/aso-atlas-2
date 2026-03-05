"""
Blanket patenting evidence: per-gene in vitro inhibition distributions.

Box plots with jittered dots showing that for each target gene, Ionis patents
ASOs spanning the full activity range — including many poor performers —
consistent with systematic tiling for IP coverage rather than cherry-picking.

Uses the same "major" gene set as the gene circle in Figure 1.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
OUT_DIR = _root / "typst/plots/fig_blanket"


def _get_major_genes():
    """Return the major gene list from the gene circle (same cutoff logic)."""
    data_dir = _root / "data/oligostack/processed"

    iv = pd.read_parquet(data_dir / "in_vitro_inhibition_processed.parquet")
    dr = pd.read_parquet(data_dir / "dose_response_processed.parquet")
    hep = pd.read_parquet(data_dir / "hepatictoxicity_processed.parquet")
    neuro = pd.read_parquet(data_dir / "neurotoxicity_processed.parquet")

    iv_genomic = pd.read_parquet(
        data_dir / "in_vitro_inhibition_processed_with_genomic_data.parquet"
    )
    dr_genomic = pd.read_parquet(data_dir / "dose_response_with_genomic.parquet")

    gene_map = (
        pd.concat([
            iv_genomic[["target_RNA", "gene_symbol"]],
            dr_genomic[["target_RNA", "gene_symbol"]],
        ])
        .dropna(subset=["gene_symbol"])
        .drop_duplicates("target_RNA")
        .set_index("target_RNA")["gene_symbol"]
    )

    def resolve_gene(target_rna):
        if pd.isna(target_rna):
            return None
        return gene_map.get(target_rna, target_rna)

    iv_genes = iv["target_RNA"].map(resolve_gene)
    dr_genes = dr["target_RNA"].map(resolve_gene)
    iv_counts = iv_genes.groupby(iv_genes).size()
    dr_counts = dr_genes.groupby(dr_genes).size()

    biomarker_cols = ["ALT", "AST", "ALB", "TBIL", "BUN", "CREA"]

    def has_measurement(val):
        if val is None:
            return 0
        if isinstance(val, (list, np.ndarray)) and len(val) > 0:
            return 1
        return 0

    hep = hep.copy()
    hep["_meas"] = hep[biomarker_cols].apply(
        lambda row: sum(has_measurement(row[col]) for col in biomarker_cols), axis=1
    )
    hep["_gene"] = hep["target_RNA"].map(resolve_gene)
    hep_counts = hep.groupby("_gene")["_meas"].sum()

    neuro = neuro.copy()
    neuro["_meas"] = neuro["FOB_score"].apply(has_measurement)
    neuro["_gene"] = neuro["target_RNA"].map(resolve_gene)
    neuro_counts = neuro.groupby("_gene")["_meas"].sum()

    gene_meas = (
        pd.concat([iv_counts, dr_counts, hep_counts, neuro_counts])
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )

    total_all = gene_meas.sum()
    cumulative_angle = 0
    cutoff_idx = len(gene_meas)
    for i, count in enumerate(gene_meas.values):
        wedge_angle = 360 * count / total_all
        mid_angle = cumulative_angle + wedge_angle / 2
        if (mid_angle % 360) > 270:
            cutoff_idx = i
            break
        cumulative_angle += wedge_angle

    return list(gene_meas.iloc[:cutoff_idx].index), gene_map


def main():
    data_dir = _root / "data/oligostack/processed"
    major_genes, gene_map = _get_major_genes()

    iv = pd.read_parquet(data_dir / "in_vitro_inhibition_processed.parquet")
    iv["gene"] = iv["target_RNA"].map(
        lambda x: gene_map.get(x, x) if pd.notna(x) else None
    )

    # Per-compound max inhibition (one value per ASO per gene)
    max_inhib = iv.groupby(["gene", "Compound ID"])["Inhibition_pct"].max().reset_index()
    max_inhib = max_inhib[max_inhib["gene"].isin(major_genes)]

    # Sort genes by median inhibition
    medians = max_inhib.groupby("gene")["Inhibition_pct"].median().sort_values()
    gene_order = list(medians.index)

    # Count ASOs per gene for labels
    aso_counts = max_inhib.groupby("gene")["Compound ID"].nunique()

    fig, ax = plt.subplots(figsize=(7, 12), dpi=300)

    rng = np.random.default_rng(42)
    c_dot = "#4878A8"
    c_box = "#2a2a2a"

    for i, gene in enumerate(gene_order):
        vals = np.clip(
            max_inhib[max_inhib["gene"] == gene]["Inhibition_pct"].values, 0, 100,
        )

        # Subsample dots if too many (for readability)
        if len(vals) > 200:
            dot_vals = rng.choice(vals, 200, replace=False)
        else:
            dot_vals = vals

        jitter = rng.normal(0, 0.12, size=len(dot_vals))
        ax.scatter(
            dot_vals, i + jitter,
            s=1.5, alpha=0.15, color=c_dot, rasterized=True, zorder=1,
        )

        # Box plot (horizontal)
        ax.boxplot(
            vals, positions=[i], widths=0.5, patch_artist=True,
            showfliers=False, zorder=2, vert=False,
            boxprops=dict(facecolor="white", edgecolor=c_box, linewidth=0.8, alpha=0.85),
            whiskerprops=dict(color=c_box, linewidth=0.8),
            capprops=dict(color=c_box, linewidth=0.8),
            medianprops=dict(color="#c44e52", linewidth=1.5),
        )

    # 80% threshold line
    ax.axvline(80, color="#c44e52", linewidth=1, linestyle="--", alpha=0.7, zorder=3)
    ax.text(
        82, len(gene_order) - 0.5, "80% threshold",
        va="top", fontsize=8, color="#c44e52", style="italic", rotation=90,
    )

    # Gene labels with ASO counts
    labels = [f"{g}  ({aso_counts[g]:,})" for g in gene_order]
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Max in vitro inhibition (%)", fontsize=10)
    ax.set_ylim(-0.7, len(gene_order) - 0.3)
    ax.set_xlim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "blanket_patenting.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
