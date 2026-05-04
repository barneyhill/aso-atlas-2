"""
Figure 1: Dataset overview.

Panel A — Pipeline diagram (external SVG, rasterised into top row).
Panel B — Sankey diagram showing data flow across assay stages.
Panel C — Gene circle donut chart showing measurements per target gene.
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, PathPatch, Wedge
from matplotlib.path import Path as MPath

from analyses.utils.compounds import has_measurement

_root = Path(__file__).resolve().parents[2]


def draw_sankey_tapered(flows, node_positions, node_heights, node_labels, node_colors,
                        flow_colors, invisible_flows=None, ax=None):
    if invisible_flows is None:
        invisible_flows = set()

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0.05, 0.90)
    ax.set_ylim(0.15, 1.02)
    ax.axis('off')

    node_flow_positions = {idx: {'out': 0, 'in': 0} for idx in node_positions.keys()}

    for source_idx, target_idx, start_prop, end_prop in flows:
        is_invisible = (source_idx, target_idx) in invisible_flows

        source_x, source_y = node_positions[source_idx]
        target_x, target_y = node_positions[target_idx]

        source_height = node_heights[source_idx]
        target_height = node_heights[target_idx]
        flow_start_h = start_prop * source_height
        flow_end_h = end_prop * target_height

        source_y_start = source_y - source_height/2 + node_flow_positions[source_idx]['out']
        target_y_start = target_y - target_height/2 + node_flow_positions[target_idx]['in']

        node_flow_positions[source_idx]['out'] += flow_start_h
        node_flow_positions[target_idx]['in'] += flow_end_h

        source_right = source_x + 0.02
        target_left = target_x - 0.02
        ctrl_x = (source_right + target_left) / 2

        verts = [
            (source_right, source_y_start),
            (ctrl_x, source_y_start),
            (ctrl_x, target_y_start),
            (target_left, target_y_start),
            (target_left, target_y_start + flow_end_h),
            (ctrl_x, target_y_start + flow_end_h),
            (ctrl_x, source_y_start + flow_start_h),
            (source_right, source_y_start + flow_start_h),
            (source_right, source_y_start),
        ]
        codes = [
            MPath.MOVETO,
            MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.LINETO,
            MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.CLOSEPOLY,
        ]
        path = MPath(verts, codes)

        if is_invisible:
            patch = PathPatch(path, facecolor='none', edgecolor='none', alpha=0)
        else:
            color = flow_colors.get((source_idx, target_idx), 'gray')
            patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.4)
        ax.add_patch(patch)

    for node_idx, (x, y) in node_positions.items():
        if node_idx == 5:
            continue
        height = node_heights[node_idx]
        width = 0.04
        rect = mpatches.Rectangle(
            (x - width/2, y - height/2), width, height,
            facecolor=node_colors[node_idx], edgecolor='black', linewidth=1.5, zorder=10
        )
        ax.add_patch(rect)
        label = node_labels[node_idx]
        ax.text(x, y + height/2 + 0.05, label,
                ha='center', va='bottom', fontsize=12, zorder=11)

    return ax


def draw_gene_circle(in_vitro_df, dose_response_df, hepatictox_df, neurotox_df,
                     biomarker_cols, has_measurement, ax=None):
    """Draw a donut chart of measurements per target gene across all 4 data types."""
    # Load genomic-enriched parquets for gene_symbol mapping
    data_dir = _root / "data/oligostack/processed"
    iv_genomic = pd.read_parquet(data_dir / "in_vitro_inhibition_processed_with_genomic_data.parquet")
    dr_genomic = pd.read_parquet(data_dir / "dose_response_with_genomic.parquet")

    # Build target_RNA → gene_symbol lookup (use gene_symbol where available, else target_RNA)
    gene_map = pd.concat([
        iv_genomic[['target_RNA', 'gene_symbol']],
        dr_genomic[['target_RNA', 'gene_symbol']],
    ]).dropna(subset=['gene_symbol']).drop_duplicates('target_RNA').set_index('target_RNA')['gene_symbol']
    def resolve_gene(target_rna):
        if pd.isna(target_rna):
            return None
        return gene_map.get(target_rna, target_rna)

    # Count measurements per gene (using mapped gene_symbol)
    # In vitro / dose response: 1 row = 1 measurement
    iv_genes = in_vitro_df['target_RNA'].map(resolve_gene)
    dr_genes = dose_response_df['target_RNA'].map(resolve_gene)
    iv_counts = iv_genes.groupby(iv_genes).size()
    dr_counts = dr_genes.groupby(dr_genes).size()

    # Hepatic: sum of non-null biomarker columns per row
    hepatictox_df = hepatictox_df.copy()
    hepatictox_df["_meas"] = hepatictox_df[biomarker_cols].apply(
        lambda row: sum(has_measurement(row[col]) for col in biomarker_cols), axis=1
    )
    hepatictox_df["_gene"] = hepatictox_df['target_RNA'].map(resolve_gene)
    hep_counts = hepatictox_df.groupby('_gene')['_meas'].sum()

    # Neurotox: count rows with valid FOB_score
    neurotox_df = neurotox_df.copy()
    neurotox_df["_meas"] = neurotox_df["FOB_score"].apply(has_measurement)
    neurotox_df["_gene"] = neurotox_df['target_RNA'].map(resolve_gene)
    neuro_counts = neurotox_df.groupby('_gene')['_meas'].sum()

    # Combine all counts per gene
    gene_meas = (
        pd.concat([iv_counts, dr_counts, hep_counts, neuro_counts])
        .groupby(level=0).sum()
        .sort_values(ascending=False)
    )
    n_genes = len(gene_meas)
    total_meas = int(gene_meas.sum())

    # Split major vs minor: keep genes individually labelled until their
    # label midpoint enters the bottom-right of the circle (past 270°).
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
    major = gene_meas.iloc[:cutoff_idx]
    minor = gene_meas.iloc[cutoff_idx:]

    genes = list(major.index)
    counts = list(major.values.astype(int))
    genes.append('Other')
    counts.append(int(minor.sum()))
    other_count = len(minor)
    total = sum(counts)

    # Angles
    angles = [360 * c / total for c in counts]
    cumulative = [0]
    for a in angles:
        cumulative.append(cumulative[-1] + a)

    # Colors: rainbow (matching original notebook style), gray for Other
    n_major = len(major)
    colors = list(cm.rainbow(np.linspace(0, 1, n_major))) + ['#CCCCCC']

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect('equal')

    radius = 1
    inner_radius = 0.6

    for i, (gene, count, color) in enumerate(zip(genes, counts, colors)):
        start_angle = cumulative[i]
        end_angle = cumulative[i + 1]

        wedge = Wedge((0, 0), radius, start_angle, end_angle,
                       facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(wedge)

        mid_angle = (start_angle + end_angle) / 2
        mid_rad = np.radians(mid_angle)

        if gene == 'Other':
            label_dist = (radius + inner_radius) / 2
            x = label_dist * np.cos(mid_rad)
            y = label_dist * np.sin(mid_rad)
            ax.text(x, y, f'Other\ngenes\n(N={other_count})',
                    ha='center', va='center', fontsize=10, color='black')
        else:
            label_gap = 0.05
            label_dist = radius + label_gap
            x = label_dist * np.cos(mid_rad)
            y = label_dist * np.sin(mid_rad)
            on_left = 90 < (mid_angle % 360) < 270
            ha = 'right' if on_left else 'left'
            rotation = (mid_angle + 180) if on_left else mid_angle
            ax.text(x, y, gene, ha=ha, va='center',
                    rotation=rotation, rotation_mode='anchor',
                    fontsize=10, fontweight='bold')

    # Donut hole
    ax.add_patch(Circle((0, 0), inner_radius, facecolor='white', edgecolor='white'))

    ax.text(0, 0.08, f'{total_meas:,}',
            ha='center', va='center', fontsize=22, fontweight='bold')
    ax.text(0, -0.1, f'measurements across\n{n_genes:,} target genes',
            ha='center', va='center', fontsize=12)

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.15, 1.75)
    ax.axis('off')
    return ax


def main():
    data_dir = _root / "data/oligostack/processed"
    in_vitro_df = pd.read_parquet(data_dir / "in_vitro_inhibition_processed.parquet")
    dose_response_df = pd.read_parquet(data_dir / "dose_response_processed.parquet")
    neurotox_df = pd.read_parquet(data_dir / "neurotoxicity_processed.parquet")
    hepatictox_df = pd.read_parquet(data_dir / "hepatictoxicity_processed.parquet")

    biomarker_cols = ['ALB', 'ALT', 'AST', 'BUN', 'CREA', 'TBIL', 'PC_ratio']

    hepatictox_df["_measurements"] = hepatictox_df[biomarker_cols].apply(
        lambda row: sum(has_measurement(row[col]) for col in biomarker_cols), axis=1
    )
    neurotox_df["_measurements"] = neurotox_df["FOB_score"].apply(has_measurement)

    print(f"In vitro inhibition: {len(in_vitro_df):,} measurements")
    print(f"Dose response: {len(dose_response_df):,} measurements")
    print(f"Neurotox: {neurotox_df['_measurements'].sum():,} measurements")
    print(f"Hepatic tox: {hepatictox_df['_measurements'].sum():,} measurements")

    # Compute flow proportions
    in_vitro_compounds = set(in_vitro_df["Compound ID"].unique())
    dose_response_compounds = set(dose_response_df["Compound ID"].unique())
    neurotox_compounds = set(neurotox_df["Compound ID"].unique())
    hepatictox_compounds = set(hepatictox_df["Compound ID"].unique())

    in_vitro_to_dose = in_vitro_compounds & dose_response_compounds
    dose_to_neurotox = dose_response_compounds & neurotox_compounds
    dose_to_hepatic = dose_response_compounds & hepatictox_compounds

    in_vitro_total_meas = len(in_vitro_df)
    dose_total_meas = len(dose_response_df)
    hepatic_total_meas = hepatictox_df["_measurements"].sum()
    neurotox_total_meas = neurotox_df["_measurements"].sum()

    in_vitro_shared_meas = len(in_vitro_df[in_vitro_df["Compound ID"].isin(in_vitro_to_dose)])
    dose_from_in_vitro_meas = len(dose_response_df[dose_response_df["Compound ID"].isin(in_vitro_to_dose)])
    dose_to_hepatic_meas = len(dose_response_df[dose_response_df["Compound ID"].isin(dose_to_hepatic)])
    dose_to_neurotox_meas = len(dose_response_df[dose_response_df["Compound ID"].isin(dose_to_neurotox)])
    hepatic_from_dose_meas = hepatictox_df[hepatictox_df["Compound ID"].isin(dose_to_hepatic)]["_measurements"].sum()
    neurotox_from_dose_meas = neurotox_df[neurotox_df["Compound ID"].isin(dose_to_neurotox)]["_measurements"].sum()

    flow_proportions = {
        (0, 1): (in_vitro_shared_meas / in_vitro_total_meas, dose_from_in_vitro_meas / dose_total_meas),
        (1, 2): (dose_to_hepatic_meas / dose_total_meas, hepatic_from_dose_meas / hepatic_total_meas),
        (1, 3): (dose_to_neurotox_meas / dose_total_meas, neurotox_from_dose_meas / neurotox_total_meas),
    }

    # Build bottom row: sankey + gene circle
    fig, (ax_circle, ax_sankey) = plt.subplots(1, 2, figsize=(16, 8), dpi=300,
                                                gridspec_kw={'width_ratios': [3, 4], 'wspace': 0.05})

    flows = [
        (0, 1, flow_proportions[(0, 1)][0], flow_proportions[(0, 1)][1]),
        (1, 2, flow_proportions[(1, 2)][0], flow_proportions[(1, 2)][1]),
        (1, 3, flow_proportions[(1, 3)][0], flow_proportions[(1, 3)][1]),
        (0, 5, 1 - flow_proportions[(0, 1)][0], 0.5),
        (1, 5, 1 - flow_proportions[(1, 2)][0] - flow_proportions[(1, 3)][0], 0.5),
        (5, 2, 0.5, 1 - flow_proportions[(1, 2)][1]),
        (5, 3, 0.5, 1 - flow_proportions[(1, 3)][1]),
    ]

    x_start = 0.22
    x_gap = 0.25
    y_split = 0.15  # vertical offset for the hepatic/neuro fork
    node_positions = {
        0: (x_start, 0.5),
        1: (x_start + x_gap, 0.5),
        2: (x_start + 2 * x_gap, 0.5 - y_split),
        3: (x_start + 2 * x_gap, 0.5 + y_split),
        5: (1.5, 0.5),
    }

    max_meas = max(in_vitro_total_meas, dose_total_meas, hepatic_total_meas, neurotox_total_meas)
    node_heights = {
        0: in_vitro_total_meas / max_meas * 0.6,
        1: dose_total_meas / max_meas * 0.6,
        2: hepatic_total_meas / max_meas * 0.6,
        3: neurotox_total_meas / max_meas * 0.6,
        5: 0.3,
    }

    n_in_vitro_asos = in_vitro_df["Compound ID"].nunique()
    n_dose_asos = dose_response_df["Compound ID"].nunique()
    n_hepatic_asos = hepatictox_df["Compound ID"].nunique()
    n_neurotox_asos = neurotox_df["Compound ID"].nunique()

    node_labels = {
        0: f"$\\bf{{\\it{{In\\ vitro}}}}$\n$\\bf{{hit\\ screening}}$\n{in_vitro_total_meas:,} measurements\nacross {n_in_vitro_asos:,} ASOs",
        1: f"$\\bf{{\\it{{In\\ vitro}}}}$\n$\\bf{{dose\\ response}}$\n{dose_total_meas:,} measurements\nacross {n_dose_asos:,} ASOs",
        2: f"$\\bf{{\\it{{In\\ vivo}}}}$\n$\\bf{{hepatorenal\\ toxicity}}$\n{hepatic_total_meas:,} measurements\nacross {n_hepatic_asos:,} ASOs",
        3: f"$\\bf{{\\it{{In\\ vivo}}}}$\n$\\bf{{neuro\\ tolerability}}$\n{neurotox_total_meas:,} measurements\nacross {n_neurotox_asos:,} ASOs",
        5: "",
    }

    node_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 5: "none"}
    flow_colors = {(0, 1): "#ff7f0e", (1, 2): "#2ca02c", (1, 3): "#d62728"}
    invisible_flows = {(0, 5), (1, 5), (5, 2), (5, 3)}

    draw_sankey_tapered(flows, node_positions, node_heights, node_labels,
                        node_colors, flow_colors, invisible_flows, ax=ax_sankey)

    # Panel C: Gene circle donut chart
    draw_gene_circle(in_vitro_df, dose_response_df, hepatictox_df, neurotox_df,
                     biomarker_cols, has_measurement, ax=ax_circle)

    # Panel labels — use figure coords for vertical alignment
    bb_b = ax_sankey.get_position()
    bb_c = ax_circle.get_position()
    label_y = max(bb_b.y1, bb_c.y1) + 0.01
    fig.text(bb_c.x0, label_y, "B", fontsize=16, fontweight="bold", va="bottom")
    fig.text(bb_b.x0, label_y, "C", fontsize=16, fontweight="bold", va="bottom")

    sankey_gene_path = _root / "typst/plots/fig_atlas" / "sankey_gene_circle.svg"
    fig.savefig(sankey_gene_path, format="svg", bbox_inches='tight')
    fig.savefig(sankey_gene_path.with_suffix(".png"), format="png", dpi=300, bbox_inches='tight')
    print(f"Saved {sankey_gene_path}")


if __name__ == "__main__":
    main()
