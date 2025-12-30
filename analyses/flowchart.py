import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Flowchart: In Vitro Inhibition → Dose Response → (Hepatic Tox + Neuro Tox)

    - Box heights proportional to number of measurements
    - Flow heights proportional to measurements with shared ASOs / total measurements
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # Load processed data
    in_vitro_df = pd.read_parquet("../data/oligostack/processed/in_vitro_inhibition_processed.parquet")
    dose_response_df = pd.read_parquet("../data/oligostack/processed/dose_response_processed.parquet")
    neurotox_df = pd.read_parquet("../data/oligostack/processed/neurotoxicity_processed.parquet")
    hepatictox_df = pd.read_parquet("../data/oligostack/processed/hepatictoxicity_processed.parquet")

    # For in_vitro and dose_response: 1 row = 1 measurement
    # For hepatic: count non-empty values across biomarker columns (each is a list)
    # For neurotox: count values in FOB_score lists

    biomarker_cols = ['ALB', 'ALT', 'AST', 'BUN', 'CREA', 'TBIL', 'PC_ratio']

    def has_measurement(val):
        """Check if a list/array has any non-NaN values (counts as 1 measurement)."""
        if val is None:
            return 0
        if isinstance(val, (list, np.ndarray)):
            has_valid = any(x is not None and not (isinstance(x, float) and np.isnan(x)) for x in val)
            return 1 if has_valid else 0
        return 0

    # Add measurement counts to dataframes
    # Each non-empty list counts as 1 measurement
    hepatictox_df["_measurements"] = hepatictox_df[biomarker_cols].apply(
        lambda row: sum(has_measurement(row[col]) for col in biomarker_cols), axis=1
    )
    neurotox_df["_measurements"] = neurotox_df["FOB_score"].apply(has_measurement)

    print(f"In vitro inhibition: {len(in_vitro_df):,} measurements")
    print(f"Dose response: {len(dose_response_df):,} measurements")
    print(f"Neurotox: {neurotox_df['_measurements'].sum():,} measurements")
    print(f"Hepatic tox: {hepatictox_df['_measurements'].sum():,} measurements")
    return dose_response_df, hepatictox_df, in_vitro_df, neurotox_df


@app.cell
def _(dose_response_df, hepatictox_df, in_vitro_df, neurotox_df):
    # Get compound sets
    in_vitro_compounds = set(in_vitro_df["Compound ID"].unique())
    dose_response_compounds = set(dose_response_df["Compound ID"].unique())
    neurotox_compounds = set(neurotox_df["Compound ID"].unique())
    hepatictox_compounds = set(hepatictox_df["Compound ID"].unique())

    # Shared ASO sets
    in_vitro_to_dose = in_vitro_compounds & dose_response_compounds
    dose_to_neurotox = dose_response_compounds & neurotox_compounds
    dose_to_hepatic = dose_response_compounds & hepatictox_compounds

    # Total measurements per dataset
    in_vitro_total_meas = len(in_vitro_df)
    dose_total_meas = len(dose_response_df)
    hepatic_total_meas = hepatictox_df["_measurements"].sum()
    neurotox_total_meas = neurotox_df["_measurements"].sum()

    # Measurements with shared ASOs
    in_vitro_shared_meas = len(in_vitro_df[in_vitro_df["Compound ID"].isin(in_vitro_to_dose)])
    dose_from_in_vitro_meas = len(dose_response_df[dose_response_df["Compound ID"].isin(in_vitro_to_dose)])
    dose_to_hepatic_meas = len(dose_response_df[dose_response_df["Compound ID"].isin(dose_to_hepatic)])
    dose_to_neurotox_meas = len(dose_response_df[dose_response_df["Compound ID"].isin(dose_to_neurotox)])
    hepatic_from_dose_meas = hepatictox_df[hepatictox_df["Compound ID"].isin(dose_to_hepatic)]["_measurements"].sum()
    neurotox_from_dose_meas = neurotox_df[neurotox_df["Compound ID"].isin(dose_to_neurotox)]["_measurements"].sum()

    # Flow proportions based on measurements
    # Flow A->B: start_prop = meas_with_shared_ASOs/total_meas_A, end_prop = meas_with_shared_ASOs/total_meas_B
    flow_proportions = {
        (0, 1): (in_vitro_shared_meas / in_vitro_total_meas, dose_from_in_vitro_meas / dose_total_meas),
        (1, 2): (dose_to_hepatic_meas / dose_total_meas, hepatic_from_dose_meas / hepatic_total_meas),
        (1, 3): (dose_to_neurotox_meas / dose_total_meas, neurotox_from_dose_meas / neurotox_total_meas),
    }

    print(f"Total measurements: in_vitro={in_vitro_total_meas:,}, dose={dose_total_meas:,}, hepatic={hepatic_total_meas:,}, neurotox={neurotox_total_meas:,}")
    print()
    print(f"In vitro -> Dose: {flow_proportions[(0,1)][0]:.1%} of in_vitro meas -> {flow_proportions[(0,1)][1]:.1%} of dose meas")
    print(f"Dose -> Hepatic: {flow_proportions[(1,2)][0]:.1%} of dose meas -> {flow_proportions[(1,2)][1]:.1%} of hepatic meas")
    print(f"Dose -> Neurotox: {flow_proportions[(1,3)][0]:.1%} of dose meas -> {flow_proportions[(1,3)][1]:.1%} of neurotox meas")
    return (
        dose_total_meas,
        flow_proportions,
        hepatic_total_meas,
        in_vitro_total_meas,
        neurotox_total_meas,
    )


@app.cell
def _(
    dose_response_df,
    dose_total_meas,
    flow_proportions,
    hepatic_total_meas,
    hepatictox_df,
    in_vitro_df,
    in_vitro_total_meas,
    neurotox_df,
    neurotox_total_meas,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import PathPatch, Polygon
    from matplotlib.path import Path

    def draw_sankey_tapered(flows, node_positions, node_heights, node_labels, node_colors,
                            flow_colors, invisible_flows=None, figsize=(14, 8)):
        """
        Custom Sankey diagram with tapering flows based on ASO proportions.
        Flow start/end heights are proportional to shared ASOs relative to each node's total.
        """
        if invisible_flows is None:
            invisible_flows = set()

        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')

        # Track vertical positions for flows at each node
        node_flow_positions = {idx: {'out': 0, 'in': 0} for idx in node_positions.keys()}

        # Draw flows
        for source_idx, target_idx, start_prop, end_prop in flows:
            is_invisible = (source_idx, target_idx) in invisible_flows

            source_x, source_y = node_positions[source_idx]
            target_x, target_y = node_positions[target_idx]

            # Calculate flow heights at source and target based on proportions
            source_height = node_heights[source_idx]
            target_height = node_heights[target_idx]
            flow_start_h = start_prop * source_height
            flow_end_h = end_prop * target_height

            # Get current positions on source and target nodes
            source_y_start = source_y - source_height/2 + node_flow_positions[source_idx]['out']
            target_y_start = target_y - target_height/2 + node_flow_positions[target_idx]['in']

            # Update positions
            node_flow_positions[source_idx]['out'] += flow_start_h
            node_flow_positions[target_idx]['in'] += flow_end_h

            # Create bezier curve for tapering flow
            source_right = source_x + 0.01
            target_left = target_x - 0.01

            # Control points for Bezier curves
            ctrl_x = (source_right + target_left) / 2

            # Bottom edge: source_y_start to target_y_start
            # Top edge: source_y_start + flow_start_h to target_y_start + flow_end_h
            verts = [
                (source_right, source_y_start),                    # Start bottom
                (ctrl_x, source_y_start),                          # Control 1
                (ctrl_x, target_y_start),                          # Control 2
                (target_left, target_y_start),                     # End bottom
                (target_left, target_y_start + flow_end_h),        # End top
                (ctrl_x, target_y_start + flow_end_h),             # Control 2 (return)
                (ctrl_x, source_y_start + flow_start_h),           # Control 1 (return)
                (source_right, source_y_start + flow_start_h),     # Start top
                (source_right, source_y_start),                    # Close
            ]

            codes = [
                Path.MOVETO,
                Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.LINETO,
                Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.CLOSEPOLY,
            ]

            path = Path(verts, codes)

            if is_invisible:
                patch = PathPatch(path, facecolor='none', edgecolor='none', alpha=0)
            else:
                color = flow_colors.get((source_idx, target_idx), 'gray')
                patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.4)

            ax.add_patch(patch)

        # Draw nodes (on top of flows)
        for node_idx, (x, y) in node_positions.items():
            if node_idx == 5:  # Skip sink node
                continue

            height = node_heights[node_idx]
            width = 0.02

            rect = mpatches.Rectangle(
                (x - width/2, y - height/2),
                width,
                height,
                facecolor=node_colors[node_idx],
                edgecolor='black',
                linewidth=1.5,
                zorder=10
            )
            ax.add_patch(rect)

            # Add label above the node
            label = node_labels[node_idx]
            ax.text(x, y + height/2 + 0.05, label,
                    ha='center', va='bottom', fontsize=10, zorder=11)

        plt.tight_layout()
        return fig

    # Define flows: (source_idx, target_idx, start_proportion, end_proportion)
    # Proportions are based on shared ASOs relative to each node's total ASOs
    # 0: In vitro inhibition
    # 1: Dose response
    # 2: Hepatic tox
    # 3: Neuro tox
    # 5: Sink (invisible, off-screen)

    flows = [
        (0, 1, flow_proportions[(0, 1)][0], flow_proportions[(0, 1)][1]),  # In vitro -> Dose
        (1, 2, flow_proportions[(1, 2)][0], flow_proportions[(1, 2)][1]),  # Dose -> Hepatic
        (1, 3, flow_proportions[(1, 3)][0], flow_proportions[(1, 3)][1]),  # Dose -> Neuro
        # Invisible flows for sizing (use remaining proportions)
        (0, 5, 1 - flow_proportions[(0, 1)][0], 0.5),  # In vitro -> Sink
        (1, 5, 1 - flow_proportions[(1, 2)][0] - flow_proportions[(1, 3)][0], 0.5),  # Dose -> Sink
        (5, 2, 0.5, 1 - flow_proportions[(1, 2)][1]),  # Sink -> Hepatic
        (5, 3, 0.5, 1 - flow_proportions[(1, 3)][1]),  # Sink -> Neuro
    ]

    node_positions = {
        0: (0.15, 0.5),   # In vitro inhibition
        1: (0.15+(0.8-0.15)/2, 0.5),   # Dose response
        2: (0.80, 0.35),  # Hepatic tox (lower branch)
        3: (0.80, 0.65),  # Neuro tox (upper branch)
        5: (1.5, 0.5),    # Sink (off-screen)
    }

    # Node heights proportional to number of measurements
    max_meas = max(in_vitro_total_meas, dose_total_meas, hepatic_total_meas, neurotox_total_meas)
    node_heights = {
        0: in_vitro_total_meas / max_meas * 0.6,
        1: dose_total_meas / max_meas * 0.6,
        2: hepatic_total_meas / max_meas * 0.6,
        3: neurotox_total_meas / max_meas * 0.6,
        5: 0.3,  # Sink
    }

    # Labels with measurement counts and ASO counts
    n_in_vitro_asos = in_vitro_df["Compound ID"].nunique()
    n_dose_asos = dose_response_df["Compound ID"].nunique()
    n_hepatic_asos = hepatictox_df["Compound ID"].nunique()
    n_neurotox_asos = neurotox_df["Compound ID"].nunique()

    node_labels = {
        0: f"$\\bf{{In\\ vitro}}$\n$\\bf{{hit\\ screening}}$\n{in_vitro_total_meas:,} measurements\nacross {n_in_vitro_asos:,} ASOs",
        1: f"$\\bf{{In\\ vitro}}$\n$\\bf{{dose\\ response}}$\n{dose_total_meas:,} measurements\nacross {n_dose_asos:,} ASOs",
        2: f"$\\bf{{In \\ vivo}}$\n$\\bf{{hepatorenal\\ toxicity}}$\n{hepatic_total_meas:,} measurements\nacross {n_hepatic_asos:,} ASOs",
        3: f"$\\bf{{In \\ vivo}}$\n$\\bf{{neurological\\ toxicity}}$\n{neurotox_total_meas:,} measurements\nacross {n_neurotox_asos:,} ASOs",
        5: "",
    }

    node_colors = {
        0: "#1f77b4",  # Blue
        1: "#ff7f0e",  # Orange
        2: "#2ca02c",  # Green
        3: "#d62728",  # Red
        5: "none",
    }

    flow_colors = {
        (0, 1): "#ff7f0e",  # Orange (Dose Response destination)
        (1, 2): "#2ca02c",  # Green (Hepatic tox destination)
        (1, 3): "#d62728",  # Red (Neuro tox destination)
    }

    invisible_flows = {
        (0, 5),
        (1, 5),
        (5, 2),
        (5, 3),
    }

    fig = draw_sankey_tapered(flows, node_positions, node_heights, node_labels,
                              node_colors, flow_colors, invisible_flows)

    fig.savefig("../plots/flowchart_sankey.svg", format="svg")
    fig
    return


if __name__ == "__main__":
    app.run()
