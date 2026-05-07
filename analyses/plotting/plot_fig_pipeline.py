"""
Figure 2: The Pipeline Problem.

Panel A — Pipeline attrition funnel (tapered flow connecting boxes at each stage).
Panel B — Stage measurement distributions (8-panel histograms with thresholds).

Reads pre-computed results from data/results/pipeline_results.json.
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

_root = Path(__file__).resolve().parents[2]
RESULTS_PATH = _root / "data/results/pipeline_results.json"
OUT_DIR = _root / "typst/plots/fig_pipeline"


def draw_pipeline_attrition(results, stages, ax, x_positions=None):
    """Draw the pipeline attrition funnel on the given axes."""
    n_stages = len(stages)
    asos = results["asos_at_stage"]
    props = results["proportions"]
    costs = results["costs_per_stage"]

    log_asos = np.array([np.log10(max(a, 1)) for a in asos])
    min_log, max_log = log_asos.min(), log_asos.max()
    rng = max_log - min_log if max_log > min_log else 1
    box_heights = 0.15 + (log_asos - min_log) / rng * 0.40

    if x_positions is None:
        x_positions = np.linspace(0.07, 0.93, n_stages + 1)
    y_center = 0.45
    box_width = 0.035

    # Tapered flows between stages
    for i in range(n_stages):
        x1 = x_positions[i] + box_width / 2
        x2 = x_positions[i + 1] - box_width / 2
        h1 = box_heights[i]
        h2 = box_heights[i + 1]
        ctrl_x = (x1 + x2) / 2

        verts = [
            (x1, y_center - h1 / 2), (ctrl_x, y_center - h1 / 2),
            (ctrl_x, y_center - h2 / 2), (x2, y_center - h2 / 2),
            (x2, y_center + h2 / 2), (ctrl_x, y_center + h2 / 2),
            (ctrl_x, y_center + h1 / 2), (x1, y_center + h1 / 2),
            (x1, y_center - h1 / 2),
        ]
        codes = [
            MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.CLOSEPOLY,
        ]
        path = MPath(verts, codes)
        patch = PathPatch(path, facecolor="#e0e0e0", edgecolor="none", alpha=0.7)
        ax.add_patch(patch)

        filter_text = stages[i]["threshold"]
        prop_text = f"{props[i] * 100:.0f}% ASOs\nhave\n{filter_text}"
        ax.text((x1 + x2) / 2, y_center, prop_text,
                ha="center", va="center", fontsize=12.1, color="#404040")

    # Stage boxes
    for i in range(n_stages + 1):
        x = x_positions[i]
        h = box_heights[i]

        if i < n_stages:
            color = stages[i]["color"]
            label_name = stages[i]["short_name"]
        else:
            color = "#C44E52"
            label_name = "Clinical\ncandidate"

        rect = mpatches.Rectangle(
            (x - box_width / 2, y_center - h / 2), box_width, h,
            facecolor=color, edgecolor="white", linewidth=1, zorder=10,
        )
        ax.add_patch(rect)

        n_label = f"{asos[i]:,}" if asos[i] != 1 else "1"
        aso_label = "ASOs" if asos[i] != 1 else "ASO"
        ax.text(x, y_center + 0.005, n_label, ha="center", va="bottom",
                fontsize=12.1, fontweight="bold", color="white", zorder=11)
        ax.text(x, y_center - 0.005, aso_label, ha="center", va="top",
                fontsize=7.5, fontweight="bold", color="white", zorder=11)

        if i < n_stages:
            ax.text(x, y_center + h / 2 + 0.095, label_name,
                    ha="center", va="center", fontsize=12.1, fontweight="bold", color="#202020")
            c = costs[i]
            cost_total = f"{c / 1e6:.1f}M" if c >= 1e6 else f"{c / 1e3:.0f}K"
            cpa = stages[i]["cost_per_aso"]
            cost_per = f"{cpa / 1e3:.0f}K" if cpa >= 1000 else f"{cpa:.0f}"
            cost_text = "\\$" + cost_total + " @ \\$" + cost_per + "/ASO"
            ax.text(x, y_center + h / 2 + 0.035, cost_text,
                    ha="center", va="center", fontsize=11, color="#303030")
        else:
            prev_h = box_heights[i - 1]
            ax.text(x, y_center + prev_h / 2 + 0.095, label_name,
                    ha="center", va="center", fontsize=12.1, fontweight="bold", color="#202020")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_stage_distributions(distributions, sample_sizes, stages, axes):
    """Draw 7 histograms on a single-row grid."""
    thresholds = [s["threshold_value"] for s in stages]
    threshold_ops = [s["threshold_op"] for s in stages]

    for i in range(7):
        ax = axes[i]
        data = np.array(distributions[i])
        stage = stages[i]
        n_pass, n_total = sample_sizes[i]

        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(f"{stage['short_name']}\n(n=0)")
            continue

        if i == 0:
            # Efficacy: clip to [0, 100]
            data_plot = np.clip(data[~np.isnan(data)], 0, 100)
            threshold_plot = thresholds[i]
            xlabel = "Inhibition (%)"
        elif i == 1:
            # Potency: log10
            data_plot = np.log10(data[data > 0])
            threshold_plot = np.log10(thresholds[i])
            xlabel = "IC\u2085\u2080 (nM)"
        elif i in [2, 4, 6]:
            # ALT stages: log10
            data_plot = np.log10(data[(~np.isnan(data)) & (data > 0)])
            threshold_plot = np.log10(thresholds[i])
            xlabel = "ALT (IU/L)"
        elif i == 3:
            data_plot = np.round(data[~np.isnan(data)]).astype(int)
            threshold_plot = 1.5  # <=1 threshold: line between 1 and 2
            xlabel = "bFOB Score"
        else:
            data_plot = np.round(data[~np.isnan(data)]).astype(int)
            threshold_plot = 1.5  # <=1 threshold: line between 1 and 2
            xlabel = "mFOB Score"

        if i in [3, 5]:
            bins = np.arange(-0.5, data_plot.max() + 1.5, 1)
            ax.hist(data_plot, bins=bins, color=stage["color"], alpha=0.7, edgecolor="black")
            ax.set_xticks(range(0, int(data_plot.max()) + 1))
        elif i == 6:
            ax.hist(data_plot, bins=12, color=stage["color"], alpha=0.7, edgecolor="black")
        else:
            ax.hist(data_plot, bins=30, color=stage["color"], alpha=0.7, edgecolor="black")
        ax.axvline(threshold_plot, color="red", linestyle="--", linewidth=2)

        # Set xlim: efficacy [0,100], log panels [1, ...], FOB [-0.5, ...]
        xlim = ax.get_xlim()
        if i == 0:
            ax.set_xlim(0, 100)
            xlim = (0, 100)
        elif i in [1, 2, 4, 6]:
            ax.set_xlim(1, xlim[1])
            xlim = (1, xlim[1])
        elif i in [3, 5]:
            ax.set_xlim(-0.5, xlim[1])
            xlim = (-0.5, xlim[1])
        else:
            ax.set_xlim(0, xlim[1])
            xlim = (0, xlim[1])

        # Log-scale panels: show real values on the axis (10, 100, 1000, …)
        if i in [1, 2, 4, 6]:
            lo, hi = ax.get_xlim()
            ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([
                f"{10**int(t)/1000:g}K" if 10**int(t) >= 1000 else f"{10**int(t):g}"
                for t in ticks
            ])

        # Shade the fail region in grey
        if threshold_ops[i] == ">":
            ax.axvspan(xlim[0], threshold_plot, alpha=0.15, color="grey")
        elif threshold_ops[i] == "<":
            ax.axvspan(threshold_plot, xlim[1], alpha=0.15, color="grey")
        elif threshold_ops[i] == "<=":
            ax.axvspan(threshold_plot, xlim[1], alpha=0.15, color="grey")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Two-line title: stage name (bold) + "(X% of ASOs have <threshold>)"
        # with threshold portion in red
        prop = n_pass / n_total * 100 if n_total > 0 else 0
        title_name = stage["short_name"].replace("\n", " ")
        threshold_text = stage["threshold"]
        ax.set_title("")
        ax.text(0.5, 1.20, title_name, fontsize=12.1, fontweight="bold",
                ha="center", va="bottom", transform=ax.transAxes)
        prefix = f"({prop:.0f}% of ASOs have "
        ta_prefix = TextArea(prefix, textprops=dict(fontsize=9, color="black"))
        ta_thresh = TextArea(threshold_text, textprops=dict(fontsize=9, color="red",
                                                             fontweight="bold"))
        ta_close = TextArea(")", textprops=dict(fontsize=9, color="black"))
        pack = HPacker(children=[ta_prefix, ta_thresh, ta_close],
                       pad=0, sep=0, align="baseline")
        ab = AnchoredOffsetbox(loc="lower center", child=pack,
                               frameon=False,
                               bbox_to_anchor=(0.5, 1.03),
                               bbox_transform=ax.transAxes)
        ax.add_artist(ab)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run `just analysis` first."
        )

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    baseline = data["baseline"]
    stages = data["stages"]
    distributions = data["distributions"]
    sample_sizes = data["sample_sizes"]

    # Two-row figure: Panel A (7 histograms) on top, Panel B (funnel) below
    fig = plt.figure(figsize=(18, 11), dpi=300)

    # Funnel x-positions: 8 evenly-spaced stage-box centres
    n_stages = len(stages)
    shared_x = np.linspace(0.07, 0.93, n_stages + 1)
    # Flow midpoints — arrow targets (between consecutive stage boxes)
    flow_cx = (shared_x[:-1] + shared_x[1:]) / 2

    # Histogram centres — wider spread, symmetric about 0.5 so the
    # middle histogram (mouse neurotox) sits exactly above its flow.
    hist_x = np.linspace(0.06, 0.94, n_stages)

    # Panel A — histograms at hist_x positions, sized from funnel spacing
    spacing = shared_x[1] - shared_x[0]
    pa_gap = 0.02
    pa_w = spacing - pa_gap
    fig_w, fig_h = fig.get_size_inches()
    pa_h = pa_w * fig_w / fig_h
    pa_bot = 0.73 - pa_h
    axes_dist = []
    for i in range(n_stages):
        x0 = hist_x[i] - pa_w / 2
        axes_dist.append(fig.add_axes([x0, pa_bot, pa_w, pa_h]))
    draw_stage_distributions(distributions, sample_sizes, stages, axes_dist)
    for ax in axes_dist:
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.08)

    # Panel B — funnel below, using shared_x converted to funnel axes coords
    funnel_left, funnel_width = 0.05, 0.90
    ax_funnel = fig.add_axes([funnel_left, 0.02, funnel_width, 0.50])
    funnel_x = (shared_x - funnel_left) / funnel_width
    draw_pipeline_attrition(baseline, stages, ax_funnel, x_positions=funnel_x)

    # Bézier S-curve arrows from each histogram to its flow centre.
    # Arrow end sits a fixed distance above each flow's top edge.
    funnel_bot, funnel_h_size = 0.02, 0.50
    y_center_funnel = 0.45
    asos = baseline["asos_at_stage"]
    log_asos = np.array([np.log10(max(a, 1)) for a in asos])
    rng = log_asos.max() - log_asos.min() or 1
    box_h = 0.15 + (log_asos - log_asos.min()) / rng * 0.40
    arrow_gap = 0.025  # fixed gap above flow top (in funnel axes coords)

    arrow_y0 = pa_bot - 0.05
    for i in range(n_stages):
        x0 = hist_x[i]
        x1 = flow_cx[i]
        # Flow top at its centre = y_center + (h_left + h_right) / 4
        flow_top = y_center_funnel + (box_h[i] + box_h[i + 1]) / 4
        arrow_y1 = funnel_bot + funnel_h_size * (flow_top + arrow_gap)
        arrow_y_mid = (arrow_y0 + arrow_y1) / 2

        arrow_path = MPath(
            [(x0, arrow_y0), (x0, arrow_y_mid), (x1, arrow_y_mid), (x1, arrow_y1)],
            [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
        )
        arrow = mpatches.FancyArrowPatch(
            path=arrow_path,
            arrowstyle="-|>",
            mutation_scale=25,
            color="#bbbbbb",
            linewidth=3,
            alpha=1.0,
            zorder=0,
        )
        arrow.set_transform(fig.transFigure)
        fig.add_artist(arrow)

    # Panel labels — aligned in x; B sits just above the tallest funnel stage label
    a_label_y = 0.73 + pa_h * 0.20 + 0.03
    b_label_y = funnel_bot + funnel_h_size * (y_center_funnel + box_h.max() / 2 + 0.12)
    fig.text(0.005, a_label_y, "A", fontsize=16, fontweight="bold", va="bottom")
    fig.text(0.005, b_label_y, "B", fontsize=16, fontweight="bold", va="bottom")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig_pipeline.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
