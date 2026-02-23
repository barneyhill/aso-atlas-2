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
OUT_DIR = _root / "typst/plots/fig2"


def draw_pipeline_attrition(results, stages, ax):
    """Draw the pipeline attrition funnel on the given axes."""
    n_stages = len(stages)
    asos = results["asos_at_stage"]
    props = results["proportions"]
    costs = results["costs_per_stage"]

    log_asos = np.array([np.log10(max(a, 1)) for a in asos])
    min_log, max_log = log_asos.min(), log_asos.max()
    rng = max_log - min_log if max_log > min_log else 1
    box_heights = 0.15 + (log_asos - min_log) / rng * 0.40

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
        prop_text = f"{props[i] * 100:.0f}% ASOs\ntested have\n{filter_text}"
        ax.text((x1 + x2) / 2, y_center, prop_text,
                ha="center", va="center", fontsize=8, color="#404040")

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

        if i < n_stages:
            ax.text(x, y_center + h / 2 + 0.095, label_name,
                    ha="center", va="center", fontsize=10, fontweight="bold", color="#202020")
            cost_total = f"${costs[i] / 1e6:.1f}M" if costs[i] >= 1e6 else f"${costs[i] / 1e3:.0f}K"
            cpa = stages[i]["cost_per_aso"]
            cost_per = f"${cpa / 1e3:.0f}K" if cpa >= 1000 else f"${cpa:.0f}"
            cost_text = f"{asos[i]:,} ASOs cost {cost_total}\nat {cost_per}/ASO"
            ax.text(x, y_center + h / 2 + 0.035, cost_text,
                    ha="center", va="center", fontsize=8, color="#303030")
        else:
            prev_h = box_heights[i - 1]
            ax.text(x, y_center + prev_h / 2 + 0.095, label_name,
                    ha="center", va="center", fontsize=10, fontweight="bold", color="#202020")
            ax.text(x, y_center + prev_h / 2 + 0.035, f"n={asos[i]:,}",
                    ha="center", va="center", fontsize=8, color="#303030")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_stage_distributions(distributions, sample_sizes, stages, axes):
    """Draw 7 histograms on a single-row grid."""
    thresholds = [80, 500, 150, 1, 78, 1, 206]
    threshold_ops = [">", "<", "<", "<=", "<", "<=", "<"]

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
            xlabel = "log\u2081\u2080(IC\u2085\u2080, nM)"
        elif i in [2, 4, 6]:
            # ALT stages: log10
            data_plot = np.log10(data[(~np.isnan(data)) & (data > 0)])
            threshold_plot = np.log10(thresholds[i])
            xlabel = "log\u2081\u2080(ALT, IU/L)"
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

        # Shade the fail region in grey
        if threshold_ops[i] == ">":
            ax.axvspan(xlim[0], threshold_plot, alpha=0.15, color="grey")
        elif threshold_ops[i] == "<":
            ax.axvspan(threshold_plot, xlim[1], alpha=0.15, color="grey")
        elif threshold_ops[i] == "<=":
            ax.axvspan(threshold_plot, xlim[1], alpha=0.15, color="grey")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Two-line title: stage name (bold) + "(X% of ASOs have <threshold>)"
        # with threshold portion in red
        prop = n_pass / n_total * 100 if n_total > 0 else 0
        title_name = stage["short_name"].replace("\n", " ")
        threshold_text = stage["threshold"]
        ax.set_title("")
        ax.text(0.5, 1.20, title_name, fontsize=9, fontweight="bold",
                ha="center", va="bottom", transform=ax.transAxes)
        prefix = f"({prop:.0f}% of ASOs have "
        ta_prefix = TextArea(prefix, textprops=dict(fontsize=8, color="black"))
        ta_thresh = TextArea(threshold_text, textprops=dict(fontsize=8, color="red",
                                                             fontweight="bold"))
        ta_close = TextArea(")", textprops=dict(fontsize=8, color="black"))
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

    # Panel A — manual equal-spaced square axes
    n_panels = 7
    pa_left, pa_right = 0.05, 0.97
    pa_gap = 0.045
    pa_total_gap = pa_gap * (n_panels - 1)
    pa_w = (pa_right - pa_left - pa_total_gap) / n_panels
    # Make height equal to width in inches, then convert to figure fraction
    fig_w, fig_h = fig.get_size_inches()
    pa_h = pa_w * fig_w / fig_h
    pa_bot = 0.73 - pa_h
    axes_dist = []
    for c in range(n_panels):
        x0 = pa_left + c * (pa_w + pa_gap)
        axes_dist.append(fig.add_axes([x0, pa_bot, pa_w, pa_h]))
    draw_stage_distributions(distributions, sample_sizes, stages, axes_dist)
    for ax in axes_dist:
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.08)

    # Panel B — funnel below
    ax_funnel = fig.add_axes([0.05, 0.02, 0.90, 0.50])
    draw_pipeline_attrition(baseline, stages, ax_funnel)

    # Panel labels
    # Place "A" just above the title text (which sits at ~1.20 in axes coords)
    a_label_y = 0.73 + pa_h * 0.20 + 0.03
    fig.text(0.02, a_label_y, "A", fontsize=16, fontweight="bold", va="bottom")
    ax_funnel.text(-0.02, 1.0, "B", fontsize=16, fontweight="bold", va="top",
                   transform=ax_funnel.transAxes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig2.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # ── Variant: panel A only ──
    # Compute square panel size first, then set figure height to fit
    fig_a_w = 18
    left, right = 0.05, 0.97
    gap = 0.045
    total_gap = gap * (n_panels - 1)
    ax_w_frac = (right - left - total_gap) / n_panels
    ax_w_in = ax_w_frac * fig_a_w  # width of one panel in inches
    bot = 0.15
    top_margin = 0.30  # space above for titles
    fig_a_h = ax_w_in / (1 - bot - top_margin)  # figure height so panel is square
    ax_h_frac = ax_w_in / fig_a_h
    fig_a = plt.figure(figsize=(fig_a_w, fig_a_h), dpi=300)
    axes_a = []
    for c in range(n_panels):
        x0 = left + c * (ax_w_frac + gap)
        axes_a.append(fig_a.add_axes([x0, bot, ax_w_frac, ax_h_frac]))
    draw_stage_distributions(distributions, sample_sizes, stages, axes_a)
    for ax in axes_a:
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.08)

    a_path = OUT_DIR / "fig2-just-A.svg"
    fig_a.savefig(a_path, format="svg", bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved {a_path}")


if __name__ == "__main__":
    main()
