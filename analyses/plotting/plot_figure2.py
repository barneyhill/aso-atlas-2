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
    box_heights = 0.08 + (log_asos - min_log) / rng * 0.47

    x_positions = np.linspace(0.07, 0.93, n_stages + 1)
    y_center = 0.45
    box_width = 0.035

    tc = results['target_candidates']
    cand_word = "candidate" if tc == 1 else "candidates"
    summary_text = (
        f"To obtain {tc} expected human development {cand_word} we require "
        f"{results['n_initial']:,} initial ASO candidates with a ${results['total_cost'] / 1e6:.1f}M "
        f"estimated total cost."
    )
    ax.text(0.5, 0.95, summary_text, ha="center", va="top", fontsize=12, transform=ax.transAxes)

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
            color = "#6B8E6B"
            label_name = "Human\ndevelopment\ncandidates"

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
    """Draw 7 histograms + legend panel on an 8-axis grid."""
    thresholds = [80, 500, 100, 1, 100, 1, 100]
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

        if i == 1:
            data_plot = np.log10(data[data > 0])
            threshold_plot = np.log10(thresholds[i])
            xlabel = "log10(IC50, nM)"
        else:
            data_plot = data[~np.isnan(data)]
            threshold_plot = thresholds[i]
            if i == 0:
                xlabel = "Inhibition (%)"
            elif i in [2, 4, 6]:
                xlabel = "ALT (IU/L)"
            else:
                xlabel = "FOB Score"

        ax.hist(data_plot, bins=30, color=stage["color"], alpha=0.7, edgecolor="black")
        ax.axvline(threshold_plot, color="red", linestyle="--", linewidth=2)

        xlim = ax.get_xlim()
        if threshold_ops[i] == ">":
            ax.axvspan(threshold_plot, xlim[1], alpha=0.2, color="green")
        elif threshold_ops[i] == "<":
            ax.axvspan(xlim[0], threshold_plot, alpha=0.2, color="green")
        elif threshold_ops[i] == "<=":
            ax.axvspan(xlim[0], threshold_plot + 0.01, alpha=0.2, color="green")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        prop = n_pass / n_total if n_total > 0 else 0
        title = stage["short_name"].replace("\n", " ")
        ax.set_title(f"{title}\n(n={n_total:,}, {prop:.1%} pass)", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Legend panel
    axes[7].axis("off")
    axes[7].text(
        0.5, 0.5,
        "Stage Thresholds:\n\n"
        "1. Inhibition > 80%\n"
        "2. IC50 < 500 nM\n"
        "3. Mouse ALT < 100 IU/L\n"
        "4. Mouse FOB <= 1\n"
        "5. Rat ALT < 100 IU/L\n"
        "6. Rat FOB <= 1\n"
        "7. Monkey ALT < 100 IU/L",
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
        transform=axes[7].transAxes,
    )


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

    # Two-row figure: Panel A (funnel) on top, Panel B (distributions) below
    fig = plt.figure(figsize=(16, 17), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.15)

    # Panel A
    ax_funnel = fig.add_subplot(gs[0])
    ax_funnel.set_title("A", fontsize=16, fontweight="bold", loc="left")
    draw_pipeline_attrition(baseline, stages, ax_funnel)

    # Panel B
    gs_bottom = gs[1].subgridspec(2, 4, hspace=0.4, wspace=0.3)
    axes_dist = [fig.add_subplot(gs_bottom[r, c]) for r in range(2) for c in range(4)]
    fig.text(0.01, 0.52, "B", fontsize=16, fontweight="bold")
    draw_stage_distributions(distributions, sample_sizes, stages, axes_dist)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig2.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
