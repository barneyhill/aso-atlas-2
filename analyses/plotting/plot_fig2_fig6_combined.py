"""
Temporary combined figure: Fig2 (panels A+B) on left, Fig6 cost comparison (panel C) on right.
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "analyses" / "plotting"))

from plot_figure2 import draw_pipeline_attrition, draw_stage_distributions
from plot_figure6 import draw_cost_comparison

RESULTS_PATH = _root / "data/results/pipeline_results.json"
OUT_DIR = _root / "typst/plots"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    baseline = data["baseline"]
    stages = data["stages"]
    distributions = data["distributions"]
    sample_sizes = data["sample_sizes"]
    oligoai_tox = data.get("oligoai_tox")
    oligoai = data.get("oligoai")
    combined = data.get("combined")

    scenarios = [("Baseline", baseline)]
    if oligoai:
        scenarios.append(("OligoAI", oligoai))
    if oligoai_tox:
        scenarios.append(("OligoAI-tox", oligoai_tox))
    if combined:
        scenarios.append(("Combined", combined))

    # Layout: fig2 takes left 74%, fig6 takes right 26%
    fig_w, fig_h = 24, 11
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # ── Left side: Fig2 content (panels A + B) ──
    # Scale fig2 coordinates into left 72% of the figure
    left_frac = 0.72
    n_stages = len(stages)
    shared_x = np.linspace(0.07, 0.93, n_stages + 1)
    flow_cx = (shared_x[:-1] + shared_x[1:]) / 2
    hist_x = np.linspace(0.06, 0.94, n_stages)

    # Scale all x-positions into left portion
    def scale_x(x):
        return x * left_frac

    spacing = shared_x[1] - shared_x[0]
    pa_gap = 0.02
    pa_w = (spacing - pa_gap) * left_frac
    pa_h = pa_w * fig_w / fig_h
    pa_bot = 0.73 - pa_h

    # Panel A — histograms
    axes_dist = []
    for i in range(n_stages):
        x0 = scale_x(hist_x[i]) - pa_w / 2
        axes_dist.append(fig.add_axes([x0, pa_bot, pa_w, pa_h]))
    draw_stage_distributions(distributions, sample_sizes, stages, axes_dist)
    for ax in axes_dist:
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.08)

    # Panel B — funnel
    funnel_left = scale_x(0.05)
    funnel_width = left_frac * 0.90
    ax_funnel = fig.add_axes([funnel_left, 0.02, funnel_width, 0.50])
    funnel_x = (np.array([scale_x(x) for x in shared_x]) - funnel_left) / funnel_width
    draw_pipeline_attrition(baseline, stages, ax_funnel, x_positions=funnel_x)

    # Bézier arrows from histograms to funnel flows
    from matplotlib.path import Path as MPath
    import matplotlib.patches as mpatches

    funnel_bot, funnel_h_size = 0.02, 0.50
    y_center_funnel = 0.45
    asos = baseline["asos_at_stage"]
    log_asos = np.array([np.log10(max(a, 1)) for a in asos])
    rng = log_asos.max() - log_asos.min() or 1
    box_h = 0.15 + (log_asos - log_asos.min()) / rng * 0.40
    arrow_gap = 0.025

    arrow_y0 = pa_bot - 0.05
    for i in range(n_stages):
        x0 = scale_x(hist_x[i])
        x1 = scale_x(flow_cx[i])
        flow_top = y_center_funnel + (box_h[i] + box_h[i + 1]) / 4
        arrow_y1 = funnel_bot + funnel_h_size * (flow_top + arrow_gap)
        arrow_y_mid = (arrow_y0 + arrow_y1) / 2

        arrow_path = MPath(
            [(x0, arrow_y0), (x0, arrow_y_mid), (x1, arrow_y_mid), (x1, arrow_y1)],
            [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
        )
        arrow = mpatches.FancyArrowPatch(
            path=arrow_path, arrowstyle="-|>", mutation_scale=25,
            color="#bbbbbb", linewidth=3, alpha=1.0, zorder=0,
        )
        arrow.set_transform(fig.transFigure)
        fig.add_artist(arrow)

    # ── Right side: Fig6 cost comparison (panel C) ──
    right_left = 0.76
    right_width = 0.22
    right_h = 0.62
    right_bot = 0.14
    ax_cost = fig.add_axes([right_left, right_bot, right_width, right_h])
    draw_cost_comparison(scenarios, stages, ax_cost)

    # ── Panel labels ──
    a_label_y = 0.73 + pa_h * 0.20 + 0.03
    fig.text(0.005, a_label_y, "A", fontsize=18, fontweight="bold", va="bottom")
    ax_funnel.text(-0.04, 1.0, "B", fontsize=18, fontweight="bold", va="top",
                   transform=ax_funnel.transAxes)
    fig.text(right_left - 0.02, a_label_y, "C", fontsize=18, fontweight="bold", va="bottom")

    out_path = OUT_DIR / "fig2_fig6_combined.svg"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
