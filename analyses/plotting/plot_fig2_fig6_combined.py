"""
Figure 2: The ASO preclinical screening pipeline.

Panel A — Stage measurement distributions (7 histograms with thresholds).
Panel B — Pipeline attrition funnel (tapered flows connecting stage boxes).
Panel C — Cost savings from computational pre-screening (stacked bars).

Single self-contained script. Reads data/results/pipeline_results.json.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

matplotlib.use("Agg")

# ── Paths ──
_root = Path(__file__).resolve().parents[2]
RESULTS_PATH = _root / "data/results/pipeline_results.json"
OUT_DIR = _root / "typst/plots"

# ── Style constants ──
TEXT_SIZE = 16          # unified size for titles, axis labels, cost labels, flow text
PANEL_LABEL_SIZE = 22   # A, B, C panel letters
BOX_NUMBER_SIZE = 16    # ASO count inside funnel boxes
BOX_WORD_SIZE = 11      # "ASOs" word inside funnel boxes

# Aliases for clarity in code
TITLE_SIZE = TEXT_SIZE
LABEL_SIZE = TEXT_SIZE
COST_LABEL_SIZE = 20
FLOW_TEXT_SIZE = TEXT_SIZE
COST_TEXT_SIZE = TEXT_SIZE


# ---------------------------------------------------------------------------
# Panel A: Stage distributions
# ---------------------------------------------------------------------------

def _transform_data(raw, stage):
    """Apply stage-specific transform to raw distribution data."""
    data = np.array(raw)
    t = stage["transform"]
    tv = stage["threshold_value"]

    if t == "clip01":
        clean = data[~np.isnan(data)]
        return np.clip(clean, 0, 100), tv
    elif t == "log10":
        clean = data[(~np.isnan(data)) & (data > 0)]
        return np.log10(clean), np.log10(tv)
    elif t == "integer":
        clean = np.round(data[~np.isnan(data)]).astype(int)
        return clean, 1.5  # line between 1 and 2 for <=1
    else:
        return data[~np.isnan(data)], tv


def draw_histograms(fig, stages, distributions, sample_sizes, positions, bot, width, height):
    """Draw one histogram per stage at the given figure positions."""
    axes = []
    for i, stage in enumerate(stages):
        ax = fig.add_axes([positions[i] - width / 2, bot, width, height])
        axes.append(ax)

        raw = distributions[i]
        if len(raw) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=LABEL_SIZE)
            continue

        data_plot, threshold_plot = _transform_data(raw, stage)
        bins = stage["bins"]

        if bins == "integer":
            b = np.arange(-0.5, data_plot.max() + 1.5, 1)
            ax.hist(data_plot, bins=b, color=stage["color"], alpha=0.7, edgecolor="black")
            ax.set_xticks(range(0, int(data_plot.max()) + 1))
        else:
            ax.hist(data_plot, bins=bins, color=stage["color"], alpha=0.7, edgecolor="black")

        ax.axvline(threshold_plot, color="red", linestyle="--", linewidth=2)

        # X-limits
        if stage["transform"] == "clip01":
            ax.set_xlim(0, 100)
        elif stage.get("xlim_lo") is not None:
            ax.set_xlim(stage["xlim_lo"], ax.get_xlim()[1])

        # Log-scale tick labels
        if stage["transform"] == "log10":
            lo, hi = ax.get_xlim()
            ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([
                f"{10**int(t)/1000:g}K" if 10**int(t) >= 1000 else f"{10**int(t):g}"
                for t in ticks
            ])

        # Shade fail region
        xl = ax.get_xlim()
        if stage["threshold_op"] == ">":
            ax.axvspan(xl[0], threshold_plot, alpha=0.15, color="grey")
        else:
            ax.axvspan(threshold_plot, xl[1], alpha=0.15, color="grey")

        ax.set_xlabel(stage["xlabel"], fontsize=TEXT_SIZE)
        if i == 0:
            ax.set_ylabel("Count", fontsize=TEXT_SIZE)
        ax.tick_params(axis="both", labelsize=TEXT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 1.06, stage["short_name"],
                fontsize=TITLE_SIZE + 2, fontweight="bold",
                ha="center", va="bottom", transform=ax.transAxes)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.08)

    return axes


# ---------------------------------------------------------------------------
# Panel B: Pipeline attrition funnel
# ---------------------------------------------------------------------------

def draw_funnel(ax, baseline, stages, x_positions):
    """Draw the pipeline attrition funnel on the given axes.

    x_positions: array of n_stages+1 values in axes coords (0-1 range).
    """
    n = len(stages)
    asos = baseline["asos_at_stage"]
    props = baseline["proportions"]

    log_asos = np.array([np.log10(max(a, 1)) for a in asos])
    rng = log_asos.max() - log_asos.min() or 1
    box_h = 0.15 + (log_asos - log_asos.min()) / rng * 0.40
    y_center = 0.45
    bw = 0.035

    # Tapered flows
    for i in range(n):
        x1, x2 = x_positions[i] + bw / 2, x_positions[i + 1] - bw / 2
        h1, h2 = box_h[i], box_h[i + 1]
        cx = (x1 + x2) / 2
        verts = [
            (x1, y_center - h1/2), (cx, y_center - h1/2),
            (cx, y_center - h2/2), (x2, y_center - h2/2),
            (x2, y_center + h2/2), (cx, y_center + h2/2),
            (cx, y_center + h1/2), (x1, y_center + h1/2),
            (x1, y_center - h1/2),
        ]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor="#e0e0e0", edgecolor="none", alpha=0.7))

        text = f"{props[i]*100:.0f}% ASOs\nhave\n{stages[i]['threshold']}"
        ax.text((x1 + x2) / 2, y_center, text, ha="center", va="center",
                fontsize=FLOW_TEXT_SIZE - 1, color="#404040")

    # Stage boxes
    for i in range(n + 1):
        x, h = x_positions[i], box_h[i]
        color = stages[i]["color"] if i < n else "#C44E52"
        label = stages[i]["short_name"] if i < n else "Clinical\ncandidate"

        ax.add_patch(mpatches.Rectangle(
            (x - bw/2, y_center - h/2), bw, h,
            facecolor=color, edgecolor="white", linewidth=1, zorder=10))

        count = f"{asos[i]:,}" if asos[i] != 1 else "1"
        word = "ASOs" if asos[i] != 1 else "ASO"
        ax.text(x, y_center + 0.005, count, ha="center", va="bottom",
                fontsize=BOX_NUMBER_SIZE, fontweight="bold", color="white", zorder=11)
        ax.text(x, y_center - 0.005, word, ha="center", va="top",
                fontsize=BOX_WORD_SIZE, fontweight="bold", color="white", zorder=11)

        ref_h = h if i < n else box_h[i - 1]
        ax.text(x, y_center + ref_h/2 + 0.115,
                label,
                ha="center", va="center", fontsize=TITLE_SIZE + 2, fontweight="bold", color="#202020")
        if i < n:
            cpa = stages[i]["cost_per_aso"]
            cost_str = f"${cpa/1e3:.0f}K/ASO" if cpa >= 1000 else f"${cpa:.0f}/ASO"
            ax.text(x, y_center + h/2 + 0.035, cost_str,
                    ha="center", va="center", fontsize=COST_TEXT_SIZE, color="#303030")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return box_h


# ---------------------------------------------------------------------------
# Arrows connecting A → B
# ---------------------------------------------------------------------------

def draw_arrows(fig, hist_x, flow_cx, box_h, arrow_y0, funnel_bot, funnel_h):
    """Draw Bézier S-curves from each histogram to the corresponding funnel flow."""
    y_center = 0.45
    n = len(hist_x)
    for i in range(n):
        x0 = hist_x[i]
        x1 = flow_cx[i]
        flow_top = y_center + (box_h[i] + box_h[i + 1]) / 4
        y1 = funnel_bot + funnel_h * (flow_top + 0.025)
        ym = (arrow_y0 + y1) / 2

        path = MPath(
            [(x0, arrow_y0), (x0, ym), (x1, ym), (x1, y1)],
            [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
        )
        arrow = mpatches.FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=25,
            color="#bbbbbb", linewidth=3, alpha=1.0, zorder=0)
        arrow.set_transform(fig.transFigure)
        fig.add_artist(arrow)


# ---------------------------------------------------------------------------
# Panel C: Cost comparison stacked bars
# ---------------------------------------------------------------------------

def draw_cost_bars(ax, scenarios, stages, bar_width=0.45):
    """Stacked bar chart of pipeline costs by scenario with inline legend."""
    categories = [s["short_name"].replace("\n", " ") for s in reversed(stages)]
    colors = [s["color"] for s in reversed(stages)]

    n = len(scenarios)
    x_pos = list(range(n))
    last = n - 1
    midpoints = []

    bottoms = [0.0] * n
    for i, (cat, color) in enumerate(zip(categories, colors)):
        for j, (label, scenario) in enumerate(scenarios):
            val = list(reversed(scenario["costs_per_stage"]))[i] / 1e6
            ax.bar(x_pos[j], val, bar_width, bottom=bottoms[j],
                   color=color, edgecolor="white", linewidth=0.5)
            if j == last:
                midpoints.append((bottoms[j] + val / 2, cat, color))
            bottoms[j] += val

    ax.set_xticks(x_pos)
    ax.set_xticklabels([label for label, _ in scenarios], fontsize=COST_LABEL_SIZE)

    totals = [sum(s["costs_per_stage"]) / 1e6 for _, s in scenarios]
    for j, total in enumerate(totals):
        ax.text(x_pos[j], total + max(totals) * 0.02, f"${total:.2f}M",
                ha="center", va="bottom", fontsize=COST_LABEL_SIZE, fontweight="bold")

    ax.set_ylim(0, 1.5)
    ax.set_ylabel("Total Cost ($M)", fontsize=COST_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=COST_LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(False)

    # Inline legend
    bar_right = x_pos[last] + bar_width / 2
    label_x = bar_right + 0.30
    x_left = x_pos[0] - 0.5
    grid_right = bar_right + 0.08
    ax.set_xlim(x_left, label_x + 1.0)
    ax.spines["bottom"].set_bounds(x_left, grid_right)

    for yt in ax.get_yticks():
        if ax.get_ylim()[0] <= yt <= ax.get_ylim()[1]:
            ax.hlines(yt, x_left, grid_right, colors="#666666",
                      linestyles="--", linewidth=0.7, alpha=0.35, zorder=0)

    n_labels = len(midpoints)
    y_min, y_max = ax.get_ylim()
    label_ys = np.linspace(y_min + 0.05 * (y_max - y_min), y_max * 0.95, n_labels)

    for (seg_y, cat, color), ly in zip(midpoints, label_ys):
        ax.plot([bar_right, label_x - 0.04], [seg_y, ly], color="black", lw=0.8, clip_on=False)
        ax.plot(label_x, ly, marker='s', markersize=8, color=color, clip_on=False, zorder=5)
        ax.annotate(cat, xy=(label_x, ly), xytext=(10, 0), textcoords="offset points",
                    fontsize=COST_LABEL_SIZE, va="center", ha="left", color="#333333", clip_on=False)


# ---------------------------------------------------------------------------
# Main: compose all panels
# ---------------------------------------------------------------------------

def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    baseline = data["baseline"]
    stages = data["stages"]
    distributions = data["distributions"]
    sample_sizes = data["sample_sizes"]

    scenarios = [("Baseline", baseline)]
    for key, label in [("oligoai", "OligoAI"), ("oligoai_tox", "OligoAI-tox")]:
        if data.get(key):
            scenarios.append((label, data[key]))
    if data.get("combined"):
        scenarios.append(("OligoAI +\nOligoAI-tox", data["combined"]))

    n_stages = len(stages)
    fig_w, fig_h = 18, 20
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # Shared x-positions (figure coordinates)
    shared_x = np.linspace(0.03, 0.97, n_stages + 1)
    hist_x = np.linspace(0.03, 0.97, n_stages)
    flow_cx = (shared_x[:-1] + shared_x[1:]) / 2

    # ── Panel A: histograms ──
    spacing = shared_x[1] - shared_x[0]
    pa_w = spacing - 0.02
    pa_h = pa_w * fig_w / fig_h
    pa_bot = 0.82 - pa_h
    draw_histograms(fig, stages, distributions, sample_sizes,
                    hist_x, pa_bot, pa_w, pa_h)

    # ── Panel B: funnel ──
    funnel_left, funnel_width = 0.01, 0.98
    funnel_bot, funnel_h = 0.36, 0.30
    ax_funnel = fig.add_axes([funnel_left, funnel_bot, funnel_width, funnel_h])
    funnel_x = (shared_x - funnel_left) / funnel_width  # convert to axes coords
    box_h = draw_funnel(ax_funnel, baseline, stages, funnel_x)

    # ── Arrows A → B ──
    draw_arrows(fig, hist_x, flow_cx, box_h, pa_bot - 0.03, funnel_bot, funnel_h)

    # ── Panel C: cost bars ──
    cost_bot, cost_h = 0.12, 0.25
    ax_cost = fig.add_axes([0.05, cost_bot, 0.90, cost_h])
    draw_cost_bars(ax_cost, scenarios, stages)

    # Reposition C so bar midpoint aligns with figure centre
    xlim = ax_cost.get_xlim()
    frac = (1.5 - xlim[0]) / (xlim[1] - xlim[0])
    cw = 0.90
    ax_cost.set_position([0.5 - cw * frac, cost_bot, cw, cost_h])

    # ── Panel labels ──
    lx = -0.01
    fig.text(lx, 0.82 + pa_h * 0.20 + 0.02, "A",
             fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="bottom")
    fig.text(lx, funnel_bot + funnel_h - 0.01, "B",
             fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top")
    fig.text(lx, cost_bot + cost_h - 0.01, "C",
             fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top")

    # ── Save ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig2_fig6_combined"
    fig.savefig(out.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.svg')}")
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
