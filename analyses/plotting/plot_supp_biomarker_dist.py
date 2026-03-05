"""
Supplementary figure: Mouse ALT and bFOB distributions side by side.

Same style as fig2 panel A, but only the mouse ALT (i=2) and mouse bFOB (i=3)
histogram panels.

Reads: data/results/pipeline_results.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
RESULTS_PATH = _root / "data/results/pipeline_results.json"
OUT_DIR = _root / "typst/plots/supp_biomarker_dist"


def _draw_panel(ax, data_raw, stage, n_pass, n_total, panel_type):
    """Draw a single histogram panel in fig2 style.

    panel_type: "alt" (log10) or "fob" (integer bins).
    """
    data = np.array(data_raw)

    if panel_type == "alt":
        data_plot = np.log10(data[(~np.isnan(data)) & (data > 0)])
        threshold_plot = np.log10(150)
        xlabel = "ALT (IU/L)"
        ax.hist(data_plot, bins=30, color=stage["color"], alpha=0.7, edgecolor="black")
        ax.axvline(threshold_plot, color="red", linestyle="--", linewidth=2)
        ax.set_xlim(1, ax.get_xlim()[1])
        # Real-value tick labels
        lo, hi = ax.get_xlim()
        ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([
            f"{10**int(t)/1000:g}K" if 10**int(t) >= 1000 else f"{10**int(t):g}"
            for t in ticks
        ])
        # Shade fail region
        ax.axvspan(threshold_plot, ax.get_xlim()[1], alpha=0.15, color="grey")
    else:
        data_plot = np.round(data[~np.isnan(data)]).astype(int)
        threshold_plot = 1.5
        xlabel = "bFOB Score"
        bins = np.arange(-0.5, data_plot.max() + 1.5, 1)
        ax.hist(data_plot, bins=bins, color=stage["color"], alpha=0.7, edgecolor="black")
        ax.axvline(threshold_plot, color="red", linestyle="--", linewidth=2)
        ax.set_xlim(-0.5, ax.get_xlim()[1])
        ax.set_xticks(range(0, int(data_plot.max()) + 1))
        # Shade fail region
        ax.axvspan(threshold_plot, ax.get_xlim()[1], alpha=0.15, color="grey")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Title: stage name (bold) + threshold info
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
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    stages = data["stages"]
    distributions = data["distributions"]
    sample_sizes = data["sample_sizes"]

    fig, (ax_alt, ax_fob) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    # i=2 → mouse ALT, i=3 → mouse bFOB
    _draw_panel(ax_alt, distributions[2], stages[2], sample_sizes[2][0], sample_sizes[2][1], "alt")
    _draw_panel(ax_fob, distributions[3], stages[3], sample_sizes[3][0], sample_sizes[3][1], "fob")

    for ax in [ax_alt, ax_fob]:
        ax.set_aspect(1.0 / ax.get_data_ratio())
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.08)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "mouse_biomarker_distributions.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
