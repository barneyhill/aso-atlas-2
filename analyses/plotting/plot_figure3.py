"""
Figure 3: ION582 benchmarking against patent ASO distributions.

Panel A — IC50 distribution (iCell GABANeurons, free uptake, UBE3A-ATS) with ION582.
Panel B — Mouse FOB score distribution (ICV, 700 ug, 3 h) with ION582.
Panel C — Rat FOB score distribution (IT, 3000 ug, 3 h) with ION582.

ION582 is marked as a vertical line with percentile label.
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)

_root = Path(__file__).resolve().parents[2]
DATA_DIR = _root / "data" / "oligostack" / "processed"
OUT_DIR = _root / "typst" / "plots" / "fig3"

ION582 = {"id": 1273062, "name": "ION582", "color": "#4DBBD5"}


# ---------------------------------------------------------------------------
# IC50 fitting (mirrors analyses/logic/pipeline.py)
# ---------------------------------------------------------------------------

def _hill(dose, bottom, top, ic50, slope):
    return bottom + (top - bottom) / (1 + (ic50 / dose) ** slope)


def _fit_ic50(doses, responses):
    mask = ~(np.isnan(doses) | np.isnan(responses))
    doses, responses = np.array(doses)[mask], np.array(responses)[mask]
    if len(doses) < 4:
        return np.nan
    min_d, max_d = doses.min(), doses.max()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _hill, doses, responses,
                p0=[np.min(responses), np.max(responses), np.median(doses), 1.0],
                bounds=([-50, 20, min_d / 100, 0.1], [50, 120, max_d * 100, 10]),
                maxfev=5000,
            )
        ic50 = popt[2]
        pred = _hill(doses, *popt)
        ss_res = np.sum((responses - pred) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 > 0.5 and min_d / 10 <= ic50 <= max_d * 10:
            return ic50
        return np.nan
    except (RuntimeError, ValueError):
        return np.nan


def _fit_ic50s_for_subset(dr):
    """Fit IC50 per compound from a pre-filtered dataframe."""
    ic50s = {}
    for cid, g in dr.groupby("Compound ID"):
        vals = []
        for _, eg in g.groupby(["USPTO ID", "Table Number"]):
            v = _fit_ic50(eg["dosage_nm"].values, eg["Inhibition_pct"].values)
            if not np.isnan(v):
                vals.append(v)
        if vals:
            ic50s[cid] = np.median(vals)
    return ic50s


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _mark_drug(ax, x_val, dist_vals, name, color, y_top_frac=0.95):
    """Mark a drug on a histogram with a vertical line and label."""
    ax.axvline(x_val, color=color, linewidth=2, linestyle="-", zorder=5)
    ylim = ax.get_ylim()
    ax.text(
        x_val, ylim[1] * y_top_frac,
        f" {name}",
        color=color, fontsize=8, fontweight="bold",
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )


def draw_ic50_panel(dr, ax):
    """Panel A: iCell GABANeurons IC50 distribution with ION582."""
    icell_free = dr[
        (dr["cell_line"] == "iCell GABANeurons")
        & ((dr["transfection_method"] == "Gymnosis") | (dr["transfection_method"].isna()))
    ]
    ic50s = _fit_ic50s_for_subset(icell_free)
    dist = np.array(list(ic50s.values()))
    log_dist = np.log10(dist)

    ax.hist(log_dist, bins=30, color="#B8B8B8", edgecolor="black",
            linewidth=0.4, alpha=0.7)

    ion582_ic50 = ic50s.get(ION582["id"])
    if ion582_ic50 and not np.isnan(ion582_ic50):
        _mark_drug(ax, np.log10(ion582_ic50), log_dist,
                   f"{ION582['name']} (IC\u2085\u2080={ion582_ic50:.0f} nM)",
                   ION582["color"])

    ax.set_xlabel("log\u2081\u2080(IC\u2085\u2080, nM)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Potency — free uptake in iCell GABANeurons\n(n={len(dist):,})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)


def draw_neuro_panel(neuro, species, ax):
    """FOB distribution with ION582 marked."""
    sub = neuro[(neuro["species"] == species) & (neuro["FOB_val"].notna())].copy()
    sub["FOB_int"] = sub["FOB_val"].round().astype(int)
    fob_vals = sub["FOB_int"].values

    ax.hist(fob_vals, bins=np.arange(-0.5, 8.5, 1),
            color="#B8B8B8", edgecolor="black", linewidth=0.4, alpha=0.7)

    rows = neuro[(neuro["Compound ID"] == ION582["id"]) & (neuro["species"] == species)]
    if len(rows) > 0:
        fob = round(rows["FOB_val"].iloc[0])
        route = rows["administration_method"].iloc[0]
        dose = rows["dosage_ug"].iloc[0]
        _mark_drug(ax, fob, fob_vals, ION582["name"], ION582["color"])

    species_detail = {
        "Mouse": "ICV 700 \u00b5g, 3 h post-dose",
        "Rat": "IT 3000 \u00b5g, 3 h post-dose",
    }
    ax.set_xlabel("FOB score")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{species} neurotoxicity — {species_detail[species]}\n(n={len(sub):,})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)


def main():
    dr = pd.read_parquet(DATA_DIR / "dose_response_processed.parquet")
    neuro = pd.read_parquet(DATA_DIR / "neurotoxicity_processed.parquet")
    neuro["FOB_val"] = neuro["FOB_score"].apply(
        lambda x: float(x[0]) if isinstance(x, np.ndarray) and len(x) > 0 else np.nan
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    draw_ic50_panel(dr, axes[0])
    draw_neuro_panel(neuro, "Mouse", axes[1])
    draw_neuro_panel(neuro, "Rat", axes[2])

    for i, letter in enumerate("ABC"):
        axes[i].text(-0.08, 1.08, letter, transform=axes[i].transAxes,
                     fontsize=14, fontweight="bold")

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / "fig3.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
