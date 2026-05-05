"""
Supplementary figure: precision vs selection fraction with per-fold uncertainty.

For each pipeline stage with a predictive model, sweeps the selection fraction
K from 5% to 95% and plots precision@K (= pass rate among selected) with
mean ± 1 s.d. across 5 folds.

- Panels A-B: OligoAI (efficacy, potency) — stratified by patent table / unstratified
- Panels C-F: Random Forest in vivo (mouse ALT, mouse FOB, rat ALT, rat FOB)
- Panel G: Pipeline savings vs selection fraction (combined strategy)

The base_rate operating point is marked on each panel.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyses.logic.enrichment import (
    enrichment_at_top_k,
    stage_for_dataset,
    stratified_enrichment_at_top_k,
)
from analyses.logic.ef_table import DATASET_STRATA_KEY, OLIGOAI_TOX_MODEL, per_fold_ef_and_prec
from analyses.logic.pipeline import PIPELINE_STAGES

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
BENCH_PATH = _root / "data/results/oligogym_benchmark.json"
PIPE_PATH = _root / "data/results/pipeline_results.json"
EF_TABLE_PATH = _root / "data/results/ef_table.json"
OLIGOAI_FOLD_PATHS = [_root / f"data/results/oligoai_fold{i}_predictions.parquet" for i in range(5)]
OUT_DIR = _root / "typst/plots/fig_enrichment"

FRACTIONS = np.linspace(0.05, 0.95, 37)

IN_VIVO_DATASETS = ["mouse_hepatic", "mouse_neuro", "rat_hepatic", "rat_neuro"]
STAGE_MAP = {"mouse_hepatic": 2, "mouse_neuro": 3, "rat_hepatic": 4, "rat_neuro": 5}

PANEL_COLORS = {
    "efficacy": "#4878A8",
    "potency": "#6A9BC3",
    "mouse_hepatic": "#D4A574",
    "mouse_neuro": "#E8B88A",
    "rat_hepatic": "#7BAA97",
    "rat_neuro": "#94C4A7",
}
PANEL_LABELS = {
    "efficacy": "In vitro efficacy",
    "potency": "In vitro potency",
    "mouse_hepatic": "Mouse ALT",
    "mouse_neuro": "Mouse bFOB",
    "rat_hepatic": "Rat ALT",
    "rat_neuro": "Rat mFOB",
}


# ── OligoAI per-fold sweep ──────────────────────────────────────────────────

def _oligoai_efficacy_sweep_fold(df: pd.DataFrame, fractions: np.ndarray) -> list[float]:
    """Compute precision@K for efficacy at each fraction, stratified by dosage."""
    doses_per_table = df.groupby("custom_id")["dosage"].nunique()
    single_ids = doses_per_table[doses_per_table == 1].index
    sub = df[df["custom_id"].isin(single_ids)]

    precs = []
    for k in fractions:
        ef = stratified_enrichment_at_top_k(
            sub["inhibition_percent"].to_numpy(dtype=float),
            sub["prediction"].to_numpy(dtype=float),
            PIPELINE_STAGES[0],
            strata=sub["dosage"].to_numpy(dtype=float),
            k=float(k),
        )
        precs.append(ef.get("selected_pass_rate", np.nan))
    return precs


def _oligoai_potency_sweep_fold(df: pd.DataFrame, fractions: np.ndarray) -> list[float]:
    """Compute precision@K for potency at each fraction via 4PL IC50 fits.

    Filtered to electroporation tables only (matching pipeline stage definition).
    """
    from analyses.logic.models.oligoai_efs import _electroporation_table_ids, _fit_group

    electro_ids = _electroporation_table_ids()
    doses_per_table = df.groupby("custom_id")["dosage"].nunique()
    multi_ids = doses_per_table[doses_per_table > 1].index
    multi = df[df["custom_id"].isin(multi_ids) & df["custom_id"].isin(electro_ids)]

    ic50_true, ic50_pred = [], []
    for _, g in multi.groupby(["helm_annotation", "cell_line", "target_RNA"], sort=False):
        if len(g) < 4:
            continue
        doses = g["dosage"].to_numpy(dtype=float)
        true_fit = _fit_group(doses, g["inhibition_percent"].to_numpy(dtype=float))
        pred_fit = _fit_group(doses, g["prediction"].to_numpy(dtype=float))
        if true_fit is None or pred_fit is None:
            continue
        ic50_true.append(true_fit[0])
        ic50_pred.append(pred_fit[0])

    if not ic50_true:
        return [np.nan] * len(fractions)

    yt = np.asarray(ic50_true)
    yp = np.asarray(ic50_pred)
    precs = []
    for k in fractions:
        ef = enrichment_at_top_k(yt, yp, PIPELINE_STAGES[1], k=float(k))
        precs.append(ef.get("selected_pass_rate", np.nan))
    return precs


def compute_oligoai_sweeps() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {panel: (mean_prec, std_prec)} across folds for OligoAI."""
    available = [p for p in OLIGOAI_FOLD_PATHS if p.exists()]
    if not available:
        return {}

    eff_folds, pot_folds = [], []
    for p in available:
        df = pd.read_parquet(p)
        eff_folds.append(_oligoai_efficacy_sweep_fold(df, FRACTIONS))
        pot_folds.append(_oligoai_potency_sweep_fold(df, FRACTIONS))

    out = {}
    if eff_folds:
        arr = np.array(eff_folds)
        out["efficacy"] = (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1))
    if pot_folds:
        arr = np.array(pot_folds)
        out["potency"] = (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1))
    return out


# ── OligoGym tox model per-fold sweep ──────────────────────────────────────

def compute_tox_sweeps(bench: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {dataset: (mean_prec, std_prec)} across folds for tox model in vivo.

    Hepatic datasets are stratified by dosage (matching ef_table).
    """
    out = {}
    for r in bench.get("all_results", []):
        if r["model"] != OLIGOAI_TOX_MODEL or r["dataset"] not in IN_VIVO_DATASETS:
            continue
        stage = stage_for_dataset(r["dataset"])
        strata_key = DATASET_STRATA_KEY.get(r["dataset"])
        fold_precs_by_k = []
        for k in FRACTIONS:
            _, precs = per_fold_ef_and_prec(
                r.get("fold_metrics", []), stage, strata_key=strata_key, k=float(k))
            fold_precs_by_k.append(precs)
        if fold_precs_by_k and fold_precs_by_k[0]:
            n_folds = len(fold_precs_by_k[0])
            arr = np.array([[fold_precs_by_k[ki][fi] for ki in range(len(FRACTIONS))]
                            for fi in range(n_folds)])
            out[r["dataset"]] = (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1))
    return out


# ── Base rates ───────────────────────────────────────────────────────────────

def get_base_rates(pipe: dict) -> dict[str, float]:
    """Get the base rate (pass fraction) per stage from pipeline results."""
    proportions = pipe["proportions"]
    return {
        "efficacy": proportions[0],
        "potency": proportions[1],
        "mouse_hepatic": proportions[2],
        "mouse_neuro": proportions[3],
        "rat_hepatic": proportions[4],
        "rat_neuro": proportions[5],
    }


# ── Plot ─────────────────────────────────────────────────────────────────────

def main():
    if not BENCH_PATH.exists() or not PIPE_PATH.exists():
        raise FileNotFoundError("Missing result files. Run `just analysis` and `just oligogym` first.")

    bench = json.loads(BENCH_PATH.read_text())
    pipe = json.loads(PIPE_PATH.read_text())
    base_rates = get_base_rates(pipe)

    print("Computing OligoAI sweeps...")
    oligoai_sweeps = compute_oligoai_sweeps()
    print("Computing in vivo sweeps...")
    tox_sweeps = compute_tox_sweeps(bench)

    panels = ["efficacy", "potency", "mouse_hepatic", "mouse_neuro", "rat_hepatic", "rat_neuro"]
    all_sweeps = {**oligoai_sweeps, **tox_sweeps}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=300)
    panel_axes = [axes[0, 0], axes[0, 1], axes[0, 2],
                  axes[1, 0], axes[1, 1], axes[1, 2]]
    letters = "ABCDEF"

    x_pct = FRACTIONS * 100

    for i, panel in enumerate(panels):
        ax = panel_axes[i]
        color = PANEL_COLORS[panel]
        label = PANEL_LABELS[panel]

        if panel in all_sweeps:
            mean, std = all_sweeps[panel]
            ax.plot(x_pct, mean * 100, linewidth=2, color=color)
            ax.fill_between(x_pct, (mean - std) * 100, (mean + std) * 100,
                            alpha=0.25, color=color)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

        br = base_rates.get(panel)
        if br is not None:
            ax.axvline(br * 100, color="#444444", linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(br * 100, color="#999999", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_title(f"{letters[i]}. {label}", fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Selection fraction (%)")
        ax.set_ylabel("Precision (%)")
        ax.set_xlim(100, 0)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "fig_enrichment.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
