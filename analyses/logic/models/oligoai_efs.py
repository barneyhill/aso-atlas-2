"""Compute OligoAI enrichment factors (efficacy + potency) from held-out predictions.

Reads ``data/results/oligoai_fold{0..4}_predictions.parquet`` (written by
``analyses.logic.models.oligoai_eval`` from each RunPod fold run) and produces
``data/results/oligoai_efs.json``:

- **Efficacy EF**: top-20% selection by predicted inhibition on single-dose
  screening rows (pure ASO ranking; the model's dose feature is effectively
  constant within each screen). Maps to pipeline stage 0 (inhibition > 80%).
- **Potency EF**: top-20% selection by predicted IC50 on multi-dose
  dose-response curves. Each (helm, cell, target) curve is fitted twice with
  the same 4PL used by oligoai_eval (observed and predicted). Maps to pipeline
  stage 1 (IC50 < 500 nM).

Reuses ``enrichment_at_top_k`` from analyses.logic.enrichment and ``_fit_group``
from oligoai_eval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analyses.logic.enrichment import enrichment_at_top_k, stratified_enrichment_at_top_k
from analyses.logic.pipeline import PIPELINE_STAGES


# ── 4PL fit ──────────────────────────────────────────────────────────────────
# Inlined from analyses.logic.models.oligoai_eval so this module doesn't pull
# in the pod-side rinalmo/torch deps. Kept bit-for-bit identical so local
# recomputation matches the oligoai.json numbers.

def _four_pl(dose, baseline, emax, ic50, hill):
    d = np.clip(dose, 1e-9, None)
    return baseline + emax / (1.0 + (ic50 / d) ** hill)


def _fit_group(doses: np.ndarray, inh: np.ndarray) -> tuple[float, float] | None:
    doses = np.asarray(doses, dtype=float)
    inh = np.asarray(inh, dtype=float)
    if doses.size < 4:
        return None
    unique_doses = np.unique(doses[np.isfinite(doses) & (doses > 0)])
    if unique_doses.size < 4:
        return None
    if not np.isfinite(inh).any():
        return None
    if np.nanmax(inh) - np.nanmin(inh) < 40.0:
        return None

    d_min = float(unique_doses.min())
    d_max = float(unique_doses.max())
    p0 = [
        float(np.nanmin(inh)),
        float(np.nanmax(inh) - np.nanmin(inh)),
        float(np.sqrt(d_min * d_max)),
        1.0,
    ]
    bounds = ([0.0, 0.0, d_min, 0.1], [100.0, 100.0, d_max, 10.0])
    try:
        popt, _ = curve_fit(_four_pl, doses, inh, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError):
        return None
    baseline, emax, ic50, hill = popt
    if not (np.isfinite(ic50) and np.isfinite(emax)):
        return None
    return float(ic50), float(emax)

_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _ROOT / "data/oligostack/processed"


def _electroporation_table_ids() -> set[str]:
    """Return custom_id values whose dose-response rows are electroporation."""
    import re
    dr = pd.read_parquet(_DATA_DIR / "dose_response_processed.parquet")
    electro_tables = dr[dr["transfection_method"] == "Electroporation"]
    ids = set()
    for (pat, tbl), _ in electro_tables.groupby(["USPTO ID", "Table Number"]):
        ids.add(f"our-data/inhibition_tables/{pat}_table_{int(tbl):05d}.xml")
    return ids
# Per-fold predictions for the 5-fold patent-split runs. The naming matches
# what oligoai_eval.py writes when invoked with --out oligoai_fold{i}.json
# (it derives `<stem>_predictions.parquet` automatically). Missing folds
# are tolerated so a 1- or 2-fold partial run still populates oligoai_efs.json.
PREDICTIONS_PATHS_5FOLD = [
    _ROOT / f"data/results/oligoai_fold{i}_predictions.parquet" for i in range(5)
]
# Fold-0 fallback when no fold parquets are present at all (kept so the
# aggregator still has something to chew on after a fresh checkout).
PREDICTIONS_PATH = PREDICTIONS_PATHS_5FOLD[0]
OUT_PATH = _ROOT / "data/results/oligoai_efs.json"

_SOURCE_NOTE = (
    "5-fold GroupKFold patent split with HELM-level dedup (seed=42); "
    "K=0.20 (top-20% selection)"
)
_SOURCE_NOTE_SINGLE = (
    "fold 0 of patent-level 5-fold GroupKFold with HELM-level dedup "
    "(seed=42); K=0.20 (top-20% selection); "
    "folds 1-4 not yet trained — launch `just oligoai-launch-folds` for parity"
)


def _efficacy_ef(df: pd.DataFrame) -> dict:
    """EF at pipeline stage 0, stratified by dosage.

    Uses single-dose-table rows and ranks within each dosage group so
    the model cannot exploit dose variation for ranking.
    """
    doses_per_table = df.groupby("custom_id")["dosage"].nunique()
    single_ids = doses_per_table[doses_per_table == 1].index
    sub = df[df["custom_id"].isin(single_ids)]
    ef = stratified_enrichment_at_top_k(
        sub["inhibition_percent"].to_numpy(dtype=float),
        sub["prediction"].to_numpy(dtype=float),
        PIPELINE_STAGES[0],
        strata=sub["dosage"].to_numpy(dtype=float),
    )
    ef["stage"] = 0
    ef["stage_name"] = PIPELINE_STAGES[0].name
    ef["threshold"] = (
        f"inhibition {PIPELINE_STAGES[0].threshold_op} "
        f"{PIPELINE_STAGES[0].threshold_value}"
    )
    ef["n_tables"] = int(len(single_ids))
    ef["source"] = _SOURCE_NOTE
    return ef


def _potency_ef(df: pd.DataFrame, electro_ids: set[str] | None = None) -> dict:
    """EF at pipeline stage 1 using 4PL IC50 fits on electroporation multi-dose curves."""
    if electro_ids is None:
        electro_ids = _electroporation_table_ids()
    doses_per_table = df.groupby("custom_id")["dosage"].nunique()
    multi_ids = doses_per_table[doses_per_table > 1].index
    multi = df[df["custom_id"].isin(multi_ids) & df["custom_id"].isin(electro_ids)]

    ic50_true: list[float] = []
    ic50_pred: list[float] = []
    n_candidates = 0
    n_converged = 0
    for _, g in multi.groupby(
        ["helm_annotation", "cell_line", "target_RNA"], sort=False
    ):
        if len(g) < 4:
            continue
        n_candidates += 1
        doses = g["dosage"].to_numpy(dtype=float)
        true_fit = _fit_group(doses, g["inhibition_percent"].to_numpy(dtype=float))
        pred_fit = _fit_group(doses, g["prediction"].to_numpy(dtype=float))
        if true_fit is None or pred_fit is None:
            continue
        ic50_true.append(true_fit[0])
        ic50_pred.append(pred_fit[0])
        n_converged += 1

    if not ic50_true:
        return {
            "enrichment_factor": float("nan"),
            "n": 0,
            "n_candidates": n_candidates,
            "stage": 1,
            "stage_name": PIPELINE_STAGES[1].name,
            "source": _SOURCE_NOTE,
        }

    ef = enrichment_at_top_k(
        np.asarray(ic50_true), np.asarray(ic50_pred), PIPELINE_STAGES[1]
    )
    ef["stage"] = 1
    ef["stage_name"] = PIPELINE_STAGES[1].name
    ef["threshold"] = (
        f"IC50 {PIPELINE_STAGES[1].threshold_op} "
        f"{PIPELINE_STAGES[1].threshold_value} nM"
    )
    ef["n_candidates"] = n_candidates
    ef["n_converged"] = n_converged
    ef["source"] = _SOURCE_NOTE
    return ef


def _aggregate_5fold() -> dict | None:
    """Compute pooled OOF point estimates and retain fold-level variability.

    The primary point estimate is calculated once from the concatenated held-out
    predictions, matching the pooled OOF convention used by the OligoGym models.
    Fold-level EFs and selected-pass rates are retained for uncertainty analyses.
    Returns ``None`` when no fold predictions are available.
    """
    available = [p for p in PREDICTIONS_PATHS_5FOLD if p.exists()]
    if not available:
        return None
    try:
        electro_ids = _electroporation_table_ids()
    except FileNotFoundError:
        electro_ids = None
    eff_efs: list[float] = []
    eff_precs: list[float] = []
    eff_base_rates: list[float] = []
    pot_efs: list[float] = []
    pot_precs: list[float] = []
    pot_base_rates: list[float] = []
    frames: list[pd.DataFrame] = []
    per_fold = []
    for p in available:
        df = pd.read_parquet(p)
        frames.append(df)
        eff = _efficacy_ef(df)
        pot = _potency_ef(df, electro_ids=electro_ids)
        eff_efs.append(float(eff["enrichment_factor"]))
        eff_precs.append(float(eff["selected_pass_rate"]))
        eff_base_rates.append(float(eff["base_rate"]))
        pot_efs.append(float(pot["enrichment_factor"]))
        pot_precs.append(float(pot["selected_pass_rate"]))
        pot_base_rates.append(float(pot["base_rate"]))
        per_fold.append({
            "path": str(p.relative_to(_ROOT)),
            "efficacy_ef": eff["enrichment_factor"],
            "efficacy_precision": eff["selected_pass_rate"],
            "efficacy_base_rate": eff["base_rate"],
            "potency_ef": pot["enrichment_factor"],
            "potency_precision": pot["selected_pass_rate"],
            "potency_base_rate": pot["base_rate"],
            "n_efficacy": eff["n"],
            "n_potency": pot["n"],
        })

    pooled_df = pd.concat(frames, ignore_index=True)
    pooled_eff = _efficacy_ef(pooled_df)
    pooled_pot = _potency_ef(pooled_df, electro_ids=electro_ids)

    def _agg(lst):
        arr = np.asarray(lst, dtype=float)
        return {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "per_fold": list(lst),
        }

    return {
        "efficacy": {
            "enrichment_factor": pooled_eff["enrichment_factor"],
            "selected_pass_rate": pooled_eff["selected_pass_rate"],
            "base_rate": pooled_eff["base_rate"],
            "n": pooled_eff["n"],
            "n_selected": pooled_eff["n_selected"],
            "enrichment_factor_std": _agg(eff_efs)["std"],
            "fold_efs": eff_efs,
            "fold_precs": eff_precs,
            "fold_base_rates": eff_base_rates,
            "n_folds": len(eff_efs),
            "source": _SOURCE_NOTE + "; primary point estimate pooled across OOF predictions",
        },
        "potency": {
            "enrichment_factor": pooled_pot["enrichment_factor"],
            "selected_pass_rate": pooled_pot["selected_pass_rate"],
            "base_rate": pooled_pot["base_rate"],
            "n": pooled_pot["n"],
            "n_selected": pooled_pot["n_selected"],
            "enrichment_factor_std": _agg(pot_efs)["std"],
            "fold_efs": pot_efs,
            "fold_precs": pot_precs,
            "fold_base_rates": pot_base_rates,
            "n_folds": len(pot_efs),
            "source": _SOURCE_NOTE + "; primary point estimate pooled across OOF predictions",
        },
        "meta": {
            "mode": "5fold_patent_split",
            "per_fold": per_fold,
        },
    }


def main() -> None:
    # Prefer 5-fold aggregation when per-fold parquets are present.
    agg = _aggregate_5fold()
    if agg is not None:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(agg, indent=2))
        print(f"Wrote {OUT_PATH} (5-fold aggregation; "
              f"n_folds={agg['efficacy']['n_folds']})")
        print(f"  pooled efficacy EF = {agg['efficacy']['enrichment_factor']:.3f} "
              f"± {agg['efficacy']['enrichment_factor_std']:.3f}")
        print(f"  pooled potency  EF = {agg['potency']['enrichment_factor']:.3f} "
              f"± {agg['potency']['enrichment_factor_std']:.3f}")
        return

    if not PREDICTIONS_PATH.exists():
        raise SystemExit(
            f"Predictions parquet missing: {PREDICTIONS_PATH}. "
            "Run `just oligoai-launch-folds` (or fetch the run's artifacts) first."
        )
    df = pd.read_parquet(PREDICTIONS_PATH)
    out = {
        "efficacy": {**_efficacy_ef(df), "source": _SOURCE_NOTE_SINGLE},
        "potency": {**_potency_ef(df), "source": _SOURCE_NOTE_SINGLE},
        "meta": {
            "mode": "single_fold_patent_split_fold0",
            "predictions": str(PREDICTIONS_PATH.relative_to(_ROOT)),
            "n_rows": int(len(df)),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    eff = out["efficacy"]["enrichment_factor"]
    pot = out["potency"]["enrichment_factor"]
    print(f"Wrote {OUT_PATH} (single-split fallback)")
    print(f"  efficacy EF = {eff:.3f}x")
    print(f"  potency  EF = {pot:.3f}x")


if __name__ == "__main__":
    main()
