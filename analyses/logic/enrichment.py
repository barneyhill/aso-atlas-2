"""Enrichment factor (EF) computation under top-20% selection.

Selection rule (single definition used everywhere in this project):
- Rank compounds by predicted score.
- Select the top 20% (``SELECTION_FRACTION``).
- ``EF = P(pass | selected) / P(pass overall)``.

For stages where pass = low value (ALT, IC50, FOB), selection picks the
*lowest* predictions; for stages where pass = high value (inhibition),
selection picks the *highest* predictions. Direction comes from
``stage.threshold_op``.
"""

from __future__ import annotations

import numpy as np

from analyses.logic.pipeline import PIPELINE_STAGES, PipelineStage

SELECTION_FRACTION = 0.20


def _passes(y_true: np.ndarray, stage: PipelineStage) -> np.ndarray:
    op = stage.threshold_op
    t = stage.threshold_value
    if op == "<":
        return y_true < t
    if op == "<=":
        return y_true <= t
    if op == ">":
        return y_true > t
    if op == ">=":
        return y_true >= t
    raise ValueError(f"Unsupported threshold_op: {op!r}")


def enrichment_at_top_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage: PipelineStage,
    k: float | None = None,
) -> dict:
    """Compute EF at the top-K predicted-safest fraction.

    Selection policy:
    - For "pass when low" stages (ALT, IC50, FOB) we rank ascending and take the
      lowest predictions; for "pass when high" stages (inhibition) we rank
      descending. Threshold direction is derived from ``stage.threshold_op``.
    - Default K = ``SELECTION_FRACTION`` (20%). Pass a float to override.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    n = len(y_true)
    if n == 0:
        return {"enrichment_factor": float("nan"), "n": 0}

    passes = _passes(y_true, stage)
    base_rate = float(passes.mean())

    k_eff = SELECTION_FRACTION if k is None else float(k)
    if not np.isfinite(k_eff) or k_eff <= 0:
        return {"enrichment_factor": float("nan"), "n": int(n),
                "base_rate": round(base_rate, 4)}

    n_sel = max(1, int(np.floor(k_eff * n)))
    ascending = stage.threshold_op in ("<", "<=")
    order = np.argsort(y_pred, kind="mergesort")
    selected = order[:n_sel] if ascending else order[-n_sel:]

    selected_pass_rate = float(passes[selected].mean())
    ef = selected_pass_rate / base_rate if base_rate > 0 else float("nan")

    return {
        "enrichment_factor": round(ef, 3),
        "base_rate": round(base_rate, 4),
        "selected_pass_rate": round(selected_pass_rate, 4),
        "n": int(n),
        "n_selected": int(n_sel),
        "selection_policy": "top_k_lowest" if ascending else "top_k_highest",
        "selection_fraction": round(k_eff, 4),
    }


# Maps benchmark dataset names → (pipeline stage index, PipelineStage)
DATASET_TO_STAGE: dict[str, int] = {
    "in_vitro_inhibition": 0,
    "potency": 1,
    "mouse_hepatic": 2,
    "mouse_neuro": 3,
    "rat_hepatic": 4,
    "rat_neuro": 5,
}


def stage_for_dataset(ds_name: str) -> PipelineStage:
    return PIPELINE_STAGES[DATASET_TO_STAGE[ds_name]]


def stratified_enrichment_at_top_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage: PipelineStage,
    strata: np.ndarray,
    k: float | None = None,
    min_stratum: int = 10,
) -> dict:
    """EF with within-stratum ranking to control for experimental confounders.

    Within each unique value of ``strata`` (e.g. dosage group), compounds are
    ranked by ``y_pred`` and the top ``SELECTION_FRACTION`` are selected per
    stratum. All per-stratum selections are pooled, then:

        EF = pooled_selected_pass_rate / overall_base_rate

    Strata with fewer than ``min_stratum`` samples are dropped (too noisy for
    reliable within-group ranking).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    strata = np.asarray(strata)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred, strata = y_true[valid], y_pred[valid], strata[valid]

    n = len(y_true)
    if n == 0:
        return {"enrichment_factor": float("nan"), "n": 0}

    passes = _passes(y_true, stage)
    base_rate = float(passes.mean())
    if not (base_rate > 0):
        return {"enrichment_factor": float("nan"), "n": int(n),
                "base_rate": round(base_rate, 4)}

    k_eff = SELECTION_FRACTION if k is None else float(k)
    ascending = stage.threshold_op in ("<", "<=")
    selected_mask = np.zeros(n, dtype=bool)
    n_strata_used = 0
    n_dropped = 0

    for g in np.unique(strata):
        g_idx = np.where(strata == g)[0]
        n_g = len(g_idx)
        if n_g < min_stratum:
            n_dropped += n_g
            continue
        n_strata_used += 1
        n_sel_g = max(1, int(np.floor(k_eff * n_g)))
        g_order = np.argsort(y_pred[g_idx], kind="mergesort")
        g_selected = g_order[:n_sel_g] if ascending else g_order[-n_sel_g:]
        selected_mask[g_idx[g_selected]] = True

    n_sel = int(selected_mask.sum())
    if n_sel == 0:
        return {"enrichment_factor": float("nan"), "n": int(n),
                "base_rate": round(base_rate, 4)}

    selected_pass_rate = float(passes[selected_mask].mean())
    ef = selected_pass_rate / base_rate

    return {
        "enrichment_factor": round(ef, 3),
        "base_rate": round(base_rate, 4),
        "selected_pass_rate": round(selected_pass_rate, 4),
        "n": int(n),
        "n_selected": n_sel,
        "n_strata": n_strata_used,
        "n_dropped": n_dropped,
        "selection_policy": ("stratified_top_k_lowest" if ascending
                             else "stratified_top_k_highest"),
        "selection_fraction": round(k_eff, 4),
    }


