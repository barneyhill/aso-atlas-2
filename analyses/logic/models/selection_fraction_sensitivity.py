"""Recompute held-out model precision across selection fractions.

This module contains only the K-sweep used by the focused AC analysis. It is
separate from the superseded uncertainty-budget analysis so that the latter can
be removed without retaining its rejected bootstrap and floor arguments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyses.logic.enrichment import (
    enrichment_at_top_k,
    stage_for_dataset,
    stratified_enrichment_at_top_k,
)
from analyses.logic.ef_table import (
    DATASET_STRATA_KEY,
    _best_config_per_dataset,
)
from analyses.logic.models.oligoai_efs import (
    PREDICTIONS_PATHS_5FOLD,
    _electroporation_table_ids,
    _fit_group,
)
from analyses.logic.pipeline import PIPELINE_STAGES


IN_VIVO = {2: "mouse_hepatic", 3: "mouse_neuro", 4: "rat_hepatic", 5: "rat_neuro"}
TOX_MODEL = "XGBoost"


def oligoai_precision_by_k(ks):
    """Return pooled held-out OligoAI efficacy and potency precision at each K."""
    try:
        electro_ids = _electroporation_table_ids()
    except FileNotFoundError:
        electro_ids = set()

    frames = [
        pd.read_parquet(path)
        for path in PREDICTIONS_PATHS_5FOLD
        if path.exists()
    ]
    if not frames:
        return {}, {}

    # Match the paper's primary method: combine all held-out predictions before
    # measuring performance instead of averaging five fold percentages.
    frame = pd.concat(frames, ignore_index=True)
    n_doses = frame.groupby("custom_id")["dosage"].nunique()
    single = frame[frame["custom_id"].isin(n_doses[n_doses == 1].index)]

    multiple = frame[
        frame["custom_id"].isin(n_doses[n_doses > 1].index)
        & frame["custom_id"].isin(electro_ids)
    ]
    true_ic50, predicted_ic50 = [], []
    for _, group in multiple.groupby(
        ["helm_annotation", "cell_line", "target_RNA"], sort=False,
    ):
        if len(group) < 4:
            continue
        dose = group["dosage"].to_numpy(dtype=float)
        true_fit = _fit_group(
            dose, group["inhibition_percent"].to_numpy(dtype=float),
        )
        predicted_fit = _fit_group(
            dose, group["prediction"].to_numpy(dtype=float),
        )
        if true_fit and predicted_fit:
            true_ic50.append(true_fit[0])
            predicted_ic50.append(predicted_fit[0])

    efficacy = {}
    potency = {}
    for k in ks:
        result = stratified_enrichment_at_top_k(
            single["inhibition_percent"].to_numpy(dtype=float),
            single["prediction"].to_numpy(dtype=float),
            PIPELINE_STAGES[0],
            single["dosage"].to_numpy(dtype=float),
            k=k,
        )
        if result.get("selected_pass_rate") is not None:
            efficacy[k] = float(result["selected_pass_rate"])

        if true_ic50:
            result = enrichment_at_top_k(
                np.asarray(true_ic50),
                np.asarray(predicted_ic50),
                PIPELINE_STAGES[1],
                k=k,
            )
            if result.get("selected_pass_rate") is not None:
                potency[k] = float(result["selected_pass_rate"])

    return efficacy, potency


def _pooled_precision_at_k(row, dataset, k):
    """Measure precision after combining all held-out folds, as in the paper."""
    y_true, y_pred = [], []
    strata_parts = []
    strata_key = DATASET_STRATA_KEY.get(dataset)
    for fold in row.get("fold_metrics", []):
        if "y_true" not in fold or "y_pred" not in fold:
            continue
        y_true.extend(fold["y_true"])
        y_pred.extend(fold["y_pred"])
        if strata_key:
            strata = fold.get("conditions", {}).get(strata_key)
            if strata is not None:
                strata_parts.extend(strata)

    if not y_true:
        return None

    stage = stage_for_dataset(dataset)
    if strata_key and len(strata_parts) == len(y_true):
        result = stratified_enrichment_at_top_k(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
            stage,
            np.asarray(strata_parts),
            k=k,
        )
    else:
        result = enrichment_at_top_k(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
            stage,
            k=k,
        )
    precision = result.get("selected_pass_rate")
    return float(precision) if precision is not None else None


def k_sweep(benchmark, ks=(0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)):
    """Return held-out in-vitro and in-vivo stage precisions at each K."""
    best = _best_config_per_dataset(benchmark)
    efficacy, potency = oligoai_precision_by_k(ks)

    rows = []
    for k in ks:
        in_vitro = {}
        if k in efficacy:
            in_vitro[0] = efficacy[k]
        if k in potency:
            in_vitro[1] = potency[k]

        in_vivo = {}
        for stage, dataset in IN_VIVO.items():
            row = best.get((dataset, TOX_MODEL))
            if row is None:
                continue
            precision = _pooled_precision_at_k(row, dataset, k)
            if precision is not None:
                in_vivo[stage] = precision

        rows.append({
            "K": k,
            "in_vitro_precisions": {
                str(stage): value for stage, value in sorted(in_vitro.items())
            },
            "in_vivo_precisions": {
                str(stage): value for stage, value in sorted(in_vivo.items())
            },
        })
    return rows
