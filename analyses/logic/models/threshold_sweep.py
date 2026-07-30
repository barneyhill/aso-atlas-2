"""Threshold-sensitivity sweep (rebuttal S3).

For each pipeline stage, sweep its pass/fail threshold over plausible values
(all other stages held at canonical thresholds), recomputing FROM DATA both:
  * the baseline pass rate (from the stage's readout distribution), and
  * the model's selected-pass-rate precision at that threshold (from pooled
    out-of-fold predictions: OligoAI in vitro, XGBoost in vivo),
then the cost recursion (ceiling and expected-count), without flooring an
underperforming model at EF=1. This isolates each stage's threshold as a
one-at-a-time sensitivity (a tornado), answering UEDf W3 / AC #5.

The lenient rat-neuro FOB<=2 gate that would pass the Phase-3 candidate ION582
is included explicitly.

    uv run python -m analyses.logic.models.threshold_sweep
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.logic.enrichment import (
    _passes,
    enrichment_at_top_k,
    stratified_enrichment_at_top_k,
)
from analyses.logic.pipeline import (
    MONKEY_ALT_ULN,
    MOUSE_ALT_ULN,
    PIPELINE_STAGES,
    RAT_ALT_ULN,
    TARGET_CANDIDATES,
)

_root = Path(__file__).resolve().parents[3]
RESULTS_DIR = _root / "data/results"

# stage index -> XGBoost in-vivo dataset (+ whether ranking is dosage-stratified)
IN_VIVO = {2: ("mouse_hepatic", True), 3: ("mouse_neuro", False),
           4: ("rat_hepatic", True), 5: ("rat_neuro", False)}

# Plausible threshold sets per stage. Hepatic/NHP are ULN multipliers -> IU/L.
SWEEPS = {
    0: ("Efficacy (inhib >)", [70, 80, 90]),
    1: ("Potency (IC50 < nM)", [250, 500, 1000]),
    2: ("Mouse hepatic (ALT xULN)", [1.5 * MOUSE_ALT_ULN, 2 * MOUSE_ALT_ULN, 3 * MOUSE_ALT_ULN]),
    3: ("Mouse neuro (bFOB <=)", [1, 2, 3]),
    4: ("Rat hepatic (ALT xULN)", [1.5 * RAT_ALT_ULN, 2 * RAT_ALT_ULN, 3 * RAT_ALT_ULN]),
    5: ("Rat neuro (mFOB <=)", [1, 2, 3]),
    6: ("NHP hepatic (ALT xULN)", [1.5 * MONKEY_ALT_ULN, 2 * MONKEY_ALT_ULN, 3 * MONKEY_ALT_ULN]),
}
CANONICAL_LABELS = {0: 80, 1: 500, 2: 1.5 * MOUSE_ALT_ULN, 3: 1,
                    4: 1.5 * RAT_ALT_ULN, 5: 1, 6: 1.5 * MONKEY_ALT_ULN}
STAGE_DISPLAY_NAMES = {
    0: "Efficacy",
    1: "Potency",
    2: "Mouse ALT",
    3: "Mouse bFOB",
    4: "Rat ALT",
    5: "Rat mFOB",
    6: "NHP ALT",
}
ALT_ULN_BY_STAGE = {2: MOUSE_ALT_ULN, 4: RAT_ALT_ULN, 6: MONKEY_ALT_ULN}


def _number(value):
    """Compact, deterministic display for a threshold value."""
    return f"{float(value):g}"


def _threshold_display(stage_idx, value):
    """Human-readable gate label derived from the analysed numeric threshold."""
    if stage_idx == 0:
        return f">{_number(value)}%"
    if stage_idx == 1:
        return f"<{_number(value)} nM"
    if stage_idx in ALT_ULN_BY_STAGE:
        multiple = float(value) / ALT_ULN_BY_STAGE[stage_idx]
        return f"<{_number(multiple)}×ULN"
    return f"≤{_number(value)}"


def _load_xgb_pooled(bench):
    """Pooled (concatenated over folds) XGBoost preds per in-vivo dataset."""
    best = {}
    for row in bench["all_results"]:
        if row["model"] != "XGBoost":
            continue
        sp = row.get("spearman")
        if sp is None or (isinstance(sp, float) and math.isnan(sp)):
            continue
        best.setdefault(row["dataset"], row)  # one XGBoost config per dataset
    pooled = {}
    for idx, (ds, strat) in IN_VIVO.items():
        row = best.get(ds)
        if row is None:
            continue
        yt, yp, strata = [], [], []
        skey = "dosage_mg_per_kg"
        for fm in row["fold_metrics"]:
            yt.extend(fm["y_true"]); yp.extend(fm["y_pred"])
            cond = fm.get("conditions", {}).get(skey)
            if cond is not None:
                strata.extend(cond)
        pooled[idx] = {
            "y_true": np.asarray(yt, float),
            "y_pred": np.asarray(yp, float),
            "strata": np.asarray(strata, float) if len(strata) == len(yt) else None,
            "fold_metrics": row["fold_metrics"],
            "stratified": strat,
        }
    return pooled


def _load_oligoai_predictions():
    """Pooled held-out efficacy rows and potency curves from the five OligoAI folds."""
    from analyses.logic.models.oligoai_efs import _electroporation_table_ids, _fit_group

    try:
        electro_ids = _electroporation_table_ids()
    except FileNotFoundError:
        electro_ids = set()

    frames = []
    for i in range(5):
        p = RESULTS_DIR / f"oligoai_fold{i}_predictions.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return None, None

    # Concatenate raw held-out rows before deriving either endpoint. This is the
    # same pooled-OOF path used for the primary OligoAI point estimates.
    pooled = pd.concat(frames, ignore_index=True)
    doses_per_table = pooled.groupby("custom_id")["dosage"].nunique()
    single_ids = doses_per_table[doses_per_table == 1].index
    efficacy = pooled[pooled["custom_id"].isin(single_ids)].copy()

    ic50_true, ic50_pred = [], []
    multi_ids = doses_per_table[doses_per_table > 1].index
    multi = pooled[
        pooled["custom_id"].isin(multi_ids) & pooled["custom_id"].isin(electro_ids)
    ]
    for _, g in multi.groupby(
        ["helm_annotation", "cell_line", "target_RNA"], sort=False
    ):
        if len(g) < 4:
            continue
        doses = g["dosage"].to_numpy(dtype=float)
        true_fit = _fit_group(doses, g["inhibition_percent"].to_numpy(dtype=float))
        pred_fit = _fit_group(doses, g["prediction"].to_numpy(dtype=float))
        if true_fit is not None and pred_fit is not None:
            ic50_true.append(true_fit[0])
            ic50_pred.append(pred_fit[0])

    potency = None
    if ic50_true:
        potency = {
            "y_true": np.asarray(ic50_true, dtype=float),
            "y_pred": np.asarray(ic50_pred, dtype=float),
        }
    return efficacy, potency


def _precision(stage_idx, stage_obj, xgb, oligoai_eff, oligoai_pot):
    """Pooled model selected-pass-rate at the given (possibly swept) threshold."""
    if stage_idx == 0 and oligoai_eff is not None:
        r = stratified_enrichment_at_top_k(
            oligoai_eff["inhibition_percent"].to_numpy(float),
            oligoai_eff["prediction"].to_numpy(float),
            stage_obj, strata=oligoai_eff["dosage"].to_numpy(float))
        return r.get("selected_pass_rate")
    if stage_idx == 1 and oligoai_pot is not None:
        r = enrichment_at_top_k(
            oligoai_pot["y_true"], oligoai_pot["y_pred"], stage_obj
        )
        return r.get("selected_pass_rate")
    if stage_idx in xgb:
        d = xgb[stage_idx]
        if d["stratified"] and d["strata"] is not None:
            r = stratified_enrichment_at_top_k(d["y_true"], d["y_pred"], stage_obj, d["strata"])
        else:
            r = enrichment_at_top_k(d["y_true"], d["y_pred"], stage_obj)
        return r.get("selected_pass_rate")
    return None


def _recompute(base_rates, precisions, use_ceil):
    """Total cost with model precision used directly, including EF < 1."""
    n = len(PIPELINE_STAGES)
    eff = list(base_rates)
    for i, prec in precisions.items():
        if prec is not None and base_rates[i] > 0:
            eff[i] = min(max(prec, 1e-12), 1.0)
    asos = [0.0] * (n + 1)
    asos[-1] = TARGET_CANDIDATES
    for i in range(n - 1, -1, -1):
        if eff[i] <= 0:
            return float("inf")
        asos[i] = math.ceil(asos[i + 1] / eff[i]) if use_ceil else asos[i + 1] / eff[i]
    return sum(asos[i] * PIPELINE_STAGES[i].cost_per_aso for i in range(n))


def _savings(base_rates, precisions, use_ceil):
    base_cost = _recompute(base_rates, {}, use_ceil)          # no model
    model_cost = _recompute(base_rates, precisions, use_ceil)  # with model
    return 100.0 * (1 - model_cost / base_cost)


def _rate_fields(base_rate, precision):
    """Report the quantities the AC requested at each threshold."""
    selected = None if precision is None else float(precision)
    enrichment = (
        None if selected is None or base_rate <= 0 else selected / float(base_rate)
    )
    return {
        "baseline_pass_rate": round(float(base_rate), 4),
        "model_selected_pass_rate": (
            None if selected is None else round(selected, 4)
        ),
        "enrichment_factor": (
            None if enrichment is None else round(enrichment, 3)
        ),
    }


def compute():
    pipe = json.loads((RESULTS_DIR / "pipeline_results.json").read_text())
    dists = [np.asarray(d, float) for d in pipe["distributions"]]
    canon_base = list(pipe["proportions"])

    bench = json.loads((RESULTS_DIR / "oligogym_benchmark.json").read_text())
    xgb = _load_xgb_pooled(bench)
    oligoai_eff, oligoai_pot = _load_oligoai_predictions()

    # Canonical precisions from pooled out-of-fold predictions for every enriched stage.
    canon_prec = {}
    for i in [0, 1, 2, 3, 4, 5]:
        canon_prec[i] = _precision(
            i, PIPELINE_STAGES[i], xgb, oligoai_eff, oligoai_pot
        )

    canonical_ceiling = _savings(canon_base, canon_prec, True)
    canonical_expected = _savings(canon_base, canon_prec, False)

    # The threshold sweep and primary economic projection must use the same
    # pooled-OOF point estimates at the canonical gates.
    ef_table_path = RESULTS_DIR / "ef_table.json"
    if ef_table_path.exists():
        rows = json.loads(ef_table_path.read_text())
        combined = next((row for row in rows if row.get("group") == "combined"), None)
        if combined is not None:
            primary_prec = {
                int(stage): value
                for stage, value in combined["prec_by_stage"].items()
                if value is not None
            }
            mismatches = {
                stage: (canon_prec.get(stage), primary_prec.get(stage))
                for stage in range(6)
                if not np.isclose(canon_prec.get(stage), primary_prec.get(stage), atol=5e-5)
            }
            if mismatches:
                raise ValueError(
                    "canonical threshold precisions do not match the primary pooled-OOF "
                    f"precisions: {mismatches}"
                )

    # ---- tornado: sweep each stage one at a time ----
    tornado = []
    for idx, (label, values) in SWEEPS.items():
        pts = []
        for t in values:
            stage_t = replace(PIPELINE_STAGES[idx], threshold_value=t)
            base = list(canon_base)
            base[idx] = float(_passes(dists[idx], stage_t).mean())
            prec = dict(canon_prec)
            if idx != 6:  # NHP unenriched
                prec[idx] = _precision(
                    idx, stage_t, xgb, oligoai_eff, oligoai_pot
                )
            pts.append({
                "threshold": round(float(t), 2),
                "threshold_display": _threshold_display(idx, t),
                "is_canonical": abs(t - CANONICAL_LABELS[idx]) < 1e-6,
                **_rate_fields(base[idx], prec.get(idx)),
                "ceiling_savings_pct": round(_savings(base, prec, True), 1),
                "expected_savings_pct": round(_savings(base, prec, False), 1),
            })
        cvals = [p["ceiling_savings_pct"] for p in pts]
        tornado.append({
            "stage_idx": idx,
            "label": label,
            "display_name": STAGE_DISPLAY_NAMES[idx],
            "points": pts,
            "ceiling_min": min(cvals), "ceiling_max": max(cvals),
        })

    # ---- ION582 lenient gate: rat-neuro mFOB <= 2 (would pass ION582) ----
    stage5_lenient = replace(PIPELINE_STAGES[5], threshold_value=2)
    base = list(canon_base); base[5] = float(_passes(dists[5], stage5_lenient).mean())
    prec = dict(canon_prec)
    prec[5] = _precision(5, stage5_lenient, xgb, oligoai_eff, oligoai_pot)
    ion582 = {
        "gate": "rat neuro mFOB <= 2",
        **_rate_fields(base[5], prec[5]),
        "ceiling_savings_pct": round(_savings(base, prec, True), 1),
        "expected_savings_pct": round(_savings(base, prec, False), 1),
    }

    all_ceiling = [p["ceiling_savings_pct"] for tb in tornado for p in tb["points"]]
    all_expected = [p["expected_savings_pct"] for tb in tornado for p in tb["points"]]
    return {
        "design": "one-at-a-time threshold sensitivity using pooled out-of-fold predictions",
        "ef_floor": False,
        "canonical_ceiling_savings_pct": round(canonical_ceiling, 1),
        "canonical_expected_savings_pct": round(canonical_expected, 1),
        "tornado": tornado,
        "ion582_lenient": ion582,
        "overall_ceiling_min": min(all_ceiling),
        "overall_ceiling_max": max(all_ceiling),
        "overall_expected_min": min(all_expected),
        "overall_expected_max": max(all_expected),
    }


def main():
    res = compute()
    (RESULTS_DIR / "threshold_sweep.json").write_text(json.dumps(res, indent=2))
    print(f"Wrote {RESULTS_DIR / 'threshold_sweep.json'}")
    print(f"  canonical: ceiling {res['canonical_ceiling_savings_pct']}%  "
          f"expected {res['canonical_expected_savings_pct']}%")
    for tb in res["tornado"]:
        vals = "/".join(f"{p['ceiling_savings_pct']}" for p in tb["points"])
        print(f"  {tb['label']:26s} ceiling {vals}")
    print(f"  ION582 lenient (mFOB<=2): {res['ion582_lenient']['ceiling_savings_pct']}% ceiling / "
          f"{res['ion582_lenient']['expected_savings_pct']}% expected")
    print(f"  overall ceiling band [{res['overall_ceiling_min']}, {res['overall_ceiling_max']}]%")


if __name__ == "__main__":
    main()
