"""Held-out enrichment curves for the paper-3 q-feasibility figure.

Reads ``data/results/oligoai_geneholdout_predictions.parquet`` — the test-set
predictions from the single gene-holdout retrain (``just oligoai-holdout-genes``)
that holds **SCN2A** and **UBE3A-ATS** entirely out of training. Because the
held-out genes never appear in any training patent, these predictions are an
honest "predict for an unseen gene" evaluation (unlike the patent-grouped
5-fold folds, where a gene can straddle train/test via a sibling patent).

Two stages, matched to the n-Lorem CNS screening cascade (NAR 2026, Table 1):

- **Efficacy (n-Lorem step 1, primary screen)** → **SCN2A**, electroporation,
  single dose 4000 nM. Pass = inhibition > 80% (pipeline stage 0). This is the
  feasibility lever for ``q``: ranking tiled ASOs by predicted inhibition and
  screening only the top-X% raises the >80% hit rate by the enrichment factor.
- **Potency (n-Lorem step 2, dose response)** → **UBE3A-ATS**, *free uptake*
  (gymnosis) in iPSC/GABA neurons. Pass = IC50 < 1 µM (n-Lorem free-uptake
  criterion; 500 nM also reported for paper-2 parity). IC50 uses paper-2's
  canonical ``fit_ic50_for_compound`` (Hill fit) on predicted vs observed
  inhibition across the dose series — the same estimator the paper-2 attrition
  pipeline reports potency with.

SCN2A has zero free-uptake data, so the two stages necessarily use different
genes/contexts — UBE3A-ATS is the only substantial neuronal free-uptake
dose-response set and matches n-Lorem's iPSC-neuron paradigm closely.

Output: ``data/results/geneholdout_enrichment.json`` (EF swept over top-X% for
each stage, with base rate, selected pass rate and n at every fraction).

Usage:
    uv run python -m analyses.logic.models.geneholdout_enrichment
    uv run python -m analyses.logic.models.geneholdout_enrichment \
        --predictions data/results/oligoai_geneholdout_predictions.parquet
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.logic.enrichment import enrichment_at_top_k
from analyses.logic.pipeline import PIPELINE_STAGES
from analyses.utils.compounds import fit_ic50_for_compound

_ROOT = Path(__file__).resolve().parents[3]
_PRED_DEFAULT = _ROOT / "data/results/oligoai_geneholdout_predictions.parquet"
_OUT_PATH = _ROOT / "data/results/geneholdout_enrichment.json"

EFFICACY_GENE = "SCN2A"
EFFICACY_DOSE_NM = 4000.0
POTENCY_GENE = "UBE3A-ATS"
# Single clean free-uptake setup: the commercial iPSC GABAergic-neuron line that
# carries ~93% of the 4PL-fittable UBE3A-ATS gymnosis curves (241 of 259). The
# small patient-derived Angelman iPSC subgroups (3-dose, mixed source) are
# excluded to keep one experimental context.
POTENCY_CELL_LINE = "iCell GABANeurons"

# Top-X% fractions swept for the enrichment curve. The smallest bins are only
# reported when they retain >= MIN_SELECTED candidates (a finite tiled pool
# can't support an arbitrarily deep top-X%).
TOP_FRACTIONS = [0.50, 0.25, 0.10, 0.05, 0.025]
MIN_SELECTED = 30

# IC50 pass thresholds (nM): n-Lorem free-uptake criterion (1 µM) is primary;
# 500 nM mirrors paper-2's potency stage.
POTENCY_THRESHOLDS_NM = [1000.0, 500.0]


def _single_dose_table_ids(df: pd.DataFrame) -> set[str]:
    """custom_ids in `df` whose rows are a single-dose screen (one dosage)."""
    n = df.groupby("custom_id")["dosage"].nunique()
    return set(n[n == 1].index)


def _sweep_efficacy(df: pd.DataFrame) -> dict:
    """EF vs top-X% for SCN2A single-dose (4000 nM) inhibition, pass = >80%."""
    stage = PIPELINE_STAGES[0]  # inhibition > 80
    single_ids = _single_dose_table_ids(df)
    sub = df[
        (df["target_RNA"] == EFFICACY_GENE)
        & (df["custom_id"].isin(single_ids))
        & np.isclose(df["dosage"].astype(float), EFFICACY_DOSE_NM)
    ].copy()
    # One row per ASO sequence (a compound can recur across patent tables at the
    # same dose); keep the mean measured inhibition, prediction is sequence-fixed.
    sub = (sub.groupby("helm_annotation", as_index=False)
              .agg(inhibition_percent=("inhibition_percent", "mean"),
                   prediction=("prediction", "mean")))
    y_true = sub["inhibition_percent"].to_numpy(float)
    y_pred = sub["prediction"].to_numpy(float)
    curve = _sweep(y_true, y_pred, stage)
    return {
        "gene": EFFICACY_GENE,
        "stage": "efficacy",
        "stage_name": stage.name,
        "assay": "electroporation, single dose 4000 nM",
        "threshold": f"inhibition {stage.threshold_op} {stage.threshold_value}%",
        "n_asos": int(len(sub)),
        "base_rate": curve["base_rate"],
        "n_pass": int((y_true > stage.threshold_value).sum()),
        "top_fraction_curve": curve["points"],
    }


def _fit_ic50s(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    """IC50 (observed, predicted) per compound via paper-2's canonical estimator.

    Uses ``fit_ic50_for_compound`` (Hill fit, fitted IC50 parameter, R²>0.5,
    bounds allowing sub-range potency) — the same method the paper-2 attrition
    pipeline reports potency with — for both the observed inhibition (pass
    label) and the model-predicted inhibition (ranking). Compounds where either
    fit fails are dropped.
    """
    ic50_true: list[float] = []
    ic50_pred: list[float] = []
    n_candidates = 0
    n_converged = 0
    for _, g in df.groupby(["helm_annotation", "cell_line", "target_RNA"], sort=False):
        if g["dosage"].nunique() < 4:
            continue
        n_candidates += 1
        doses = g["dosage"].to_numpy(float)
        it = fit_ic50_for_compound(doses, g["inhibition_percent"].to_numpy(float))
        ip = fit_ic50_for_compound(doses, g["prediction"].to_numpy(float))
        if np.isnan(it) or np.isnan(ip):
            continue
        ic50_true.append(it)
        ic50_pred.append(ip)
        n_converged += 1
    return np.asarray(ic50_true), np.asarray(ic50_pred), n_candidates, n_converged


def _sweep_potency(df: pd.DataFrame) -> dict:
    """EF vs top-X% for UBE3A-ATS free-uptake IC50, pass = IC50 < threshold.

    Restricted to the single iCell GABANeurons free-uptake context.
    """
    multi = df[
        (df["target_RNA"] == POTENCY_GENE)
        & (df["cell_line"] == POTENCY_CELL_LINE)
    ].copy()
    ic50_true, ic50_pred, n_cand, n_conv = _fit_ic50s(multi)

    thresholds = {}
    for t_nm in POTENCY_THRESHOLDS_NM:
        stage = replace(PIPELINE_STAGES[1], threshold_value=t_nm)  # IC50 < t_nm
        if ic50_true.size:
            curve = _sweep(ic50_true, ic50_pred, stage)
            base_rate = curve["base_rate"]
            n_pass = int((ic50_true < t_nm).sum())
            points = curve["points"]
        else:
            base_rate, n_pass, points = float("nan"), 0, []
        thresholds[f"ic50_lt_{int(t_nm)}nM"] = {
            "threshold": f"IC50 < {int(t_nm)} nM",
            "base_rate": base_rate,
            "n_pass": n_pass,
            "top_fraction_curve": points,
        }

    return {
        "gene": POTENCY_GENE,
        "stage": "potency",
        "stage_name": PIPELINE_STAGES[1].name,
        "assay": f"free uptake (gymnosis), {POTENCY_CELL_LINE}",
        "cell_line": POTENCY_CELL_LINE,
        "n_curves_fitted": n_conv,
        "n_curves_candidate": n_cand,
        "thresholds": thresholds,
    }


def _sweep(y_true: np.ndarray, y_pred: np.ndarray, stage) -> dict:
    """Run enrichment_at_top_k over TOP_FRACTIONS; flag bins below the floor."""
    points = []
    base_rate = None
    for k in TOP_FRACTIONS:
        ef = enrichment_at_top_k(y_true, y_pred, stage, k=k)
        base_rate = ef.get("base_rate", base_rate)
        pt = {
            "top_fraction": k,
            "enrichment_factor": ef.get("enrichment_factor"),
            "selected_pass_rate": ef.get("selected_pass_rate"),
            "n_selected": ef.get("n_selected"),
            "below_floor": bool(ef.get("n_selected", 0) < MIN_SELECTED),
        }
        points.append(pt)
    return {"base_rate": base_rate, "points": points}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default=str(_PRED_DEFAULT),
                    help="Held-out predictions parquet (default: the geneholdout run).")
    ap.add_argument("--out", default=str(_OUT_PATH))
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(
            f"Predictions parquet missing: {pred_path}\n"
            "Run the gene-holdout retrain first: `just oligoai-holdout-genes` "
            "(builds the split + launches one RunPod training run)."
        )
    df = pd.read_parquet(pred_path)
    for col in ("target_RNA", "custom_id", "dosage", "inhibition_percent",
                "prediction", "helm_annotation", "cell_line"):
        if col not in df.columns:
            raise SystemExit(f"Predictions parquet missing column {col!r}")

    efficacy = _sweep_efficacy(df)
    potency = _sweep_potency(df)

    out = {
        "efficacy": efficacy,
        "potency": potency,
        "meta": {
            "predictions": str(pred_path.relative_to(_ROOT))
            if pred_path.is_relative_to(_ROOT) else str(pred_path),
            "holdout_genes": [EFFICACY_GENE, POTENCY_GENE],
            "top_fractions": TOP_FRACTIONS,
            "min_selected_floor": MIN_SELECTED,
            "note": ("Honest gene-level holdout: SCN2A and UBE3A-ATS held fully "
                     "out of training. Efficacy and potency use different "
                     "genes/contexts (SCN2A electroporation primary screen; "
                     "UBE3A-ATS free-uptake iPSC-neuron dose response)."),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(f"  efficacy (SCN2A >80%, n={efficacy['n_asos']}): "
          f"base={efficacy['base_rate']}")
    for pt in efficacy["top_fraction_curve"]:
        flag = "  (below floor)" if pt["below_floor"] else ""
        print(f"    top-{pt['top_fraction']*100:>4.1f}%: "
              f"EF={pt['enrichment_factor']}x  hit={pt['selected_pass_rate']}  "
              f"n_sel={pt['n_selected']}{flag}")
    pot1 = potency["thresholds"].get("ic50_lt_1000nM", {})
    print(f"  potency (UBE3A-ATS IC50<1µM, curves={potency['n_curves_fitted']}): "
          f"base={pot1.get('base_rate')}")
    for pt in pot1.get("top_fraction_curve", []):
        flag = "  (below floor)" if pt["below_floor"] else ""
        print(f"    top-{pt['top_fraction']*100:>4.1f}%: "
              f"EF={pt['enrichment_factor']}x  hit={pt['selected_pass_rate']}  "
              f"n_sel={pt['n_selected']}{flag}")


if __name__ == "__main__":
    main()
