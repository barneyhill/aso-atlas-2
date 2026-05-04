"""
ASO Development Pipeline Attrition Analysis.

Calculate success proportions at each stage, back-calculate initial ASOs
needed for target candidates, and save results to data/results/.

Pipeline Stages (Sequential):
1. Inhibition >80%
2. IC50 <500nM (electroporation)
3. Mouse ALT <1.5×ULN (90 IU/L)
4. Mouse Neuro FOB <=1
5. Rat ALT <1.5×ULN (58.5 IU/L)
6. Rat Neuro FOB <=1
7. Monkey ALT <100 IU/L
"""

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.utils.compounds import (
    compound_ic50s,
    compound_max_inhibition,
    compound_mean_biomarker,
    fit_ic50_for_compound,
    hill_equation,
)

_root = Path(__file__).resolve().parents[2]
DATA_DIR = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Pipeline stage definitions
# ---------------------------------------------------------------------------

# Literature ULN (IU/L): nonparametric 97.5th percentile, sex-averaged.
# Mouse (C57BL/6N): Otto et al. 2016, JAALAS 55(4):375-386
# Rat (Sprague-Dawley): He et al. 2017, PLoS ONE 12(12):e0189837
# Monkey (cynomolgus, ≥4yr): Bakker et al. 2023, Animals 13(3):445
MOUSE_ALT_ULN = 70   # (82 + 57) / 2
RAT_ALT_ULN = 39     # (47 + 30) / 2
MONKEY_ALT_ULN = 103  # (92 + 113) / 2

ALT_ULN_MULT = 1.5


@dataclass
class PipelineStage:
    name: str
    short_name: str
    threshold: str
    cost_per_aso: float
    color: str
    # Plotting metadata
    threshold_value: float = 0.0
    threshold_op: str = "<"
    transform: str = "none"       # "clip01", "log10", "integer"
    xlabel: str = ""
    bins: int | str = 30           # int or "integer" for unit-spaced
    xlim_lo: float | None = None   # explicit lower xlim (None = auto)


PIPELINE_STAGES = [
    PipelineStage("In vitro efficacy", "In vitro\nefficacy", "inhibition >80%", 500, "#4878A8",
                  threshold_value=80, threshold_op=">", transform="clip01", xlabel="Inhibition (%)", xlim_lo=0),
    PipelineStage("In vitro potency", "In vitro\npotency", "IC50 <500nM", 2000, "#6A9BC3",
                  threshold_value=500, threshold_op="<", transform="log10", xlabel="IC\u2085\u2080 (nM)", xlim_lo=1),
    PipelineStage("Mouse liver toxicity", "Mouse liver\ntoxicity", f"ALT <{ALT_ULN_MULT}×ULN", 15000, "#D4A574",
                  threshold_value=ALT_ULN_MULT * MOUSE_ALT_ULN, threshold_op="<", transform="log10", xlabel="ALT (IU/L)", xlim_lo=1),
    PipelineStage("Mouse neuro tolerability", "Mouse neuro\ntolerability", "bFOB <=1", 20000, "#E8B88A",
                  threshold_value=1, threshold_op="<=", transform="integer", xlabel="bFOB Score", bins="integer", xlim_lo=-0.5),
    PipelineStage("Rat liver toxicity", "Rat liver\ntoxicity", f"ALT <{ALT_ULN_MULT}×ULN", 25000, "#7BAA97",
                  threshold_value=ALT_ULN_MULT * RAT_ALT_ULN, threshold_op="<", transform="log10", xlabel="ALT (IU/L)", xlim_lo=1),
    PipelineStage("Rat neuro tolerability", "Rat neuro\ntolerability", "mFOB <=1", 30000, "#94C4A7",
                  threshold_value=1, threshold_op="<=", transform="integer", xlabel="mFOB Score", bins="integer", xlim_lo=-0.5),
    PipelineStage("Monkey liver toxicity", "Monkey liver\ntoxicity", f"ALT <{ALT_ULN_MULT}×ULN", 100000, "#9B8AA6",
                  threshold_value=ALT_ULN_MULT * MONKEY_ALT_ULN, threshold_op="<", transform="log10", xlabel="ALT (IU/L)", bins=12, xlim_lo=1),
]

TARGET_CANDIDATES = 1

# OligoAI enrichment factors.
# When ``data/results/oligoai_efs.json`` is available (written by
# analyses.logic.models.oligoai_efs from the fine-tune's held-out predictions),
# use those values so the strategy table reflects our own run on ASO Atlas 2.0.
# Otherwise fall back to Hill et al. 2025's published 3.14× for in vitro
# efficacy (external benchmark; no potency number reported).
def _load_oligoai_enrichment() -> dict:
    efs_path = RESULTS_DIR / "oligoai_efs.json"
    if not efs_path.exists():
        # Empty when the fold runs haven't landed yet — strategy_table will
        # skip the OligoAI row entirely rather than substitute an external
        # fallback, so the table only ever reports what we actually trained.
        return {}
    efs = json.loads(efs_path.read_text())
    out = {
        "inhibition": {
            "enrichment_factor": efs["efficacy"]["enrichment_factor"],
            "source": efs["efficacy"].get("source", ""),
        },
    }
    if "fold_efs" in efs["efficacy"]:
        out["inhibition"]["fold_efs"] = efs["efficacy"]["fold_efs"]
        out["inhibition"]["ef_mean"] = efs["efficacy"]["enrichment_factor"]
        out["inhibition"]["ef_std"] = efs["efficacy"].get("enrichment_factor_std")
    if "potency" in efs and np.isfinite(efs["potency"].get("enrichment_factor", float("nan"))):
        out["potency"] = {
            "enrichment_factor": efs["potency"]["enrichment_factor"],
            "source": efs["potency"].get("source", ""),
        }
        if "fold_efs" in efs["potency"]:
            out["potency"]["fold_efs"] = efs["potency"]["fold_efs"]
            out["potency"]["ef_mean"] = efs["potency"]["enrichment_factor"]
            out["potency"]["ef_std"] = efs["potency"].get("enrichment_factor_std")
    return out


OLIGOAI_ENRICHMENT = _load_oligoai_enrichment()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def calculate_proportions(in_vitro_df, dose_response_df, hepatic_df, neuro_df):
    """Calculate pass rates and distributions at each pipeline stage."""
    # Stage 1: Inhibition >80%
    iv_by_compound = compound_max_inhibition(in_vitro_df)
    n1_total = len(iv_by_compound)
    n1_pass = int((iv_by_compound > 80).sum())
    p1 = n1_pass / n1_total if n1_total > 0 else 0

    # Stage 2: IC50 <500nM (electroporation)
    ic50_series = compound_ic50s(dose_response_df)
    n2_total = len(ic50_series)
    n2_pass = int((ic50_series < 500).sum())
    p2 = n2_pass / n2_total if n2_total > 0 else 0

    # Stage 3: Mouse ALT
    valid_mouse_alt = compound_mean_biomarker(
        hepatic_df[hepatic_df["species"] == "mouse"], "ALT",
    )
    n3_total = len(valid_mouse_alt)
    n3_pass = int((valid_mouse_alt < PIPELINE_STAGES[2].threshold_value).sum())
    p3 = n3_pass / n3_total if n3_total > 0 else 0

    # Stage 4: Mouse FOB <=1
    valid_mouse_fob = compound_mean_biomarker(
        neuro_df[
            (neuro_df["species"] == "Mouse")
            & (neuro_df["dosage_ug"] == 700)
            & (neuro_df["latency_time_hours"] == 3)
            & (neuro_df["administration_method"] == "ICV")
        ], "FOB_score",
    )
    n4_total = len(valid_mouse_fob)
    n4_pass = int((valid_mouse_fob <= 1).sum())
    p4 = n4_pass / n4_total if n4_total > 0 else 0

    # Stage 5: Rat ALT
    valid_rat_alt = compound_mean_biomarker(
        hepatic_df[hepatic_df["species"] == "rat"], "ALT",
    )
    n5_total = len(valid_rat_alt)
    n5_pass = int((valid_rat_alt < PIPELINE_STAGES[4].threshold_value).sum())
    p5 = n5_pass / n5_total if n5_total > 0 else 0

    # Stage 6: Rat FOB <=1
    valid_rat_fob = compound_mean_biomarker(
        neuro_df[
            (neuro_df["species"] == "Rat")
            & (neuro_df["dosage_ug"] == 3000)
            & (neuro_df["latency_time_hours"] == 3)
        ], "FOB_score",
    )
    n6_total = len(valid_rat_fob)
    n6_pass = int((valid_rat_fob <= 1).sum())
    p6 = n6_pass / n6_total if n6_total > 0 else 0

    # Stage 7: Monkey ALT
    valid_monkey_alt = compound_mean_biomarker(
        hepatic_df[hepatic_df["species"] == "monkey"], "ALT",
    )
    n7_total = len(valid_monkey_alt)
    n7_pass = int((valid_monkey_alt < PIPELINE_STAGES[6].threshold_value).sum())
    p7 = n7_pass / n7_total if n7_total > 0 else 0

    proportions = [p1, p2, p3, p4, p5, p6, p7]
    sample_sizes = [
        (n1_pass, n1_total), (n2_pass, n2_total),
        (n3_pass, n3_total), (n4_pass, n4_total),
        (n5_pass, n5_total), (n6_pass, n6_total),
        (n7_pass, n7_total),
    ]
    distributions = [
        iv_by_compound.tolist(),
        ic50_series.tolist(),
        valid_mouse_alt.tolist(),
        valid_mouse_fob.tolist(),
        valid_rat_alt.tolist(),
        valid_rat_fob.tolist(),
        valid_monkey_alt.tolist(),
    ]

    return proportions, sample_sizes, distributions


def compute_conditional_rates(hepatic_df, neuro_df):
    """P(pass rat | pass mouse) on overlap compounds for correlated stage pairs.

    In the sequential pipeline, compounds reaching rat stages have already
    passed mouse stages. Where mouse–rat outcomes are positively correlated,
    the conditional pass rate exceeds the marginal, making the independence
    assumption conservative.
    """
    out = {}

    # --- Hepatic: mouse ALT (stage 2) → rat ALT (stage 4) ---
    m_alt = compound_mean_biomarker(
        hepatic_df[hepatic_df["species"] == "mouse"], "ALT",
    )
    r_alt = compound_mean_biomarker(
        hepatic_df[hepatic_df["species"] == "rat"], "ALT",
    )

    ids = m_alt.index.intersection(r_alt.index)
    if len(ids) > 0:
        m_ok = m_alt.loc[ids] < PIPELINE_STAGES[2].threshold_value
        r_ok = r_alt.loc[ids] < PIPELINE_STAGES[4].threshold_value
        pm, pr = float(m_ok.mean()), float(r_ok.mean())
        out["rat_ALT"] = {
            "conditional_rate": float(r_ok[m_ok].mean()) if m_ok.sum() > 0 else float("nan"),
            "marginal_rate_overlap": pr,
            "n_overlap": len(ids),
            "n_mouse_pass": int(m_ok.sum()),
            "correlation_factor": round(float((m_ok & r_ok).mean()) / (pm * pr), 3) if pm * pr > 0 else float("nan"),
            "stage_idx": 4,
        }

    # --- Neuro: mouse FOB (stage 3) → rat FOB (stage 5) ---
    m_fob = compound_mean_biomarker(
        neuro_df[
            (neuro_df["species"] == "Mouse")
            & (neuro_df["dosage_ug"] == 700)
            & (neuro_df["latency_time_hours"] == 3)
            & (neuro_df["administration_method"] == "ICV")
        ], "FOB_score",
    )
    r_fob = compound_mean_biomarker(
        neuro_df[
            (neuro_df["species"] == "Rat")
            & (neuro_df["dosage_ug"] == 3000)
            & (neuro_df["latency_time_hours"] == 3)
        ], "FOB_score",
    )

    ids = m_fob.index.intersection(r_fob.index)
    if len(ids) > 0:
        m_ok = m_fob.loc[ids] <= PIPELINE_STAGES[3].threshold_value
        r_ok = r_fob.loc[ids] <= PIPELINE_STAGES[5].threshold_value
        pm, pr = float(m_ok.mean()), float(r_ok.mean())
        out["rat_FOB"] = {
            "conditional_rate": float(r_ok[m_ok].mean()) if m_ok.sum() > 0 else float("nan"),
            "marginal_rate_overlap": pr,
            "n_overlap": len(ids),
            "n_mouse_pass": int(m_ok.sum()),
            "correlation_factor": round(float((m_ok & r_ok).mean()) / (pm * pr), 3) if pm * pr > 0 else float("nan"),
            "stage_idx": 5,
        }

    return out


def back_calculate(proportions):
    """Back-calculate initial ASOs needed for TARGET_CANDIDATES development candidates."""
    n_stages = len(PIPELINE_STAGES)

    cumulative_survival = []
    running = 1.0
    for i in range(n_stages):
        running *= proportions[i]
        cumulative_survival.append(running)

    # Back-calculate from target: how many ASOs must enter each stage?
    asos_at_stage = [0] * (n_stages + 1)
    asos_at_stage[-1] = TARGET_CANDIDATES
    for i in range(n_stages - 1, -1, -1):
        asos_at_stage[i] = math.ceil(asos_at_stage[i + 1] / proportions[i]) if proportions[i] > 0 else float("inf")
    n_initial = asos_at_stage[0]

    costs_per_stage = [asos_at_stage[i] * PIPELINE_STAGES[i].cost_per_aso for i in range(n_stages)]
    total_cost = sum(costs_per_stage)

    return {
        "n_initial": n_initial,
        "asos_at_stage": asos_at_stage,
        "proportions": proportions,
        "cumulative_survival": cumulative_survival,
        "costs_per_stage": costs_per_stage,
        "total_cost": total_cost,
        "target_candidates": TARGET_CANDIDATES,
    }


def compute_classifier_enrichment(preds_data: dict) -> dict:
    """Compute enrichment factors from Hagedorn classifier predictions.

    For each endpoint, the selected pool is the top-K safest compounds
    (lowest predicted P(high)), where ``K = base_rate`` for that stage — i.e.
    we select as many ASOs as would pass without enrichment, concentrated to
    the highest-predicted. This ties EF to realistic screening budgets: with
    perfect prediction, every selected ASO passes (max EF = 1/base_rate).

    Returns dict mapping task name → {enrichment_factor, base_rate, selected_pass_rate, n}.
    """
    enrichment = {}
    for task, data in preds_data.items():
        predictions = np.array(data["predictions"])
        labels = np.array(data["labels"])  # 1 = high toxicity, 0 = low toxicity

        # Base rate: fraction that are actually low toxicity (pass)
        base_pass_rate = float((labels == 0).mean())

        # Selected: top-K safest compounds by predicted P(high), K = base_rate.
        n = len(labels)
        if not (base_pass_rate > 0):
            continue
        k = max(1, int(np.floor(base_pass_rate * n)))
        selected_idx = np.argsort(predictions, kind="mergesort")[:k]
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[selected_idx] = True
        if selected_mask.sum() == 0:
            continue

        # Among selected, fraction that actually pass (are truly low toxicity)
        selected_pass_rate = float((labels[selected_mask] == 0).mean())

        ef = selected_pass_rate / base_pass_rate

        enrichment[task] = {
            "enrichment_factor": round(ef, 3),
            "base_rate": round(base_pass_rate, 3),
            "selected_pass_rate": round(selected_pass_rate, 3),
            "n": int(len(labels)),
            "n_selected": int(selected_mask.sum()),
            "selection_policy": "top_k_safest",
            "selection_fraction": round(base_pass_rate, 4),
            "k_mode": "base_rate",
        }

    return enrichment


def back_calculate_enriched(proportions, enrichment, stage_map):
    """Back-calculate pipeline with computational pre-screening.

    The classifier screens candidates for free, selecting those predicted
    to pass each covered stage. This increases the effective pass rate
    at enriched stages (= base_rate * enrichment_factor).
    """
    n_stages = len(PIPELINE_STAGES)

    enriched_stages = {}
    for task_name, stage_idx in stage_map.items():
        if task_name in enrichment:
            enriched_stages[stage_idx] = enrichment[task_name]

    eff_proportions = []
    for i in range(n_stages):
        if i in enriched_stages:
            ef = enriched_stages[i]["enrichment_factor"]
            eff_proportions.append(min(proportions[i] * ef, 1.0))
        else:
            eff_proportions.append(proportions[i])

    cumulative_survival = []
    running = 1.0
    for i in range(n_stages):
        running *= eff_proportions[i]
        cumulative_survival.append(running)

    # Back-calculate from target: how many ASOs must enter each stage?
    asos_at_stage = [0] * (n_stages + 1)
    asos_at_stage[-1] = TARGET_CANDIDATES
    for i in range(n_stages - 1, -1, -1):
        if eff_proportions[i] > 0 and math.isfinite(asos_at_stage[i + 1]):
            asos_at_stage[i] = math.ceil(asos_at_stage[i + 1] / eff_proportions[i])
        else:
            asos_at_stage[i] = float("inf")
    n_initial = asos_at_stage[0]

    costs_per_stage = [
        asos_at_stage[i] * PIPELINE_STAGES[i].cost_per_aso
        if math.isfinite(asos_at_stage[i]) else float("inf")
        for i in range(n_stages)
    ]
    total_cost = float("inf") if any(not math.isfinite(c) for c in costs_per_stage) else sum(costs_per_stage)

    return {
        "n_initial": n_initial,
        "asos_at_stage": asos_at_stage,
        "proportions": eff_proportions,
        "cumulative_survival": cumulative_survival,
        "costs_per_stage": costs_per_stage,
        "total_cost": total_cost,
        "target_candidates": TARGET_CANDIDATES,
        "enriched_stages": {
            str(i): enriched_stages[i] for i in enriched_stages
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    in_vitro_df = pd.read_parquet(DATA_DIR / "in_vitro_inhibition_processed.parquet")
    dose_response_df = pd.read_parquet(DATA_DIR / "dose_response_processed.parquet")
    hepatic_df = pd.read_parquet(DATA_DIR / "hepatictoxicity_processed.parquet")
    neuro_df = pd.read_parquet(DATA_DIR / "neurotoxicity_processed.parquet")

    print(f"Loaded: {len(in_vitro_df):,} in_vitro, {len(dose_response_df):,} dose_response, "
          f"{len(hepatic_df):,} hepatic, {len(neuro_df):,} neuro")

    # Calculate proportions
    proportions, sample_sizes, distributions = calculate_proportions(
        in_vitro_df, dose_response_df, hepatic_df, neuro_df,
    )
    for i in range(7):
        n_pass, n_total = sample_sizes[i]
        print(f"  Stage {i+1} ({PIPELINE_STAGES[i].name}): {n_pass:,}/{n_total:,} = {proportions[i]:.1%}")

    # Compute conditional rates as metadata (not applied to proportions).
    # Rat-tested compounds in the dataset already reflect pipeline survivorship
    # (93% hepatic / 82% neuro were also tested in mouse), so the observed
    # marginal rates approximate the conditional population.
    cond_rates = compute_conditional_rates(hepatic_df, neuro_df)
    for key, data in cond_rates.items():
        idx = data["stage_idx"]
        print(f"  {PIPELINE_STAGES[idx].name}: marginal={proportions[idx]:.1%}, "
              f"conditional={data['conditional_rate']:.1%} (CF={data['correlation_factor']}, n={data['n_overlap']})")

    # Baseline back-calculation (using marginal rates)
    baseline = back_calculate(proportions)
    print(f"Baseline: {baseline['n_initial']:,} initial ASOs, ${baseline['total_cost']/1e6:.1f}M total")

    # Save results (OligoAI-tox and combined scenarios are now canonical
    # in ef_table.json, built by analyses.logic.ef_table)
    results = {
        "proportions": proportions,
        "conditional_rates": cond_rates,
        "sample_sizes": sample_sizes,
        "distributions": distributions,
        "baseline": baseline,
        "stages": [
            {"name": s.name, "short_name": s.short_name, "threshold": s.threshold,
             "cost_per_aso": s.cost_per_aso, "color": s.color,
             "threshold_value": s.threshold_value, "threshold_op": s.threshold_op,
             "transform": s.transform, "xlabel": s.xlabel, "bins": s.bins,
             "xlim_lo": s.xlim_lo}
            for s in PIPELINE_STAGES
        ],
    }

    out_path = RESULTS_DIR / "pipeline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
