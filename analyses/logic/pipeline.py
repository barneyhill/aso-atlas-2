"""
ASO Development Pipeline Attrition Analysis.

Calculate success proportions at each stage, back-calculate initial ASOs
needed for target candidates, and save results to data/results/.

Pipeline Stages (Sequential):
1. Inhibition >80%
2. IC50 <500nM (electroporation)
3. Mouse ALT <100 IU/L
4. Mouse Neuro FOB <=1
5. Rat ALT <100 IU/L
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
from scipy.optimize import curve_fit

_root = Path(__file__).resolve().parents[2]
DATA_DIR = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Pipeline stage definitions
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    name: str
    short_name: str
    threshold: str
    cost_per_aso: float
    color: str


PIPELINE_STAGES = [
    PipelineStage("In vitro efficacy", "In vitro\nefficacy", "Inhibition >80%", 500, "#4878A8"),
    PipelineStage("In vitro potency", "In vitro\npotency", "IC50 <500nM", 2000, "#6A9BC3"),
    PipelineStage("Mouse liver toxicity", "Mouse liver\ntoxicity", "ALT <2×ULN", 15000, "#D4A574"),
    PipelineStage("Mouse neuro tolerability", "Mouse neuro\ntolerability", "bFOB <=1", 20000, "#E8B88A"),
    PipelineStage("Rat liver toxicity", "Rat liver\ntoxicity", "ALT <2×ULN", 25000, "#7BAA97"),
    PipelineStage("Rat neuro tolerability", "Rat neuro\ntolerability", "mFOB <=1", 30000, "#94C4A7"),
    PipelineStage("Monkey liver toxicity", "Monkey liver\ntoxicity", "ALT <2×ULN", 100000, "#9B8AA6"),
]

TARGET_CANDIDATES = 1
TOP_K_SELECTION_FRACTION = 0.25

# Maps prediction keys → pipeline stage indices (OligoAI-tox RF)
STAGE_MAP = {"ALT": 2, "FOB": 3, "rat_ALT": 4, "rat_FOB": 5}

# OligoAI (Hill et al. 2025, bioRxiv 10.1101/2025.10.29.685292)
# Top-10% enrichment factor for in vitro efficacy across 299 held-out screens
OLIGOAI_ENRICHMENT = {
    "inhibition": {
        "enrichment_factor": 3.14,
        "source": "Hill et al. 2025, Table 1",
    },
}
OLIGOAI_STAGE_MAP = {"inhibition": 0}


# ---------------------------------------------------------------------------
# IC50 fitting
# ---------------------------------------------------------------------------

def hill_equation(dose, bottom, top, ic50, hill):
    return bottom + (top - bottom) / (1 + (ic50 / dose) ** hill)


def fit_ic50_for_compound(doses, responses):
    mask = ~(np.isnan(doses) | np.isnan(responses))
    doses = np.array(doses)[mask]
    responses = np.array(responses)[mask]

    if len(doses) < 4:
        return np.nan

    pos_mask = doses > 0
    doses = doses[pos_mask]
    responses = responses[pos_mask]

    if len(doses) < 4:
        return np.nan

    try:
        min_dose, max_dose = np.min(doses), np.max(doses)
        bounds = ([0, 0, min_dose / 100, 0.1], [100, 100, max_dose * 100, 10])
        popt, _ = curve_fit(
            hill_equation, doses, responses,
            p0=[np.min(responses), np.max(responses), np.median(doses), 1.0],
            bounds=bounds, maxfev=5000,
        )
        ic50 = popt[2]
        predicted = hill_equation(doses, *popt)
        ss_res = np.sum((responses - predicted) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if r_squared > 0.5 and min_dose / 10 <= ic50 <= max_dose * 10:
            return ic50
        return np.nan
    except (RuntimeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_mean(row, col):
    val = row[col]
    if isinstance(val, (list, np.ndarray)):
        valid = [v for v in val if v is not None and not np.isnan(v)]
        return np.mean(valid) if valid else np.nan
    if val is not None and not np.isnan(val):
        return val
    return np.nan


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def calculate_proportions(in_vitro_df, dose_response_df, hepatic_df, neuro_df):
    """Calculate pass rates and distributions at each pipeline stage."""
    # Stage 1: Inhibition >80%
    iv_by_compound = in_vitro_df.groupby("Compound ID")["Inhibition_pct"].max()
    n1_total = len(iv_by_compound)
    n1_pass = int((iv_by_compound > 80).sum())
    p1 = n1_pass / n1_total if n1_total > 0 else 0

    # Stage 2: IC50 <500nM (electroporation)
    dr_elec = dose_response_df[dose_response_df["transfection_method"] == "Electroporation"]
    ic50_results = []
    for compound_id, group in dr_elec.groupby("Compound ID"):
        ic50 = fit_ic50_for_compound(group["dosage_nm"].values, group["Inhibition_pct"].values)
        if not np.isnan(ic50):
            ic50_results.append({"Compound ID": compound_id, "ic50_nm": ic50})
    ic50_df = pd.DataFrame(ic50_results)
    n2_total = len(ic50_df)
    n2_pass = int((ic50_df["ic50_nm"] < 500).sum()) if len(ic50_df) > 0 else 0
    p2 = n2_pass / n2_total if n2_total > 0 else 0

    # Stage 3: Mouse ALT < 2×ULN (150 IU/L)
    # ULN = 75 IU/L: mean of male (94) + female (56) 97.5th percentiles
    # for C57BL/6J mice (Otto et al. 2016, JAALAS 55(4):375-386)
    MOUSE_ALT_ULN = 75
    mouse_hep = hepatic_df[hepatic_df["species"] == "mouse"].copy()
    mouse_hep["ALT_mean"] = mouse_hep.apply(_extract_mean, axis=1, col="ALT")
    valid_mouse_alt = mouse_hep.groupby("Compound ID")["ALT_mean"].mean().dropna()
    n3_total = len(valid_mouse_alt)
    n3_pass = int((valid_mouse_alt < 2 * MOUSE_ALT_ULN).sum())
    p3 = n3_pass / n3_total if n3_total > 0 else 0

    # Stage 4: Mouse FOB <=1
    mouse_neuro = neuro_df[
        (neuro_df["species"] == "Mouse")
        & (neuro_df["dosage_ug"] == 700)
        & (neuro_df["latency_time_hours"] == 3)
    ].copy()
    mouse_neuro["FOB_mean"] = mouse_neuro.apply(_extract_mean, axis=1, col="FOB_score")
    valid_mouse_fob = mouse_neuro.groupby("Compound ID")["FOB_mean"].mean().dropna()
    n4_total = len(valid_mouse_fob)
    n4_pass = int((valid_mouse_fob <= 1).sum())
    p4 = n4_pass / n4_total if n4_total > 0 else 0

    # Stage 5: Rat ALT < 2×ULN (78 IU/L)
    # ULN = 39 IU/L: mean of male (47) + female (30) 97.5th percentiles
    # for Sprague-Dawley rats (He et al. 2017, PLoS ONE 12(12):e0189837)
    RAT_ALT_ULN = 39
    rat_hep = hepatic_df[hepatic_df["species"] == "rat"].copy()
    rat_hep["ALT_mean"] = rat_hep.apply(_extract_mean, axis=1, col="ALT")
    valid_rat_alt = rat_hep.groupby("Compound ID")["ALT_mean"].mean().dropna()
    n5_total = len(valid_rat_alt)
    n5_pass = int((valid_rat_alt < 2 * RAT_ALT_ULN).sum())
    p5 = n5_pass / n5_total if n5_total > 0 else 0

    # Stage 6: Rat FOB <=1
    rat_neuro = neuro_df[
        (neuro_df["species"] == "Rat")
        & (neuro_df["dosage_ug"] == 3000)
        & (neuro_df["latency_time_hours"] == 3)
    ].copy()
    rat_neuro["FOB_mean"] = rat_neuro.apply(_extract_mean, axis=1, col="FOB_score")
    valid_rat_fob = rat_neuro.groupby("Compound ID")["FOB_mean"].mean().dropna()
    n6_total = len(valid_rat_fob)
    n6_pass = int((valid_rat_fob <= 1).sum())
    p6 = n6_pass / n6_total if n6_total > 0 else 0

    # Stage 7: Monkey ALT < 2×ULN (206 IU/L)
    # ULN = 103 IU/L: mean of male (92) + female (113) 97.5th percentiles
    # for adult cynomolgus macaques (Bakker et al. 2023, Animals 13(3):445)
    MONKEY_ALT_ULN = 103
    monkey_hep = hepatic_df[hepatic_df["species"] == "monkey"].copy()
    monkey_hep["ALT_mean"] = monkey_hep.apply(_extract_mean, axis=1, col="ALT")
    valid_monkey_alt = monkey_hep.groupby("Compound ID")["ALT_mean"].mean().dropna()
    n7_total = len(valid_monkey_alt)
    n7_pass = int((valid_monkey_alt < 2 * MONKEY_ALT_ULN).sum())
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
        ic50_df["ic50_nm"].tolist() if len(ic50_df) > 0 else [],
        valid_mouse_alt.tolist(),
        valid_mouse_fob.tolist(),
        valid_rat_alt.tolist(),
        valid_rat_fob.tolist(),
        valid_monkey_alt.tolist(),
    ]

    return proportions, sample_sizes, distributions


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
    (lowest predicted P(high)), where K = floor(25% of available compounds).
    Among those, the fraction that actually pass the pipeline threshold gives
    the enrichment factor.

    Returns dict mapping task name → {enrichment_factor, base_rate, selected_pass_rate, n}.
    """
    enrichment = {}
    for task, data in preds_data.items():
        predictions = np.array(data["predictions"])
        labels = np.array(data["labels"])  # 1 = high toxicity, 0 = low toxicity

        # Base rate: fraction that are actually low toxicity (pass)
        base_pass_rate = float((labels == 0).mean())

        # Selected: top-K safest compounds by predicted P(high).
        n = len(labels)
        k = max(1, int(np.floor(TOP_K_SELECTION_FRACTION * n)))
        selected_idx = np.argsort(predictions, kind="mergesort")[:k]
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[selected_idx] = True
        if selected_mask.sum() == 0:
            continue

        # Among selected, fraction that actually pass (are truly low toxicity)
        selected_pass_rate = float((labels[selected_mask] == 0).mean())

        ef = selected_pass_rate / base_pass_rate if base_pass_rate > 0 else 1.0

        enrichment[task] = {
            "enrichment_factor": round(ef, 3),
            "base_rate": round(base_pass_rate, 3),
            "selected_pass_rate": round(selected_pass_rate, 3),
            "n": int(len(labels)),
            "n_selected": int(selected_mask.sum()),
            "selection_policy": "top_k_safest",
            "selection_fraction": TOP_K_SELECTION_FRACTION,
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
        asos_at_stage[i] = math.ceil(asos_at_stage[i + 1] / eff_proportions[i]) if eff_proportions[i] > 0 else float("inf")
    n_initial = asos_at_stage[0]

    costs_per_stage = [asos_at_stage[i] * PIPELINE_STAGES[i].cost_per_aso for i in range(n_stages)]
    total_cost = sum(costs_per_stage)

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

    # Baseline back-calculation
    baseline = back_calculate(proportions)
    print(f"Baseline: {baseline['n_initial']:,} initial ASOs, ${baseline['total_cost']/1e6:.1f}M total")

    # OligoAI-tox RF enrichment (hepatotox + neurotox)
    oligoai_tox = None
    oligoai_tox_enrichment = {}
    hepatotox_path = RESULTS_DIR / "hepatotox.json"
    neurotox_path = RESULTS_DIR / "neurotox.json"

    if hepatotox_path.exists() and neurotox_path.exists():
        with open(hepatotox_path) as f:
            hepatotox_data = json.load(f)
        with open(neurotox_path) as f:
            neurotox_data = json.load(f)
        oligoai_tox_enrichment.update(compute_classifier_enrichment(hepatotox_data["predictions"]))
        oligoai_tox_enrichment.update(compute_classifier_enrichment(neurotox_data["predictions"]))

        # Rat predictions (independent models trained on rat data)
        if "rat_predictions" in hepatotox_data:
            oligoai_tox_enrichment.update(compute_classifier_enrichment(hepatotox_data["rat_predictions"]))
        if "rat_predictions" in neurotox_data:
            oligoai_tox_enrichment.update(compute_classifier_enrichment(neurotox_data["rat_predictions"]))

        if oligoai_tox_enrichment:
            oligoai_tox = back_calculate_enriched(proportions, oligoai_tox_enrichment, STAGE_MAP)
            savings_pct = (1 - oligoai_tox["total_cost"] / baseline["total_cost"]) * 100
            print(f"OligoAI-tox: {oligoai_tox['n_initial']:,} initial ASOs, ${oligoai_tox['total_cost']/1e6:.1f}M total ({savings_pct:.1f}% reduction)")
            for task, e in oligoai_tox_enrichment.items():
                if task in STAGE_MAP:
                    print(f"  {task}: EF={e['enrichment_factor']:.2f}x (base={e['base_rate']:.2f}, selected={e['selected_pass_rate']:.2f})")
    else:
        print("OligoAI-tox results not found — run `just hagerdorn` first")

    # OligoAI enrichment (in vitro efficacy — Hill et al. 2025)
    oligoai = back_calculate_enriched(proportions, OLIGOAI_ENRICHMENT, OLIGOAI_STAGE_MAP)
    oligoai_savings = (1 - oligoai["total_cost"] / baseline["total_cost"]) * 100
    print(f"OligoAI: {oligoai['n_initial']:,} initial ASOs, ${oligoai['total_cost']/1e6:.1f}M total ({oligoai_savings:.1f}% reduction)")
    print(f"  inhibition: EF={OLIGOAI_ENRICHMENT['inhibition']['enrichment_factor']:.2f}x (hardcoded from Hill et al. 2025)")

    # Combined: OligoAI (inhibition) + OligoAI-tox (ALT, FOB)
    combined = None
    if oligoai_tox_enrichment:
        combined_enrichment = dict(OLIGOAI_ENRICHMENT)
        for task in STAGE_MAP:
            if task in oligoai_tox_enrichment:
                combined_enrichment[task] = oligoai_tox_enrichment[task]
        combined_stage_map = {**OLIGOAI_STAGE_MAP, **STAGE_MAP}
        combined = back_calculate_enriched(proportions, combined_enrichment, combined_stage_map)
        combined_savings = (1 - combined["total_cost"] / baseline["total_cost"]) * 100
        print(f"Combined: {combined['n_initial']:,} initial ASOs, ${combined['total_cost']/1e6:.1f}M total ({combined_savings:.1f}% reduction)")

    # Save results
    results = {
        "proportions": proportions,
        "sample_sizes": sample_sizes,
        "distributions": distributions,
        "baseline": baseline,
        "oligoai_tox": oligoai_tox,
        "oligoai": oligoai,
        "combined": combined,
        "stages": [
            {"name": s.name, "short_name": s.short_name, "threshold": s.threshold,
             "cost_per_aso": s.cost_per_aso, "color": s.color}
            for s in PIPELINE_STAGES
        ],
    }

    out_path = RESULTS_DIR / "pipeline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
