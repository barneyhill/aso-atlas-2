"""Focused uncertainty analysis requested by the NeurIPS AC.

This deliberately keeps statistical and structural uncertainty separate.  It reports:

1. the corrected expected-count point estimate and a cross-fold-combination
   sensitivity analysis;
2. ordinary bootstrap uncertainty for the observed baseline pass/fail records,
   propagated jointly through the expected-count recursion;
3. the existing +/-50% stage-cost sensitivity, recomputed for the corrected model; and
4. model-choice sensitivity after applying expected counts and a model-specific linked-stage
   conditional correction to every OligoGym architecture.

It also combines (1) and (2): each draw samples one of the 3,125 valid empirical
cross-endpoint fold combinations and resamples every stage's observed baseline pass/fail
records, then propagates both through the corrected expected-count calculation with fixed
costs.  This is the narrow, requested all-stage uncertainty interval.  Costs and model
choice remain separate sensitivities rather than being assigned sampling distributions.

The three conditional-composition uplifts are held fixed.  Resampling those sparse linked
subsets is a separate structural question, not smuggled into the sampling interval here.

The endpoint benchmarks used independently generated GroupKFold partitions, so fold labels
are not aligned across endpoints.  Pairing fold ``i`` from every endpoint would therefore be
arbitrary.  Model variability is instead summarised over the exact Cartesian product of five
fold blocks: the paired OligoAI efficacy/potency fold and one independently varying fold for
each of the four in-vivo XGBoost endpoints (5^5 = 3,125 combinations).  This is a sensitivity
distribution, not a confidence interval.

Run with::

    uv run python -m analyses.logic.models.ac_uncertainty
"""
from __future__ import annotations

import json
import itertools
import math
from pathlib import Path

import numpy as np

from analyses.logic.ef_table import oligogym_strategy_data
from analyses.logic.models.selection_fraction_sensitivity import IN_VIVO, k_sweep
from analyses.logic.pipeline import OLIGOAI_ENRICHMENT, PIPELINE_STAGES


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "data/results"
N_DRAWS = 100_000
SEED = 4274


def _load_model_inputs():
    combined = next(
        row for row in json.loads((RESULTS / "ef_table.json").read_text())
        if row.get("group") == "combined"
    )
    benchmark = json.loads((RESULTS / "oligogym_benchmark.json").read_text())
    return {
        "canon_precs": {
            int(stage): value
            for stage, value in combined["prec_by_stage"].items()
            if value is not None
        },
        "bench": benchmark,
        "xgb": oligogym_strategy_data(benchmark)["XGBoost"],
    }


def _cost(base_rates, precisions, costs, *, use_ceil=False, target=1):
    """Pipeline cost with no EF floor."""
    effective = np.asarray(base_rates, dtype=float).copy()
    for stage, precision in precisions.items():
        if precision is not None and np.isfinite(precision):
            effective[stage] = np.clip(precision, 1e-12, 1.0)
    required = float(target)
    total = 0.0
    for stage in range(len(effective) - 1, -1, -1):
        quotient = required / effective[stage]
        required = math.ceil(quotient) if use_ceil else quotient
        total += required * costs[stage]
    return float(total)


def _savings(base_rates, precisions, costs, *, use_ceil=False, target=1):
    baseline = _cost(base_rates, {}, costs, use_ceil=use_ceil, target=target)
    model = _cost(base_rates, precisions, costs, use_ceil=use_ceil, target=target)
    return 100.0 * (1.0 - model / baseline)


def _conditional_factors(s1):
    """Return fixed baseline/model uplifts keyed by downstream stage."""
    baseline, model = {}, {}
    for pair in s1["pairs"].values():
        downstream = int(pair["idx"][1])
        if pair.get("baseline_uplift") is not None:
            baseline[downstream] = float(pair["baseline_uplift"])
        if pair.get("model_uplift") is not None:
            model[downstream] = float(pair["model_uplift"])
    return baseline, model


def _apply_fixed_correction(base_rates, precisions, baseline_factors, model_factors):
    base = np.asarray(base_rates, dtype=float).copy()
    prec = dict(precisions)
    for stage, factor in baseline_factors.items():
        base[stage] = min(base[stage] * factor, 1.0)
    for stage, factor in model_factors.items():
        if stage in prec and prec[stage] is not None:
            prec[stage] = min(float(prec[stage]) * factor, 1.0)
    return base, prec


def _summarise(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(a.mean()), 1),
        "sd": round(float(a.std(ddof=1)), 1),
        "min": round(float(a.min()), 1),
        "max": round(float(a.max()), 1),
        "p2.5": round(float(np.percentile(a, 2.5)), 1),
        "median": round(float(np.median(a)), 1),
        "p97.5": round(float(np.percentile(a, 97.5)), 1),
    }


def _expected_cost_draws(rates, costs):
    """Vectorised expected-count costs for arrays shaped (draw, stage)."""
    required = np.ones(rates.shape[0], dtype=float)
    total = np.zeros(rates.shape[0], dtype=float)
    for stage in range(rates.shape[1] - 1, -1, -1):
        required /= rates[:, stage]
        total += required * costs[:, stage]
    return total


def compute():
    pipeline = json.loads((RESULTS / "pipeline_results.json").read_text())
    ef_rows = json.loads((RESULTS / "ef_table.json").read_text())
    s1 = json.loads((RESULTS / "s1_conditional_pipeline.json").read_text())
    data = _load_model_inputs()

    base_marginal = np.asarray(pipeline["proportions"], dtype=float)
    sample_sizes = [tuple(map(int, x)) for x in pipeline["sample_sizes"]]
    costs = np.asarray([s.cost_per_aso for s in PIPELINE_STAGES], dtype=float)
    canonical_marginal = data["canon_precs"]
    baseline_factors, model_factors = _conditional_factors(s1)
    base_corrected, canonical_corrected = _apply_fixed_correction(
        base_marginal, canonical_marginal, baseline_factors, model_factors,
    )

    canonical = {
        "marginal_ceiling_savings_pct": round(
            _savings(base_marginal, canonical_marginal, costs, use_ceil=True), 1,
        ),
        "marginal_expected_savings_pct": round(
            _savings(base_marginal, canonical_marginal, costs), 1,
        ),
        "corrected_ceiling_savings_pct": round(
            _savings(base_corrected, canonical_corrected, costs, use_ceil=True), 1,
        ),
        "corrected_expected_savings_pct": round(
            _savings(base_corrected, canonical_corrected, costs), 1,
        ),
        "baseline_rates_marginal": [round(float(x), 6) for x in base_marginal],
        "baseline_rates_corrected": [round(float(x), 6) for x in base_corrected],
        "model_precisions_corrected": {
            str(k): round(float(v), 6) for k, v in sorted(canonical_corrected.items())
        },
    }

    # Derived comparisons must use the same corrected pipeline as the headline rather than
    # the superseded marginal 77.5% calculation.
    corrected_full = canonical["corrected_expected_savings_pct"]
    stage_contributions = []
    for stage in sorted(canonical_corrected):
        without = {k: v for k, v in canonical_corrected.items() if k != stage}
        savings_without = _savings(base_corrected, without, costs)
        stage_contributions.append({
            "stage": stage,
            "stage_name": PIPELINE_STAGES[stage].name,
            "savings_without_stage_pct": round(savings_without, 1),
            "contribution_pt": round(corrected_full - savings_without, 1),
        })
    corrected_ordering = {
        "in_vitro_only_pct": round(
            _savings(
                base_corrected,
                {k: v for k, v in canonical_corrected.items() if k < 2},
                costs,
            ),
            1,
        ),
        "in_vivo_only_pct": round(
            _savings(
                base_corrected,
                {k: v for k, v in canonical_corrected.items() if 2 <= k <= 5},
                costs,
            ),
            1,
        ),
        "combined_pct": corrected_full,
    }
    ceiling_vs_expected = []
    for target in (1, 2, 5, 10, 20):
        ceiling = _savings(
            base_corrected, canonical_corrected, costs,
            use_ceil=True, target=target,
        )
        expected = _savings(
            base_corrected, canonical_corrected, costs,
            use_ceil=False, target=target,
        )
        ceiling_vs_expected.append({
            "target_candidates": target,
            "ceiling_savings_pct": round(ceiling, 1),
            "expected_savings_pct": round(expected, 1),
            "gap_pt": round(expected - ceiling, 1),
        })

    # Cross-fold model sensitivity. OligoAI's efficacy and potency results come from the
    # same five trained models and therefore remain paired. Each in-vivo endpoint used an
    # independently generated GroupKFold partition, so its fold index varies independently.
    # Enumerating the Cartesian product avoids assigning meaning to coincident fold labels.
    eff_folds = OLIGOAI_ENRICHMENT["inhibition"].get("fold_precs")
    if eff_folds is None:
        eff_folds = [base_marginal[0] * ef for ef in OLIGOAI_ENRICHMENT["inhibition"]["fold_efs"]]
    pot_folds = OLIGOAI_ENRICHMENT.get("potency", {}).get("fold_precs")
    if pot_folds is None and OLIGOAI_ENRICHMENT.get("potency", {}).get("fold_efs") is not None:
        pot_folds = [
            base_marginal[1] * ef
            for ef in OLIGOAI_ENRICHMENT["potency"]["fold_efs"]
        ]
    n_oligoai_folds = len(eff_folds)
    if pot_folds is not None and len(pot_folds) != n_oligoai_folds:
        raise ValueError("OligoAI efficacy and potency fold counts differ")

    in_vivo_blocks = list(IN_VIVO.items())
    fold_ranges = [range(n_oligoai_folds)] + [
        range(len(data["xgb"][ds]["fold_precs"])) for _stage, ds in in_vivo_blocks
    ]
    combination_savings = []
    combination_precisions = []
    for fold_choice in itertools.product(*fold_ranges):
        oligoai_fold, *in_vivo_folds = fold_choice
        marginal = {0: eff_folds[oligoai_fold]}
        if pot_folds is not None:
            marginal[1] = pot_folds[oligoai_fold]
        for (stage, ds), fold in zip(in_vivo_blocks, in_vivo_folds):
            marginal[stage] = data["xgb"][ds]["fold_precs"][fold]
        _, corrected = _apply_fixed_correction(
            base_marginal, marginal, baseline_factors, model_factors,
        )
        combination_savings.append(_savings(base_corrected, corrected, costs))
        combination_precisions.append([
            corrected.get(stage, np.nan) for stage in range(len(PIPELINE_STAGES))
        ])

    # Endpoint-wise fold precision summaries. These retain the valid fold variability for
    # each endpoint without pretending that fold labels align between endpoints.
    stage_fold_precisions = {stage: [] for stage in range(6)}
    for fold in range(n_oligoai_folds):
        marginal = {0: eff_folds[fold]}
        if pot_folds is not None:
            marginal[1] = pot_folds[fold]
        _, corrected = _apply_fixed_correction(
            base_marginal, marginal, baseline_factors, model_factors,
        )
        for stage in (0, 1):
            if stage in corrected:
                stage_fold_precisions[stage].append(corrected[stage])
    for stage, ds in in_vivo_blocks:
        for precision in data["xgb"][ds]["fold_precs"]:
            _, corrected = _apply_fixed_correction(
                base_marginal, {stage: precision}, baseline_factors, model_factors,
            )
            stage_fold_precisions[stage].append(corrected[stage])

    per_stage_precision = {}
    for stage, values in stage_fold_precisions.items():
        if values:
            stats = _summarise(np.asarray(values) * 100.0)
            per_stage_precision[str(stage)] = {
                "stage_name": PIPELINE_STAGES[stage].name,
                "per_fold_pct": [round(100.0 * float(x), 1) for x in values],
                **stats,
            }

    # K sensitivity with the pooled conditional factors held fixed. This varies the
    # empirically estimated per-stage precision curves without pretending the sparse linked
    # subsets can support a fresh conditional-uplift estimate at every K.
    corrected_k_sensitivity = []
    for row in k_sweep(data["bench"]):
        in_vitro = {int(k): v for k, v in row["in_vitro_precisions"].items()}
        in_vivo = {int(k): v for k, v in row["in_vivo_precisions"].items()}
        both = {**in_vitro, **in_vivo}

        def corrected_savings(precs):
            _, corrected = _apply_fixed_correction(
                base_marginal, precs, baseline_factors, model_factors,
            )
            return round(_savings(base_corrected, corrected, costs), 1)

        corrected_k_sensitivity.append({
            "K": row["K"],
            "in_vitro_only_pct": corrected_savings(in_vitro),
            "in_vivo_only_pct": corrected_savings(in_vivo),
            "combined_pct": corrected_savings(both),
            "conditional_uplifts": "fixed at pooled K=0.20 point estimates",
        })

    # Ordinary nonparametric bootstrap of each stage's observed binary pass/fail
    # records. For binary values, the resampled number of passes has exactly this
    # binomial count distribution; this is a computational shortcut for sampling the
    # observed zeros and ones with replacement, not a fitted parametric model or prior.
    # The fixed conditional baseline uplifts are then applied draw-by-draw.
    rng = np.random.default_rng(SEED)
    base_draws = np.column_stack([
        rng.binomial(n_total, n_pass / n_total, size=N_DRAWS) / n_total
        for n_pass, n_total in sample_sizes
    ])
    for stage, factor in baseline_factors.items():
        base_draws[:, stage] = np.minimum(base_draws[:, stage] * factor, 1.0)

    model_draws = base_draws.copy()
    for stage, precision in canonical_corrected.items():
        model_draws[:, stage] = precision
    fixed_cost_draws = np.broadcast_to(costs, (N_DRAWS, len(costs)))
    baseline_costs = _expected_cost_draws(base_draws, fixed_cost_draws)
    model_costs = _expected_cost_draws(model_draws, fixed_cost_draws)
    baseline_savings = 100.0 * (1.0 - model_costs / baseline_costs)

    # Narrow joint interval requested by AC #3: empirical held-out fold performance
    # at every modelled stage plus ordinary bootstrap uncertainty in every baseline
    # pass rate, propagated together with fixed costs. The conditional corrections
    # remain the pooled linked-data point estimates from AC #4; their small-sample
    # structural sensitivity is reported separately rather than mixed into this CI.
    combination_precisions = np.asarray(combination_precisions, dtype=float)
    sampled_combinations = combination_precisions[
        rng.integers(0, len(combination_precisions), size=N_DRAWS)
    ]
    joint_model_draws = base_draws.copy()
    for stage in range(sampled_combinations.shape[1]):
        present = np.isfinite(sampled_combinations[:, stage])
        joint_model_draws[present, stage] = sampled_combinations[present, stage]
    joint_model_costs = _expected_cost_draws(joint_model_draws, fixed_cost_draws)
    joint_savings = 100.0 * (1.0 - joint_model_costs / baseline_costs)

    rate_intervals = []
    for stage, (n_pass, n_total) in enumerate(sample_sizes):
        rate_intervals.append({
            "stage": stage,
            "stage_name": PIPELINE_STAGES[stage].name,
            "n_pass": n_pass,
            "n_total": n_total,
            "corrected_point": round(float(base_corrected[stage]), 4),
            "p2.5": round(float(np.percentile(base_draws[:, stage], 2.5)), 4),
            "p97.5": round(float(np.percentile(base_draws[:, stage], 97.5)), 4),
        })

    # Stage-cost sensitivity, retaining the paper's independent U[0.5, 1.5]
    # perturbation but using expected counts and the corrected pipeline.
    cost_multipliers = rng.uniform(0.5, 1.5, size=(N_DRAWS, len(costs)))
    cost_draws = costs * cost_multipliers
    corrected_base_matrix = np.broadcast_to(base_corrected, (N_DRAWS, len(costs)))
    corrected_model_matrix = corrected_base_matrix.copy()
    for stage, precision in canonical_corrected.items():
        corrected_model_matrix[:, stage] = precision
    cost_savings = 100.0 * (
        1.0
        - _expected_cost_draws(corrected_model_matrix, cost_draws)
        / _expected_cost_draws(corrected_base_matrix, cost_draws)
    )

    # Model choice is a structural sensitivity, not a sampling distribution.  Apply the same
    # expected-count and two-arm conditional calculation as the canonical result, but recover
    # the hepatic and neuro model-arm uplifts separately for every OligoGym architecture.
    # The efficacy/potency model remains OligoAI in every row, so its linked-stage uplift is
    # shared.  No retraining occurs: _benchmark_model_uplift uses stored out-of-fold
    # predictions and reattaches compound identifiers by replaying deterministic GroupKFold.
    from analyses.logic.models.s1_conditional_pipeline import _benchmark_model_uplift

    oligoai = next(r for r in ef_rows if r.get("group") == "external")
    oligoai_precs = {int(k): v for k, v in oligoai["prec_by_stage"].items()
                     if v is not None and int(k) < 2}
    efficacy_pair = s1["pairs"]["efficacy_potency"]
    efficacy_stage = int(efficacy_pair["idx"][1])
    shared_model_factors = {efficacy_stage: float(efficacy_pair["model_uplift"])}
    toxicity_pairs = {
        "hepatic": ("mouse_hepatic", "rat_hepatic"),
        "neuro": ("mouse_neuro", "rat_neuro"),
    }
    model_choice = []
    for row in ef_rows:
        if row.get("group") != "oligogym":
            continue
        prec = dict(oligoai_precs)
        prec.update({int(k): v for k, v in row["prec_by_stage"].items()
                     if v is not None and 2 <= int(k) <= 5})

        choice_model_factors = dict(shared_model_factors)
        linked_stage_details = {}
        for pair_name, (upstream_dataset, downstream_dataset) in toxicity_pairs.items():
            pair = s1["pairs"][pair_name]
            upstream_stage, downstream_stage = map(int, pair["idx"])
            uplift = _benchmark_model_uplift(
                upstream_dataset,
                downstream_dataset,
                upstream_stage,
                downstream_stage,
                model=row["name"],
            )
            if not uplift or uplift.get("model_uplift") is None:
                raise ValueError(
                    f"missing {pair_name} linked-stage uplift for {row['name']}"
                )
            choice_model_factors[downstream_stage] = float(uplift["model_uplift"])
            linked_stage_details[pair_name] = {
                key: uplift.get(key)
                for key in (
                    "model_precision_marginal",
                    "model_precision_gate_matched",
                    "model_uplift",
                    "n_gate_matched",
                )
            }

        choice_base, choice_prec = _apply_fixed_correction(
            base_marginal,
            prec,
            baseline_factors,
            choice_model_factors,
        )
        model_choice.append({
            "model": row["name"],
            "marginal_ceiling_savings_pct": round(
                _savings(base_marginal, prec, costs, use_ceil=True), 1,
            ),
            "marginal_expected_savings_pct": round(
                _savings(base_marginal, prec, costs), 1,
            ),
            "corrected_ceiling_savings_pct": round(
                _savings(choice_base, choice_prec, costs, use_ceil=True), 1,
            ),
            "corrected_expected_savings_pct": round(
                _savings(choice_base, choice_prec, costs), 1,
            ),
            "linked_stage_model_uplifts": linked_stage_details,
        })

    xgb_choice = next(row for row in model_choice if row["model"] == "XGBoost")
    if xgb_choice["corrected_expected_savings_pct"] != canonical["corrected_expected_savings_pct"]:
        raise ValueError(
            "model-choice XGBoost result does not reproduce canonical corrected savings: "
            f"{xgb_choice['corrected_expected_savings_pct']} vs "
            f"{canonical['corrected_expected_savings_pct']}"
        )

    return {
        "method": {
            "primary_recursion": "expected counts",
            "ef_floor": False,
            "conditional_uplifts": "fixed at linked-compound point estimates",
            "baseline_rate_distribution": (
                "ordinary nonparametric bootstrap of observed binary pass/fail "
                "records, independently by stage"
            ),
            "baseline_rate_draws": N_DRAWS,
            "cost_distribution": "independent U[0.5, 1.5] multiplier by stage",
            "model_choice_sensitivity": (
                "expected-count recursion with shared conditional baseline uplifts, "
                "shared OligoAI efficacy/potency uplift, and model-specific hepatic "
                "and neuro uplifts from stored out-of-fold predictions"
            ),
            "seed": SEED,
        },
        "canonical": canonical,
        "corrected_ordering": corrected_ordering,
        "corrected_stage_contributions": stage_contributions,
        "corrected_ceiling_vs_expected": ceiling_vs_expected,
        "corrected_cross_fold_combinations": {
            "design": (
                "Exact Cartesian product of one paired OligoAI efficacy/potency fold "
                "and independently varying folds for mouse hepatic, mouse neuro, rat "
                "hepatic, and rat neuro; fixed pooled conditional uplifts"
            ),
            "interpretation": "cross-fold sensitivity distribution, not a confidence interval",
            "n_combinations": len(combination_savings),
            **_summarise(combination_savings),
        },
        "per_stage_corrected_precision": per_stage_precision,
        "corrected_k_sensitivity": corrected_k_sensitivity,
        "baseline_pass_rate_intervals": rate_intervals,
        "baseline_pass_rate_uncertainty_savings_pct": _summarise(baseline_savings),
        "joint_model_fold_and_baseline_bootstrap_savings_pct": {
            "interpretation": (
                "95% resampling interval for all-stage held-out model performance "
                "and baseline pass-rate sampling, with fixed linked-data conditional "
                "corrections and fixed stage costs"
            ),
            **_summarise(joint_savings),
        },
        "stage_cost_sensitivity_savings_pct": _summarise(cost_savings),
        "model_choice_sensitivity": model_choice,
    }


def main():
    result = compute()
    path = RESULTS / "ac_uncertainty.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {path}")
    print("canonical corrected:", result["canonical"]["corrected_expected_savings_pct"])
    print("corrected cross-fold combinations:",
          result["corrected_cross_fold_combinations"])
    print("baseline-rate uncertainty:", result["baseline_pass_rate_uncertainty_savings_pct"])
    print("joint model+baseline interval:",
          result["joint_model_fold_and_baseline_bootstrap_savings_pct"])
    print("cost sensitivity:", result["stage_cost_sensitivity_savings_pct"])
    print("model choice:", result["model_choice_sensitivity"])


if __name__ == "__main__":
    main()
