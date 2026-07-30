"""Conditional pass rates in BOTH arms of the cost comparison (rebuttal S1, AC #4).

``s1_ef_composition_bias`` conditions the model arm on upstream model selection and
finds the rat-neuro effective pass rate rising from 0.789 to 0.857. Taken alone that
makes multiplicative composition look conservative. But it conditions one arm only.
The no-model baseline is also a sequential pipeline: compounds only reach the rat
stage after passing the mouse gate, and when the two species' outcomes are correlated
that gate raises the baseline rat pass rate too. Crediting the model arm with a
conditional uplift while holding the baseline at its marginal rate manufactures
savings out of the correlation. That is the double-count UEDf W2 warns about, and
once both arms are conditioned it runs in the direction they predicted.

The four cells estimated here, on the compounds carrying both species' endpoints:

                        baseline arm                model arm
    marginal        P(rat pass)                 precision(top-K by score)
    conditional     P(rat pass | mouse pass)    precision(top-K | mouse-selected)

Stage populations are the pipeline's own (``calculate_proportions``), so the base
rates conditioned here are the ones the published recursion consumes. Both correlated
species pairs are covered: neurotoxicity (mouse bFOB -> rat mFOB, the strong pair) and
hepatotoxicity (mouse ALT -> rat ALT, the weak one).

    uv run python -m analyses.logic.models.s1_conditional_pipeline
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from analyses.logic.models.neurotox import hagedorn_score
from analyses.logic.pipeline import PIPELINE_STAGES, TARGET_CANDIDATES
from analyses.utils.compounds import compound_mean_biomarker
from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

K = 0.20
_root = Path(__file__).resolve().parents[3]
PROCESSED = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"

# (upstream stage index, downstream stage index)
NEURO_PAIR = (3, 5)       # mouse bFOB -> rat mFOB, the strong pair (rho = 0.648)
HEPATIC_PAIR = (2, 4)     # mouse ALT -> rat ALT (rho = 0.295)
EFFICACY_PAIR = (0, 1)    # inhibition -> IC50: not cross-species, but the same underlying
                          # property measured twice, so the baseline gate enriches here too
PAIR_DISPLAY_NAMES = {
    "efficacy_potency": "Efficacy → potency",
    "hepatic": "Mouse → rat hepatic",
    "neuro": "Mouse → rat neuro",
}


def _cost(base_rates, precs, use_ceil, target=TARGET_CANDIDATES):
    n = len(PIPELINE_STAGES)
    eff = list(base_rates)
    for i, p in precs.items():
        if p is not None and base_rates[i] > 0:
            eff[i] = min(max(p, base_rates[i]), 1.0)
    asos = [0.0] * (n + 1)
    asos[-1] = target
    for i in range(n - 1, -1, -1):
        if eff[i] <= 0:
            return float("inf")
        asos[i] = math.ceil(asos[i + 1] / eff[i]) if use_ceil else asos[i + 1] / eff[i]
    return sum(asos[i] * PIPELINE_STAGES[i].cost_per_aso for i in range(n))


def _passes(values: pd.Series, stage_idx: int) -> pd.Series:
    st = PIPELINE_STAGES[stage_idx]
    op, t = st.threshold_op, st.threshold_value
    if op == "<":
        return values < t
    if op == "<=":
        return values <= t
    if op == ">":
        return values > t
    if op == ">=":
        return values >= t
    raise ValueError(f"Unsupported threshold_op: {op!r}")


# ---------------------------------------------------------------------------
# Stage populations, matching pipeline.calculate_proportions exactly
# ---------------------------------------------------------------------------

def _neuro_populations(neuro_df):
    mouse = compound_mean_biomarker(
        neuro_df[(neuro_df["species"] == "Mouse") & (neuro_df["dosage_ug"] == 700)
                 & (neuro_df["latency_time_hours"] == 3)
                 & (neuro_df["administration_method"] == "ICV")], "FOB_score")
    rat = compound_mean_biomarker(
        neuro_df[(neuro_df["species"] == "Rat") & (neuro_df["dosage_ug"] == 3000)
                 & (neuro_df["latency_time_hours"] == 3)], "FOB_score")
    return mouse, rat


def _efficacy_populations():
    """Stage 0 and stage 1 populations, per pipeline.calculate_proportions."""
    from analyses.logic.pipeline import compound_max_inhibition, compound_ic50s
    iv = pd.read_parquet(PROCESSED / "in_vitro_inhibition_processed.parquet")
    dr = pd.read_parquet(PROCESSED / "dose_response_processed.parquet")
    return compound_max_inhibition(iv), compound_ic50s(dr)


def _efficacy_model_uplift(k: float = K) -> dict | None:
    """OligoAI's potency precision, marginal versus gate-matched.

    The generic helper cannot do this link: the model ranks by predicted inhibition at stage 0
    and by predicted IC50 at stage 1, so there is no single score column. We therefore read the
    held-out fold predictions directly. Marginal is the top-K of the whole potency pool; the
    gate-matched pool is the compounds that both passed the efficacy gate and fell in OligoAI's
    top-K at stage 0, which is what the model arm actually presents to the potency stage.
    """
    import numpy as np
    from analyses.logic.models.oligoai_efs import (
        PREDICTIONS_PATHS_5FOLD, _electroporation_table_ids, _fit_group)
    try:
        electro = _electroporation_table_ids()
    except FileNotFoundError:
        return None

    marg, gated, ns = [], [], []
    for path in PREDICTIONS_PATHS_5FOLD:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        nd = df.groupby("custom_id")["dosage"].nunique()
        single = df[df["custom_id"].isin(nd[nd == 1].index)]
        multi = df[df["custom_id"].isin(nd[nd > 1].index) & df["custom_id"].isin(electro)]
        eff = single.groupby("helm_annotation").agg(obs=("inhibition_percent", "max"),
                                                    pred=("prediction", "max"))
        curves: dict[str, list] = {}
        for (h, _c, _t), g in multi.groupby(["helm_annotation", "cell_line", "target_RNA"],
                                            sort=False):
            if len(g) < 4:
                continue
            d = g["dosage"].to_numpy(dtype=float)
            a = _fit_group(d, g["inhibition_percent"].to_numpy(dtype=float))
            b = _fit_group(d, g["prediction"].to_numpy(dtype=float))
            if a and b:
                curves.setdefault(h, []).append((a[0], b[0]))
        if not curves:
            continue
        pot = pd.DataFrame({h: {"obs": np.mean([x[0] for x in v]),
                                "pred": np.mean([x[1] for x in v])}
                            for h, v in curves.items()}).T

        thr = PIPELINE_STAGES[1].threshold_value
        n_sel = max(1, int(round(len(pot) * k)))
        marg.append(float((pot.nsmallest(n_sel, "pred")["obs"] < thr).mean()))

        selected = set(eff.nlargest(max(1, int(round(len(eff) * k))), "pred").index)
        gate_pass = set(eff.index[_passes(eff["obs"], 0)])
        g_pool = pot[pot.index.isin(selected & gate_pass)]
        ns.append(int(len(g_pool)))
        if len(g_pool) >= 10:
            n_g = max(1, int(round(len(g_pool) * k)))
            gated.append(float((g_pool.nsmallest(n_g, "pred")["obs"] < thr).mean()))

    if not marg or not gated:
        return None
    m, g = float(np.mean(marg)), float(np.mean(gated))
    return {"model_precision_marginal": round(m, 3),
            "model_precision_gate_matched": round(g, 3),
            "n_gate_matched": int(sum(ns)),
            "model_uplift": round(g / m, 3)}


def _benchmark_model_uplift(ds_up, ds_dn, up_idx, dn_idx, model="XGBoost", k=K):
    """Model-arm uplift from the actual benchmark predictions.

    Uses ``benchmark_ids`` to reattach compound identifiers to the stored out-of-fold
    predictions, which removes the need for the Hagedorn proxy. Predictions are averaged
    per HELM because a compound can appear at several doses. Ranking is global rather
    than dose-stratified, so the precision *levels* differ slightly from the published
    table; we consume only the ratio, which is insensitive to that choice.
    """
    import numpy as np
    from analyses.logic.models.benchmark_ids import labelled_predictions

    def pool(ds):
        d = labelled_predictions(ds, model)
        return d.groupby("helm").agg(y=("y_true", "mean"), p=("y_pred", "mean"))

    try:
        up, dn = pool(ds_up), pool(ds_dn)
    except (LookupError, ValueError, FileNotFoundError):
        return None

    n_sel = max(1, int(round(len(dn) * k)))
    marginal = float(_passes(dn.nsmallest(n_sel, "p")["y"], dn_idx).mean())
    if not marginal > 0:
        return None

    selected = set(up.nsmallest(max(1, int(round(len(up) * k))), "p").index)
    gate_pass = set(up.index[_passes(up["y"], up_idx)])
    gated = dn[dn.index.isin(selected & gate_pass)]
    if len(gated) < 10:
        return {"n_gate_matched": int(len(gated))}

    n_g = max(1, int(round(len(gated) * k)))
    g = float(_passes(gated.nsmallest(n_g, "p")["y"], dn_idx).mean())
    return {"model_precision_marginal": round(marginal, 3),
            "model_precision_gate_matched": round(g, 3),
            "n_gate_matched": int(len(gated)),
            "model_uplift": round(g / marginal, 3),
            "model_source": model}


def _hepatic_populations(hep_df):
    mouse = compound_mean_biomarker(hep_df[hep_df["species"] == "mouse"], "ALT")
    rat = compound_mean_biomarker(hep_df[hep_df["species"] == "rat"], "ALT")
    return mouse, rat


def _scores_for(neuro_df) -> pd.Series:
    """Hagedorn sequence score per compound (higher = predicted safer)."""
    sub = neuro_df[neuro_df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    sub["score"] = hagedorn_score(sub)
    return sub.dropna(subset=["score"]).groupby("Compound ID")["score"].first()


def _arm_uplifts(mouse, rat, scores, mouse_idx, rat_idx):
    """Baseline-arm and model-arm conditioning uplifts for one species pair."""
    shared_idx = mouse.index.intersection(rat.index)
    if len(shared_idx) < 30:
        return None
    m, r = mouse[shared_idx], rat[shared_idx]
    rho, _ = spearmanr(m, r)
    m_pass, r_pass = _passes(m, mouse_idx), _passes(r, rat_idx)

    q_rat_marginal = float(r_pass.mean())
    q_rat_given_pass = float(r_pass[m_pass].mean()) if m_pass.any() else None
    baseline_uplift = (q_rat_given_pass / q_rat_marginal
                       if q_rat_marginal > 0 and q_rat_given_pass is not None else None)

    # Model arm. The comparison has to be pool-matched: in BOTH arms the compounds
    # that reach the rat stage have passed the mouse gate, so the baseline rate and
    # the model precision must be read off populations that differ only by the model
    # selection. Three quantities:
    #   prec_marg  - top-K of the whole rat pool          (what the paper uses)
    #   prec_cond  - top-K of the mouse-model-selected pool (the one-arm version)
    #   prec_gated - top-K of {mouse gate passed AND mouse-model-selected}, which is
    #                the population the model arm actually presents to the rat stage
    model_uplift = prec_marg = prec_cond = prec_gated = None
    n_gated = 0
    common = scores.index.intersection(rat.index)
    if len(common) >= 30:
        rat_s = pd.DataFrame({"y": rat[common], "score": scores[common]})
        n_sel = max(1, int(round(len(rat_s) * K)))
        prec_marg = float(_passes(rat_s.nlargest(n_sel, "score")["y"], rat_idx).mean())

        mouse_common = scores.index.intersection(mouse.index)
        mouse_s = pd.DataFrame({"score": scores[mouse_common]})
        n_sel_m = max(1, int(round(len(mouse_s) * K)))
        selected = set(mouse_s.nlargest(n_sel_m, "score").index)

        after = rat_s[rat_s.index.isin(selected)]
        if len(after) >= 10:
            n_sel_r = max(1, int(round(len(after) * K)))
            prec_cond = float(_passes(after.nlargest(n_sel_r, "score")["y"], rat_idx).mean())

        gate_pass = set(m.index[m_pass])
        gated = rat_s[rat_s.index.isin(selected & gate_pass)]
        n_gated = int(len(gated))
        if n_gated >= 10:
            n_sel_g = max(1, int(round(n_gated * K)))
            prec_gated = float(_passes(gated.nlargest(n_sel_g, "score")["y"], rat_idx).mean())
            model_uplift = prec_gated / prec_marg if prec_marg > 0 else None

    return {
        "n_mouse": int(len(mouse)), "n_rat": int(len(rat)), "n_shared": int(len(shared_idx)),
        "spearman_rho": round(float(rho), 3),
        "n_mouse_pass": int(m_pass.sum()),
        "frac_rat_tested_failing_mouse_gate": round(float((~m_pass).mean()), 3),
        "q_rat_marginal_in_shared": round(q_rat_marginal, 3),
        "q_rat_given_mouse_pass": round(q_rat_given_pass, 3) if q_rat_given_pass else None,
        "baseline_uplift": round(baseline_uplift, 3) if baseline_uplift else None,
        "model_precision_marginal": round(prec_marg, 3) if prec_marg else None,
        "model_precision_conditional": round(prec_cond, 3) if prec_cond else None,
        "model_precision_gate_matched": round(prec_gated, 3) if prec_gated else None,
        "n_gate_matched": n_gated,
        "model_uplift": round(model_uplift, 3) if model_uplift else None,
    }


def compute() -> dict:
    neuro_df = pd.read_parquet(PROCESSED / "neurotoxicity_processed.parquet")
    hep_df = pd.read_parquet(PROCESSED / "hepatictoxicity_processed.parquet")
    scores = _scores_for(neuro_df)

    m_neu, r_neu = _neuro_populations(neuro_df)
    m_hep, r_hep = _hepatic_populations(hep_df)
    up_eff, dn_eff = _efficacy_populations()

    # Model-arm uplifts: the two toxicity links use the actual XGBoost predictions, with
    # identifiers recovered by replaying the deterministic GroupKFold split (see
    # analyses.logic.models.benchmark_ids). Efficacy/potency uses the OligoAI held-out fold
    # predictions. All three links are therefore corrected symmetrically.
    no_scores = pd.Series(dtype=float)
    neuro = _arm_uplifts(m_neu, r_neu, scores, *NEURO_PAIR) or {}
    # Keep the Hagedorn-proxy estimate as a cross-check, then supersede it with the
    # actual XGBoost predictions now that identifiers can be recovered.
    neuro["model_uplift_hagedorn_proxy"] = neuro.get("model_uplift")
    neuro.update(_benchmark_model_uplift("mouse_neuro", "rat_neuro", *NEURO_PAIR) or {})

    pairs = {
        "neuro": {"idx": NEURO_PAIR, **neuro},
        "hepatic": {"idx": HEPATIC_PAIR,
                    **(_arm_uplifts(m_hep, r_hep, no_scores, *HEPATIC_PAIR) or {}),
                    **(_benchmark_model_uplift("mouse_hepatic", "rat_hepatic",
                                               *HEPATIC_PAIR) or {})},
        "efficacy_potency": {"idx": EFFICACY_PAIR,
                             **(_arm_uplifts(up_eff, dn_eff, no_scores, *EFFICACY_PAIR) or {}),
                             **(_efficacy_model_uplift() or {})},
    }
    for name, info in pairs.items():
        info["display_name"] = PAIR_DISPLAY_NAMES[name]
        baseline_uplift = info.get("baseline_uplift")
        model_uplift = info.get("model_uplift")
        if baseline_uplift and model_uplift:
            # EF_cond / EF_marg = (precision_cond / precision_marg) /
            #                         (base_cond / base_marg).
            info["conditional_to_marginal_ef_ratio"] = round(
                model_uplift / baseline_uplift, 3,
            )

    pipe = json.loads((RESULTS_DIR / "pipeline_results.json").read_text())
    base = list(pipe["proportions"])
    combined = next(r for r in json.loads((RESULTS_DIR / "ef_table.json").read_text())
                    if r.get("group") == "combined")
    canon = {int(k): v for k, v in combined["prec_by_stage"].items() if v is not None}

    def cell(applied, baseline_conditional, model_conditional, use_ceil):
        b, p = list(base), dict(canon)
        for name in applied:
            info = pairs[name]
            rat_idx = info["idx"][1]
            if baseline_conditional and info.get("baseline_uplift"):
                b[rat_idx] = min(base[rat_idx] * info["baseline_uplift"], 1.0)
            if model_conditional and info.get("model_uplift"):
                p[rat_idx] = min(canon[rat_idx] * info["model_uplift"], 1.0)
        return 100.0 * (1 - _cost(b, p, use_ceil) / _cost(b, {}, use_ceil))

    out_cells = {}
    # "symmetric" is the one to quote: only links where BOTH arms are estimable. Applying a
    # baseline uplift at a link whose model uplift we cannot measure repeats, in mirror image,
    # the one-sided error this module exists to correct. It matters at the hepatic link
    # specifically: conditioning the baseline to 0.445 lifts it above the model's 0.406
    # precision, so the EF floor zeroes the stage out and the model is charged a penalty with
    # no offsetting credit.
    for applied, label in ((("neuro",), "neuro_only"),
                           (("neuro", "efficacy_potency"), "symmetric"),
                           (("neuro", "hepatic"), "tox_pairs"),
                           (("neuro", "hepatic", "efficacy_potency"), "all_pairs")):
        out_cells[label] = {}
        for use_ceil, rec in ((True, "ceiling"), (False, "expected")):
            out_cells[label][rec] = {
                "marginal_marginal": round(cell(applied, False, False, use_ceil), 1),
                "marginal_conditional": round(cell(applied, False, True, use_ceil), 1),
                "conditional_marginal": round(cell(applied, True, False, use_ceil), 1),
                "conditional_conditional": round(cell(applied, True, True, use_ceil), 1),
            }

    # Does the paper's qualitative claim survive the correction? The claim is an
    # ordering (in vivo enrichment is worth more than in vitro enrichment), not the
    # 71% itself, so it has to be re-tested under the conditioned base rates rather
    # than assumed to carry over.
    b_cond = list(base)
    for info in pairs.values():
        if info.get("baseline_uplift"):
            ri = info["idx"][1]
            b_cond[ri] = min(base[ri] * info["baseline_uplift"], 1.0)
    in_vitro = {i: canon[i] for i in (0, 1) if i in canon}
    in_vivo = {i: canon[i] for i in (2, 3, 4, 5) if i in canon}

    def sav(b, p, use_ceil):
        return round(100.0 * (1 - _cost(b, p, use_ceil) / _cost(b, {}, use_ceil)), 1)

    ordering = {}
    for rec, use_ceil in (("ceiling", True), ("expected", False)):
        ordering[rec] = {
            "published_base_rates": {
                "in_vitro_only": sav(base, in_vitro, use_ceil),
                "in_vivo_only": sav(base, in_vivo, use_ceil),
                "combined": sav(base, canon, use_ceil),
            },
            "conditioned_base_rates": {
                "in_vitro_only": sav(b_cond, in_vitro, use_ceil),
                "in_vivo_only": sav(b_cond, in_vivo, use_ceil),
                "combined": sav(b_cond, canon, use_ceil),
            },
        }

    return {
        "K": K,
        "pairs": pairs,
        "published_base_rates": {str(i): round(base[i], 4) for i in (2, 3, 4, 5)},
        "conditioned_base_rates": {str(i): round(b_cond[i], 4) for i in (2, 3, 4, 5)},
        "savings_by_cell": out_cells,
        "ordering_check": ordering,
    }


def main() -> None:
    r = compute()
    (RESULTS_DIR / "s1_conditional_pipeline.json").write_text(json.dumps(r, indent=2))

    for name, info in r["pairs"].items():
        if not info.get("baseline_uplift"):
            print(f"{name}: insufficient overlap")
            continue
        print(f"\n{name} pair (stages {info['idx'][0]} -> {info['idx'][1]}), "
              f"rho = {info['spearman_rho']}, n shared = {info['n_shared']}")
        print(f"  {info['frac_rat_tested_failing_mouse_gate']:.1%} of downstream-tested "
              f"compounds had already FAILED the upstream gate")
        print(f"  baseline arm: P(rat pass) {info['q_rat_marginal_in_shared']} -> "
              f"P(rat pass | mouse pass) {info['q_rat_given_mouse_pass']}  "
              f"(uplift {info['baseline_uplift']})")
        print(f"  model arm:    precision {info['model_precision_marginal']} -> "
              f"{info['model_precision_conditional']} (model-selected only) -> "
              f"{info['model_precision_gate_matched']} (gate-matched, n={info['n_gate_matched']})"
              f"  uplift {info['model_uplift']}")

    for label, recs in r["savings_by_cell"].items():
        for rec, c in recs.items():
            print(f"\n{label} / {rec} recursion")
            print(f"  baseline marginal    | model marginal    {c['marginal_marginal']}%"
                  f"   <- published")
            print(f"  baseline marginal    | model conditional {c['marginal_conditional']}%")
            print(f"  baseline conditional | model marginal    {c['conditional_marginal']}%")
            print(f"  baseline conditional | model conditional {c['conditional_conditional']}%"
                  f"   <- internally consistent")


    print("\n  ordering check (does in vivo still beat in vitro?)")
    for rec, o in r["ordering_check"].items():
        for which, v in o.items():
            print(f"    {rec:<9} {which:<24} in vitro {v['in_vitro_only']:>5}%  "
                  f"in vivo {v['in_vivo_only']:>5}%  combined {v['combined']:>5}%")


if __name__ == "__main__":
    main()
