"""q-feasibility cascade: how far modelling lowers the n-Lorem screening rate q.

The n-Lorem primary screen tiles a target with ~500 ASOs, advances >80%
knockdown, and after dose-response and rodent tolerability yields ~1 viable
lead: an unguided per-ASO success rate of q ~ 1/500. Ranking candidates before
assay raises that rate by the enrichment of the ranked subset.

This module measures the enrichment at the top decile for each n-Lorem gate,
using only leakage-free predictors, and compounds the two independent ones:

  * Efficacy (primary screen, inhibition > 80%) -- OligoAI, SCN2A held fully out
    of training. Strong enrichment.
  * Potency (dose-response IC50) -- OligoAI, SCN2A held out. No enrichment: the
    model resolves activity, not relative potency among actives (reported, not
    compounded).
  * Tolerability (rodent FOB < 1) -- Hagedorn 2022 fixed-coefficient score, an
    external sequence-intrinsic model (no fit, no leakage). Reported for SCN2A
    and for the full neurotox corpus.

Efficacy and tolerability score independent axes (target activity vs CNS
toxicity), so their enrichments multiply; the compounded factor maps q ~ 1/500
to the improved rate. 95% CIs are bootstrap over compounds.

Output: data/results/cascade_qfeasibility.json
Usage:  uv run python -m analyses.logic.models.cascade_qfeasibility
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array
from analyses.utils.compounds import fit_ic50_for_compound
from analyses.logic.models.neurotox import hagedorn_score

_ROOT = Path(__file__).resolve().parents[3]
_PRED = _ROOT / "data/results/oligoai_geneholdout_predictions.parquet"
_NEURO = _ROOT / "data/oligostack/processed/neurotoxicity_processed.parquet"
_OUT = _ROOT / "data/results/cascade_qfeasibility.json"

TOP_FRACTION = 0.10          # operating point: top decile
FRACTIONS = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.10, 0.05]   # enrichment-curve sweep
N_BOOT = 5000
_RNG = np.random.default_rng(0)


def _ef(score, passed, frac=TOP_FRACTION):
    score = np.asarray(score, float)
    passed = np.asarray(passed, bool)
    base = passed.mean()
    if base == 0:
        return np.nan
    k = max(1, int(round(frac * len(score))))
    order = np.argsort(-score)            # best (highest score) first
    return passed[order[:k]].mean() / base


def _bootstrap(score, passed, frac=TOP_FRACTION):
    """Point EF + bootstrap-over-compounds samples at a single fraction."""
    score = np.asarray(score, float)
    passed = np.asarray(passed, bool)
    n = len(score)
    pt = _ef(score, passed, frac)
    samp = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = _RNG.integers(0, n, n)
        samp[b] = _ef(score[idx], passed[idx], frac)
    samp = samp[~np.isnan(samp)]
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return pt, float(lo), float(hi), samp


def _sweep(score, passed):
    """Enrichment curve over FRACTIONS (shared bootstrap indices for smooth
    bands). Returns (curve_points, op_samp) where op_samp is the bootstrap
    sample at the operating decile, used for compounding."""
    score = np.asarray(score, float)
    passed = np.asarray(passed, bool)
    n = len(score)
    boot = [_RNG.integers(0, n, n) for _ in range(N_BOOT)]
    curve, op_samp = [], None
    for f in FRACTIONS:
        pt = _ef(score, passed, f)
        samp = np.array([_ef(score[idx], passed[idx], f) for idx in boot])
        samp = samp[~np.isnan(samp)]
        lo, hi = np.percentile(samp, [2.5, 97.5])
        curve.append({"top_fraction": f, "ef": round(pt, 2),
                      "ef_ci": [round(float(lo), 2), round(float(hi), 2)]})
        if abs(f - TOP_FRACTION) < 1e-9:
            op_samp = samp
    return curve, op_samp


def _op(curve):
    """The operating-decile point of a sweep curve."""
    return next(p for p in curve if abs(p["top_fraction"] - TOP_FRACTION) < 1e-9)


def _efficacy(pred):
    """SCN2A held-out, single-dose 4000 nM tables, pass = inhibition > 80%."""
    ndose = pred.groupby("custom_id")["dosage"].nunique()
    single = set(ndose[ndose == 1].index)
    scn = pred[pred.target_RNA == "SCN2A"]
    sub = scn[scn.custom_id.isin(single) & np.isclose(scn.dosage.astype(float), 4000.0)]
    sub = sub.groupby("helm_annotation", as_index=False).agg(
        inh=("inhibition_percent", "mean"), pr=("prediction", "mean"))
    passed = sub.inh.values > 80
    base = passed.mean()
    curve, samp = _sweep(sub.pr.values, passed)
    op = _op(curve)
    k = max(1, int(round(TOP_FRACTION * len(sub))))
    hit = passed[np.argsort(-sub.pr.values)[:k]].mean()
    return {
        "gene": "SCN2A", "assay": "electroporation, single dose 4000 nM",
        "threshold": "inhibition > 80%", "n_asos": int(len(sub)),
        "base_rate_pct": round(base * 100, 1), "hit_rate_pct": round(hit * 100, 1),
        "ef": op["ef"], "ef_ci": op["ef_ci"], "curve": curve,
    }, samp


def _potency(pred):
    """SCN2A held-out dose-response: report rank-correlation; no enrichment claim."""
    scn = pred[pred.target_RNA == "SCN2A"]
    it_l, ip_l = [], []
    for _, g in scn.groupby("helm_annotation", sort=False):
        if g.dosage.nunique() < 4:
            continue
        d = g.dosage.to_numpy(float)
        it = fit_ic50_for_compound(d, g.inhibition_percent.to_numpy(float))
        ip = fit_ic50_for_compound(d, g.prediction.to_numpy(float))
        if np.isnan(it) or np.isnan(ip):
            continue
        it_l.append(it); ip_l.append(ip)
    it = np.array(it_l); ip = np.array(ip_l)
    rho = float(spearmanr(it, ip)[0])
    # EF at top decile against the n-Lorem-calibrated base rate (~30%)
    thr = float(np.quantile(it, 0.30))
    pt, lo, hi, _ = _bootstrap(-ip, it < thr)
    return {
        "gene": "SCN2A", "assay": "electroporation dose-response, SH-SY5Y",
        "n_curves": int(len(it)), "spearman_ic50": round(rho, 2),
        "ef": round(pt, 2), "ef_ci": [round(lo, 2), round(hi, 2)],
        "note": "model resolves activity, not relative potency; gate kept at unguided rate",
    }


def _tolerability(scope_mask, neuro):
    """Hagedorn FOB<1 enrichment for a subset (HELM-deduped)."""
    d = neuro[scope_mask]
    g = d.groupby("HELM Annotation").agg(
        fob=("mean_FOB", "mean"), hage=("hage", "first")).reset_index()
    passed = (g.fob < 1.0).values
    curve, samp = _sweep(g.hage.values, passed)
    op = _op(curve)
    return {
        "n": int(len(g)), "base_rate_pct": round(passed.mean() * 100, 1),
        "ef": op["ef"], "ef_ci": op["ef_ci"], "curve": curve,
    }, samp


def _compound(eff_samp, tol_samp, eff_ef, tol_ef):
    """Compounded enrichment of two independent gates, with bootstrap CI.

    This factor multiplies the per-ASO preclinical pass rate q (and divides the
    K/q screen size); the q-translation against the attrition baseline is done
    downstream in paper-3's export so q ~ 1/100 stays a single source of truth.
    """
    m = min(len(eff_samp), len(tol_samp))
    comb = eff_samp[:m] * tol_samp[:m]
    pt = eff_ef * tol_ef
    c_lo, c_hi = np.percentile(comb, [2.5, 97.5])
    return {"ef": round(pt, 2), "ef_ci": [round(float(c_lo), 2), round(float(c_hi), 2)]}


def main():
    pred = pd.read_parquet(_PRED)

    efficacy, eff_samp = _efficacy(pred)
    potency = _potency(pred)

    neuro = pd.read_parquet(_NEURO)
    neuro = neuro[neuro["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    neuro["mean_FOB"] = neuro["FOB_score"].apply(mean_of_array)
    neuro = neuro[neuro["mean_FOB"].notna()].copy()
    neuro["hage"] = hagedorn_score(neuro)
    neuro = neuro[neuro["hage"].notna()].copy()
    mouse = ((neuro.species == "Mouse") & (neuro.dosage_ug == 700)
             & (neuro.administration_method == "ICV"))
    scn_tol, scn_samp = _tolerability(mouse & (neuro.target_RNA == "SCN2A"), neuro)
    cor_tol, cor_samp = _tolerability(mouse, neuro)

    out = {
        "operating_top_pct": int(TOP_FRACTION * 100),
        "efficacy": efficacy,
        "potency": potency,
        "tolerability": {
            "assay": "Hagedorn fixed-coefficient score, rodent FOB < 1 (mouse 700ug ICV)",
            "scn2a": scn_tol, "corpus": cor_tol,
        },
        "compounded": {
            "scn2a": _compound(eff_samp, scn_samp, efficacy["ef"], scn_tol["ef"]),
            "corpus": _compound(eff_samp, cor_samp, efficacy["ef"], cor_tol["ef"]),
        },
        "meta": {
            "bootstrap_reps": N_BOOT,
            "ci": "95% bootstrap over compounds",
            "predictions": str(_PRED.relative_to(_ROOT)),
            "note": ("Efficacy and tolerability score independent axes (OligoAI "
                     "target activity vs Hagedorn CNS toxicity) and compound; "
                     "potency is the same OligoAI signal as efficacy and is not "
                     "compounded. SCN2A and UBE3A-ATS held fully out of training."),
        },
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {_OUT}")
    e, c = out["efficacy"], out["compounded"]
    print(f"  efficacy   EF {e['ef']}x CI{e['ef_ci']} (base {e['base_rate_pct']}% -> {e['hit_rate_pct']}%)")
    print(f"  potency    EF {potency['ef']}x (Spearman {potency['spearman_ic50']}) -- not compounded")
    print(f"  tol SCN2A  EF {scn_tol['ef']}x CI{scn_tol['ef_ci']} (base {scn_tol['base_rate_pct']}%)")
    print(f"  tol corpus EF {cor_tol['ef']}x CI{cor_tol['ef_ci']} (base {cor_tol['base_rate_pct']}%)")
    print(f"  compounded SCN2A  {c['scn2a']['ef']}x CI{c['scn2a']['ef_ci']}")
    print(f"  compounded corpus {c['corpus']['ef']}x CI{c['corpus']['ef_ci']}")


if __name__ == "__main__":
    main()
