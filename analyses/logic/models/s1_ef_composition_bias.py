"""S1: Does multiplicative EF composition over- or under-estimate combined
savings for the correlated mouse-neuro -> rat-neuro stages?

Same Hagedorn score (G-content dominated) is used at both neuro stages, and
cross-species FOB concordance is strong (rho=0.647). The cost model multiplies
per-stage effective pass rates as if independent. We test the sign of the bias
empirically on the compounds that actually traverse both stages.

The decisive follow-up (that the sign is immaterial to the headline because of the
integer ceiling) is in analyses.logic.models.s1_plateau.

    uv run python -m analyses.logic.models.s1_ef_composition_bias
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from analyses.logic.models.neurotox import hagedorn_score
from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

K = 0.20
_root = Path(__file__).resolve().parents[3]
D = _root / "data/oligostack/processed/neurotoxicity_processed.parquet"
RESULTS_DIR = _root / "data/results"


def _pass_frac(g):                 # base rate: FOB <= 1
    return float((g["mean_FOB"] <= 1).mean())


def _topk_precision(g, k=K):       # select top-k safest by score, fraction passing
    n = max(1, int(round(len(g) * k)))
    sel = g.nlargest(n, "score")
    return float((sel["mean_FOB"] <= 1).mean()), n


def compute() -> dict:
    df = pd.read_parquet(D)
    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df["score"] = hagedorn_score(df)  # higher = safer

    def stage_pop(species, dose, icv):
        m = (df["species"] == species) & (df["dosage_ug"] == dose) & df["HELM Annotation"].apply(Helm.valid_chemistry)
        if icv:
            m &= (df["administration_method"] == "ICV")
        sub = df[m & df["mean_FOB"].notna() & df["score"].notna()].copy()
        return sub.groupby("Compound ID").agg(mean_FOB=("mean_FOB", "mean"), score=("score", "first"))

    mouse = stage_pop("Mouse", 700, True)
    rat = stage_pop("Rat", 3000, False)

    # marginal per-stage numbers (what the cost model uses)
    q_mouse, (qt_mouse, _) = _pass_frac(mouse), _topk_precision(mouse)
    q_rat, (qt_rat, _) = _pass_frac(rat), _topk_precision(rat)
    ef_mouse = qt_mouse / q_mouse
    ef_rat = qt_rat / q_rat

    # compounds traversing BOTH stages
    shared = mouse.join(rat, how="inner", lsuffix="_m", rsuffix="_r")
    shared["score"] = shared["score_m"]
    rho, p = spearmanr(shared["mean_FOB_m"], shared["mean_FOB_r"])

    n_sel = max(1, int(round(len(shared) * K)))
    sel = shared.nlargest(n_sel, "score")
    p_mouse_sel = float((sel["mean_FOB_m"] <= 1).mean())
    p_rat_sel = float((sel["mean_FOB_r"] <= 1).mean())
    p_both_sel = float(((sel["mean_FOB_m"] <= 1) & (sel["mean_FOB_r"] <= 1)).mean())
    realized_over_product = p_both_sel / (p_mouse_sel * p_rat_sel)

    # residual rat enrichment after mouse model-selection (UEDf mechanism)
    mouse_sel_ids = set(mouse.nlargest(max(1, int(round(len(mouse) * K))), "score").index)
    rat_after = rat[rat.index.isin(mouse_sel_ids)]
    ef_rat_cond = q_rat_cond = qt_rat_cond = None
    if len(rat_after) >= 10:
        q_rat_cond = _pass_frac(rat_after)
        qt_rat_cond, _ = _topk_precision(rat_after)
        ef_rat_cond = qt_rat_cond / q_rat_cond if q_rat_cond > 0 else None

    return {
        "n_mouse": int(len(mouse)), "n_rat": int(len(rat)), "n_shared": int(len(shared)),
        "cross_species_rho": round(float(rho), 3),
        "rat_q_marginal": round(qt_rat, 3), "rat_ef_marginal": round(ef_rat, 3),
        "rat_q_conditional": round(qt_rat_cond, 3) if qt_rat_cond else None,
        "rat_ef_conditional": round(ef_rat_cond, 3) if ef_rat_cond else None,
        "rat_ef_ratio_cond_marg": round(ef_rat_cond / ef_rat, 3) if ef_rat_cond else None,
        "realized_joint_pass_rate": round(p_both_sel, 3),
        "independent_product": round(p_mouse_sel * p_rat_sel, 3),
        "realized_over_product": round(realized_over_product, 3),
    }


def main():
    r = compute()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "s1_ef_composition_bias.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    print("\n  marginal rat EF {rat_ef_marginal} -> conditional {rat_ef_conditional} "
          "(UEDf: signal spent at mouse); realized/product {realized_over_product} "
          "(>1 => positive dependence, paper's direction). Sign resolved as immaterial "
          "in s1_plateau.".format(**r))


if __name__ == "__main__":
    main()
