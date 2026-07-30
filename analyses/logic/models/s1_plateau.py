"""EF-composition plateau (rebuttal S1, decisive point).

The same-endpoint interaction that UEDf and the paper dispute (does multiplicative
EF composition over- or under-count sequential mouse->rat neuro savings?) is
immaterial to the headline: because the late-stage animal counts are tiny and
integer-rounded, the combined savings is a STEP FUNCTION of the rat-neuro effective
pass rate, flat across a wide plateau. Marginal (0.79), conditional (0.86), the
cost-model XGBoost value (0.75) and the XGBoost CI floor all sit on the same 71.0%
plateau, so the sign of the composition bias does not move the headline.

Sweeps the rat-neuro effective pass rate through the recursion (all else canonical)
and reports the plateau. Reference q values come from
analyses.logic.models.s1_ef_composition_bias.

    uv run python -m analyses.logic.models.s1_plateau
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analyses.logic.pipeline import PIPELINE_STAGES, TARGET_CANDIDATES

_root = Path(__file__).resolve().parents[3]
RESULTS_DIR = _root / "data/results"

# reference rat-neuro effective pass rates (from s1_ef_composition_bias on the
# 1,456 shared mouse+rat neuro compounds; Hagedorn-score proxy, K=0.20):
REF_Q = {
    "XGBoost cost-model input": 0.749,   # ef_table combined precision, stage 5
    "marginal (paper assumption)": 0.789,
    "conditional (UEDf mechanism)": 0.857,
    "XGBoost CI floor": 0.586,            # 0.749 - 0.163 (per-fold s.d. floor)
}


def _cost(base_rates, precisions, use_ceil=True, target=TARGET_CANDIDATES):
    n = len(PIPELINE_STAGES)
    eff = list(base_rates)
    for i, prec in precisions.items():
        if prec is not None and base_rates[i] > 0:
            eff[i] = min(max(prec, base_rates[i]), 1.0)
    asos = [0.0] * (n + 1)
    asos[-1] = target
    for i in range(n - 1, -1, -1):
        if eff[i] <= 0:
            return float("inf")
        asos[i] = math.ceil(asos[i + 1] / eff[i]) if use_ceil else asos[i + 1] / eff[i]
    return sum(asos[i] * PIPELINE_STAGES[i].cost_per_aso for i in range(n))


def compute():
    pipe = json.loads((RESULTS_DIR / "pipeline_results.json").read_text())
    base = list(pipe["proportions"])
    baseline_cost = pipe["baseline"]["total_cost"]

    combined = next(r for r in json.loads((RESULTS_DIR / "ef_table.json").read_text())
                    if r.get("group") == "combined")
    canon = {int(k): v for k, v in combined["prec_by_stage"].items() if v is not None}

    def savings(q):
        prec = dict(canon); prec[5] = q
        return 100.0 * (1 - _cost(base, prec, True) / baseline_cost)

    def asos_rat(q):
        return math.ceil(TARGET_CANDIDATES / max(q, base[5]))

    grid = np.round(np.arange(0.40, 1.0001, 0.01), 2)
    sweep = [{"q": float(q), "asos_rat_neuro": asos_rat(q),
              "savings_pct": round(savings(q), 1)} for q in grid]

    # plateau = the contiguous q-range that yields the modal (canonical) savings
    canonical_sav = round(savings(canon[5]), 1)
    plateau_qs = [s["q"] for s in sweep if s["savings_pct"] == canonical_sav]
    refs = {name: {"q": q, "savings_pct": round(savings(q), 1),
                   "on_plateau": round(savings(q), 1) == canonical_sav}
            for name, q in REF_Q.items()}

    # The plateau exists only under the integer ceiling. Repeat the marginal ->
    # conditional comparison on the no-ceiling expected-count recursion (the one
    # the S2 reply adopts), and confirm the delta is invariant to candidate yield.
    def savings_nc(q, target):
        prec = dict(canon); prec[5] = q
        bl = _cost(base, {}, use_ceil=False, target=target)
        return 100.0 * (1 - _cost(base, prec, use_ceil=False, target=target) / bl)

    q_marg, q_cond = REF_Q["marginal (paper assumption)"], REF_Q["conditional (UEDf mechanism)"]
    nc = {
        "q_marginal": q_marg,
        "q_conditional": q_cond,
        "savings_marginal_pct": round(savings_nc(q_marg, TARGET_CANDIDATES), 1),
        "savings_conditional_pct": round(savings_nc(q_cond, TARGET_CANDIDATES), 1),
        "delta_pt": round(savings_nc(q_cond, TARGET_CANDIDATES)
                          - savings_nc(q_marg, TARGET_CANDIDATES), 2),
        "delta_by_target": {
            str(t): round(savings_nc(q_cond, t) - savings_nc(q_marg, t), 2)
            for t in (1, 10, 100, 1000)
        },
    }

    return {
        "canonical_savings_pct": canonical_sav,
        "plateau_q_lo": min(plateau_qs) if plateau_qs else None,
        "plateau_q_hi": max(plateau_qs) if plateau_qs else None,
        "references": refs,
        "no_ceiling": nc,
        "sweep": sweep,
    }


def main():
    res = compute()
    (RESULTS_DIR / "s1_plateau.json").write_text(json.dumps(res, indent=2))
    print(f"Wrote {RESULTS_DIR / 's1_plateau.json'}")
    print(f"  plateau savings {res['canonical_savings_pct']}% across rat-neuro "
          f"q~ in [{res['plateau_q_lo']}, {res['plateau_q_hi']}]")
    for name, r in res["references"].items():
        print(f"  {name:32s} q~={r['q']:.3f} -> {r['savings_pct']}%  "
              f"{'ON plateau' if r['on_plateau'] else 'off plateau'}")
    # a few step points
    for q in (0.45, 0.55, 0.95, 1.00):
        s = next(x for x in res["sweep"] if abs(x["q"] - q) < 1e-6)
        print(f"  q~={q:.2f}: asos@rat-neuro={s['asos_rat_neuro']}  savings {s['savings_pct']}%")


if __name__ == "__main__":
    main()
