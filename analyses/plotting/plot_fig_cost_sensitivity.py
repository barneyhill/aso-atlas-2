"""
Supplementary figure: sensitivity of combined pipeline savings to
per-stage cost assumptions.

Monte Carlo joint ±50 % uniform perturbation of all stage costs
(10,000 draws) for the combined (OligoAI + OligoAI-tox) scenario.
"""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
PIPE_PATH = _root / "data/results/pipeline_results.json"
EF_TABLE_PATH = _root / "data/results/ef_table.json"
OUT_DIR = _root / "typst/plots/fig_cost_sensitivity"

N_MC = 10_000
DELTA = 0.5  # ±50 %


def _total_cost(proportions, costs):
    n = len(proportions)
    asos = [0] * (n + 1)
    asos[-1] = 1
    for i in range(n - 1, -1, -1):
        asos[i] = (
            math.ceil(asos[i + 1] / proportions[i])
            if proportions[i] > 0
            else float("inf")
        )
    return sum(asos[i] * costs[i] for i in range(n))


def _savings(base_p, enr_p, costs):
    bc = _total_cost(base_p, costs)
    ec = _total_cost(enr_p, costs)
    return (1 - ec / bc) * 100 if bc > 0 else 0.0


def main():
    if not PIPE_PATH.exists():
        raise FileNotFoundError("Run `just analysis` first.")
    if not EF_TABLE_PATH.exists():
        raise FileNotFoundError("Run `just analysis` first (ef_table.json missing).")

    pipe = json.loads(PIPE_PATH.read_text())
    base_p = pipe["proportions"]
    costs0 = [s["cost_per_aso"] for s in pipe["stages"]]
    n = len(base_p)

    ef_rows = json.loads(EF_TABLE_PATH.read_text())
    comb_row = next((r for r in ef_rows if r.get("group") == "combined"), None)
    if comb_row is None:
        raise RuntimeError("Combined row missing from ef_table.json.")
    comb_p = comb_row["proportions"]

    rng = np.random.default_rng(42)
    mc_comb = []
    for _ in range(N_MC):
        m = rng.uniform(1 - DELTA, 1 + DELTA, n)
        c = [costs0[i] * m[i] for i in range(n)]
        mc_comb.append(_savings(base_p, comb_p, c))
    mc_comb = np.array(mc_comb)

    med = np.median(mc_comb)

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=300)

    ax.hist(mc_comb, bins=50, alpha=0.85, color="#4878A8")
    ax.axvline(med, color="#E76F51", linestyle="--", linewidth=1.5, label=f"Median {med:.1f}%")

    ax.set_xlabel("Combined pipeline cost savings (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Joint ±{int(DELTA * 100)}% cost perturbation ({N_MC:,} draws)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_DIR / "fig_cost_sensitivity.svg"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {svg}")

    stats_path = _root / "data/results/cost_sensitivity.json"
    stats = {
        "n_mc": N_MC,
        "delta": DELTA,
        "median_savings_pct": round(float(med), 1),
        "min_savings_pct": round(float(mc_comb.min()), 1),
        "p5_savings_pct": round(float(np.percentile(mc_comb, 5)), 1),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
