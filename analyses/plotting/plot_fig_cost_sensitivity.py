"""Plot the existing corrected joint and cost sensitivity distributions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyses.logic.paper_pipeline import load_corrected_analysis

ROOT = Path(__file__).resolve().parents[2]


def main():
    report = load_corrected_analysis()
    for histogram_key, summary_key, name, title in [
        ("joint_model_and_baseline", "joint_model_fold_and_baseline_bootstrap_savings_pct",
         "fig_uncertainty", "Joint model-fold and baseline pass-rate sensitivity"),
        ("stage_costs", "stage_cost_sensitivity_savings_pct",
         "fig_cost_sensitivity", "Independent ±50% stage-cost sensitivity"),
    ]:
        histogram = report["sensitivity_histograms"][histogram_key]
        summary = report[summary_key]
        fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
        ax.stairs(histogram["counts"], histogram["edges"], fill=True,
                  color="#4878A8", alpha=0.85)
        ax.axvspan(summary["p2.5"], summary["p97.5"], color="#E76F51", alpha=0.12,
                   label=f'Middle 95%: {summary["p2.5"]:.1f}–{summary["p97.5"]:.1f}%')
        ax.axvline(summary["median"], color="#E76F51", linestyle="--",
                   label=f'Median {summary["median"]:.1f}%')
        ax.set_xlabel("Projected cost savings (%)")
        ax.set_ylabel("Draw count")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        out = ROOT / "typst/plots" / name / name
        out.parent.mkdir(parents=True, exist_ok=True)
        for suffix in [".svg", ".png"]:
            fig.savefig(out.with_suffix(suffix), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
