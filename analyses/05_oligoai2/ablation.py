"""In vitro base ablation study for OligoAI2.

Three conditions compare the impact of in vitro inhibition data on vivo prediction:
  - full:       warmup (10 ep) + IV in joint phase  (current best pipeline)
  - no_warmup:  no warmup (0 ep) + IV in joint phase
  - vivo_only:  no IV data at all

Runs each as a subprocess (clean MPS state between runs). Skips completed runs.
Writes ablation_results/summary.json with per-condition vivo Spearman metrics.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "ablation_results"


@dataclass
class AblationConfig:
    """Configuration for a single ablation condition."""
    name: str
    encoder_lr: float = 1e-4
    warmup_epochs: int | None = None
    vivo_only: bool = False

    def to_cli_args(self) -> list[str]:
        args = ["--encoder-lr", str(self.encoder_lr)]
        if self.warmup_epochs is not None:
            args.extend(["--warmup-epochs", str(self.warmup_epochs)])
        if self.vivo_only:
            args.append("--vivo-only")
        return args


CONFIGS = [
    AblationConfig(name="full", encoder_lr=1e-4, warmup_epochs=10),
    AblationConfig(name="no_warmup", encoder_lr=1e-4, warmup_epochs=0),
    AblationConfig(name="vivo_only", encoder_lr=1e-4, vivo_only=True),
]


def run_config(cfg: AblationConfig) -> dict | None:
    """Run a single ablation condition as a subprocess."""
    output_json = RESULTS_DIR / f"{cfg.name}.json"

    if output_json.exists():
        print(f"  [skip] {cfg.name} — already completed")
        with open(output_json) as f:
            return json.load(f)

    cli_args = [
        sys.executable, "-m", "analyses.05_oligoai2.train",
        "--run-name", cfg.name,
        "--output-json", str(output_json),
        *cfg.to_cli_args(),
    ]

    print(f"\n{'='*60}")
    print(f"  Ablation: {cfg.name}")
    print(f"  Args: {' '.join(cfg.to_cli_args())}")
    print(f"{'='*60}")

    result = subprocess.run(
        cli_args,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
    )

    if result.returncode != 0:
        print(f"  [FAIL] {cfg.name} exited with code {result.returncode}")
        return None

    if output_json.exists():
        with open(output_json) as f:
            return json.load(f)
    return None


def _vivo_spearmans(results: dict) -> dict:
    """Extract per-task vivo Spearman values from a run's results."""
    vivo = results.get("invivo", {})
    out = {}
    for task in ("ALT", "AST", "FOB"):
        out[f"{task.lower()}_spearman"] = (
            vivo.get(task, {}).get("median_spearman", float("nan"))
            if isinstance(vivo, dict) else float("nan")
        )
    vals = [v for v in out.values() if np.isfinite(v)]
    out["median_vivo_spearman"] = float(np.median(vals)) if vals else float("nan")
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for cfg in CONFIGS:
        result = run_config(cfg)
        if result is not None:
            all_results[cfg.name] = _vivo_spearmans(result)

    if not all_results:
        print("No ablation results collected.")
        return

    # Print summary
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':<16} {'ALT ρ':>8} {'AST ρ':>8} {'FOB ρ':>8} {'Median ρ':>10}")
    print(f"  {'-'*54}")
    for name, metrics in all_results.items():
        print(
            f"  {name:<16} {metrics['alt_spearman']:>8.4f} "
            f"{metrics['ast_spearman']:>8.4f} {metrics['fob_spearman']:>8.4f} "
            f"{metrics['median_vivo_spearman']:>10.4f}"
        )

    # Write summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
