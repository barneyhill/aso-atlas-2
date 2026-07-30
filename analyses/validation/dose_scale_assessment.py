"""Measure 1,000-fold dose-scale discrepancies in the canonical audit pairs.

This is deliberately narrower than general dose-field accuracy.  It asks the
unit-normalisation question directly: among canonical Gold/Atlas pairs for
which both sides state a dose, how often is the Atlas dose exactly 1,000 times
or one-thousandth of the manually verified Gold dose?
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "validation" / "canonical_source_match_ledger.csv"
OUTPUT = ROOT / "data" / "results" / "dose_scale_assessment.json"


def _dose(identity_json: str):
    return json.loads(identity_json).get("dose")


def _wilson_errors(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"low_pct": None, "high_pct": None}
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "low_pct": round(100 * max(0.0, centre - half), 4),
        "high_pct": round(100 * min(1.0, centre + half), 4),
    }


def assess(ledger: pd.DataFrame) -> dict:
    gold = ledger[
        ledger["side"].eq("gold")
        & ledger["status"].fillna("").str.startswith("matched_")
    ].copy()
    atlas = ledger[ledger["side"].eq("atlas")].set_index("measurement_id")

    counts = Counter()
    by_endpoint: dict[str, Counter] = defaultdict(Counter)
    other_tables: Counter = Counter()
    scale_examples = []

    for row in gold.itertuples(index=False):
        gold_dose = _dose(row.identity_json)
        atlas_identity = atlas.loc[row.counterpart_id, "identity_json"]
        atlas_dose = _dose(atlas_identity)
        endpoint = row.endpoint

        if gold_dose is None and atlas_dose is None:
            coverage = "both_missing"
        elif gold_dose is None:
            coverage = "gold_missing"
        elif atlas_dose is None:
            coverage = "atlas_missing"
        else:
            coverage = "both_stated"
        counts[coverage] += 1
        by_endpoint[endpoint][coverage] += 1
        if coverage != "both_stated":
            continue

        ratio = float(atlas_dose) / float(gold_dose)
        is_scale_error = math.isclose(
            ratio, 1000.0, rel_tol=1e-9, abs_tol=1e-9
        ) or math.isclose(ratio, 0.001, rel_tol=1e-9, abs_tol=1e-9)
        if is_scale_error:
            result = "thousand_fold_error"
            if len(scale_examples) < 20:
                scale_examples.append({
                    "patent": row.patent,
                    "table": row.table,
                    "endpoint": endpoint,
                    "channel": row.channel,
                    "compound": str(row.compound),
                    "gold_dose": gold_dose,
                    "atlas_dose": atlas_dose,
                    "atlas_to_gold_ratio": ratio,
                })
        elif math.isclose(ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            result = "exact_agreement"
        else:
            result = "other_dose_discrepancy"
            other_tables[(endpoint, row.patent, str(row.table))] += 1
        counts[result] += 1
        by_endpoint[endpoint][result] += 1

    denominator = counts["both_stated"]
    errors = counts["thousand_fold_error"]
    return {
        "status": "complete",
        "estimand": (
            "1,000-fold dose-scale error rate among one-to-one canonical "
            "Gold/Atlas measurement pairs where both sides state dose"
        ),
        "definition": "Atlas dose / Gold dose equals 1000 or 0.001",
        "paired_measurements": int(len(gold)),
        "both_state_dose": int(denominator),
        "atlas_dose_missing": int(counts["atlas_missing"]),
        "gold_dose_missing": int(counts["gold_missing"]),
        "both_dose_missing": int(counts["both_missing"]),
        "thousand_fold_errors": int(errors),
        "thousand_fold_error_rate_pct": round(100 * errors / denominator, 4),
        "thousand_fold_error_rate_ci95": _wilson_errors(errors, denominator),
        "exact_dose_agreements": int(counts["exact_agreement"]),
        "other_dose_discrepancies": int(counts["other_dose_discrepancy"]),
        "by_endpoint": {
            endpoint: {key: int(value) for key, value in sorted(values.items())}
            for endpoint, values in sorted(by_endpoint.items())
        },
        "other_discrepancies_by_table": [
            {
                "endpoint": endpoint,
                "patent": patent,
                "table": table,
                "count": int(count),
            }
            for (endpoint, patent, table), count in sorted(other_tables.items())
        ],
        "thousand_fold_error_examples": scale_examples,
        "interpretation": (
            "Missing doses and non-scale dose discrepancies are reported separately; "
            "they are not counted as successful unit conversions."
        ),
    }


def main() -> dict:
    ledger = pd.read_csv(LEDGER)
    report = assess(ledger)
    report["source"] = {
        "ledger": str(LEDGER.relative_to(ROOT)),
        "sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
