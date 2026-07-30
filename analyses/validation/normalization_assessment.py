"""Write the focused sequence, chemistry, and unit-normalisation assessment.

These checks are deliberately separate from the production-canonical occurrence
audit: chemistry is a compound-level annotation, while conversion orientation is
tested only where a source and released readout are numerically comparable.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.validation import gold_standard_audit as diagnostic


OUTPUT = ROOT / "data" / "results" / "normalization_assessment.json"
CANONICAL_LEDGER = ROOT / "data" / "validation" / "canonical_source_match_ledger.csv"
LINEAGE_LEDGER = ROOT / "data" / "validation" / "lineage_rescued_recall.csv"


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
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


def canonical_percent_control_assessment(
    canonical_ledger: pd.DataFrame,
    lineage_ledger: pd.DataFrame,
) -> dict:
    """Check % control orientation only within canonical one-to-one pairs.

    The former release-wide diagnostic searched every row for the same compound
    and dose.  That allowed a real 33% inhibition measurement in a different
    cell line to masquerade as an unflipped counterpart to a missing 67% row.
    Here the frozen canonical audit supplies the counterpart before values are
    compared, while the lineage ledger supplies the source-derived table label.
    """

    labels = (
        lineage_ledger[["gold_id", "unit_conversion"]]
        .drop_duplicates("gold_id")
        .rename(columns={"gold_id": "measurement_id"})
    )
    gold = canonical_ledger[
        canonical_ledger["side"].eq("gold")
        & canonical_ledger["status"].fillna("").str.startswith("matched_")
    ].merge(labels, on="measurement_id", how="left", validate="one_to_one")
    atlas = canonical_ledger[canonical_ledger["side"].eq("atlas")].set_index(
        "measurement_id"
    )
    gold["atlas_outcome"] = gold["counterpart_id"].map(atlas["outcome"])

    total_gold = int(
        lineage_ledger["unit_conversion"].eq("% control → % inhibition").sum()
    )
    compared = gold[gold["unit_conversion"].eq("% control → % inhibition")].copy()
    compared["gold_value"] = pd.to_numeric(compared["outcome"], errors="coerce")
    compared["atlas_value"] = pd.to_numeric(
        compared["atlas_outcome"], errors="coerce"
    )
    compared = compared[
        compared["gold_value"].notna() & compared["atlas_value"].notna()
    ].copy()

    same = (compared["atlas_value"] - compared["gold_value"]).abs() < 1e-9
    uninformative = compared["gold_value"].eq(50.0)
    unflipped = (
        ~uninformative
        & ~same
        & (
            compared["atlas_value"] - (100.0 - compared["gold_value"])
        ).abs().lt(1e-9)
    )
    other_wrong = ~same & ~unflipped
    informative_n = int((~uninformative).sum())
    unflipped_n = int(unflipped.sum())
    compared["conversion_classification"] = "correct"
    compared.loc[unflipped, "conversion_classification"] = "stored as % control"
    compared.loc[other_wrong, "conversion_classification"] = "other value discrepancy"

    examples = []
    for row in compared[unflipped | other_wrong].itertuples(index=False):
        examples.append({
            "patent": row.patent,
            "table": row.table,
            "compound": str(row.compound),
            "gold_inhibition_pct": row.gold_value,
            "atlas_value": row.atlas_value,
            "classification": row.conversion_classification,
        })

    return {
        "definition": (
            "Within frozen canonical one-to-one pairs from tables labelled as "
            "% control, an unflipped error has Atlas value = 100 - Gold inhibition."
        ),
        "gold_readouts_in_percent_control_tables": total_gold,
        "canonical_pairs_with_numeric_values": int(len(compared)),
        "correct_normalized_values": int(same.sum()),
        "other_value_discrepancies": int(other_wrong.sum()),
        "orientation_uninformative_at_50_pct": int(uninformative.sum()),
        "orientation_informative_pairs": informative_n,
        "confirmed_unflipped_errors": unflipped_n,
        "unflipped_error_rate_pct": round(100 * unflipped_n / informative_n, 4),
        "unflipped_error_rate_ci95": _wilson(unflipped_n, informative_n),
        "examples": examples,
        "provenance_guardrail": (
            "No release-wide compound/dose substitution; cell line and table provenance "
            "are fixed by the canonical match before comparing values."
        ),
    }


def main() -> dict:
    report = diagnostic.main(seed=2026, n_perm=1)
    canonical_ledger = pd.read_csv(CANONICAL_LEDGER)
    lineage_ledger = pd.read_csv(LINEAGE_LEDGER)
    focused = {
        "status": "complete",
        "role": "normalisation accuracy; not the primary precision/recall estimand",
        "sequence_and_chemistry": report["chemistry_normalisation"],
        "percent_control_orientation": canonical_percent_control_assessment(
            canonical_ledger, lineage_ledger
        ),
        "source": {
            "gold_compounds": report["corpus"]["gold_chemistry_compounds"],
            "gold_dir": report["provenance"]["gold_dir"],
            "release_dir": report["provenance"]["release_dir"],
            "canonical_ledger": str(CANONICAL_LEDGER.relative_to(ROOT)),
            "lineage_ledger": str(LINEAGE_LEDGER.relative_to(ROOT)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(focused, indent=2) + "\n")
    print(json.dumps({
        "status": focused["status"],
        "sequence": focused["sequence_and_chemistry"]["sequence"],
        "sugar_motif_where_stated": focused["sequence_and_chemistry"]["sugar_motif_where_stated"],
        "linkage_motif_where_stated": focused["sequence_and_chemistry"]["linkage_motif_where_stated"],
        "percent_control_orientation": focused["percent_control_orientation"],
    }, indent=2))
    return focused


if __name__ == "__main__":
    main()
