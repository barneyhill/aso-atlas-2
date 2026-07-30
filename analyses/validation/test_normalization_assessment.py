import pandas as pd

from analyses.validation.normalization_assessment import (
    canonical_percent_control_assessment,
)


def test_percent_control_orientation_uses_only_canonical_counterparts():
    canonical = pd.DataFrame(
        [
            {
                "side": "gold",
                "status": "matched_exact",
                "measurement_id": f"g{i}",
                "counterpart_id": f"a{i}",
                "outcome": gold,
                "patent": "P1",
                "table": i,
                "compound": f"C{i}",
            }
            for i, gold in enumerate([67.0, 67.0, 50.0, 59.0], start=1)
        ]
        + [
            {
                "side": "atlas",
                "status": "matched_exact",
                "measurement_id": f"a{i}",
                "counterpart_id": f"g{i}",
                "outcome": atlas,
                "patent": "P1",
                "table": i,
                "compound": f"C{i}",
            }
            for i, atlas in enumerate([67.0, 33.0, 50.0, 50.0], start=1)
        ]
    )
    lineage = pd.DataFrame(
        {
            "gold_id": ["g1", "g2", "g3", "g4", "g5"],
            "unit_conversion": ["% control → % inhibition"] * 5,
        }
    )

    result = canonical_percent_control_assessment(canonical, lineage)

    assert result["gold_readouts_in_percent_control_tables"] == 5
    assert result["canonical_pairs_with_numeric_values"] == 4
    assert result["correct_normalized_values"] == 2
    assert result["orientation_uninformative_at_50_pct"] == 1
    assert result["orientation_informative_pairs"] == 3
    assert result["confirmed_unflipped_errors"] == 1
    assert result["other_value_discrepancies"] == 1
    assert [example["classification"] for example in result["examples"]] == [
        "stored as % control",
        "other value discrepancy",
    ]
