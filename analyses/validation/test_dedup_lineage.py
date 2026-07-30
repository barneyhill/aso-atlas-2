"""Focused tests for the audit-only deduplication-lineage replay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyses.validation.dedup_lineage import (
    PIPELINE_KEYS,
    _with_raw_ids,
    hepatic_contributor_rows,
    prepare_hepatotoxicity,
    replay_endpoint,
)


def _raw(rows, endpoint="test", filename="raw.csv"):
    return _with_raw_ids(pd.DataFrame(rows), filename, endpoint)


def test_production_keys_are_exposed_exactly():
    assert PIPELINE_KEYS["in_vitro_inhibition"] == [
        "Compound ID", "Inhibition_pct"
    ]
    assert PIPELINE_KEYS["dose_response"] == [
        "Compound ID", "dosage_nm", "Inhibition_pct"
    ]
    assert "FOB_score" not in PIPELINE_KEYS["neurotoxicity"]
    assert "USPTO ID" not in PIPELINE_KEYS["hepatotoxicity"]


def test_in_vitro_descending_patent_wins_and_context_is_ignored():
    raw = _raw([
        {
            "USPTO ID": "US20200000001A1", "Table Number": 1, "Compound ID": 7,
            "HELM Annotation": np.nan, "Inhibition_pct": 50,
            "dosage_nm": 10, "cell_line": "A",
        },
        {
            "USPTO ID": "US20240000001A1", "Table Number": 9, "Compound ID": 7,
            "HELM Annotation": np.nan, "Inhibition_pct": 50,
            "dosage_nm": 1000, "cell_line": "B",
        },
    ], endpoint="in_vitro_inhibition")
    lineage, groups = replay_endpoint("in_vitro_inhibition", raw)
    assert len(groups) == 1
    assert groups.iloc[0]["winner_patent"] == "US20240000001A1"
    assert set(lineage["disposition"]) == {"kept_processed", "discarded_duplicate"}


def test_dose_response_value_and_dose_are_both_in_key():
    base = {
        "USPTO ID": "US20240000001A1", "Table Number": 1, "Compound ID": 7,
        "HELM Annotation": np.nan,
    }
    raw = _raw([
        {**base, "dosage_nm": 10, "Inhibition_pct": 50},
        {**base, "dosage_nm": 10, "Inhibition_pct": 51},
        {**base, "dosage_nm": 20, "Inhibition_pct": 50},
    ], endpoint="dose_response")
    _, groups = replay_endpoint("dose_response", raw)
    assert len(groups) == 3


def test_neurotoxicity_preserves_input_order_and_ignores_outcome_and_timepoint():
    key = {
        "HELM Annotation": np.nan, "species": "mouse",
        "administration_method": "intrathecal", "tolerability_score_type": "FOB",
        "dosage_ug": 100,
    }
    raw = _raw([
        {**key, "USPTO ID": "US20240000001A1", "Table Number": 1,
         "Compound ID": 7, "FOB_score": "1", "latency_time_hours": 4},
        {**key, "USPTO ID": "US20250000001A1", "Table Number": 2,
         "Compound ID": 7, "FOB_score": "7", "latency_time_hours": 24},
    ], endpoint="neurotoxicity")
    lineage, groups = replay_endpoint("neurotoxicity", raw)
    assert len(groups) == 1
    assert groups.iloc[0]["winner_patent"] == "US20240000001A1"
    assert lineage.iloc[1]["disposition"] == "discarded_duplicate"


def test_hepatic_rows_remain_contributors_and_biomarker_filter_is_visible():
    base = {
        "USPTO ID": "US20240000001A1", "Table Number": 1, "Compound ID": 7,
        "HELM Annotation": np.nan, "dosage": "5 mg/kg", "species": "mouse",
        "num_doses": 2, "dosing_period_days": 7, "measurement_source": "plasma",
        "adminstration_method": "subcutaneous", "ALB": 20, "AST": 30,
        "BUN": 4, "CREA": 5, "TBIL": 6, "PC_ratio": 7,
    }
    raw = _raw([
        {**base, "ALT": 5},
        {**base, "USPTO ID": "US20250000001A1", "Table Number": 2, "ALT": 20},
    ], endpoint="hepatotoxicity")
    lineage, groups = replay_endpoint("hepatotoxicity", raw)
    assert len(groups) == 1
    assert set(lineage["disposition"]) == {"kept_hepatic_contributor"}
    members = hepatic_contributor_rows(lineage)
    alt = members[members["biomarker"] == "ALT"].sort_values("raw_row_id")
    assert alt["contributes_nonnull_value"].tolist() == [False, True]


def test_hepatic_weekly_dose_conversion_matches_production_formula():
    raw = _raw([{
        "USPTO ID": "US20240000001A1", "Table Number": 1, "Compound ID": 7,
        "HELM Annotation": np.nan, "dosage": "14 mg/kg/wk", "species": "mouse",
        "num_doses": 2, "dosing_period_days": 7, "measurement_source": "plasma",
        "adminstration_method": "subcutaneous", "ALB": 20, "ALT": 20,
        "AST": 30, "BUN": 4, "CREA": 5, "TBIL": 6, "PC_ratio": 7,
    }], endpoint="hepatotoxicity")

    prepared, reasons = prepare_hepatotoxicity(raw)

    assert reasons.iloc[0] == ""
    assert prepared.iloc[0]["dosage_mg_per_kg"] == 7.0
