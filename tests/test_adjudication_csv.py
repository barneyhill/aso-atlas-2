import csv
from pathlib import Path

import pytest

from analyses.validation.canonical_error_adjudication import run


ROOT = Path(__file__).resolve().parents[1]


def test_adjudication_explanations_have_no_truncated_csv_tail():
    path = ROOT / "data/validation/canonical_error_adjudication.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(None not in row and all(v is not None for v in row.values()) for row in rows)
    issue = next(row for row in rows if row["issue_id"] == "C9_INH_T17")
    assert "4,000 nM dose" in issue["adjudication"]


def test_adjudication_rejects_unquoted_excess_columns(tmp_path):
    matches = tmp_path / "matches.csv"
    matches.write_text("patent,table,side,value_correct,context_correct,status\n")
    adjudication = tmp_path / "adjudication.csv"
    adjudication.write_text("issue_id,adjudication\nbad,unquoted,comma\n")
    with pytest.raises(ValueError, match="Malformed adjudication CSV row bad"):
        run(matches, adjudication, tmp_path / "report.json")
