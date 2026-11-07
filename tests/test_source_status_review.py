import csv
import xml.etree.ElementTree as ET

import pytest

from analyses.validation.source_status_review import GOLD, run_review, source_records


def test_three_review_states_preserve_source_confirmed_frozen_outcomes(tmp_path):
    report = run_review(output=tmp_path / "review.json")
    assert report["tables_reconciled"] == 3
    assert report["measurements_source_confirmed"] == 225
    assert report["frozen_measurements_changed"] == 0
    assert len(report["other_live_review_tables_excluded"]) == 12
    assert all(not t["chemistry_and_complete_context_signed_off"] for t in report["tables"])


def test_source_reader_retains_footnotes_and_does_not_count_continuation_sites():
    root = ET.fromstring("""<source><tables><row>
      <entry>910000†</entry><entry>N/A</entry><entry>N/A</entry>
      <entry>100</entry><entry>119</entry><entry>ACGT</entry>
      <entry>124</entry><entry>9</entry></row>
      <row><entry/><entry/><entry/><entry>123456</entry><entry>123475</entry>
      <entry/><entry/><entry/></row></tables></source>""")
    rows = source_records(root)
    assert rows == [{"compound": "910000", "sequence": "ACGT",
                     "printed_percent_control": 124.0, "inhibition": -24.0}]


def test_review_rejects_changed_frozen_measurement(tmp_path):
    with GOLD.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    affected = next(row for row in rows if row["side"] == "gold"
                    and row["patent"] == "US20210038631A1" and int(row["table"]) == 1)
    affected["outcome"] = str(float(affected["outcome"]) + 1)
    changed = tmp_path / "changed.csv"
    with changed.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Frozen Gold differs from source"):
        run_review(gold=changed, output=tmp_path / "review.json")
