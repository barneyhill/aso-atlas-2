"""Validate and summarize manual XML adjudication of canonical-audit errors.

The manual ledger groups repeated row errors only when one source statement and
one extraction behaviour explain the entire group.  This script makes that
grouping auditable: every represented table with a wrong matched value, strict
context mismatch, or scoped Atlas extra must occur exactly once, and the
grouped counts must reproduce the machine match ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHES = ROOT / "data" / "validation" / "canonical_source_match_ledger.csv"
DEFAULT_ADJUDICATION = ROOT / "data" / "validation" / "canonical_error_adjudication.csv"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "canonical_error_adjudication.json"


def table_ref(row: dict[str, str]) -> str:
    return f"{row['patent']}:{int(row['table'])}"


def observed_counts(path: Path) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ref = table_ref(row)
            if row["side"] == "gold" and row["value_correct"] == "false":
                counts[ref]["matched_value_errors"] += 1
            if row["side"] == "gold" and row["context_correct"] == "false":
                counts[ref]["matched_context_errors"] += 1
            if row["side"] == "atlas" and row["status"] == "extra":
                counts[ref]["atlas_extras"] += 1
    return counts


def run(matches: Path, adjudication: Path, output: Path) -> dict:
    observed = observed_counts(matches)
    assigned: dict[str, str] = {}
    totals = Counter()
    verdicts = Counter()
    issues = []

    with adjudication.open(newline="") as handle:
        for row in csv.DictReader(handle):
            refs = row["table_refs"].split(";")
            for ref in refs:
                if ref in assigned:
                    raise ValueError(f"{ref} assigned to both {assigned[ref]} and {row['issue_id']}")
                assigned[ref] = row["issue_id"]
            stated = Counter({
                field: int(row[field])
                for field in ("matched_value_errors", "matched_context_errors", "atlas_extras")
            })
            actual = sum((observed.get(ref, Counter()) for ref in refs), Counter())
            if stated != actual:
                raise ValueError(f"{row['issue_id']}: stated {stated}, observed {actual}")
            totals.update(stated)
            verdicts[row["verdict"]] += 1
            issues.append(row)

    missing = sorted(set(observed) - set(assigned))
    surplus = sorted(set(assigned) - set(observed))
    if missing or surplus:
        raise ValueError(f"adjudication coverage mismatch: missing={missing}, surplus={surplus}")

    report = {
        "status": "complete",
        "scope": (
            "all represented canonical tables containing a matched value error, "
            "strict context mismatch, or scoped Atlas extra"
        ),
        "manual_method": (
            "row discrepancies grouped only by shared source statement and extraction "
            "behaviour, then checked against the original USPTO full-text XML"
        ),
        "tables_adjudicated": len(observed),
        "issue_groups": len(issues),
        "matched_value_errors_adjudicated": totals["matched_value_errors"],
        "matched_context_errors_adjudicated": totals["matched_context_errors"],
        "atlas_extras_adjudicated": totals["atlas_extras"],
        "issue_groups_by_verdict": dict(sorted(verdicts.items())),
        "false_negative_note": (
            "Gold-only missing occurrences are separately covered by the exhaustive XML "
            "cell-grid check and dedup_lineage audit; they are not represented-row error "
            "pairs and are therefore outside this manual discrepancy ledger."
        ),
        "issues": issues,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.matches, args.adjudication, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in {"issues"}}, indent=2))


if __name__ == "__main__":
    main()
