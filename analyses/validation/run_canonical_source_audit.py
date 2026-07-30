"""Run the primary ASO Atlas 2.0 Gold validation in production table scope.

The table scope is loaded from the retained pre-transcription
``canonical_link`` ledger.  Scope is therefore fixed before released Atlas
rows are read.  Rows link only by canonical table, endpoint, channel, compound
and multiplicity; dose and assay context are downstream correctness checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.validation import gold_standard_audit as legacy
from analyses.validation.canonical_source_audit import (
    freeze_canonical_scope,
    load_canonical_source_mappings,
    score_canonical_source,
)
from analyses.validation.measurement_records import (
    RELEASE,
    RULES,
    atlas_measurements,
    gold_measurements,
)
from analyses.validation.validation_strata import build_validation_strata


DEFAULT_MAP = ROOT / "data" / "validation" / "production_table_canonical_map.csv"
DEFAULT_JSON = ROOT / "data" / "results" / "canonical_source_audit.json"
DEFAULT_LEDGER = ROOT / "data" / "validation" / "canonical_source_match_ledger.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "verified_measurement_tables": len(rows),
        "by_production_dedup_category": dict(sorted(Counter(
            row["production_dedup_category"] for row in rows
        ).items())),
        "by_relationship": dict(sorted(Counter(
            row["relationship"] for row in rows
        ).items())),
        "mapping_sha256": sha256(path),
    }


def match_ledger(score, gold, atlas) -> list[dict[str, Any]]:
    records = {record.measurement_id: record for record in (*gold, *atlas)}
    pairs = {}
    for pair in score.matches.pairs:
        pairs[pair.gold_id] = pair
        pairs[pair.atlas_id] = pair

    rows = []
    for disposition in (*score.matches.gold, *score.matches.atlas):
        record = records[disposition.measurement_id]
        pair = pairs.get(disposition.measurement_id)
        rows.append({
            "side": disposition.side,
            "measurement_id": disposition.measurement_id,
            "patent": record.patent_id,
            "table": record.table_id,
            "endpoint": record.endpoint,
            "channel": record.channel,
            "compound": record.compound_id,
            "identity_json": json.dumps(record.identity, default=str, sort_keys=True),
            "outcome": str(record.outcome),
            "status": disposition.status,
            "counterpart_id": disposition.counterpart_id or "",
            "reason": disposition.reason or "",
            "value_correct": "" if pair is None else str(pair.value_correct).lower(),
            "context_correct": "" if pair is None else str(pair.context_correct).lower(),
            "all_fields_correct": "" if pair is None else str(pair.fully_correct).lower(),
        })
    return rows


def run(
    mapping_path: Path = DEFAULT_MAP,
    release: Path = RELEASE,
    gold_runs: Path = legacy.GOLD_RUNS,
    out_json: Path = DEFAULT_JSON,
    out_ledger: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    mappings = load_canonical_source_mappings(mapping_path)
    frozen = freeze_canonical_scope(mappings)
    gold = gold_measurements(gold_runs)
    atlas = atlas_measurements(release)
    validation_strata = build_validation_strata(frozen, gold_runs)
    score = score_canonical_source(
        gold,
        atlas,
        frozen_scope=frozen,
        rules=RULES,
        stratifiers=validation_strata.stratifiers,
    )

    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    report = {
        "method": {
            "scope": (
                "retained production table-level canonical_link decisions, fixed before "
                "reading Atlas rows"
            ),
            "linkage": (
                "canonical table + endpoint + channel + compound + occurrence multiplicity"
            ),
            "outcome_and_context": (
                "outcome, dose and assay context are evaluated only after one-to-one linkage"
            ),
            "precision_denominator": "all Atlas readouts in the frozen canonical tables",
            "recall_denominator": "all 22,000 verified Gold readouts",
        },
        "versions": {
            "release_version": manifest.get("version"),
            "release_commit": manifest.get("commit"),
        },
        "table_mapping": mapping_summary(mapping_path),
        "stratification": validation_strata.metadata,
        **score.report,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_ledger.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    rows = match_ledger(score, gold, atlas)
    with out_ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--gold-runs", type=Path, default=legacy.GOLD_RUNS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    report = run(
        mapping_path=args.mapping,
        release=args.release,
        gold_runs=args.gold_runs,
        out_json=args.out_json,
        out_ledger=args.out_ledger,
    )
    print(json.dumps({
        key: report[key] for key in (
            "primary_status", "gold_readouts_scored", "scoped_atlas_readouts",
            "precision_pct", "precision_ci95", "recall_pct", "recall_ci95",
            "identity_precision_pct", "identity_recall_pct",
            "value_accuracy_given_identity_match_pct",
            "context_accuracy_given_identity_match_pct",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
