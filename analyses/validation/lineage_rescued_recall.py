"""Measure content recall by following the production deduplication lineage.

For each verified Gold readout, this audit first finds the corresponding raw
collation content only within its predeclared production table-equivalence
class.  It then follows that raw row's actual pipeline group to the released
winner and verifies that the outcome survives there.  A value found elsewhere
without this lineage is never credited.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.validation import dedup_lineage
from analyses.validation import gold_standard_audit as legacy
from analyses.validation.canonical_source_audit import (
    freeze_canonical_scope,
    load_canonical_source_mappings,
)
from analyses.validation.family_matcher import TableRef
from analyses.validation.measurement_records import gold_measurements
from analyses.validation.validation_strata import build_validation_strata


DEFAULT_MAP = ROOT / "data" / "validation" / "production_table_canonical_map.csv"
DEFAULT_LINEAGE = ROOT / "data" / "validation" / "dedup_lineage"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "lineage_rescued_recall.json"
DEFAULT_LEDGER = ROOT / "data" / "validation" / "lineage_rescued_recall.csv"


def _table_ref(patent: Any, table: Any) -> TableRef | None:
    if patent is None or table is None or pd.isna(patent) or pd.isna(table):
        return None
    return TableRef(str(patent), str(int(float(table))))


def _class_id(table_class) -> str:
    return f"{table_class.family_id}\u241f{table_class.table_class_id}"


def _cid(value: Any) -> str | None:
    parsed = legacy.cid_of(value)
    return str(parsed) if parsed is not None else None


def _contains(container: Any, wanted: Any) -> bool:
    if legacy.same_value(container, wanted):
        return True
    values = legacy.as_vector(container)
    if values is not None and legacy.as_vector(wanted) is None:
        return any(legacy.same_value(value, wanted) for value in values)
    return False


def _dose_matches(gold_dose: Any, raw_dose: Any, required: bool) -> bool:
    if not required:
        return True
    left, right = legacy.as_scalar(gold_dose), legacy.as_scalar(raw_dose)
    return left is not None and right is not None and legacy.same_value(left, right)


def _release_values() -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for endpoint in dedup_lineage.PIPELINE_KEYS:
        release = dedup_lineage._release_rows(endpoint)
        keys = dedup_lineage.PIPELINE_KEYS[endpoint]
        release["pipeline_key_json"] = release.apply(
            lambda row: dedup_lineage._key_json(row, keys), axis=1
        )
        release["pipeline_group_id"] = release["pipeline_key_json"].map(
            lambda key: dedup_lineage._group_id(endpoint, key)
        )
        for _, row in release.iterrows():
            if endpoint in {"in_vitro_inhibition", "dose_response"}:
                content = {"Inhibition_pct": row.get("Inhibition_pct")}
            elif endpoint == "neurotoxicity":
                content = {"FOB_score": row.get("FOB_score")}
            else:
                content = {channel: row.get(channel) for channel in dedup_lineage.BIOMARKERS}
            values[str(row["pipeline_group_id"])] = content
    return values


def _raw_outcome(row: dict[str, Any], gold: dict[str, Any]) -> Any:
    if gold["klass"] == "inhibition":
        return row.get("Inhibition_pct")
    if gold["klass"] == "neurotox":
        return row.get("FOB_score")
    return row.get(gold["channel"])


def _release_outcome(
    release_values: dict[str, dict[str, Any]], row: dict[str, Any], gold: dict[str, Any]
) -> Any:
    content = release_values.get(str(row.get("pipeline_group_id")), {})
    channel = (
        "Inhibition_pct" if gold["klass"] == "inhibition"
        else "FOB_score" if gold["klass"] == "neurotox"
        else gold["channel"]
    )
    return content.get(channel)


def _endpoint_candidates(klass: str) -> tuple[str, ...]:
    return {
        "inhibition": ("in_vitro_inhibition", "dose_response"),
        "neurotox": ("neurotoxicity",),
        "hepatotox": ("hepatotoxicity",),
    }[klass]


def _assay(gold: dict[str, Any]) -> str:
    if gold["klass"] == "inhibition":
        return "inhibition"
    if gold["klass"] == "neurotox":
        return "FOB score"
    return "ALT" if gold["channel"] == "ALT" else "other hepatic biomarker"


def _rate(successes: int, total: int) -> dict[str, Any]:
    _, low, high = legacy.wilson(successes, total)
    return {
        "recovered": successes,
        "gold_readouts": total,
        "recall_pct": round(100 * successes / total, 2) if total else None,
        "ci95": {"low_pct": round(low, 2), "high_pct": round(high, 2)},
    }


def run(
    mapping_path: Path = DEFAULT_MAP,
    lineage_dir: Path = DEFAULT_LINEAGE,
    output: Path = DEFAULT_OUTPUT,
    ledger: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    mappings = load_canonical_source_mappings(mapping_path)
    frozen = freeze_canonical_scope(mappings)
    legacy.GOLD_RUNS = legacy.GOLD_RUNS
    gold, _, _ = legacy.load_gold()
    gold_measurement_rows = gold_measurements(legacy.GOLD_RUNS)
    if len(gold) != len(gold_measurement_rows):
        raise AssertionError("Gold record representations have different lengths")
    validation_strata = build_validation_strata(frozen, legacy.GOLD_RUNS)

    raw = pd.read_parquet(lineage_dir / "raw_row_lineage.parquet")
    release_values = _release_values()

    ref_to_class = {
        ref: _class_id(table_class)
        for ref, table_class in frozen.table_scope.memberships.items()
    }
    raw_records: list[dict[str, Any]] = []
    raw_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw.to_dict("records"):
        ref = _table_ref(row.get("source_patent"), row.get("source_table"))
        class_id = ref_to_class.get(ref)
        compound = _cid(row.get("compound_id"))
        if class_id is None or compound is None:
            continue
        row["_class_id"] = class_id
        row["_compound"] = compound
        raw_records.append(row)
        raw_index[(class_id, str(row["endpoint"]), compound)].append(row)

    dispositions = []
    for index, record in enumerate(gold):
        measurement = gold_measurement_rows[index]
        if (
            measurement.patent_id != record["pid"]
            or int(measurement.table_id) != int(record["table"])
            or str(measurement.compound_id) != str(record["cid"])
            or measurement.channel != record["channel"]
        ):
            raise AssertionError(f"Gold record order diverged at index {index}")
        ref = TableRef(record["pid"], str(int(record["table"])))
        table_class = frozen.table_scope.class_for(ref)
        class_id = _class_id(table_class) if table_class is not None else None
        candidates = []
        if class_id is not None:
            for endpoint in _endpoint_candidates(record["klass"]):
                candidates.extend(raw_index.get((class_id, endpoint, str(record["cid"])), []))

        exact = []
        for row in candidates:
            require_dose = row["endpoint"] == "dose_response"
            if not _dose_matches(record.get("dose"), row.get("dosage_nm"), require_dose):
                continue
            if _contains(_raw_outcome(row, record), record["value"]):
                exact.append(row)

        surviving = [
            row for row in exact
            if bool(row.get("eligible_for_pipeline_group"))
            and bool(row.get("in_release"))
            and _contains(_release_outcome(release_values, row, record), record["value"])
        ]
        direct = [
            row for row in surviving
            if bool(row.get("is_pipeline_representative"))
        ]
        if direct:
            status, chosen = "retained_direct", direct[0]
        elif surviving:
            status, chosen = "lineage_rescued_deduplicated", surviving[0]
        elif exact and all(not bool(row.get("eligible_for_pipeline_group")) for row in exact):
            status, chosen = "filtered_before_deduplication", exact[0]
        elif exact and any(bool(row.get("in_release")) for row in exact):
            status, chosen = "content_not_preserved_in_winner", exact[0]
        elif exact:
            status, chosen = "eligible_group_not_released", exact[0]
        elif candidates:
            status, chosen = "raw_row_found_but_content_differs", candidates[0]
        else:
            status, chosen = "no_raw_collation_lineage", None

        row = {
            "gold_id": f"gold:{index}",
            "gold_patent": record["pid"],
            "gold_table": int(record["table"]),
            "endpoint": record["klass"],
            "assay_readout": _assay(record),
            "channel": record["channel"],
            "compound": record["cid"],
            "dose": record.get("dose"),
            "outcome": record["value"],
            "status": status,
            "raw_row_id": "" if chosen is None else chosen.get("raw_row_id", ""),
            "pipeline_group_id": "" if chosen is None else chosen.get("pipeline_group_id", ""),
            "winner_patent": "" if chosen is None else chosen.get("winner_patent", ""),
            "winner_table": "" if chosen is None else chosen.get("winner_table", ""),
            "release_row_id": "" if chosen is None else chosen.get("release_row_id", ""),
        }
        for name, (gold_label, _) in validation_strata.stratifiers.items():
            row[name] = gold_label(measurement)
        dispositions.append(row)

    recovered_statuses = {"retained_direct", "lineage_rescued_deduplicated"}
    recovered = sum(row["status"] in recovered_statuses for row in dispositions)

    def stratified(field: str) -> dict[str, Any]:
        result = {}
        values = sorted({str(row[field]) for row in dispositions})
        for value in values:
            subset = [row for row in dispositions if str(row[field]) == value]
            result[value] = _rate(
                sum(row["status"] in recovered_statuses for row in subset), len(subset)
            )
        return result

    report = {
        "status": "complete",
        "estimand": (
            "fraction of verified Gold readout occurrences whose content reaches a released "
            "row through an exact source/canonical raw-collation and production-group lineage"
        ),
        "matching_guardrail": (
            "content is searched only inside the Gold table's frozen production-equivalence "
            "class; unrelated release-wide value matches are never credited"
        ),
        "overall": _rate(recovered, len(dispositions)),
        "directly_retained": sum(row["status"] == "retained_direct" for row in dispositions),
        "rescued_through_deduplication": sum(
            row["status"] == "lineage_rescued_deduplicated" for row in dispositions
        ),
        "losses_by_reason": dict(sorted(Counter(
            row["status"] for row in dispositions if row["status"] not in recovered_statuses
        ).items())),
        "by_endpoint": stratified("endpoint"),
        "by_assay_readout": stratified("assay_readout"),
        "by_table_format": stratified("table_format"),
        "by_table_layout": stratified("table_layout"),
        "by_unit_conversion": stratified("unit_conversion"),
        "by_chemistry_class": stratified("chemistry_class"),
        "gold_readouts_reconciled": len(dispositions),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    pd.DataFrame(dispositions).to_csv(ledger, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--lineage-dir", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    report = run(args.mapping, args.lineage_dir, args.output, args.ledger)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
