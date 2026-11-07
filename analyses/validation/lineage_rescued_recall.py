"""Audit final-release measurements through deterministic production lineage.

The manually verified patent tables are the source side of the audit.  The
Atlas side is built without consulting their outcomes: every final-release
measurement in a production group reached from a mapped canonical table is
included.  Table-level duplicate mappings are unfolded so each audited patent
table is evaluated against the canonical table that production selected for
it.  Where several source tables are merged into one release group, each
table is credited only with the values its own source rows contributed.

Patent and Atlas measurements are paired one-to-one within table mapping,
endpoint, channel and compound.  Repeated indistinguishable occurrences are
scored by multiset agreement after their lineage-defined cardinality is fixed.
The resulting precision and recall therefore use the same population without
using a measured value to establish scope, identity or physical-row lineage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.validation import dedup_lineage
from analyses.validation import gold_standard_audit as legacy
from analyses.validation.canonical_source_audit import (
    FrozenCanonicalScope,
    _primary_comparison_rules,
    freeze_canonical_scope,
    load_canonical_source_mappings,
)
from analyses.validation.family_matcher import (
    MatchedPair,
    Measurement,
    RecordDisposition,
    TableRef,
    match_measurements,
)
from analyses.validation.measurement_records import RULES
from analyses.validation.validation_strata import build_validation_strata


DEFAULT_MAP = ROOT / "data" / "validation" / "production_table_canonical_map.csv"
DEFAULT_LINEAGE = ROOT / "data" / "validation" / "dedup_lineage"
DEFAULT_GOLD_LEDGER = ROOT / "data" / "validation" / "canonical_source_match_ledger.csv"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "lineage_rescued_recall.json"
DEFAULT_LEDGER = ROOT / "data" / "validation" / "lineage_rescued_recall.csv"


@dataclass(frozen=True)
class LiftedMeasurement:
    """One actual release measurement reached from a canonical source table."""

    measurement: Measurement
    raw_row_ids: tuple[str, ...]
    pipeline_group_id: str
    release_row_id: str
    release_endpoint: str


def load_frozen_gold(path: Path) -> list[Measurement]:
    """Load the versioned 22,000-measurement corpus, not mutable work files."""

    measurements = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["side"] != "gold":
                continue
            measurements.append(Measurement(
                measurement_id=row["measurement_id"],
                patent_id=row["patent"],
                table_id=str(int(float(row["table"]))),
                endpoint=row["endpoint"],
                channel=row["channel"],
                compound_id=row["compound"],
                identity=json.loads(row["identity_json"]),
                outcome=row["outcome"],
            ))
    if len(measurements) != 22_000:
        raise ValueError(
            f"frozen Gold ledger contains {len(measurements):,} measurements, "
            "expected 22,000"
        )
    return measurements


def _table_ref(patent: Any, table: Any) -> TableRef | None:
    if patent is None or table is None or pd.isna(patent) or pd.isna(table):
        return None
    return TableRef(str(patent), str(int(float(table))))


def _compound(value: Any) -> str | None:
    parsed = legacy.cid_of(value)
    return str(parsed) if parsed is not None else None


def _logical_values(value: Any, *, vector_is_one: bool = False) -> list[Any]:
    """Return logical measurement outcomes at the paper's stated grain."""

    vector = legacy.as_vector(value)
    if vector_is_one:
        if vector is not None:
            return [value]
        return [value] if legacy.as_scalar(value) is not None else []
    if vector is not None:
        return vector
    scalar = legacy.as_scalar(value)
    return [] if scalar is None else [scalar]


def _release_rows_by_group() -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for endpoint, keys in dedup_lineage.PIPELINE_KEYS.items():
        release = dedup_lineage._release_rows(endpoint)
        release["pipeline_key_json"] = release.apply(
            lambda row: dedup_lineage._key_json(row, keys), axis=1
        )
        release["pipeline_group_id"] = release["pipeline_key_json"].map(
            lambda key: dedup_lineage._group_id(endpoint, key)
        )
        for row in release.to_dict("records"):
            group_id = str(row["pipeline_group_id"])
            if group_id in rows:
                raise AssertionError(f"duplicate release pipeline group: {group_id}")
            rows[group_id] = row
    return rows


def _identity(endpoint: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if endpoint in {"in_vitro_inhibition", "dose_response"}:
        return {
            "dose": legacy.as_scalar(row.get("dosage_nm")),
            "cell_line": row.get("cell_line"),
        }
    if endpoint == "hepatotoxicity":
        return {
            "dose": legacy.as_scalar(row.get("dosage_mg_per_kg")),
            "species": row.get("species"),
            "route": row.get("adminstration_method"),
            "num_doses": row.get("num_doses"),
            "period": row.get("dosing_period_days"),
            "sample": row.get("measurement_source"),
        }
    return {
        "dose": legacy.as_scalar(row.get("dosage_ug")),
        "species": row.get("species"),
        "route": row.get("administration_method"),
        "score_type": row.get("tolerability_score_type"),
        "timepoint": row.get("latency_time_hours"),
    }


def _value_key(value: Any) -> str:
    vector = legacy.as_vector(value)
    if vector is not None:
        value = [float(item) for item in vector]
    else:
        scalar = legacy.as_scalar(value)
        value = scalar if scalar is not None else str(value)
    return json.dumps(value, sort_keys=True, default=str)


def _assert_hepatic_release_values(
    raw: pd.DataFrame,
    releases: Mapping[str, Mapping[str, Any]],
) -> None:
    """Verify that contributor values exactly reconstruct released vectors."""

    expected: dict[tuple[str, str], list[Any]] = defaultdict(list)
    hepatic = raw[
        raw["endpoint"].eq("hepatotoxicity")
        & raw["eligible_for_pipeline_group"].astype(bool)
        & raw["in_release"].astype(bool)
    ]
    for row in hepatic.to_dict("records"):
        group_id = str(row["pipeline_group_id"])
        for channel in dedup_lineage.BIOMARKERS:
            expected[(group_id, channel)].extend(
                _logical_values(row.get(f"_clean_{channel}"))
            )

    for (group_id, channel), values in expected.items():
        observed = _logical_values(releases[group_id].get(channel))
        if Counter(map(_value_key, values)) != Counter(map(_value_key, observed)):
            raise AssertionError(
                f"hepatic lineage does not reconstruct release values: "
                f"{group_id} {channel}"
            )


def build_lifted_measurements(
    raw: pd.DataFrame,
    releases: Mapping[str, Mapping[str, Any]],
) -> list[LiftedMeasurement]:
    """Enumerate actual release measurements in lineage-connected groups."""

    required_context = {
        "cell_line", "species", "species_strain", "administration_method",
        "tolerability_score_type", "latency_time_hours", "dosage_mg_per_kg",
        "num_doses", "dosing_period_days", "measurement_source",
        "adminstration_method",
    }
    missing_context = required_context - set(raw.columns)
    if missing_context:
        raise ValueError(
            "dedup lineage is stale; regenerate it to include context fields: "
            + ", ".join(sorted(missing_context))
        )
    missing_clean = {
        f"_clean_{channel}" for channel in dedup_lineage.BIOMARKERS
    } - set(raw.columns)
    if missing_clean:
        raise ValueError(
            "dedup lineage is stale; regenerate it to include cleaned hepatic values"
        )

    _assert_hepatic_release_values(raw, releases)
    connected: dict[tuple[TableRef, str], dict[str, Any]] = {}
    eligible = raw[
        raw["eligible_for_pipeline_group"].astype(bool)
        & raw["in_release"].astype(bool)
    ]
    for row in eligible.to_dict("records"):
        group_id = str(row["pipeline_group_id"])
        table = _table_ref(row.get("source_patent"), row.get("source_table"))
        if table is None:
            continue
        key = (table, group_id)
        if key not in connected:
            connected[key] = {
                "endpoint": str(row["endpoint"]),
                "raw_row_ids": [],
                "rows": [],
            }
        connected[key]["raw_row_ids"].append(str(row["raw_row_id"]))
        connected[key]["rows"].append(row)

    lifted: list[LiftedMeasurement] = []
    for (table, group_id), provenance in connected.items():
        release = releases.get(group_id)
        if release is None:
            raise AssertionError(f"released lineage group has no release row: {group_id}")
        compound = _compound(release.get("Compound ID"))
        if compound is None:
            continue
        endpoint = str(provenance["endpoint"])
        if endpoint in {"in_vitro_inhibition", "dose_response"}:
            channels = [("inhibition", "Inhibition_pct", [release.get("Inhibition_pct")])]
        elif endpoint == "neurotoxicity":
            channels = [("neurotox", "FOB_score", _logical_values(
                release.get("FOB_score"), vector_is_one=True
            ))]
        else:
            # A release cell lists the values of every source row merged into
            # the group (for example the same table repeated in a sibling
            # patent).  Each audited table is credited only with the values its
            # own source rows contributed, so a second source's copy of a value
            # is not counted as an extra measurement of this table.
            def _contributed(channel):
                values = []
                for contributor in provenance["rows"]:
                    values.extend(_logical_values(contributor.get(f"_clean_{channel}")))
                return values
            channels = [("hepatotox", channel, _contributed(channel)) for channel in dedup_lineage.BIOMARKERS]

        for audit_endpoint, channel, outcomes in channels:
            for occurrence, outcome in enumerate(outcomes):
                if not _logical_values(outcome, vector_is_one=audit_endpoint == "neurotox"):
                    continue
                measurement_id = (
                    f"lifted:{table.patent_id}:{table.table_id}:{group_id}:"
                    f"{channel}:{occurrence:06d}"
                )
                measurement = Measurement(
                    measurement_id=measurement_id,
                    patent_id=table.patent_id,
                    table_id=table.table_id,
                    endpoint=audit_endpoint,
                    channel=channel,
                    compound_id=compound,
                    identity=_identity(endpoint, release),
                    outcome=outcome,
                )
                lifted.append(LiftedMeasurement(
                    measurement=measurement,
                    raw_row_ids=tuple(provenance["raw_row_ids"]),
                    pipeline_group_id=group_id,
                    release_row_id=str(row["release_row_id"]),
                    release_endpoint=endpoint,
                ))
    return lifted


def _clone_for_mapping(item: LiftedMeasurement, mapping_id: str) -> LiftedMeasurement:
    source = item.measurement
    clone = Measurement(
        measurement_id=f"{mapping_id}|{source.measurement_id}",
        patent_id=source.patent_id,
        table_id=source.table_id,
        endpoint=source.endpoint,
        channel=source.channel,
        compound_id=source.compound_id,
        identity=source.identity,
        outcome=source.outcome,
    )
    return LiftedMeasurement(
        measurement=clone,
        raw_row_ids=item.raw_row_ids,
        pipeline_group_id=item.pipeline_group_id,
        release_row_id=item.release_row_id,
        release_endpoint=item.release_endpoint,
    )


def _pct(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(
    gold_dispositions: Iterable[RecordDisposition],
    atlas_dispositions: Iterable[RecordDisposition],
    pairs: Iterable[MatchedPair],
) -> dict[str, Any]:
    gold = tuple(gold_dispositions)
    atlas = tuple(atlas_dispositions)
    pairs = tuple(pairs)
    outcome_correct = sum(pair.value_correct for pair in pairs)
    context_correct = sum(pair.context_correct for pair in pairs)
    all_correct = sum(pair.value_correct and pair.context_correct for pair in pairs)
    matched = len(pairs)
    return {
        "patent_measurements": len(gold),
        "atlas_measurements": len(atlas),
        "identity_matched": matched,
        "missing_patent_measurements": sum(row.status == "missing" for row in gold),
        "extra_atlas_measurements": sum(row.status == "extra" for row in atlas),
        "outcome_correct": outcome_correct,
        "outcome_incorrect": matched - outcome_correct,
        "outcome_precision_pct": _pct(outcome_correct, len(atlas)),
        "outcome_recall_pct": _pct(outcome_correct, len(gold)),
        "context_correct": context_correct,
        "context_incorrect": matched - context_correct,
        "context_accuracy_pct": _pct(context_correct, matched),
        "all_fields_correct": all_correct,
        "all_fields_precision_pct": _pct(all_correct, len(atlas)),
        "all_fields_recall_pct": _pct(all_correct, len(gold)),
    }


def _stratified(
    gold: Mapping[str, Measurement],
    atlas: Mapping[str, Measurement],
    gold_dispositions: Iterable[RecordDisposition],
    atlas_dispositions: Iterable[RecordDisposition],
    pairs: Iterable[MatchedPair],
    gold_label: Callable[[Measurement], Any],
    atlas_label: Callable[[Measurement], Any],
) -> dict[str, dict[str, Any]]:
    pair_by_gold = {pair.gold_id: pair for pair in pairs}
    pair_by_atlas = {pair.atlas_id: pair for pair in pairs}
    grouped_gold: dict[str, list[RecordDisposition]] = defaultdict(list)
    grouped_atlas: dict[str, list[RecordDisposition]] = defaultdict(list)
    grouped_pairs: dict[str, list[MatchedPair]] = defaultdict(list)
    for row in gold_dispositions:
        key = str(gold_label(gold[row.measurement_id]))
        grouped_gold[key].append(row)
        pair = pair_by_gold.get(row.measurement_id)
        if pair is not None:
            grouped_pairs[key].append(pair)
    for row in atlas_dispositions:
        key = str(atlas_label(atlas[row.measurement_id]))
        grouped_atlas[key].append(row)
        pair = pair_by_atlas.get(row.measurement_id)
        if pair is not None:
            gold_key = str(gold_label(gold[pair.gold_id]))
            if key != gold_key:
                raise AssertionError(f"stratum mismatch for matched pair: {gold_key} != {key}")
    keys = sorted(set(grouped_gold) | set(grouped_atlas))
    return {
        key: _summary(grouped_gold[key], grouped_atlas[key], grouped_pairs[key])
        for key in keys
    }


def _ledger_rows(
    records: Mapping[str, Measurement],
    dispositions: Iterable[RecordDisposition],
    pairs: Mapping[str, MatchedPair],
    mapping_ids: Mapping[str, str],
    lifted: Mapping[str, LiftedMeasurement],
    stratifiers: Mapping[
        str,
        tuple[Callable[[Measurement], Any], Callable[[Measurement], Any]],
    ],
) -> list[dict[str, Any]]:
    rows = []
    for disposition in dispositions:
        record = records[disposition.measurement_id]
        pair = pairs.get(disposition.measurement_id)
        item = lifted.get(disposition.measurement_id)
        row = {
            "side": disposition.side,
            "mapping_id": mapping_ids[disposition.measurement_id],
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
            "value_correct": "" if pair is None else str(pair.value_correct).lower(),
            "context_correct": "" if pair is None else str(pair.context_correct).lower(),
            "raw_row_id": "" if item is None else "|".join(item.raw_row_ids),
            "pipeline_group_id": "" if item is None else item.pipeline_group_id,
            "release_row_id": "" if item is None else item.release_row_id,
        }
        for name, (gold_label, atlas_label) in stratifiers.items():
            label = gold_label if disposition.side == "gold" else atlas_label
            row[name] = str(label(record))
        rows.append(row)
    return rows


def run(
    mapping_path: Path = DEFAULT_MAP,
    lineage_dir: Path = DEFAULT_LINEAGE,
    gold_ledger: Path = DEFAULT_GOLD_LEDGER,
    output: Path = DEFAULT_OUTPUT,
    ledger: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    mappings = load_canonical_source_mappings(mapping_path)
    frozen: FrozenCanonicalScope = freeze_canonical_scope(mappings)
    gold_rows = load_frozen_gold(gold_ledger)
    gold_by_table: dict[TableRef, list[Measurement]] = defaultdict(list)
    for measurement in gold_rows:
        gold_by_table[measurement.table_ref].append(measurement)

    raw = pd.read_parquet(lineage_dir / "raw_row_lineage.parquet")
    releases = _release_rows_by_group()
    lifted_rows = build_lifted_measurements(raw, releases)
    lifted_by_table: dict[TableRef, list[LiftedMeasurement]] = defaultdict(list)
    for item in lifted_rows:
        lifted_by_table[item.measurement.table_ref].append(item)

    primary_rules = _primary_comparison_rules(RULES)
    all_gold: dict[str, Measurement] = {}
    all_atlas: dict[str, Measurement] = {}
    all_lifted: dict[str, LiftedMeasurement] = {}
    all_pairs: list[MatchedPair] = []
    gold_dispositions: list[RecordDisposition] = []
    atlas_dispositions: list[RecordDisposition] = []
    mapping_ids: dict[str, str] = {}

    for mapping in mappings:
        gold = list(gold_by_table.get(mapping.gold_table, ()))
        if mapping.relationship == "unresolved":
            raise ValueError(f"unresolved audit mapping: {mapping.mapping_id}")
        source_items = (
            [] if mapping.canonical_table is None
            else lifted_by_table.get(mapping.canonical_table, [])
        )
        mapped_items = [_clone_for_mapping(item, mapping.mapping_id) for item in source_items]
        atlas = [item.measurement for item in mapped_items]
        result = match_measurements(
            gold,
            atlas,
            scope=frozen.table_scope,
            rules=primary_rules,
            value_equal=legacy.same_value,
            pairing="value_then_context",
        )
        for record in gold:
            if record.measurement_id in all_gold:
                raise AssertionError(f"Gold measurement appears in two mappings: {record.measurement_id}")
            all_gold[record.measurement_id] = record
            mapping_ids[record.measurement_id] = mapping.mapping_id
        for item in mapped_items:
            record = item.measurement
            all_atlas[record.measurement_id] = record
            all_lifted[record.measurement_id] = item
            mapping_ids[record.measurement_id] = mapping.mapping_id
        all_pairs.extend(result.pairs)
        gold_dispositions.extend(result.gold)
        atlas_dispositions.extend(result.atlas)

    if len(all_gold) != len(gold_rows):
        raise AssertionError(f"audit mapped {len(all_gold)} of {len(gold_rows)} Gold measurements")

    validation_strata = build_validation_strata(frozen, legacy.GOLD_RUNS)
    release_manifest_path = dedup_lineage.RELEASE / "release_manifest.json"
    release_manifest = (
        json.loads(release_manifest_path.read_text())
        if release_manifest_path.exists() else {}
    )
    report = {
        "status": "complete",
        "inputs": {
            "release_version": release_manifest.get("version"),
            "release_commit": release_manifest.get("commit"),
            "gold_ledger": str(gold_ledger.relative_to(ROOT)),
            "gold_ledger_sha256": _sha256(gold_ledger),
            "table_mapping": str(mapping_path.relative_to(ROOT)),
            "table_mapping_sha256": _sha256(mapping_path),
            "lineage_summary": str(
                (lineage_dir / "summary.json").relative_to(ROOT)
            ),
            "lineage_summary_sha256": _sha256(lineage_dir / "summary.json"),
        },
        "method": {
            "unit": (
                "one logical measurement: one inhibition value, one FOB score vector, "
                "or one biomarker value"
            ),
            "scope": (
                "all measurements from 311 verified patent tables, evaluated against "
                "their fixed production-canonical table mappings"
            ),
            "lineage": (
                "all final-release measurements in production groups reached from each "
                "mapped canonical table are included"
            ),
            "pairing": (
                "one-to-one within mapped table, endpoint, channel and compound; scope and "
                "pairing cardinality are outcome-blind; repeated indistinguishable "
                "occurrences use multiset outcome agreement"
            ),
            "precision_denominator": (
                "all final-release measurement values contributed by the source rows of "
                "the audited table mappings"
            ),
            "recall_denominator": "all 22,000 manually verified patent measurements",
        },
        "table_mapping": {
            "verified_tables": len(mappings),
            "present_canonical_mappings": sum(item.canonical_table is not None for item in mappings),
            "absent_mappings": sum(item.canonical_table is None for item in mappings),
        },
        **_summary(gold_dispositions, atlas_dispositions, all_pairs),
        "by_endpoint": _stratified(
            all_gold, all_atlas, gold_dispositions, atlas_dispositions, all_pairs,
            lambda record: record.endpoint, lambda record: record.endpoint,
        ),
        **{
            f"by_{name}": _stratified(
                all_gold, all_atlas, gold_dispositions, atlas_dispositions, all_pairs,
                gold_label, atlas_label,
            )
            for name, (gold_label, atlas_label) in validation_strata.stratifiers.items()
        },
        "stratification": validation_strata.metadata,
    }

    pair_lookup: dict[str, MatchedPair] = {}
    for pair in all_pairs:
        pair_lookup[pair.gold_id] = pair
        pair_lookup[pair.atlas_id] = pair
    ledger_rows = _ledger_rows(
        {**all_gold, **all_atlas},
        [*gold_dispositions, *atlas_dispositions],
        pair_lookup,
        mapping_ids,
        all_lifted,
        validation_strata.stratifiers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    with ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger_rows[0]))
        writer.writeheader()
        writer.writerows(ledger_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--lineage-dir", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--gold-ledger", type=Path, default=DEFAULT_GOLD_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    report = run(
        args.mapping, args.lineage_dir, args.gold_ledger, args.output, args.ledger
    )
    print(json.dumps({
        key: report[key] for key in (
            "patent_measurements", "atlas_measurements", "identity_matched",
            "outcome_precision_pct", "outcome_recall_pct", "context_accuracy_pct",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
