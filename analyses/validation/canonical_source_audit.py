"""Unified scoring against a predeclared production-canonical source scope.

This is the primary scoring layer for the validation audit.  It deliberately
does not infer source scope from Atlas rows.  A caller must first freeze the
one-to-one ``file_to_canonical`` decision made for every verified Gold table by
``patent_collate/src/client.py::_deduplicate_files``.  That production step uses
exact file hashes followed by >=90% table-file similarity, and runs before any
table transcription or Atlas comparison.

Only after that scope is frozen does :func:`score_canonical_source` inspect
Atlas measurements.  Within the scored scope:

* every identity match is a TP;
* every other Gold measurement is an FN, including identity-unresolved Gold;
* every other scoped Atlas measurement is an FP, including identity-unresolved
  Atlas; and
* dose/assay context and outcome correctness are assessed only after the
  one-to-one identity match.

The primary linkage key is fixed to table class, endpoint, channel and compound.
Repeated occurrences retain their multiplicity.  Outcomes maximise multiset
value agreement; dose and assay fields only break ties between assignments
with the same value score.  They cannot change scope, identity cardinality, or
the number of value-correct measurements.
Headline precision and recall require a correct measured value after linkage;
identity-only precision/recall and downstream context correctness are reported
separately.  A stricter all-fields-correct diagnostic is also retained.

``equivalent`` and ``partial`` mappings both admit the declared production
canonical file; the label affects stratification, never admission.  For a
partial table, non-overlapping Gold and Atlas content naturally becomes FN and
FP.  ``absent`` means the Gold table had no valid transcribed counterpart (for
example, formatting failure or a false similarity collapse), so all Gold
content is FN.
``unresolved`` mappings are a validation-stage gap and are kept outside the
primary confusion matrix; their presence makes ``primary_status`` incomplete.

The downstream ``canonical_link`` representation does not restore duplicate
provenance: ``patent_collate/cli.py::run_step3`` filters to entries whose link is
false, so raw collations name only the selected canonical table files.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from analyses.validation import gold_standard_audit as legacy
from analyses.validation.family_matcher import (
    EquivalenceClass,
    IdentityRule,
    MatchResult,
    Measurement,
    TableEquivalenceScope,
    TableRef,
    match_measurements,
)


SCORED_RELATIONSHIPS = {"equivalent", "partial", "absent"}
VALID_RELATIONSHIPS = SCORED_RELATIONSHIPS | {"unresolved"}
REQUIRED_COLUMNS = (
    "mapping_id",
    "family_id",
    "gold_patent",
    "gold_table",
    "canonical_patent",
    "canonical_table",
    "relationship",
    "validation_stage",
    "scope_fixed_before_atlas",
    "outcome_blinded",
    "evidence",
)


def _table(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return str(int(float(text))) if text else ""


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


@dataclass(frozen=True)
class CanonicalSourceMapping:
    """A Gold-to-production source decision made before Atlas comparison."""

    mapping_id: str
    family_id: str
    gold_table: TableRef
    canonical_table: TableRef | None
    relationship: str
    validation_stage: str
    scope_fixed_before_atlas: bool
    outcome_blinded: bool
    evidence: str

    def __post_init__(self) -> None:
        if not self.mapping_id.strip():
            raise ValueError("canonical source mapping_id is required")
        if not self.family_id.strip():
            raise ValueError(f"{self.mapping_id}: family_id is required")
        if self.relationship not in VALID_RELATIONSHIPS:
            raise ValueError(
                f"{self.mapping_id}: invalid relationship {self.relationship!r}"
            )
        if not self.validation_stage.strip():
            raise ValueError(f"{self.mapping_id}: validation_stage is required")
        if not self.scope_fixed_before_atlas:
            raise ValueError(
                f"{self.mapping_id}: scope must be fixed before Atlas comparison"
            )
        if not self.evidence.strip():
            raise ValueError(f"{self.mapping_id}: evidence is required")
        if self.relationship in {"equivalent", "partial"}:
            if self.canonical_table is None:
                raise ValueError(
                    f"{self.mapping_id}: {self.relationship} requires a canonical table"
                )
            # The production file selection may inspect the complete XML,
            # including outcome cells.  That is acceptable here because the
            # selection is the pre-existing system under audit and is frozen
            # before Atlas comparison.  `outcome_blinded` records whether a
            # later equivalent/partial label was assigned blindly; it does not
            # change which canonical table enters the unified scope.
        if self.relationship == "absent" and self.canonical_table is not None:
            raise ValueError(
                f"{self.mapping_id}: absent mapping cannot name a canonical table"
            )


@dataclass(frozen=True)
class FrozenCanonicalScope:
    """Validated scope that can be reused without consulting Atlas content."""

    mappings: tuple[CanonicalSourceMapping, ...]
    table_scope: TableEquivalenceScope
    by_gold_table: Mapping[TableRef, CanonicalSourceMapping]
    by_class: Mapping[EquivalenceClass, tuple[CanonicalSourceMapping, ...]]
    canonical_tables: frozenset[TableRef]
    unresolved_canonical_tables: frozenset[TableRef]


@dataclass(frozen=True)
class CanonicalScore:
    report: Mapping[str, Any]
    matches: MatchResult


def load_canonical_source_mappings(path: Path) -> tuple[CanonicalSourceMapping, ...]:
    """Load and validate a frozen Gold-to-production table mapping CSV."""

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"canonical source map missing columns: {sorted(missing)}")
        mappings = []
        for line_number, row in enumerate(reader, 2):
            if not any(str(value).strip() for value in row.values()):
                continue
            canonical_table = None
            if row["canonical_patent"].strip() or row["canonical_table"].strip():
                if not row["canonical_patent"].strip() or not row["canonical_table"].strip():
                    raise ValueError(
                        f"canonical source map line {line_number}: canonical patent/table "
                        "must be stated together"
                    )
                canonical_table = TableRef(
                    row["canonical_patent"].strip(), _table(row["canonical_table"])
                )
            mappings.append(CanonicalSourceMapping(
                mapping_id=row["mapping_id"].strip(),
                family_id=row["family_id"].strip(),
                gold_table=TableRef(
                    row["gold_patent"].strip(), _table(row["gold_table"])
                ),
                canonical_table=canonical_table,
                relationship=row["relationship"].strip().casefold(),
                validation_stage=row["validation_stage"].strip(),
                scope_fixed_before_atlas=_truth(row["scope_fixed_before_atlas"]),
                outcome_blinded=_truth(row["outcome_blinded"]),
                evidence=row["evidence"].strip(),
            ))
    return tuple(mappings)


def freeze_canonical_scope(
    mappings: Iterable[CanonicalSourceMapping],
) -> FrozenCanonicalScope:
    """Freeze source scope without reading Atlas measurements."""

    ordered = tuple(sorted(mappings, key=lambda item: item.mapping_id))
    ids = Counter(item.mapping_id for item in ordered)
    duplicate_ids = sorted(key for key, count in ids.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate canonical mapping IDs: {duplicate_ids[:5]}")

    by_gold_lists: dict[TableRef, list[CanonicalSourceMapping]] = defaultdict(list)
    memberships: dict[TableRef, EquivalenceClass] = {}
    by_class: dict[EquivalenceClass, list[CanonicalSourceMapping]] = defaultdict(list)
    canonical_tables: set[TableRef] = set()
    unresolved_canonical_tables: set[TableRef] = set()

    for item in ordered:
        by_gold_lists[item.gold_table].append(item)
        if item.relationship == "unresolved" and item.canonical_table is not None:
            unresolved_canonical_tables.add(item.canonical_table)

    for gold_table, items in by_gold_lists.items():
        # Production file_to_canonical is a function: each input file has one
        # and only one selected canonical file.  Multiple Gold files may share
        # that canonical target, but one Gold file cannot have several targets.
        if len(items) != 1:
            raise ValueError(
                f"{gold_table}: expected one production file_to_canonical mapping, "
                f"found {len(items)}"
            )

    # Several duplicate Gold files can share one canonical target.  Build
    # connected components so one-to-one matching is global over that canonical
    # file rather than repeated independently for each duplicate source.
    blocked_gold = {
        gold_table for gold_table, items in by_gold_lists.items()
        if any(item.relationship == "unresolved" for item in items)
    }
    active = [
        item for item in ordered
        if item.gold_table not in blocked_gold
        and item.relationship in SCORED_RELATIONSHIPS
    ]
    parent: dict[TableRef, TableRef] = {}

    def find(item: TableRef) -> TableRef:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: TableRef, right: TableRef) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            parent[second] = first

    for item in active:
        find(item.gold_table)
        if item.canonical_table is not None:
            union(item.gold_table, item.canonical_table)

    component_items: dict[TableRef, list[CanonicalSourceMapping]] = defaultdict(list)
    component_refs: dict[TableRef, set[TableRef]] = defaultdict(set)
    for item in active:
        root = find(item.gold_table)
        component_items[root].append(item)
        component_refs[root].add(item.gold_table)
        if item.canonical_table is not None:
            component_refs[root].add(item.canonical_table)

    for root, items in component_items.items():
        families = {item.family_id for item in items}
        # Production file deduplication is constrained by table number and file
        # similarity, not patent family.  A shared canonical file can therefore
        # connect Gold tables carrying different reporting-family labels.  Keep
        # that component intact and label its mixed-family provenance.
        family_label = (
            next(iter(families))
            if len(families) == 1
            else "mixed:" + "|".join(sorted(families))
        )
        refs = sorted(component_refs[root])
        label = "+".join(f"{ref.patent_id}:t{ref.table_id}" for ref in refs)
        table_class = EquivalenceClass(family_label, f"production:{label}")
        for ref in refs:
            previous = memberships.setdefault(ref, table_class)
            if previous != table_class:
                raise AssertionError(f"component construction conflict for {ref}")
        by_class[table_class].extend(items)
        canonical_tables.update(
            item.canonical_table for item in items if item.canonical_table is not None
        )

    return FrozenCanonicalScope(
        mappings=ordered,
        table_scope=TableEquivalenceScope(memberships),
        by_gold_table={key: value[0] for key, value in by_gold_lists.items()},
        by_class={key: tuple(value) for key, value in by_class.items()},
        canonical_tables=frozenset(canonical_tables),
        unresolved_canonical_tables=frozenset(unresolved_canonical_tables),
    )


def _pct(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def _ci95(numerator: int, denominator: int) -> dict[str, float] | None:
    """Wilson 95% interval for a reported binomial proportion."""
    if not denominator:
        return None
    _, low, high = legacy.wilson(numerator, denominator)
    return {"low_pct": round(low, 2), "high_pct": round(high, 2)}


def _class_label(scope: FrozenCanonicalScope, record: Measurement) -> EquivalenceClass:
    table_class = scope.table_scope.class_for(record.table_ref)
    if table_class is None:
        raise ValueError(f"record outside frozen canonical scope: {record.table_ref}")
    return table_class


def _class_relationship(scope: FrozenCanonicalScope, table_class: EquivalenceClass) -> str:
    relationships = {item.relationship for item in scope.by_class[table_class]}
    if relationships == {"absent"}:
        return "absent"
    if "absent" in relationships:
        raise ValueError(
            f"production class mixes absent and present mappings: {table_class}"
        )
    # If one production table contains only a subset of any mapped Gold table,
    # the unified class must be treated as partial.
    return "partial" if "partial" in relationships else "equivalent"


def _class_stage(scope: FrozenCanonicalScope, table_class: EquivalenceClass) -> str:
    stages = sorted({item.validation_stage for item in scope.by_class[table_class]})
    return stages[0] if len(stages) == 1 else "mixed:" + "|".join(stages)


def _primary_comparison_rules(
    rules: Mapping[str, IdentityRule],
) -> dict[str, IdentityRule]:
    """Make dose/assay fields downstream checks, never primary linkage.

    This conversion is deliberately enforced inside the primary scorer.  It
    accepts older endpoint rules whose dose/context fields were declared as
    ``required_fields``, but moves those fields into ``comparison_fields``.
    The resulting linkage key is therefore always exactly table class,
    endpoint, channel and compound; multiplicity is retained by the matcher.
    """

    primary: dict[str, IdentityRule] = {}
    for endpoint, rule in rules.items():
        comparison_fields = tuple(dict.fromkeys(
            (*rule.required_fields, *rule.comparison_fields)
        ))
        primary[endpoint] = IdentityRule(
            required_fields=(),
            normalizers=rule.normalizers,
            comparison_fields=comparison_fields,
        )
    return primary


def score_canonical_source(
    gold: Iterable[Measurement],
    atlas: Iterable[Measurement],
    *,
    frozen_scope: FrozenCanonicalScope,
    rules: Mapping[str, IdentityRule],
    value_equal=legacy.same_value,
    stratifiers: Mapping[
        str,
        tuple[Callable[[Measurement], Any], Callable[[Measurement], Any]],
    ] | None = None,
) -> CanonicalScore:
    """Score a frozen production-canonical source universe one-to-one.

    Primary linkage is restricted to table class, endpoint, channel, compound
    and occurrence multiplicity.  Fields supplied in either rule field set are
    assessed only after linkage as extraction/normalisation correctness.
    """

    gold_records = tuple(gold)
    atlas_records = tuple(atlas)
    gold_tables = {record.table_ref for record in gold_records}
    missing_mappings = sorted(gold_tables - set(frozen_scope.by_gold_table))
    if missing_mappings:
        raise ValueError(
            "verified Gold tables lack canonical-source mappings: "
            f"{missing_mappings[:5]}"
        )

    unresolved_gold = tuple(
        record for record in gold_records
        if frozen_scope.by_gold_table[record.table_ref].relationship == "unresolved"
    )
    scored_gold = tuple(
        record for record in gold_records
        if frozen_scope.by_gold_table[record.table_ref].relationship
        in SCORED_RELATIONSHIPS
    )
    scoped_atlas = tuple(
        record for record in atlas_records
        if record.table_ref in frozen_scope.canonical_tables
    )
    validation_pending_atlas = tuple(
        record for record in atlas_records
        if record.table_ref in frozen_scope.unresolved_canonical_tables
        and record.table_ref not in frozen_scope.canonical_tables
    )

    primary_rules = _primary_comparison_rules(rules)
    result = match_measurements(
        scored_gold,
        scoped_atlas,
        scope=frozen_scope.table_scope,
        rules=primary_rules,
        value_equal=value_equal,
    )
    summary = result.summary
    true_positive = summary["matched_identity"]
    false_negative = len(scored_gold) - true_positive
    false_positive = len(scoped_atlas) - true_positive
    assert false_negative == summary["missing"] + summary["unresolved_gold"]
    assert false_positive == summary["extra"] + summary["unresolved_atlas"]

    gold_by_id = {record.measurement_id: record for record in scored_gold}
    atlas_by_id = {record.measurement_id: record for record in scoped_atlas}
    pair_by_gold = {pair.gold_id: pair for pair in result.pairs}

    def relation_for_gold(record: Measurement) -> str:
        return _class_relationship(frozen_scope, _class_label(frozen_scope, record))

    def stage_for_gold(record: Measurement) -> str:
        return _class_stage(frozen_scope, _class_label(frozen_scope, record))

    def relation_for_atlas(record: Measurement) -> str:
        return _class_relationship(frozen_scope, _class_label(frozen_scope, record))

    def stage_for_atlas(record: Measurement) -> str:
        return _class_stage(frozen_scope, _class_label(frozen_scope, record))

    def strata(gold_key, atlas_key) -> dict[str, dict[str, Any]]:
        cells: dict[str, Counter] = defaultdict(Counter)
        for disposition in result.gold:
            record = gold_by_id[disposition.measurement_id]
            key = str(gold_key(record))
            cells[key]["gold"] += 1
            if disposition.status.startswith("matched_"):
                cells[key]["tp"] += 1
                pair = pair_by_gold[disposition.measurement_id]
                if pair.value_correct:
                    cells[key]["value_correct"] += 1
                if pair.context_correct:
                    cells[key]["context_correct"] += 1
                if pair.fully_correct:
                    cells[key]["fully_correct"] += 1
            else:
                cells[key]["fn"] += 1
                if disposition.status == "unresolved":
                    cells[key]["identity_unresolved_gold"] += 1
        for disposition in result.atlas:
            record = atlas_by_id[disposition.measurement_id]
            key = str(atlas_key(record))
            cells[key]["atlas"] += 1
            if not disposition.status.startswith("matched_"):
                cells[key]["fp"] += 1
                if disposition.status == "unresolved":
                    cells[key]["identity_unresolved_atlas"] += 1
        return {
            key: {
                "gold_readouts": cell["gold"],
                "scoped_atlas_readouts": cell["atlas"],
                "true_positive": cell["tp"],
                "false_negative": cell["fn"],
                "false_positive": cell["fp"],
                "identity_unresolved_gold_counted_as_fn": cell[
                    "identity_unresolved_gold"
                ],
                "identity_unresolved_atlas_counted_as_fp": cell[
                    "identity_unresolved_atlas"
                ],
                "precision_pct": _pct(cell["value_correct"], cell["atlas"]),
                "recall_pct": _pct(cell["value_correct"], cell["gold"]),
                "precision_ci95": _ci95(cell["value_correct"], cell["atlas"]),
                "recall_ci95": _ci95(cell["value_correct"], cell["gold"]),
                "fully_correct_measurements": cell["value_correct"],
                "fully_correct_measurement_precision_pct": _pct(
                    cell["value_correct"], cell["atlas"]
                ),
                "fully_correct_measurement_recall_pct": _pct(
                    cell["value_correct"], cell["gold"]
                ),
                "all_fields_correct_measurements": cell["fully_correct"],
                "all_fields_correct_precision_pct": _pct(
                    cell["fully_correct"], cell["atlas"]
                ),
                "all_fields_correct_recall_pct": _pct(
                    cell["fully_correct"], cell["gold"]
                ),
                "identity_precision_pct": _pct(cell["tp"], cell["atlas"]),
                "identity_recall_pct": _pct(cell["tp"], cell["gold"]),
                "context_accuracy_given_identity_match_pct": _pct(
                    cell["context_correct"], cell["tp"]
                ),
                "value_accuracy_given_identity_match_pct": _pct(
                    cell["value_correct"], cell["tp"]
                ),
            }
            for key, cell in sorted(cells.items())
        }

    extra_strata = {
        f"by_{name}": strata(gold_key, atlas_key)
        for name, (gold_key, atlas_key) in (stratifiers or {}).items()
    }

    report = {
        "validation_role": "primary unified production-canonical source audit",
        "scope_frozen_before_atlas_comparison": True,
        "primary_status": "complete" if not unresolved_gold else "incomplete_mapping_validation",
        "scored_relationships": sorted(SCORED_RELATIONSHIPS),
        "gold_readouts_scored": len(scored_gold),
        "scoped_atlas_readouts": len(scoped_atlas),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        # Requested headline measurement rates use correct extracted values
        # over the unconditional scoped denominators.  Identity and context
        # rates remain separate diagnostics below.
        "precision_pct": _pct(summary["value_correct"], len(scoped_atlas)),
        "recall_pct": _pct(summary["value_correct"], len(scored_gold)),
        "precision_ci95": _ci95(summary["value_correct"], len(scoped_atlas)),
        "recall_ci95": _ci95(summary["value_correct"], len(scored_gold)),
        "fully_correct_measurements": summary["value_correct"],
        "fully_correct_measurement_precision_pct": _pct(
            summary["value_correct"], len(scoped_atlas)
        ),
        "fully_correct_measurement_recall_pct": _pct(
            summary["value_correct"], len(scored_gold)
        ),
        "all_fields_correct_measurements": summary["fully_correct"],
        "all_fields_correct_precision_pct": _pct(
            summary["fully_correct"], len(scoped_atlas)
        ),
        "all_fields_correct_recall_pct": _pct(
            summary["fully_correct"], len(scored_gold)
        ),
        "identity_precision_pct": _pct(true_positive, len(scoped_atlas)),
        "identity_recall_pct": _pct(true_positive, len(scored_gold)),
        "value_correct_given_identity_match": summary["value_correct"],
        "value_incorrect_given_identity_match": summary["value_incorrect"],
        "value_accuracy_given_identity_match_pct": _pct(
            summary["value_correct"], true_positive
        ),
        "context_correct_given_identity_match": summary["context_correct"],
        "context_incorrect_given_identity_match": summary["context_incorrect"],
        "context_accuracy_given_identity_match_pct": _pct(
            summary["context_correct"], true_positive
        ),
        "identity_unresolved_gold_counted_as_fn": summary["unresolved_gold"],
        "identity_unresolved_atlas_counted_as_fp": summary["unresolved_atlas"],
        "validation_unresolved_gold_excluded": len(unresolved_gold),
        "validation_unresolved_gold_tables": len({x.table_ref for x in unresolved_gold}),
        "atlas_readouts_excluded_pending_source_validation": len(
            validation_pending_atlas
        ),
        "by_relationship": strata(relation_for_gold, relation_for_atlas),
        "by_validation_stage": strata(stage_for_gold, stage_for_atlas),
        "by_endpoint": strata(lambda record: record.endpoint, lambda record: record.endpoint),
        "by_family": strata(
            lambda record: _class_label(frozen_scope, record).family_id,
            lambda record: _class_label(frozen_scope, record).family_id,
        ),
        **extra_strata,
        "definitions": {
            "primary_linkage": (
                "table class + endpoint + channel + compound + retained multiplicity only"
            ),
            "tp": "one-to-one identity match under primary linkage",
            "fn": "every other Gold readout in equivalent/partial/absent mapped tables",
            "fp": "every other Atlas readout in declared production-canonical tables",
            "precision_pct": (
                "identity-paired, value-correct measurements / all scoped Atlas measurements"
            ),
            "recall_pct": (
                "identity-paired, value-correct measurements / all scored Gold measurements"
            ),
            "fully_correct": (
                "identity-paired measurement whose outcome passes legacy.same_value"
            ),
            "all_fields_correct": (
                "value-correct pair with all declared downstream dose/assay fields correct"
            ),
            "identity_precision_pct": "identity pairs / all scoped Atlas measurements",
            "identity_recall_pct": "identity pairs / all scored Gold measurements",
            "outcome_comparator": "legacy.same_value (scalar, text, list and numpy/vector aware)",
            "partial": "fixed production canonical file is admitted; non-overlap becomes FN/FP",
            "absent": (
                "Gold table had no valid transcribed canonical counterpart (input/formatting "
                "failure or false similarity collapse); all Gold readouts become FN"
            ),
            "unresolved": "source relationship validation incomplete; excluded and blocks complete status",
            "scope_source": (
                "patent_collate _deduplicate_files file_to_canonical, before transcription"
            ),
            "downstream_provenance": (
                "raw collations contain canonical files only; canonical_link duplicates "
                "are filtered before step-3 collation"
            ),
        },
    }
    return CanonicalScore(report=report, matches=result)
