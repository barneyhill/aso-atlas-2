"""Outcome-blind, family-aware matching for the gold-standard audit.

This module deliberately knows nothing about the released Atlas schemas.  A
caller must first convert both datasets to :class:`Measurement` objects and
supply an adjudicated table-equivalence scope.  Consequently, a shared compound
or outcome value can never move a record into the audit scope.

The matching unit is one logical assay-readout occurrence.  An ordered vector
that the release defines as one readout (for example, one per-animal FOB score
vector) remains one outcome; repeated readouts get distinct ``Measurement``
IDs.  Identical identity keys are retained with their full multiplicity; they
are never deduplicated.  The size of each one-to-one assignment is fixed
without looking at outcomes.  When several observations are indistinguishable
on identity, the reported value correctness is their maximum multiset
agreement, so it does not depend on arbitrary record IDs.  Outcomes can alter
pairing *within* that fixed bucket for value scoring.  Downstream dose/context
fields cannot alter scope, identity pairing cardinality, or value scoring.

``IdentityRule.required_fields`` is intentionally allowed to be empty.  The
primary production-canonical audit links only on table class, endpoint, channel
and compound, retaining multiplicity.  Dose and assay context belong in
``comparison_fields`` there: they are checked after linkage as extraction and
normalisation outputs, rather than conditioning whether a row can match.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, order=True)
class TableRef:
    """A source table as claimed by a measurement record."""

    patent_id: str
    table_id: str


@dataclass(frozen=True, order=True)
class EquivalenceClass:
    """An adjudicated scientific-table class within a formal patent family."""

    family_id: str
    table_class_id: str


@dataclass(frozen=True)
class TableEquivalenceScope:
    """Map source tables to pre-adjudicated family/table equivalence classes.

    Legal family membership alone is insufficient: two tables match only when
    both map to the same ``(family_id, table_class_id)`` pair.
    """

    memberships: Mapping[TableRef, EquivalenceClass]

    def class_for(self, table: TableRef) -> EquivalenceClass | None:
        return self.memberships.get(table)


@dataclass(frozen=True)
class Measurement:
    """One logical assay-readout occurrence.

    ``identity`` contains dose and assay context, while ``outcome`` is the
    measured inhibition, score, or biomarker value.  Which context fields, if
    any, participate in linkage is controlled by :class:`IdentityRule`.
    ``outcome`` is never part of :func:`measurement_identity_key`.
    """

    measurement_id: str
    patent_id: str
    table_id: str
    endpoint: str
    channel: str
    compound_id: Any
    identity: Mapping[str, Any]
    outcome: Any

    @property
    def table_ref(self) -> TableRef:
        return TableRef(str(self.patent_id), str(self.table_id))


IdentityNormalizer = Callable[[Any], Any]


@dataclass(frozen=True)
class IdentityRule:
    """Endpoint-specific linkage and downstream comparison fields.

    ``required_fields`` extend the linkage key and may be empty.  All required
    fields must be stated on both sides; missing values are unresolved, never
    wildcards.  ``comparison_fields`` never create or exclude a pair.  They are
    checked after linkage and contribute to whether the paired measurement is
    fully correct.  Endpoint-specific normalizers can be shared by both sets.
    """

    required_fields: tuple[str, ...] = ()
    normalizers: Mapping[str, IdentityNormalizer] = field(default_factory=dict)
    comparison_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("IdentityRule.required_fields contains duplicates")
        if len(set(self.comparison_fields)) != len(self.comparison_fields):
            raise ValueError("IdentityRule.comparison_fields contains duplicates")


@dataclass(frozen=True)
class MatchedPair:
    gold_id: str
    atlas_id: str
    value_correct: bool
    context_correct: bool = True

    @property
    def fully_correct(self) -> bool:
        return self.value_correct and self.context_correct


@dataclass(frozen=True)
class RecordDisposition:
    measurement_id: str
    side: str
    status: str
    counterpart_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MatchResult:
    """Pairs and exhaustive per-side dispositions.

    Gold records end as ``matched_value_correct``,
    ``matched_value_incorrect``, ``missing``, or ``unresolved``.  Atlas records
    end as either matched status, ``extra``, or ``unresolved``.
    """

    pairs: tuple[MatchedPair, ...]
    gold: tuple[RecordDisposition, ...]
    atlas: tuple[RecordDisposition, ...]

    @property
    def summary(self) -> dict[str, int]:
        gold_counts = Counter(x.status for x in self.gold)
        atlas_counts = Counter(x.status for x in self.atlas)
        return {
            "matched_identity": len(self.pairs),
            "value_correct": sum(x.value_correct for x in self.pairs),
            "value_incorrect": sum(not x.value_correct for x in self.pairs),
            "context_correct": sum(x.context_correct for x in self.pairs),
            "context_incorrect": sum(not x.context_correct for x in self.pairs),
            "fully_correct": sum(x.fully_correct for x in self.pairs),
            "fully_incorrect": sum(not x.fully_correct for x in self.pairs),
            "missing": gold_counts["missing"],
            "extra": atlas_counts["extra"],
            "unresolved_gold": gold_counts["unresolved"],
            "unresolved_atlas": atlas_counts["unresolved"],
        }


_WS = re.compile(r"\s+")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().casefold() in {"na", "n/a", "none", "null"}
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def canonical_identity_value(value: Any) -> Any:
    """Conservative default normalization for an identity field."""

    if isinstance(value, str):
        return _WS.sub(" ", value.strip()).casefold()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # Inputs are expected to have canonical units.  This only makes 5 and
        # 5.0 identical; it does not introduce a fuzzy dose tolerance.
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(canonical_identity_value(v) for v in value)
    return value


def _compound_key(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).casefold()
    if isinstance(value, (int, float)) and not _missing(value):
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)
    return _WS.sub("", str(value).strip()).casefold()


def _record_problem(
    record: Measurement,
    scope: TableEquivalenceScope,
    rules: Mapping[str, IdentityRule],
) -> str | None:
    if not str(record.measurement_id).strip():
        return "missing_measurement_id"
    if _missing(record.endpoint):
        return "missing_endpoint"
    if _missing(record.channel):
        return "missing_channel"
    if _missing(record.compound_id):
        return "missing_compound_id"
    if scope.class_for(record.table_ref) is None:
        return "table_not_in_equivalence_scope"
    rule = rules.get(record.endpoint)
    if rule is None:
        return "no_identity_rule_for_endpoint"
    missing = [name for name in rule.required_fields if _missing(record.identity.get(name))]
    if missing:
        return "missing_required_identity:" + ",".join(missing)
    return None


def measurement_identity_key(
    record: Measurement,
    scope: TableEquivalenceScope,
    rules: Mapping[str, IdentityRule],
) -> tuple[Any, ...]:
    """Return the complete linkage key, explicitly excluding ``outcome``.

    Raises ``ValueError`` for records that should be classified unresolved.
    """

    problem = _record_problem(record, scope, rules)
    if problem:
        raise ValueError(f"{record.measurement_id}: {problem}")
    table_class = scope.class_for(record.table_ref)
    assert table_class is not None  # established above
    rule = rules[record.endpoint]
    fields = []
    for name in rule.required_fields:
        normalizer = rule.normalizers.get(name, canonical_identity_value)
        fields.append((name, normalizer(record.identity[name])))
    return (
        table_class.family_id,
        table_class.table_class_id,
        canonical_identity_value(record.endpoint),
        canonical_identity_value(record.channel),
        _compound_key(record.compound_id),
        *fields,
    )


def outcomes_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    """Compare outcomes after identity matching; never used to create a pair."""

    if _missing(left) or _missing(right):
        return False
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            outcomes_equal(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    try:
        a, b = float(left), float(right)
        return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return canonical_identity_value(left) == canonical_identity_value(right)


def comparison_fields_equal(
    left: Measurement,
    right: Measurement,
    rule: IdentityRule,
) -> bool:
    """Compare extraction/context fields after linkage has been fixed."""

    for name in rule.comparison_fields:
        left_value = left.identity.get(name)
        right_value = right.identity.get(name)
        left_missing = _missing(left_value)
        right_missing = _missing(right_value)
        # Gold is the reference.  If it does not state a field, that field is
        # not evaluable and cannot make the Atlas row incorrect.  Failure to
        # extract a field that Gold does state remains an error.
        if left_missing:
            continue
        if right_missing:
            return False
        normalizer = rule.normalizers.get(name, canonical_identity_value)
        try:
            if normalizer(left_value) != normalizer(right_value):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _check_unique_ids(records: Iterable[Measurement], side: str) -> None:
    ids = [str(record.measurement_id) for record in records]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {side} measurement IDs: {duplicates[:5]}")


def _pair_identity_bucket(
    gold: list[Measurement],
    atlas: list[Measurement],
    values_equal: Callable[[Measurement, Measurement], bool],
    context_equal: Callable[[Measurement, Measurement], bool],
) -> list[tuple[Measurement, Measurement]]:
    """Pair a fixed identity bucket with lexicographic multiset agreement.

    The assignment maximises value agreement first, then dose/context agreement
    among assignments with the same value score.  The bucket cardinality was
    already fixed by outcome-blind identity, so neither criterion can create a
    pair or move a row into scope.
    """

    gold = sorted(gold, key=lambda record: str(record.measurement_id))
    atlas = sorted(atlas, key=lambda record: str(record.measurement_id))
    if not gold or not atlas:
        return []
    target_size = min(len(gold), len(atlas))
    value_weight = target_size + 1
    weights = np.zeros((len(gold), len(atlas)), dtype=np.int64)
    for gold_index, gold_record in enumerate(gold):
        for atlas_index, atlas_record in enumerate(atlas):
            weights[gold_index, atlas_index] = (
                value_weight * int(values_equal(gold_record, atlas_record))
                + int(context_equal(gold_record, atlas_record))
            )
    gold_indices, atlas_indices = linear_sum_assignment(weights, maximize=True)
    return [(gold[i], atlas[j]) for i, j in zip(gold_indices, atlas_indices)]


def match_measurements(
    gold: Iterable[Measurement],
    atlas: Iterable[Measurement],
    *,
    scope: TableEquivalenceScope,
    rules: Mapping[str, IdentityRule],
    value_equal: Callable[[Any, Any], bool] = outcomes_equal,
) -> MatchResult:
    """Perform deterministic maximum-cardinality, one-to-one identity matching.

    Records are first partitioned by an outcome-blind exact identity key.  For a
    bucket with ``g`` gold and ``a`` Atlas occurrences, exactly ``min(g, a)``
    pairs are possible and produced.  Summing those maxima is the global maximum
    because no edge crosses identity buckets.  Within an identity bucket only,
    value-correct measurements are paired first to report maximum multiset
    outcome agreement.  Dose/context comparison never changes this assignment.
    """

    gold_records = tuple(gold)
    atlas_records = tuple(atlas)
    _check_unique_ids(gold_records, "gold")
    _check_unique_ids(atlas_records, "atlas")

    unresolved_gold: dict[str, str] = {}
    unresolved_atlas: dict[str, str] = {}
    gold_buckets: dict[tuple[Any, ...], list[Measurement]] = defaultdict(list)
    atlas_buckets: dict[tuple[Any, ...], list[Measurement]] = defaultdict(list)

    for record in gold_records:
        problem = _record_problem(record, scope, rules)
        if problem:
            unresolved_gold[str(record.measurement_id)] = problem
        else:
            gold_buckets[measurement_identity_key(record, scope, rules)].append(record)
    for record in atlas_records:
        problem = _record_problem(record, scope, rules)
        if problem:
            unresolved_atlas[str(record.measurement_id)] = problem
        else:
            atlas_buckets[measurement_identity_key(record, scope, rules)].append(record)

    pair_by_gold: dict[str, MatchedPair] = {}
    pair_by_atlas: dict[str, MatchedPair] = {}
    for key in sorted(set(gold_buckets) | set(atlas_buckets), key=repr):
        gs = list(gold_buckets.get(key, ()))
        ats = list(atlas_buckets.get(key, ()))
        def values_equal(g_record: Measurement, a_record: Measurement) -> bool:
            return bool(value_equal(g_record.outcome, a_record.outcome))

        def contexts_equal(g_record: Measurement, a_record: Measurement) -> bool:
            return comparison_fields_equal(
                g_record, a_record, rules[g_record.endpoint]
            )

        for g_record, a_record in _pair_identity_bucket(
            gs, ats, values_equal, contexts_equal
        ):
            rule = rules[g_record.endpoint]
            value_correct = bool(value_equal(g_record.outcome, a_record.outcome))
            context_correct = comparison_fields_equal(g_record, a_record, rule)
            pair = MatchedPair(
                gold_id=str(g_record.measurement_id),
                atlas_id=str(a_record.measurement_id),
                value_correct=value_correct,
                context_correct=context_correct,
            )
            pair_by_gold[pair.gold_id] = pair
            pair_by_atlas[pair.atlas_id] = pair

    def disposition(record: Measurement, side: str) -> RecordDisposition:
        record_id = str(record.measurement_id)
        unresolved = unresolved_gold if side == "gold" else unresolved_atlas
        if record_id in unresolved:
            return RecordDisposition(record_id, side, "unresolved", reason=unresolved[record_id])
        pair = pair_by_gold.get(record_id) if side == "gold" else pair_by_atlas.get(record_id)
        if pair is None:
            return RecordDisposition(record_id, side, "missing" if side == "gold" else "extra")
        counterpart = pair.atlas_id if side == "gold" else pair.gold_id
        status = "matched_value_correct" if pair.value_correct else "matched_value_incorrect"
        return RecordDisposition(record_id, side, status, counterpart_id=counterpart)

    pairs = tuple(sorted(pair_by_gold.values(), key=lambda pair: pair.gold_id))
    return MatchResult(
        pairs=pairs,
        gold=tuple(disposition(record, "gold") for record in gold_records),
        atlas=tuple(disposition(record, "atlas") for record in atlas_records),
    )
