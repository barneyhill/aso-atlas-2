"""Source-derived strata for the production-canonical validation audit.

Table layout and conversion labels are derived from the verified source tables
and original USPTO XML, before Atlas outcomes are consulted.  Chemistry labels
come from the independently verified compound chemistry rows.  The resulting
labels are propagated through the frozen production table-equivalence classes,
so Gold and Atlas records are always assigned by the same rule.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from analyses.validation import gold_standard_audit as legacy
from analyses.validation.canonical_source_audit import FrozenCanonicalScope
from analyses.validation.family_matcher import EquivalenceClass, Measurement, TableRef


@dataclass(frozen=True)
class ValidationStrata:
    """Callable Gold/Atlas stratifiers plus audit coverage metadata."""

    stratifiers: Mapping[
        str,
        tuple[Callable[[Measurement], str], Callable[[Measurement], str]],
    ]
    metadata: Mapping[str, Any]


def _gold_chemistry_class(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "unknown / not stated"
    sequence = legacy.clean_seq(row.get("sequence"))
    sugar = legacy.expand_sugar(row.get("sugar_motif"), len(sequence) or None)
    if not sugar:
        return "unknown / not stated"
    if re.fullmatch(r"e{5}d{10}e{5}", sugar):
        return "5-10-5 MOE gapmer"
    if "k" in sugar and "e" in sugar:
        return "mixed MOE/cEt"
    if "k" in sugar:
        return "cEt gapmer"
    if "l" in sugar:
        return "LNA-containing"
    if re.fullmatch(r"e+d+e+", sugar):
        return "MOE gapmer (other wing)"
    if set(sugar) == {"d"}:
        return "DNA"
    return "other stated chemistry"


def _class_values(
    frozen_scope: FrozenCanonicalScope,
    table_values: Mapping[TableRef, str],
) -> dict[EquivalenceClass, str]:
    """Collapse source-table labels onto frozen production components."""

    grouped: dict[EquivalenceClass, set[str]] = defaultdict(set)
    for table_ref, value in table_values.items():
        table_class = frozen_scope.table_scope.class_for(table_ref)
        if table_class is not None:
            grouped[table_class].add(value)
    output = {}
    for table_class, values in grouped.items():
        output[table_class] = (
            next(iter(values))
            if len(values) == 1
            else "mixed: " + " | ".join(sorted(values))
        )
    return output


def build_validation_strata(
    frozen_scope: FrozenCanonicalScope,
    gold_runs,
) -> ValidationStrata:
    """Build requested audit strata without consulting released Atlas values."""

    legacy.GOLD_RUNS = gold_runs
    gold_rows, gold_tables, gold_chem = legacy.load_gold()
    source = legacy.table_strata(gold_rows, gold_tables)

    def refs(field: str, transform=lambda value: value) -> dict[TableRef, str]:
        return {
            TableRef(patent, str(table)): transform(values[field])
            for (patent, table), values in source.items()
        }

    table_format = _class_values(frozen_scope, refs("format"))
    table_layout = _class_values(
        frozen_scope,
        refs(
            "multi_column",
            lambda value: "multi-column melt" if value else "single measurement column",
        ),
    )
    unit_conversion = _class_values(
        frozen_scope,
        {
            TableRef(patent, str(table)): (
                "not applicable (toxicity endpoint)"
                if gold_tables[(patent, table)]["klass"] != "inhibition"
                else "% control → % inhibition"
                if values["pct_control_conversion"]
                else "no outcome conversion"
            )
            for (patent, table), values in source.items()
        },
    )
    chem_by_compound = {
        str(compound): _gold_chemistry_class(row)
        for compound, row in gold_chem.items()
    }

    def class_label(mapping: Mapping[EquivalenceClass, str], record: Measurement) -> str:
        table_class = frozen_scope.table_scope.class_for(record.table_ref)
        return mapping.get(table_class, "unknown") if table_class is not None else "unknown"

    def table_pair(mapping):
        fn = lambda record: class_label(mapping, record)
        return fn, fn

    def chemistry(record: Measurement) -> str:
        compound = legacy.cid_of(record.compound_id)
        return chem_by_compound.get(str(compound), "unknown / not stated")

    def assay_readout(record: Measurement) -> str:
        if record.endpoint == "inhibition":
            return "inhibition"
        if record.endpoint == "neurotox":
            return "FOB score"
        if record.channel == "ALT":
            return "ALT"
        return "other hepatic biomarker"

    inhib_tables = [values for values in source.values()]
    metadata = {
        "scope": "labels derived from verified Gold rows and original USPTO XML",
        "source_tables_labelled": len(source),
        "tables_requiring_percent_control_conversion": sum(
            values["pct_control_conversion"] for values in inhib_tables
        ),
        "tables_requiring_dose_unit_conversion": sum(
            values["dose_unit_conversion"] for values in inhib_tables
        ),
        "dose_unit_conversion_note": (
            "No audited table states an input dose in uM or mM; dose-unit conversion "
            "accuracy is therefore not estimable in this four-patent corpus."
        ),
        "chemistry_compounds_labelled": sum(
            value != "unknown / not stated" for value in chem_by_compound.values()
        ),
        "chemistry_compounds_total": len(chem_by_compound),
    }
    return ValidationStrata(
        stratifiers={
            "table_format": table_pair(table_format),
            "table_layout": table_pair(table_layout),
            "unit_conversion": table_pair(unit_conversion),
            "chemistry_class": (chemistry, chemistry),
            "assay_readout": (assay_readout, assay_readout),
        },
        metadata=metadata,
    )
