"""Convert verified Gold and released Atlas data to common measurement records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from analyses.validation import gold_standard_audit as legacy
from analyses.validation.family_matcher import IdentityRule, Measurement


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "aso-atlas-2-release"
SPLITS = ("train", "validation", "test")


def _text(value: Any) -> str:
    return re.sub(r"[\s_\-/]+", " ", str(value).strip().casefold()).strip()


def _number(value: Any) -> float:
    parsed = legacy.as_scalar(value)
    if parsed is None:
        raise ValueError(f"not numeric: {value!r}")
    return float(parsed)


RULES = {
    "inhibition": IdentityRule(
        ("dose", "cell_line"),
        normalizers={"dose": _number, "cell_line": _text},
    ),
    "hepatotox": IdentityRule(
        ("dose", "species", "route", "num_doses"),
        normalizers={
            "dose": _number,
            "species": legacy.vocab,
            "route": legacy.vocab,
            "num_doses": _number,
        },
        comparison_fields=("period", "sample"),
    ),
    "neurotox": IdentityRule(
        ("dose", "species", "route", "score_type", "timepoint"),
        normalizers={
            "dose": _number,
            "species": legacy.vocab,
            "route": legacy.vocab,
            "score_type": legacy.vocab,
            "timepoint": _number,
        },
    ),
}


def _table_id(value: Any) -> str:
    return str(int(value))


def gold_measurements(gold_runs: Path) -> list[Measurement]:
    legacy.GOLD_RUNS = gold_runs
    records, _, _ = legacy.load_gold()
    output = []
    for index, record in enumerate(records):
        row = record["row"]
        if record["klass"] == "inhibition":
            identity = {"dose": record["dose"], "cell_line": row.get("cell_line")}
        elif record["klass"] == "hepatotox":
            identity = {
                "dose": record["dose"],
                "species": row.get("species"),
                "route": row.get("administration_method"),
                "num_doses": row.get("num_doses"),
                "period": row.get("dosing_period_days"),
                "sample": row.get("measurement_source"),
            }
        else:
            identity = {
                "dose": record["dose"],
                "species": row.get("species"),
                "route": row.get("administration_method"),
                "score_type": row.get("tolerability_score_type"),
                "timepoint": row.get("latency_time_hours"),
            }
        output.append(Measurement(
            measurement_id=f"gold:{index}",
            patent_id=record["pid"],
            table_id=_table_id(record["table"]),
            endpoint=record["klass"],
            channel=record["channel"],
            compound_id=record["cid"],
            identity=identity,
            outcome=record["value"],
        ))
    return output


def _read_release(release: Path, name: str) -> pd.DataFrame:
    paths = [release / name / f"{split}.parquet" for split in SPLITS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing release shards: {missing}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def atlas_measurements(release: Path) -> list[Measurement]:
    output: list[Measurement] = []
    serial = 0
    for name in ("in_vitro_inhibition", "dose_response"):
        for row in _read_release(release, name).to_dict("records"):
            output.append(Measurement(
                measurement_id=f"atlas:{serial}",
                patent_id=row["USPTO ID"],
                table_id=_table_id(row["Table Number"]),
                endpoint="inhibition",
                channel="Inhibition_pct",
                compound_id=legacy.cid_of(row["Compound ID"]),
                identity={
                    "dose": legacy.as_scalar(row.get("dosage_nm")),
                    "cell_line": row.get("cell_line"),
                },
                outcome=row.get("Inhibition_pct"),
            ))
            serial += 1

    for row in _read_release(release, "hepatotoxicity").to_dict("records"):
        identity = {
            "dose": legacy.as_scalar(row.get("dosage_mg_per_kg")),
            "species": row.get("species"),
            "route": row.get("adminstration_method"),
            "num_doses": row.get("num_doses"),
            "period": row.get("dosing_period_days"),
            "sample": row.get("measurement_source"),
        }
        for channel in legacy.BIOMARKERS.values():
            vector = legacy.as_vector(row.get(channel))
            if vector is None:
                scalar = legacy.as_scalar(row.get(channel))
                vector = [] if scalar is None else [scalar]
            for occurrence, value in enumerate(vector):
                output.append(Measurement(
                    measurement_id=f"atlas:{serial}:{channel}:{occurrence}",
                    patent_id=row["USPTO ID"],
                    table_id=_table_id(row["Table Number"]),
                    endpoint="hepatotox",
                    channel=channel,
                    compound_id=legacy.cid_of(row["Compound ID"]),
                    identity=identity,
                    outcome=value,
                ))
        serial += 1

    for row in _read_release(release, "neurotoxicity").to_dict("records"):
        if (not legacy.as_vector(row.get("FOB_score"))
                and legacy.as_scalar(row.get("FOB_score")) is None):
            serial += 1
            continue
        output.append(Measurement(
            measurement_id=f"atlas:{serial}",
            patent_id=row["USPTO ID"],
            table_id=_table_id(row["Table Number"]),
            endpoint="neurotox",
            channel="FOB_score",
            compound_id=legacy.cid_of(row["Compound ID"]),
            identity={
                "dose": legacy.as_scalar(row.get("dosage_ug")),
                "species": row.get("species"),
                "route": row.get("administration_method"),
                "score_type": row.get("tolerability_score_type"),
                "timepoint": row.get("latency_time_hours"),
            },
            outcome=row.get("FOB_score"),
        ))
        serial += 1
    return output
