"""Reconstruct raw-to-processed deduplication lineage for ASO Atlas 2.0.

This is an audit utility, not part of the production cleaning pipeline.  It
replays only the filters and transformations that determine whether a raw row
enters an endpoint's deduplication/collapse operation, then reconciles every
resulting pipeline group against the canonical processed parquet and release
shards.

Important: ``pipeline_group_id`` is *not* a scientific measurement identity.
The production in-vitro and dose-response keys contain the measured outcome,
and several endpoint keys omit assay context.  The IDs emitted here describe
what the existing pipeline did; a validation matcher must define identity
independently and without the outcome value.

Outputs (under data/validation/dedup_lineage by default):

* raw_row_lineage.parquet -- one row per raw collation row, including filtered
  rows, winner/contributor disposition, and processed/release provenance.
* pipeline_groups.parquet -- one row per eligible production dedup/collapse
  group.
* hepatic_contributors.parquet -- one row per eligible hepatic raw row and
  biomarker, preserving membership in the collapsed record.
* summary.json -- counts, keys, input hashes, and reconciliation checks.

Run from the repository root:

    uv run python analyses/validation/dedup_lineage.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analyses.utils.helm import Helm


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "oligostack" / "raw"
PROCESSED = ROOT / "data" / "oligostack" / "processed"
RELEASE = ROOT / "aso-atlas-2-release"
DEFAULT_OUT = ROOT / "data" / "validation" / "dedup_lineage"

AUDITED_PATENTS = {
    "US20160251655A1",
    "US20240026353A1",
    "US20210038631A1",
    "US20190160090A1",
}

RAW_FILES = {
    "in_vitro_inhibition": ["in_vitro_inhibition_collation_results.csv"],
    "dose_response": [
        "dose_response_nm_collation_results.csv",
        "dose_response_um_collation_results.csv",
    ],
    "neurotoxicity": ["neurotox_collation_results.csv"],
    "hepatotoxicity": ["hepatictox_collation_results.csv"],
}

PROCESSED_FILES = {
    "in_vitro_inhibition": "in_vitro_inhibition_processed.parquet",
    "dose_response": "dose_response_processed.parquet",
    "neurotoxicity": "neurotoxicity_processed.parquet",
    "hepatotoxicity": "hepatictoxicity_processed.parquet",
}

PIPELINE_KEYS = {
    "in_vitro_inhibition": ["Compound ID", "Inhibition_pct"],
    "dose_response": ["Compound ID", "dosage_nm", "Inhibition_pct"],
    "neurotoxicity": [
        "HELM Annotation",
        "species",
        "administration_method",
        "tolerability_score_type",
        "dosage_ug",
    ],
    "hepatotoxicity": [
        "Compound ID",
        "species",
        "num_doses",
        "dosing_period_days",
        "dosage_mg_per_kg",
        "measurement_source",
        "adminstration_method",
    ],
}

BIOMARKERS = ["ALB", "ALT", "AST", "BUN", "CREA", "TBIL", "PC_ratio"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scalar(value: Any) -> Any:
    """JSON-safe scalar with one representation for every missing value."""
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value == 0:
        return 0.0
    return value


def _key_json(row: pd.Series, columns: list[str]) -> str:
    return json.dumps([_scalar(row.get(c)) for c in columns], separators=(",", ":"),
                      ensure_ascii=False, default=str)


def _group_id(endpoint: str, key_json: str) -> str:
    digest = hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:24]
    return f"{endpoint}:{digest}"


def _with_raw_ids(df: pd.DataFrame, filename: str, endpoint: str) -> pd.DataFrame:
    result = df.copy()
    result["source_file"] = filename
    # CSV header is line 1, so the first data row is line 2.
    result["source_row_number"] = np.arange(2, len(result) + 2, dtype=np.int64)
    result["raw_row_id"] = (
        endpoint + ":" + filename + ":" + result["source_row_number"].astype(str)
    )
    result["_input_order"] = np.arange(len(result), dtype=np.int64)
    return result


def _helm_filter_reasons(series: pd.Series) -> pd.Series:
    """Reproduce clean._filter_helm while retaining auditable reasons."""
    cache: dict[Any, Any] = {}

    def parse(h):
        if h not in cache:
            cache[h] = Helm.parse(h) if pd.notna(h) else None
        return cache[h]

    parsed = series.map(parse)
    flags = pd.DataFrame({
        # astype("string") is equivalent for the real mixed string/null input
        # and keeps small synthetic/all-null audit frames inspectable.
        "helm_uncertain": series.astype("string").str.contains(r"\?", na=False),
        "helm_too_short": parsed.map(lambda p: p is not None and p.length <= 10),
        "helm_no_dna_gap": parsed.map(lambda p: p is not None and "DNA" not in p.sugars),
        "helm_homopolymer": parsed.map(
            lambda p: p is not None and len(set(p.bases)) == 1
        ),
        "helm_naked_dna": parsed.map(
            lambda p: p is not None and all(s == "DNA" for s in p.sugars)
        ),
    }, index=series.index)
    return flags.apply(
        lambda r: "|".join(c for c, present in r.items() if bool(present)), axis=1
    )


def _append_reason(current: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    result = current.copy()
    result.loc[mask & result.eq("")] = reason
    result.loc[mask & result.ne("")] = result.loc[mask & result.ne("")] + "|" + reason
    return result


def _base_filters(df: pd.DataFrame) -> pd.Series:
    return _helm_filter_reasons(df["HELM Annotation"])


def prepare_in_vitro(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    reasons = _base_filters(df)
    reasons = _append_reason(
        reasons, ~df["Inhibition_pct"].between(-1000, 100), "inhibition_out_of_range"
    )
    return df.copy(), reasons


def prepare_dose_response(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    result = df.copy()
    reasons = _base_filters(result)
    reasons = _append_reason(
        reasons, ~result["Inhibition_pct"].between(-1000, 100),
        "inhibition_out_of_range",
    )
    reasons = _append_reason(
        reasons, ~(result["dosage_nm"] > 0), "nonpositive_or_missing_dose"
    )
    return result, reasons


def prepare_neurotoxicity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # FOB parsing and species_strain/target enrichment occur before deduplication
    # in clean.py, but none changes a deduplication-key column.
    return df.copy(), _base_filters(df)


def _split_dosage(value: Any) -> tuple[float | None, str | None]:
    import re

    if isinstance(value, str):
        match = re.match(r"\s*([\d\.]+)\s*(.*)", value)
        if match:
            try:
                number = float(match.group(1))
            except Exception:
                number = None
            unit = match.group(2).strip() if match.group(2) else None
            return number, unit
    return None, None


def _clean_hepatic_biomarker(value: Any, biomarker: str) -> Any:
    ranges = {"ALB": (1, 100), "ALT": (10, 50000), "AST": (10, 50000)}
    if biomarker not in ranges:
        return value
    vmin, vmax = ranges[biomarker]
    if isinstance(value, list):
        return [v for v in value if pd.isna(v) or vmin <= v <= vmax]
    if pd.isna(value):
        return value
    return value if vmin <= value <= vmax else None


def prepare_hepatotoxicity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    result = df.copy()
    reasons = _base_filters(result)

    dosage = result["dosage"].map(_split_dosage)
    result["dosage_mg_per_kg"] = dosage.map(lambda x: x[0])
    result["_dosage_unit"] = dosage.map(lambda x: x[1])

    weekly = result["_dosage_unit"].eq("mg/kg/wk")
    valid_weekly = (
        weekly
        & result["dosing_period_days"].notna()
        & result["num_doses"].notna()
        & result["num_doses"].gt(0)
    )
    result.loc[valid_weekly, "dosage_mg_per_kg"] = (
        result.loc[valid_weekly, "dosage_mg_per_kg"]
        * result.loc[valid_weekly, "dosing_period_days"]
        / 7
        / result.loc[valid_weekly, "num_doses"]
    )
    result.loc[weekly, "_dosage_unit"] = "mg/kg"

    reasons = _append_reason(
        reasons, ~result["_dosage_unit"].eq("mg/kg"), "unsupported_or_missing_dose_unit"
    )
    reasons = _append_reason(
        reasons, ~(result["dosage_mg_per_kg"] <= 1e4), "dose_above_limit_or_missing"
    )

    result["adminstration_method"] = result["adminstration_method"].replace(
        {"intraperitoneally": "intraperitoneal"}
    )
    reasons = _append_reason(
        reasons,
        ~result["adminstration_method"].isin(["intraperitoneal", "subcutaneous"]),
        "unsupported_administration_method",
    )
    reasons = _append_reason(
        reasons,
        ~result["measurement_source"].isin(["plasma", "urine"]),
        "unsupported_measurement_source",
    )
    result["species"] = result["species"].replace(
        {"cynomolgus monkey": "monkey", "beagle dog": "dog"}
    )
    for biomarker in BIOMARKERS:
        result[f"_clean_{biomarker}"] = result[biomarker].map(
            lambda v, b=biomarker: _clean_hepatic_biomarker(v, b)
        )
    return result, reasons


def _load_raw(endpoint: str) -> pd.DataFrame:
    if endpoint != "dose_response":
        filename = RAW_FILES[endpoint][0]
        return _with_raw_ids(pd.read_csv(RAW / filename), filename, endpoint)

    frames = []
    for filename, multiplier in zip(RAW_FILES[endpoint], [1.0, 1000.0]):
        frame = _with_raw_ids(pd.read_csv(RAW / filename), filename, endpoint)
        frame["dosage_nm"] = frame["dosage"] * multiplier
        frames.append(frame)
    # This is the production concatenation order: nM followed by µM.  Preserve
    # our explicit _input_order across both inputs even though pandas indices
    # in the production code overlap.
    result = pd.concat(frames, ignore_index=True)
    result["_input_order"] = np.arange(len(result), dtype=np.int64)
    return result


PREPARERS = {
    "in_vitro_inhibition": prepare_in_vitro,
    "dose_response": prepare_dose_response,
    "neurotoxicity": prepare_neurotoxicity,
    "hepatotoxicity": prepare_hepatotoxicity,
}


def replay_endpoint(endpoint: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw-row lineage and one row per eligible pipeline group."""
    prepared, reasons = PREPARERS[endpoint](raw)
    prepared["discard_reason"] = reasons
    prepared["eligible_for_pipeline_group"] = reasons.eq("")
    prepared["pipeline_group_id"] = None
    prepared["pipeline_key_json"] = None
    prepared["is_pipeline_representative"] = False

    eligible = prepared[prepared["eligible_for_pipeline_group"]].copy()
    key_cols = PIPELINE_KEYS[endpoint]
    eligible["pipeline_key_json"] = eligible.apply(
        lambda r: _key_json(r, key_cols), axis=1
    )
    eligible["pipeline_group_id"] = eligible["pipeline_key_json"].map(
        lambda key: _group_id(endpoint, key)
    )

    if endpoint in {"in_vitro_inhibition", "dose_response"}:
        # Deliberately use the production default sort kind.  The summary warns
        # that ties within a patent need a raw-row ordinal for future stability.
        ordered = eligible.sort_values("USPTO ID", ascending=False)
    else:
        ordered = eligible.sort_values("_input_order", kind="stable")
    representative_ids = set(
        ordered.drop_duplicates("pipeline_group_id", keep="first")["raw_row_id"]
    )
    eligible["is_pipeline_representative"] = eligible["raw_row_id"].isin(
        representative_ids
    )

    for col in ["pipeline_group_id", "pipeline_key_json", "is_pipeline_representative"]:
        prepared.loc[eligible.index, col] = eligible[col]

    prepared["disposition"] = "discarded_filter"
    if endpoint == "hepatotoxicity":
        prepared.loc[prepared["eligible_for_pipeline_group"], "disposition"] = (
            "kept_hepatic_contributor"
        )
    else:
        prepared.loc[
            prepared["eligible_for_pipeline_group"]
            & prepared["is_pipeline_representative"].astype(bool),
            "disposition",
        ] = "kept_processed"
        prepared.loc[
            prepared["eligible_for_pipeline_group"]
            & ~prepared["is_pipeline_representative"].astype(bool),
            "disposition",
        ] = "discarded_duplicate"

    groups = []
    for group_id, members in eligible.groupby("pipeline_group_id", sort=False):
        rep = members[members["is_pipeline_representative"]].iloc[0]
        groups.append({
            "endpoint": endpoint,
            "pipeline_group_id": group_id,
            "pipeline_key_json": rep["pipeline_key_json"],
            "eligible_raw_rows": len(members),
            "source_patents": int(members["USPTO ID"].nunique(dropna=True)),
            "source_tables": int(
                members[["USPTO ID", "Table Number"]].drop_duplicates().shape[0]
            ),
            "winner_raw_row_id": rep["raw_row_id"],
            "winner_patent": rep.get("USPTO ID"),
            "winner_table": rep.get("Table Number"),
            "touches_audited_patent": bool(members["USPTO ID"].isin(AUDITED_PATENTS).any()),
            "is_hepatic_collapse": endpoint == "hepatotoxicity",
        })
    return prepared, pd.DataFrame(groups)


def _release_rows(endpoint: str) -> pd.DataFrame:
    parts = []
    for split in ["train", "validation", "test"]:
        path = RELEASE / endpoint / f"{split}.parquet"
        frame = pd.read_parquet(path)
        frame["release_split"] = split
        frame["release_row_id"] = [
            f"{endpoint}:release:{split}:{i}" for i in range(len(frame))
        ]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def reconcile_endpoint(
    endpoint: str, lineage: pd.DataFrame, groups: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    key_cols = PIPELINE_KEYS[endpoint]
    processed = pd.read_parquet(PROCESSED / PROCESSED_FILES[endpoint]).copy()
    processed["pipeline_key_json"] = processed.apply(
        lambda r: _key_json(r, key_cols), axis=1
    )
    processed["pipeline_group_id"] = processed["pipeline_key_json"].map(
        lambda key: _group_id(endpoint, key)
    )
    processed["processed_row_id"] = [
        f"{endpoint}:processed:{i}" for i in range(len(processed))
    ]
    if processed["pipeline_group_id"].duplicated().any():
        raise AssertionError(f"{endpoint}: processed pipeline keys are not unique")

    raw_ids = set(groups["pipeline_group_id"])
    processed_ids = set(processed["pipeline_group_id"])
    if raw_ids != processed_ids:
        raise AssertionError(
            f"{endpoint}: replay/processed group mismatch: "
            f"{len(raw_ids - processed_ids)} replay-only, "
            f"{len(processed_ids - raw_ids)} processed-only"
        )

    release = _release_rows(endpoint)
    release["pipeline_key_json"] = release.apply(
        lambda r: _key_json(r, key_cols), axis=1
    )
    release["pipeline_group_id"] = release["pipeline_key_json"].map(
        lambda key: _group_id(endpoint, key)
    )
    if release["pipeline_group_id"].duplicated().any():
        raise AssertionError(f"{endpoint}: release pipeline keys are not unique")
    expected_release = set(
        processed.loc[processed["HELM Annotation"].notna(), "pipeline_group_id"]
    )
    release_ids = set(release["pipeline_group_id"])
    if expected_release != release_ids:
        raise AssertionError(
            f"{endpoint}: processed/release mismatch: "
            f"{len(expected_release - release_ids)} expected-only, "
            f"{len(release_ids - expected_release)} release-only"
        )

    pcols = [
        "pipeline_group_id", "processed_row_id", "USPTO ID", "Table Number",
    ]
    pmap = processed[pcols].rename(columns={
        "USPTO ID": "processed_patent", "Table Number": "processed_table",
    })
    rmap = release[[
        "pipeline_group_id", "release_row_id", "release_split",
    ]]
    groups = groups.merge(pmap, on="pipeline_group_id", how="left", validate="one_to_one")
    groups = groups.merge(rmap, on="pipeline_group_id", how="left", validate="one_to_one")
    groups["in_release"] = groups["release_row_id"].notna()

    gmap = groups[[
        "pipeline_group_id", "winner_raw_row_id", "winner_patent", "winner_table",
        "processed_row_id", "processed_patent", "processed_table", "release_row_id",
        "release_split", "in_release",
    ]]
    lineage = lineage.merge(gmap, on="pipeline_group_id", how="left", validate="many_to_one")
    lineage["source_is_audited_patent"] = lineage["USPTO ID"].isin(AUDITED_PATENTS)

    # The production winner's retained provenance should agree with our replay.
    # Hepatic provenance is first-non-null per column, but these raw inputs state
    # both patent and table, so the representative check remains meaningful.
    provenance_mismatch = groups[
        groups["winner_patent"].astype(str).ne(groups["processed_patent"].astype(str))
        | groups["winner_table"].astype(str).ne(groups["processed_table"].astype(str))
    ]
    if len(provenance_mismatch):
        raise AssertionError(
            f"{endpoint}: {len(provenance_mismatch)} processed provenance rows "
            "do not agree with the replayed representative"
        )

    summary = {
        "raw_rows": len(lineage),
        "filtered_rows": int((lineage["disposition"] == "discarded_filter").sum()),
        "eligible_raw_rows": int(lineage["eligible_for_pipeline_group"].sum()),
        "pipeline_groups": len(groups),
        "processed_rows": len(processed),
        "released_rows": len(release),
        "discarded_duplicate_rows": int(
            (lineage["disposition"] == "discarded_duplicate").sum()
        ),
        "hepatic_contributor_rows": int(
            (lineage["disposition"] == "kept_hepatic_contributor").sum()
        ),
        "groups_touching_audited_patents": int(groups["touches_audited_patent"].sum()),
        "raw_rows_from_audited_patents": int(lineage["source_is_audited_patent"].sum()),
        "reconciliation": "exact",
    }
    return lineage, groups, summary


def hepatic_contributor_rows(lineage: pd.DataFrame) -> pd.DataFrame:
    eligible = lineage[lineage["eligible_for_pipeline_group"]].copy()
    rows = []
    for _, raw in eligible.iterrows():
        for biomarker in BIOMARKERS:
            value = raw.get(f"_clean_{biomarker}")
            rows.append({
                "raw_row_id": raw["raw_row_id"],
                "pipeline_group_id": raw["pipeline_group_id"],
                "processed_row_id": raw.get("processed_row_id"),
                "release_row_id": raw.get("release_row_id"),
                "source_patent": raw.get("USPTO ID"),
                "source_table": raw.get("Table Number"),
                "biomarker": biomarker,
                "raw_value_json": json.dumps(_scalar(raw.get(biomarker)), default=str),
                "cleaned_value_json": json.dumps(_scalar(value), default=str),
                "contributes_nonnull_value": not (
                    value is None
                    or (isinstance(value, float) and pd.isna(value))
                    or (isinstance(value, list) and len(value) == 0)
                ),
                "is_pipeline_representative": bool(raw["is_pipeline_representative"]),
            })
    return pd.DataFrame(rows)


def _public_lineage_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "endpoint", "raw_row_id", "source_file", "source_row_number", "USPTO ID", "Table Number",
        "Compound ID", "HELM Annotation", "Inhibition_pct", "dosage_nm",
        "FOB_score", "dosage_ug", *BIOMARKERS,
        "pipeline_group_id", "pipeline_key_json",
        "eligible_for_pipeline_group", "discard_reason", "disposition",
        "is_pipeline_representative", "winner_raw_row_id", "winner_patent",
        "winner_table", "processed_row_id", "processed_patent", "processed_table",
        "in_release", "release_row_id", "release_split", "source_is_audited_patent",
    ]
    return df[[c for c in columns if c in df.columns]].rename(columns={
        "USPTO ID": "source_patent", "Table Number": "source_table",
        "Compound ID": "compound_id",
    })


def generate(output_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_lineage = []
    all_groups = []
    summaries = {}
    hepatic_members = None

    for endpoint in PIPELINE_KEYS:
        raw = _load_raw(endpoint)
        lineage, groups = replay_endpoint(endpoint, raw)
        lineage, groups, summary = reconcile_endpoint(endpoint, lineage, groups)
        lineage["endpoint"] = endpoint
        groups["endpoint"] = endpoint
        summaries[endpoint] = summary
        if endpoint == "hepatotoxicity":
            hepatic_members = hepatic_contributor_rows(lineage)
        all_lineage.append(_public_lineage_columns(lineage))
        all_groups.append(groups)

    lineage_out = pd.concat(all_lineage, ignore_index=True)
    groups_out = pd.concat(all_groups, ignore_index=True)
    lineage_out.to_parquet(output_dir / "raw_row_lineage.parquet", index=False)
    groups_out.to_parquet(output_dir / "pipeline_groups.parquet", index=False)
    assert hepatic_members is not None
    hepatic_members.to_parquet(output_dir / "hepatic_contributors.parquet", index=False)

    hashes = {
        str((RAW / filename).relative_to(ROOT)): _sha256(RAW / filename)
        for filenames in RAW_FILES.values() for filename in filenames
    }
    hashes.update({
        str((PROCESSED / filename).relative_to(ROOT)): _sha256(PROCESSED / filename)
        for filename in PROCESSED_FILES.values()
    })
    for endpoint in PIPELINE_KEYS:
        for split in ("train", "validation", "test"):
            path = RELEASE / endpoint / f"{split}.parquet"
            hashes[str(path.relative_to(ROOT))] = _sha256(path)
    cleaning_code = ROOT / "analyses" / "logic" / "clean.py"
    hashes[str(cleaning_code.relative_to(ROOT))] = _sha256(cleaning_code)
    report = {
        "purpose": "pipeline deduplication lineage; not a scientific match identity",
        "audited_patent_ids": sorted(AUDITED_PATENTS),
        "pipeline_keys": PIPELINE_KEYS,
        "winner_order": {
            "in_vitro_inhibition": "USPTO ID descending; first row wins",
            "dose_response": "nM file then uM file; USPTO ID descending; first row wins",
            "neurotoxicity": "raw CSV order; first row wins",
            "hepatotoxicity": "all eligible members contribute; non-key provenance is first non-null",
        },
        "known_limits": [
            "Raw collation files identify patent and table but not original XML row/cell.",
            "Patent-family membership and table equivalence are not inferable from dedup lineage.",
            "Hepatic processed provenance stores only first-non-null source fields even when several raw rows contribute.",
            "The production USPTO sort does not request a stable algorithm for within-patent ties.",
            "Outcome-bearing pipeline keys must not be used as audit measurement identities.",
        ],
        "inputs_sha256": hashes,
        "endpoints": summaries,
        "outputs": {
            "raw_row_lineage": "raw_row_lineage.parquet",
            "pipeline_groups": "pipeline_groups.parquet",
            "hepatic_contributors": "hepatic_contributors.parquet",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = generate(args.output_dir)
    for endpoint, values in report["endpoints"].items():
        print(
            f"{endpoint}: {values['raw_rows']:,} raw -> "
            f"{values['pipeline_groups']:,} processed groups -> "
            f"{values['released_rows']:,} released rows"
        )
    print(f"Wrote lineage to {args.output_dir}")


if __name__ == "__main__":
    main()
