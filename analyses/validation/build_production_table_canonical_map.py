"""Recover the historical table-level production canonicalisation decisions.

``patent-collate`` deduplicated individual table XML files before transcription.
The resulting ``file_to_canonical`` decisions were retained in each production
run's ``step1_format_tables.json``: a false ``canonical_link`` means that the
file itself was transcribed, while a path names the table file used instead.

This script joins those retained decisions to every verified measurement table
in the four-patent Gold audit.  It does *not* infer a single canonical patent
for a family and it does not inspect Atlas rows or measurement outcomes.  A
historical similarity link records production provenance, not scientific
equivalence.  The one non-identical audited link is adjudicated from source XML
identity fields (different compound IDs), without using Atlas output values.

Usage:
    uv run python analyses/validation/build_production_table_canonical_map.py
    uv run python analyses/validation/build_production_table_canonical_map.py --check
"""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = Path.home() / "dphil" / "tox-patent-scrape" / "verify" / "data" / "runs"
DEFAULT_PRODUCTION = (
    Path.home() / "dphil" / "patent-collate-www" / "static" / "patents"
)
DEFAULT_OUTPUT = ROOT / "data" / "validation" / "production_table_canonical_map.csv"

MEASUREMENT_CLASSES = {"inhibition", "hepatotox", "neurotox"}
TABLE_FILE = re.compile(r"^(US\d+[A-Z]\d)_table_(\d+)\.xml$", re.I)
RUN_FILE = re.compile(r"^(US\d+[A-Z]\d)_t(\d+)\.json$", re.I)


@dataclass(frozen=True)
class AuditPatent:
    family_id: str
    production_run: str


AUDIT_PATENTS = {
    "US20160251655A1": AuditPatent("C9ORF72_PCT_US2014_60194", "IONIS"),
    "US20240026353A1": AuditPatent("SCN2A_PCT_US2021_44887", "IONIS"),
    "US20210038631A1": AuditPatent("IRF4_PCT_US2019_020201", "IONIS"),
    "US20190160090A1": AuditPatent("AGT_PCT_US2016_56068", "IONIS"),
}

# This retained >=.90-similarity link joins different compound/content tables;
# it is production provenance, not a scientific equivalence edge.
KNOWN_FALSE_SIMILARITY_LINKS = {("US20240026353A1", 75)}


FIELDS = (
    "mapping_id",
    "family_id",
    "gold_patent",
    "gold_table",
    "gold_status",
    "gold_class",
    "gold_rows",
    "verified_measurement_table",
    "production_run",
    "artifact_entry_present",
    "input_xml_present",
    "source_file",
    "production_link_target_file",
    "production_link_target_patent",
    "production_link_target_table",
    "canonical_file",
    "canonical_patent",
    "canonical_table",
    "canonical_link_hops",
    "production_dedup_category",
    "preview_similarity_ratio",
    "step1_transcription_status",
    "equivalence_note",
    "source_md5",
    "canonical_md5",
    "relationship",
    "validation_stage",
    "scope_fixed_before_atlas",
    "outcome_blinded",
    "evidence",
)


def digest(path: Path, algorithm: str = "md5") -> str:
    # MD5 is deliberate: this reproduces the upstream exact-file key and is
    # not being used for security.
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def table_ref(path: str | Path) -> tuple[str, int]:
    match = TABLE_FILE.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"unrecognized table filename: {path}")
    return match.group(1).upper(), int(match.group(2))


def load_artifact(path: Path) -> dict[str, dict[str, Any]]:
    content = json.loads(path.read_text())
    if not isinstance(content, dict):
        raise ValueError(f"expected an object in {path}")
    return content


def resolve_canonical(
    source: str, entries: dict[str, dict[str, Any]]
) -> tuple[str, int]:
    """Resolve retained links defensively and reject cycles."""
    current = source
    visited: set[str] = set()
    hops = 0
    while True:
        if current in visited:
            raise ValueError(f"canonical-link cycle beginning at {source}")
        visited.add(current)
        entry = entries.get(current)
        if entry is None:
            # Historical links normally point at an entry whose conversion
            # succeeded.  Keeping an absent final target still recovers the
            # file_to_canonical decision itself.
            return current, hops
        link = entry.get("canonical_link", False)
        if link is False or link is None or link == "":
            return current, hops
        if not isinstance(link, str):
            raise ValueError(f"invalid canonical_link for {current}: {link!r}")
        current = link
        hops += 1


def preview(path: Path, max_lines: int = 200) -> str:
    """Read exactly the preview consumed by upstream similarity.py."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return "".join(handle.readline() for _ in range(max_lines))


def isolated_under_upstream_rules(source: Path, peers: list[Path]) -> bool:
    """Prove that an unretained source could only map to itself.

    An isolated file has neither an exact same-number peer nor a >=.90
    first-200-line similarity edge.  It is consequently a singleton under both
    upstream deduplication stages, independent of input/set iteration order.
    """
    source_md5 = digest(source)
    source_preview = preview(source)
    for peer in peers:
        if peer == source:
            continue
        if digest(peer) == source_md5:
            return False
        peer_preview = preview(peer)
        largest = max(len(source_preview), len(peer_preview))
        if largest == 0:
            ratio_bound = 1.0
        else:
            ratio_bound = min(len(source_preview), len(peer_preview)) / largest
        if ratio_bound < 0.90:
            continue
        matcher = SequenceMatcher(None, source_preview, peer_preview, autojunk=True)
        if matcher.real_quick_ratio() >= 0.90 and matcher.ratio() >= 0.90:
            return False
    return True


def upstream_preview_similarity(left: Path, right: Path) -> float:
    return SequenceMatcher(None, preview(left), preview(right), autojunk=True).ratio()


def make_row(
    run_path: Path,
    production_root: Path,
    entries_by_run: dict[str, dict[str, dict[str, Any]]],
    disk_by_run: dict[str, dict[tuple[str, int], Path]],
) -> dict[str, Any] | None:
    match = RUN_FILE.fullmatch(run_path.name)
    if not match:
        return None
    patent, table = match.group(1).upper(), int(match.group(2))
    audit = AUDIT_PATENTS.get(patent)
    if audit is None:
        return None

    run = json.loads(run_path.read_text())
    entries = entries_by_run[audit.production_run]
    entry_by_ref = {table_ref(path): path for path in entries}
    source_key = entry_by_ref.get((patent, table))
    disk_source = disk_by_run[audit.production_run].get((patent, table))
    source_path = Path(source_key) if source_key else disk_source

    canonical_path: Path | None = None
    canonical_patent = ""
    canonical_table: int | str = ""
    hops: int | str = ""
    source_md5 = ""
    canonical_md5 = ""
    link_target_path: Path | None = None
    link_target_patent = ""
    link_target_table: int | str = ""
    dedup_category = "formatting_failed"
    similarity_ratio: float | str = ""
    step1_status = "not_in_production_input"
    equivalence_note = "no retained production table entry"
    relationship = "unresolved"
    outcome_blinded = "false"

    if source_key is not None:
        resolved, hops = resolve_canonical(source_key, entries)
        canonical_path = Path(resolved)
        canonical_patent, canonical_table = table_ref(canonical_path)
        link_target_path = canonical_path
        link_target_patent, link_target_table = canonical_patent, canonical_table
        if source_path is not None and source_path.exists():
            source_md5 = digest(source_path)
        if canonical_path.exists():
            canonical_md5 = digest(canonical_path)
        if hops == 0:
            dedup_category = "self_canonical"
            step1_status = "converted_canonical"
            similarity_ratio = 1.0
            equivalence_note = "same source file"
            relationship = "equivalent"
            outcome_blinded = "true"
        elif source_md5 and source_md5 == canonical_md5:
            dedup_category = "exact_byte_duplicate"
            step1_status = "duplicate_link_retained"
            similarity_ratio = 1.0
            equivalence_note = "source and canonical XML are byte-identical"
            relationship = "equivalent"
            # Exact byte identity is not a decision based on agreement of a
            # selected outcome field; every byte is the same.
            outcome_blinded = "true"
        else:
            if (patent, table) not in KNOWN_FALSE_SIMILARITY_LINKS:
                raise ValueError(
                    f"unadjudicated production similarity link: {patent} table {table}"
                )
            dedup_category = "false_similarity_duplicate"
            step1_status = "duplicate_link_retained"
            similarity_ratio = upstream_preview_similarity(source_path, canonical_path)
            equivalence_note = (
                "not equivalent: compound 1348447 was linked to compound 1166852"
            )
            # The production target is retained above for provenance, but it is
            # deliberately not admitted as a valid Gold counterpart.
            canonical_path = None
            canonical_patent = ""
            canonical_table = ""
            relationship = "absent"
            outcome_blinded = "true"
    elif source_path is None:
        dedup_category = "absent_from_production_input"
        relationship = "absent"
        outcome_blinded = "true"
    else:
        same_number_peers = [
            path for (other_patent, other_table), path
            in disk_by_run[audit.production_run].items()
            if other_table == table
        ]
        if isolated_under_upstream_rules(source_path, same_number_peers):
            hops = 0
            source_md5 = canonical_md5 = digest(source_path)
            dedup_category = "formatting_failed"
            step1_status = "canonical_conversion_failed"
            similarity_ratio = 1.0
            equivalence_note = (
                "dedup singleton reconstructed, but no formatted production entry"
            )
            # The file-to-canonical decision is recoverable, but no formatted
            # entry reached transcription.  All of its Gold readouts are FNs.
            relationship = "absent"
            outcome_blinded = "true"
        else:
            dedup_category = "formatting_failed"
            step1_status = "step1_entry_absent"
            relationship = "absent"
            outcome_blinded = "true"

    evidence = (
        f"retained {audit.production_run}/step1_format_tables.json; "
        f"production_dedup_category={dedup_category}"
    )
    return {
        "mapping_id": f"{patent}_t{table}",
        "family_id": audit.family_id,
        "gold_patent": patent,
        "gold_table": table,
        "gold_status": run.get("status", ""),
        "gold_class": run.get("klass", ""),
        "gold_rows": len(run.get("rows") or []),
        "verified_measurement_table": str(
            run.get("status") == "verified" and run.get("klass") in MEASUREMENT_CLASSES
        ).lower(),
        "production_run": audit.production_run,
        "artifact_entry_present": str(source_key is not None).lower(),
        "input_xml_present": str(source_path is not None and source_path.exists()).lower(),
        "source_file": str(source_path) if source_path is not None else "",
        "production_link_target_file": (
            str(link_target_path) if link_target_path is not None else ""
        ),
        "production_link_target_patent": link_target_patent,
        "production_link_target_table": link_target_table,
        "canonical_file": str(canonical_path) if canonical_path is not None else "",
        "canonical_patent": canonical_patent,
        "canonical_table": canonical_table,
        "canonical_link_hops": hops,
        "production_dedup_category": dedup_category,
        "preview_similarity_ratio": (
            f"{similarity_ratio:.9f}" if isinstance(similarity_ratio, float) else ""
        ),
        "step1_transcription_status": step1_status,
        "equivalence_note": equivalence_note,
        "source_md5": source_md5,
        "canonical_md5": canonical_md5,
        "relationship": relationship,
        "validation_stage": f"production_{dedup_category}",
        "scope_fixed_before_atlas": "true",
        "outcome_blinded": outcome_blinded,
        "evidence": evidence,
    }


def build(runs_dir: Path, production_root: Path) -> list[dict[str, Any]]:
    entries_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    disk_by_run: dict[str, dict[tuple[str, int], Path]] = {}
    for run_name in sorted({audit.production_run for audit in AUDIT_PATENTS.values()}):
        directory = production_root / run_name
        artifact = directory / "step1_format_tables.json"
        entries_by_run[run_name] = load_artifact(artifact)
        disk_by_run[run_name] = {
            table_ref(path): path for path in directory.glob("*.xml")
            if TABLE_FILE.fullmatch(path.name)
        }

    rows = []
    for path in sorted(runs_dir.glob("US*_t*.json")):
        row = make_row(path, production_root, entries_by_run, disk_by_run)
        if row is not None and row["verified_measurement_table"] == "true":
            rows.append(row)
    rows.sort(key=lambda row: (row["gold_patent"], int(row["gold_table"])))
    expected = {
        "self_canonical": 273,
        "exact_byte_duplicate": 36,
        "false_similarity_duplicate": 1,
        "formatting_failed": 1,
    }
    observed = {
        category: sum(row["production_dedup_category"] == category for row in rows)
        for category in expected
    }
    if observed != expected:
        raise ValueError(f"unexpected verified-table categories: {observed} != {expected}")
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    content = render(build(args.runs_dir, args.production_root))
    if args.check:
        if not args.output.exists() or args.output.read_text() != content:
            raise SystemExit(f"stale or missing output: {args.output}")
        print(f"OK: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"wrote {content.count(chr(10)) - 1} rows to {args.output}")


if __name__ == "__main__":
    main()
