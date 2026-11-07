"""Reconcile three frozen audit tables with later live review states.

Frozen source XML and original run snapshots are checked, followed by the
source-supported completion when present. No extraction scripts are executed;
the completion records both the agent draft and the human sign-off.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "data/validation/source_status_review"
GOLD = ROOT / "data/validation/canonical_source_match_ledger.csv"
OUTPUT = ROOT / "data/results/source_status_review.json"


def text(node):
    return " ".join("".join(node.itertext()).split())


def source_records(root):
    """Read the eight printed columns, retaining dagger-marked compounds."""
    records = []
    table = root.find("tables")
    for row in table.findall(".//row"):
        cells = [text(entry) for entry in row.findall("entry")]
        if not cells or not re.fullmatch(r"\d{6}[†‡]?", cells[0]):
            continue
        if len(cells) != 8:
            raise ValueError(f"Unexpected source column count: {cells}")
        if not re.fullmatch(r"[ACGT]+", cells[5]):
            raise ValueError(f"Invalid source sequence: {cells}")
        records.append({
            "compound": re.match(r"\d+", cells[0]).group(),
            "sequence": cells[5],
            "printed_percent_control": float(cells[6]),
            "inhibition": 100 - float(cells[6]),
        })
    if not records:
        raise ValueError("No source measurements found")
    return records


def check_table(item, gold_rows, evidence=EVIDENCE):
    for name in ("source_xml", "run_snapshot"):
        path = evidence / item[name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item[name + "_sha256"]:
            raise ValueError(f"Changed frozen evidence: {path}")
    root = ET.parse(evidence / item["source_xml"]).getroot()
    run = json.loads((evidence / item["run_snapshot"]).read_text())
    source = source_records(root)
    caption = text(root.find("tables"))
    header_cells = [row.findall("entry") for row in root.find("tables").findall(".//row")]
    if not any(len(cells) == 8 and re.search(r"control|UTC", text(cells[6]))
               for cells in header_cells):
        raise ValueError(f"Source outcome is not percent control: {item['id']}")
    prose = " ".join(text(p) for p in root.findall("p"))
    dose_match = re.search(r"at ([\d,]+) nM", caption)
    if dose_match is None:
        dose_match = re.search(r"with ([\d,]+) nM concentration", prose)
    dose = float(dose_match.group(1).replace(",", ""))
    cell_line = re.search(r"Cultured ([\w-]+) cells", prose).group(1)
    gene = re.search(r"\b(IRF4|SCN2A)\b", caption).group(1)
    source_values = Counter((r["compound"], r["inhibition"]) for r in source)
    frozen = [r for r in gold_rows if r["side"] == "gold"
              and r["patent"] == item["patent"]
              and int(r["table"]) == item["xml_table"]]
    frozen_values = Counter((r["compound"], float(r["outcome"])) for r in frozen)
    if frozen_values != source_values:
        raise ValueError(f"Frozen Gold differs from source: {item['id']}")
    for row in frozen:
        if json.loads(row["identity_json"]) != {"cell_line": cell_line, "dose": dose}:
            raise ValueError(f"Frozen assay identity differs: {item['id']}")
    expected = Counter((r["compound"], r["sequence"], r["inhibition"]) for r in source)
    actual = Counter((str(r["Compound ID"]), r["sequence"], float(r["Inhibition_pct"]))
                     for r in run["rows"])
    if actual != expected:
        raise ValueError(f"Live snapshot differs from source: {item['id']}")
    for row in run["rows"]:
        if (row["cell_line"] != cell_line or float(row["dosage_nm"]) != dose
                or row["target_RNA"] != gene):
            raise ValueError(f"Live assay identity differs: {item['id']}")
    pending = {field: sum(not row.get(field) for row in run["rows"])
               for field in ("sugar_motif", "backbone_motif", "treatment_period_hrs")}
    return {
        "id": item["id"], "patent": item["patent"], "xml_table": item["xml_table"],
        "printed_table_caption": text(root.find("tables/table/tgroup/thead")),
        "frozen_map_status": "verified", "live_snapshot_status": run["status"],
        "disposition": "retain_frozen_measurements_source_confirmed",
        "measurements_checked": len(source), "outcomes_correct": len(source),
        "sequences_correct": len(source), "assay_identities_correct": len(source),
        "source_assay_identity": {"cell_line": cell_line, "dose_nm": dose, "target": gene},
        "blank_fields_pending_separate_context_review": pending,
        "chemistry_and_complete_context_signed_off": False,
    }


def run_review(evidence=EVIDENCE, gold=GOLD, output=OUTPUT):
    manifest = json.loads((evidence / "manifest.json").read_text())
    with gold.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    tables = [check_table(item, rows, evidence) for item in manifest["tables"]]
    frozen_scope = {(r["patent"], int(r["table"])) for r in rows if r["side"] == "gold"}
    excluded = []
    for item in manifest.get("excluded_live_review_tables", []):
        path = evidence / item["source_xml"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["source_xml_sha256"]:
            raise ValueError(f"Changed frozen evidence: {path}")
        table = ET.parse(path).getroot().find("tables")
        if "proliferation" not in text(table).lower():
            raise ValueError(f"Exclusion lacks proliferation source evidence: {item['id']}")
        if (item["patent"], item["xml_table"]) in frozen_scope:
            raise ValueError(f"Proliferation table unexpectedly in frozen audit: {item['id']}")
        excluded.append({"id": item["id"], "disposition": item["disposition"],
                         "in_frozen_measurement_audit": False})
    report = {
        "scope": manifest["scope"], "method": manifest["review_method"],
        "status_explanation": manifest["status_explanation"],
        "gold_ledger_sha256": hashlib.sha256(gold.read_bytes()).hexdigest(),
        "tables_reconciled": len(tables),
        "measurements_source_confirmed": sum(t["measurements_checked"] for t in tables),
        "frozen_measurements_changed": 0,
        "tables": tables,
        "other_live_review_tables_excluded": excluded,
    }
    if (evidence / "completed/review.json").exists() and evidence == EVIDENCE:
        from analyses.validation.complete_source_review import validate_completed
        report["subsequent_chemistry_context_completion"] = validate_completed()
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run_review()
    print(json.dumps({k: result[k] for k in (
        "tables_reconciled", "measurements_source_confirmed", "frozen_measurements_changed"
    )}, indent=2))
