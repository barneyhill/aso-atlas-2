"""Complete the three source-confirmed runs from explicit patent prose.

The immutable original snapshots remain intact. The agent drafts the annotations;
a human's verification is recorded separately with --sign-off. --apply also
updates the local verifier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.validation.source_status_review import EVIDENCE, GOLD, check_table, text
from analyses.validation.gold_standard_audit import build_helm, parse_helm

DEST = EVIDENCE / "completed"
VERIFIER = Path.home() / "dphil/tox-patent-scrape/verify/data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def source_context(item):
    root = ET.parse(EVIDENCE / item["source_xml"]).getroot()
    paragraphs = [(p.attrib, text(p)) for p in root.findall("p")]
    chemistry = next((a, p) for a, p in paragraphs if "sugar motif" in p)
    assay = next((a, p) for a, p in paragraphs if "Cultured " in p)
    sugar = "e" * 5 + "d" * 10 + "e" * 5
    if sugar not in chemistry[1] or "5-methyl" not in chemistry[1]:
        raise ValueError("Source does not establish sugar/methylation")
    backbone = "soooossssssssssooss" if item["patent"] == "US20240026353A1" else "s" * 19
    if backbone != "s" * 19 and backbone not in chemistry[1]:
        raise ValueError("Mixed linkage pattern absent from source")
    if backbone == "s" * 19 and "Each internucleoside linkage is a phosphorothioate" not in chemistry[1]:
        raise ValueError("All-PS statement absent from source")
    if "approximately 24 hours" not in assay[1] or "electroporation" not in assay[1].lower():
        raise ValueError("Treatment context absent from source")
    return sugar, backbone, {"chemistry": {**chemistry[0], "text": chemistry[1]},
                             "assay": {**assay[0], "text": assay[1]}}


def prepare():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    with GOLD.open(newline="") as handle:
        gold = list(csv.DictReader(handle))
    DEST.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    review = {"reviewed_at": now, "reviewer": "Codex agent",
              "human_verified": False, "method": "Direct source-cell and patent-prose cross-reference",
              "tables": [], "excluded_tables": [], "apply_inputs": {}}
    all_rows = []
    for item in manifest["tables"]:
        check_table(item, gold)
        original = Path(item["original_patent_path"])
        if digest(original) != item["original_patent_sha256"]:
            raise ValueError(f"Original patent changed: {original}")
        live_path = VERIFIER / "runs" / (item["id"] + ".json")
        live = json.loads(live_path.read_text())
        snapshot = json.loads((EVIDENCE / item["run_snapshot"]).read_text())
        if live["rows"] != snapshot["rows"]:
            raise ValueError(f"Live rows changed: {live_path}")
        review["apply_inputs"][str(live_path)] = digest(live_path)
        sugar, backbone, context = source_context(item)
        decision = {"reviewedAt": now, "reviewer": "Codex agent", "humanVerified": False,
                    "blanksOnly": True, "sugar": {"mode": "constant", "value": sugar},
                    "backbone": {"mode": "constant", "value": backbone},
                    "context": f"Patent paragraphs {context['chemistry']['num']} (chemistry) and {context['assay']['num']} (assay); see {DEST.relative_to(ROOT)}/review.json",
                    "sourceEvidence": context}
        for row in live["rows"]:
            if len(row["sequence"]) != len(sugar) or len(backbone) != len(sugar) - 1:
                raise ValueError("Motif length mismatch")
            # Other assay fields were already correct; independently check prose.
            if str(row["cells_per_well"]) not in context["assay"]["text"].replace(",", ""):
                raise ValueError("Cell density does not match source")
            if row["cell_line"] not in context["assay"]["text"] or row["cell_line_species"] != "human":
                raise ValueError("Cell identity does not match source")
            if row["transfection_method"].lower() != "electroporation":
                raise ValueError("Transfection method does not match source")
            row.update(sugar_motif=sugar, backbone_motif=backbone,
                       cytosine_methylated="yes", treatment_period_hrs="24")
            note = "approximately 24 h; chemistry explicitly stated in preceding patent prose"
            row["measurement_note"] = row["measurement_note"] + "; " + note
            helm = build_helm(row["sequence"], sugar, backbone, True)
            parsed = parse_helm(helm)
            if (not parsed or parsed["seq"] != row["sequence"]
                    or parsed["sugars"] != sugar or parsed["links"] != backbone
                    or any(b != ("[5meC]" if s == "C" else s)
                           for b, s in zip(parsed["bases"], row["sequence"]))):
                raise ValueError("HELM failed sequence round-trip")
            all_rows.append({"patent": item["patent"], "table": item["xml_table"],
                             **row, "HELM Annotation": helm})
        live.update(status="verified", chemistryReview=decision, sourceReview=decision)
        live.setdefault("meta", {}).update(treatment_period_hrs="24", cytosine_methylated="yes")
        live.setdefault("chat", []).append({"who": "review", "text": decision["context"] + " (agent source review)"})
        (DEST / (item["id"] + ".json")).write_text(json.dumps(live, indent=2) + "\n")
        (DEST / (item["id"] + ".csv")).write_text(csv_text(live["rows"]))
        review["tables"].append({"id": item["id"], "rows": len(live["rows"]),
                                "sugar_motif": sugar, "backbone_motif": backbone,
                                "cytosine_methylated": "yes", "treatment_period_hrs": 24,
                                "treatment_period_approximate": True, "source_evidence": context,
                                "completed_run_sha256": digest(DEST / (item["id"] + ".json"))})
    for item in manifest["excluded_live_review_tables"]:
        path = EVIDENCE / item["source_xml"]
        if digest(path) != item["source_xml_sha256"]:
            raise ValueError("Excluded source changed")
        caption = text(ET.parse(path).getroot().find("tables"))
        if "proliferation" not in caption.lower():
            raise ValueError("Proliferation exclusion not supported")
        live_path = VERIFIER / "runs" / (item["id"] + ".json")
        verified_path = VERIFIER / "verified" / (item["id"] + ".csv")
        if verified_path.exists():
            review["apply_inputs"][str(verified_path)] = digest(verified_path)
        if json.loads(live_path.read_text())["status"] != "review":
            raise ValueError("Excluded table is no longer awaiting review")
        review["apply_inputs"][str(live_path)] = digest(live_path)
        review["excluded_tables"].append({"id": item["id"], "disposition": item["disposition"],
                                          "source_caption": caption[:250]})
    for name in ("classifications.csv", "chemistry_review.json"):
        review["apply_inputs"][str(VERIFIER / name)] = digest(VERIFIER / name)
    (DEST / "gold_measurements_with_chemistry.csv").write_text(csv_text(all_rows))
    (DEST / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    return review


def apply():
    review = json.loads((DEST / "review.json").read_text())
    for name, expected in review["apply_inputs"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Live file changed since review: {name}")
    for table in review["tables"]:
        if digest(DEST / (table["id"] + ".json")) != table["completed_run_sha256"]:
            raise ValueError("Completed annotation changed after review")
    backup = VERIFIER / ("_backup_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_source_review")
    backup.mkdir()
    for name in review["apply_inputs"]:
        path = Path(name)
        target = backup / path.relative_to(VERIFIER)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    chemistry = json.loads((VERIFIER / "chemistry_review.json").read_text())
    for table in review["tables"]:
        name = table["id"]
        state = json.loads((DEST / (name + ".json")).read_text())
        for suffix, folder in ((".json", "runs"), (".csv", "verified")):
            target = VERIFIER / folder / (name + suffix)
            if target.exists():
                b = backup / folder / target.name
                b.parent.mkdir(exist_ok=True)
                if not b.exists():
                    shutil.copy2(target, b)
            shutil.copy2(DEST / (name + suffix), target)
        decision = state["chemistryReview"]
        history = chemistry.get(name, {}).get("history", [])
        chemistry[name] = {**decision, "history": history + [decision]}
    with (VERIFIER / "classifications.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        classifications = list(reader)
    excluded = {t["id"] for t in review["excluded_tables"]}
    for row in classifications:
        if row["table_id"] in excluded:
            row["klass"] = "skip"
    for name in excluded:
        path = VERIFIER / "runs" / (name + ".json")
        state = json.loads(path.read_text())
        state.update(klass="skip", status="excluded", sourceReview={
            "reviewer": "Codex agent", "humanVerified": False,
            "disposition": "exclude_cell_proliferation_not_target_RNA_knockdown",
            "evidence": str(DEST / "review.json")})
        path.write_text(json.dumps(state, indent=2) + "\n")
        verified_path = VERIFIER / "verified" / (name + ".csv")
        if verified_path.exists():
            verified_path.unlink()  # Already preserved in the preflight backup.
    with (VERIFIER / "classifications.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(classifications)
    (VERIFIER / "chemistry_review.json").write_text(json.dumps(chemistry, indent=2) + "\n")
    print(f"Applied 3 completed tables and 12 exclusions; backup: {backup}")


def validate_completed():
    review = json.loads((DEST / "review.json").read_text())
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    with GOLD.open(newline="") as handle:
        gold = list(csv.DictReader(handle))
    reviewed = {t["id"]: t for t in review["tables"]}
    for item in manifest["tables"]:
        check_table(item, gold)
        state = json.loads((DEST / (item["id"] + ".json")).read_text())
        original = json.loads((EVIDENCE / item["run_snapshot"]).read_text())
        sugar, backbone, context = source_context(item)
        if len(state["rows"]) != len(original["rows"]):
            raise ValueError("Completed row count changed")
        if state["status"] != "verified" or state["sourceReview"]["humanVerified"] is not True:
            raise ValueError("Completed review attribution/status incorrect")
        if reviewed[item["id"]]["source_evidence"] != context:
            raise ValueError("Completion evidence differs from patent")
        for before, after in zip(original["rows"], state["rows"]):
            expected = {**before, "sugar_motif": sugar, "backbone_motif": backbone,
                        "cytosine_methylated": "yes", "treatment_period_hrs": "24",
                        "measurement_note": before["measurement_note"] +
                        "; approximately 24 h; chemistry explicitly stated in preceding patent prose"}
            if after != expected:
                raise ValueError("Completed annotation differs from source-supported correction")
        if digest(DEST / (item["id"] + ".json")) != reviewed[item["id"]]["completed_run_sha256"]:
            raise ValueError("Completed run hash differs")
    if review.get("human_verified") is not True:
        raise ValueError("Completed review has no human sign-off; run with --sign-off")
    return {"status": "complete", "drafted_by": review["drafted_by"],
            "reviewer": review["reviewer"], "human_verified": True,
            "tables_completed": len(reviewed), "measurements_completed": sum(t["rows"] for t in reviewed.values()),
            "tables_excluded": len(review["excluded_tables"]),
            "evidence": str((DEST / "review.json").relative_to(ROOT))}


def sign_off(reviewer: str, date: str) -> None:
    """Record a human's verification of the agent-drafted completion."""
    review = json.loads((DEST / "review.json").read_text())
    if review.get("human_verified"):
        raise ValueError("Review already signed off")
    drafted_by = review["reviewer"]
    for table in review["tables"]:
        path = DEST / (table["id"] + ".json")
        state = json.loads(path.read_text())
        for key in ("chemistryReview", "sourceReview"):
            state[key].update(draftedBy=drafted_by, reviewer=reviewer,
                              humanVerified=True, verifiedAt=date)
        path.write_text(json.dumps(state, indent=2) + "\n")
        table["completed_run_sha256"] = digest(path)
    review.update(drafted_by=drafted_by, reviewer=reviewer,
                  human_verified=True, human_verified_at=date)
    (DEST / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(f"Recorded sign-off by {reviewer} ({date}); draft by {drafted_by}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--sign-off", metavar="REVIEWER")
    parser.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()
    if args.sign_off:
        sign_off(args.sign_off, args.date)
    elif args.apply:
        validate_completed()
        apply()
    else:
        result = prepare()
        print(f"Prepared {sum(t['rows'] for t in result['tables'])} corrected measurements and 12 exclusions in {DEST}")
