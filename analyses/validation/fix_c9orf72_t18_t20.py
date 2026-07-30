"""Restore the two-cell-line split in US20160251655A1 tables 18 and 20.

Tables 17-20 of the C9ORF72 patent each print TWO measurement columns for one
compound, distinguished only by their primer probe set:

    ... | Sequence | Linkage | % inhibition (LLC-MK2) | % inhibition (HepG2) | SEQ ID
                              Primer Probe RTS3750     Primer Probe RTS3905

RTS3750 measures total C9ORF72 mRNA in rhesus LLC-MK2 cells; RTS3905 measures the
pathogenic-associated variant in human HepG2 cells. Tables 17 and 19 were rebuilt
correctly and carry both labels. Tables 18 and 20 were not: all 158 rows of each
are labelled HepG2 / human, so the 79 rhesus measurements in each table are filed
as human ones.

This is invisible to a cell-accounting check, which only ever compares numbers -
both values are present and correct, they are simply attributed to the wrong
cell line. It surfaced by adjudicating a species disagreement against the release,
where the release turned out to be right and this corpus wrong.

The dose and density come from the paragraph introducing table 17, which governs
tables 17-20 as one example (rhesus LLC-MK2 and HepG2, both at 20,000 cells per
well, both electroporated with 4,000 nM), and match what tables 17 and 19 already
record.

Usage:
    uv run python analyses/validation/fix_c9orf72_t18_t20.py --dry-run
    uv run python analyses/validation/fix_c9orf72_t18_t20.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from corpus_completeness import parse_grid

SCRAPE = Path.home() / "dphil" / "tox-patent-scrape"
RUNS = SCRAPE / "verify" / "data" / "runs"
VERIFIED = SCRAPE / "verify" / "data" / "verified"
XML = next((SCRAPE / "data" / "step2").glob("US20160251655A1-*.XML"))
PID = "US20160251655A1"
TABLES = [18, 20]

# grid column layout, identical in tables 17-20
COL_ISIS, COL_SEQ, COL_LINK, COL_MK2, COL_HEPG2 = 0, 4, 5, 6, 7

CHANNELS = [
    (COL_MK2, "LLC-MK2", "rhesus monkey",
     "RTS3750 — total C9ORF72 mRNA"),
    (COL_HEPG2, "HepG2", "human",
     "RTS3905 — C9ORF72 pathogenic associated mRNA variant"),
]
CONTEXT = {"dosage_nm": "4000", "target_RNA": "C9ORF72", "cells_per_well": "20000",
           "transfection_method": "Electroporation", "treatment_period_hrs": "24"}
FIELDS = ["Compound ID", "sequence", "sugar_motif", "backbone_motif", "cell_line",
          "cell_line_species", "dosage_nm", "target_RNA", "Inhibition_pct",
          "cells_per_well", "transfection_method", "treatment_period_hrs",
          "measurement_note"]


def build(tnum, xml, old_rows):
    # the sugar motif is stated in the patent's prose, not this grid; carry it
    # over per compound from the existing extraction rather than dropping it
    sugar = {}
    for r in old_rows:
        cid = re.sub(r"\D", "", str(r.get("Compound ID") or ""))
        if cid and r.get("sugar_motif"):
            sugar.setdefault(cid, r["sugar_motif"])

    out = []
    for cells in parse_grid(xml, tnum) or []:
        if len(cells) < 9 or not re.fullmatch(r"\d{6}\*?", cells[COL_ISIS]):
            continue
        cid = re.sub(r"\D", "", cells[COL_ISIS])
        seq = re.sub(r"[^ACGTU]", "", cells[COL_SEQ].upper())
        for col, line, species, note in CHANNELS:
            val = cells[col].strip()
            if not re.fullmatch(r"[+-]?\d*\.?\d+", val.replace(",", "")):
                continue
            out.append(dict(CONTEXT, **{
                "Compound ID": cid,
                "sequence": seq,
                "sugar_motif": sugar.get(cid, ""),
                "backbone_motif": cells[COL_LINK].strip(),
                "cell_line": line,
                "cell_line_species": species,
                "Inhibition_pct": val,
                "measurement_note": note,
            }))
    return out


def main(apply):
    xml = XML.read_text(errors="ignore")
    for tnum in TABLES:
        run = RUNS / f"{PID}_t{tnum:05d}.json"
        d = json.loads(run.read_text())
        old = d.get("rows") or []
        new = build(tnum, xml, old)

        from collections import Counter
        print(f"\n{PID} t{tnum}: {len(old)} -> {len(new)} rows")
        print(f"  before: {dict(Counter((r.get('cell_line'), r.get('cell_line_species')) for r in old))}")
        print(f"  after : {dict(Counter((r['cell_line'], r['cell_line_species']) for r in new))}")
        old_vals = sorted(str(r.get("Inhibition_pct")) for r in old)
        new_vals = sorted(str(r["Inhibition_pct"]) for r in new)
        print(f"  measurement values unchanged: {old_vals == new_vals}")
        for r in new[:2]:
            print(f'    {r["Compound ID"]}  {r["cell_line"]:8s} {r["cell_line_species"]:14s} '
                  f'{r["Inhibition_pct"]}%')
        if not apply:
            continue

        shutil.copy(run, run.with_suffix(".json.prespecies"))
        d["rows"] = new
        d["comment"] = ("rebuilt deterministically from the patent XML: the two "
                        "measurement columns (RTS3750 LLC-MK2 rhesus, RTS3905 "
                        "HepG2 human) had both been labelled HepG2/human "
                        "(analyses/validation/fix_c9orf72_t18_t20.py)")
        run.write_text(json.dumps(d, indent=1))
        with (VERIFIED / f"{PID}_t{tnum:05d}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(new)
        print(f"  wrote {run.name} and its CSV")
    if not apply:
        print("\ndry run; pass --apply to write")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().apply)
