"""Rebuild dose-series tables whose extraction dropped concentration arms.

Five tables print a compound column followed by one column per concentration
(and sometimes a trailing IC50), but the extraction captured only some of the
arms:

    SCN2A  t97   prints 39/156/625/2500/10000 nM   captured 4 of 5 (10000 lost)
    IRF4   t119  prints 8/40/200/1000 nM           captured 2 of 4
    IRF4   t125  prints 50/200/1000/5000 nM        captured 1 of 4, dose blank
    IRF4   t126  prints 50/200/1000/5000 nM        captured 1 of 4
    IRF4   t130  prints 16/80/400/2000 nM          captured 1 of 4

A dropped column is the failure mode a cell-accounting check is worst at seeing,
because a column nothing consumed looks exactly like a metadata column and gets
excluded from its own denominator. It is caught here by letting the patent's
column header decide what is a measurement column, not the extraction's own
behaviour.

All five tables report `% UTC` (percent of untreated control), so the stored
value is `100 - x`. t125 additionally had its one captured column stored
unconverted; rebuilding fixes that too.

Assay context (cell line, density, delivery) is carried over from the existing
extraction, which agrees with the patent prose for these tables; only the dose
grid and the values derived from it are rebuilt.

Usage:
    uv run python analyses/validation/fix_dose_series_tables.py --dry-run
    uv run python analyses/validation/fix_dose_series_tables.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from corpus_completeness import DOSE_TOKEN, DOSE_TO_NM, cell_number, parse_grid

SCRAPE = Path.home() / "dphil" / "tox-patent-scrape"
RUNS = SCRAPE / "verify" / "data" / "runs"
VERIFIED = SCRAPE / "verify" / "data" / "verified"
XML_DIR = SCRAPE / "data" / "step2"

TARGETS = [("US20240026353A1", 97), ("US20210038631A1", 119),
           ("US20210038631A1", 125), ("US20210038631A1", 126),
           ("US20210038631A1", 130)]

# fields copied unchanged from the existing extraction for this table
CARRY = ["cell_line", "cell_line_species", "target_RNA", "cells_per_well",
         "transfection_method", "treatment_period_hrs"]


def dose_columns(grid):
    """Column index -> concentration in nM, from the header rows."""
    out = {}
    for cells in grid:
        for j, c in enumerate(cells):
            if j in out or cell_number(c) is not None:
                continue
            m = DOSE_TOKEN.search(c)
            if m:
                out[j] = round(float(m.group(1).replace(",", ""))
                               * DOSE_TO_NM[m.group(2).lower()], 4)
    return out


def fmt(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def build(pid, tnum, xml, old):
    grid = parse_grid(xml, tnum) or []
    doses = dose_columns(grid)
    if not doses:
        raise SystemExit(f"{pid} t{tnum}: no concentration headers found")

    ctx = {k: (old[0].get(k, "") if old else "") for k in CARRY}
    seq = {}
    for r in old:
        cid = re.sub(r"\D", "", str(r.get("Compound ID") or ""))
        if cid and r.get("sequence"):
            seq.setdefault(cid, (r.get("sequence"), r.get("sugar_motif", ""),
                                 r.get("backbone_motif", "")))

    out = []
    for cells in grid:
        if not cells or not re.fullmatch(r"\d{5,7}", cells[0].strip()):
            continue
        cid = cells[0].strip()
        s, sug, bb = seq.get(cid, ("", "", ""))
        for j, dose in sorted(doses.items()):
            if j >= len(cells):
                continue
            v = cell_number(cells[j])
            if v is None:
                continue
            out.append({
                "Compound ID": cid, "sequence": s, "sugar_motif": sug,
                "backbone_motif": bb, "cell_line": ctx["cell_line"],
                "cell_line_species": ctx["cell_line_species"],
                "dosage_nm": fmt(dose), "target_RNA": ctx["target_RNA"],
                "Inhibition_pct": fmt(100 - v),
                "cells_per_well": ctx["cells_per_well"],
                "transfection_method": ctx["transfection_method"],
                "treatment_period_hrs": ctx["treatment_period_hrs"],
                "measurement_note": "converted from %UTC",
            })
    return out, doses


def main(apply):
    for pid, tnum in TARGETS:
        xml = next(XML_DIR.glob(f"{pid}-*.XML")).read_text(errors="ignore")
        run = RUNS / f"{pid}_t{tnum:05d}.json"
        d = json.loads(run.read_text())
        old = d.get("rows") or []
        new, doses = build(pid, tnum, xml, old)

        print(f"\n{pid} t{tnum}: {len(old)} -> {len(new)} rows")
        print(f"  printed doses (nM): {[fmt(x) for _, x in sorted(doses.items())]}")
        print(f"  before: {dict(Counter(str(r.get('dosage_nm')) for r in old))}")
        print(f"  after : {dict(Counter(r['dosage_nm'] for r in new))}")
        for r in new[:2]:
            print(f'    {r["Compound ID"]}  {r["dosage_nm"]:>6s} nM  '
                  f'{r["Inhibition_pct"]}%')
        if not apply:
            continue

        shutil.copy(run, run.with_suffix(".json.predose"))
        d["rows"] = new
        d["status"] = "verified"
        d["comment"] = ("rebuilt deterministically from the patent XML: dropped "
                        "concentration arms restored, %UTC converted as 100-x "
                        "(analyses/validation/fix_dose_series_tables.py)")
        run.write_text(json.dumps(d, indent=1))
        with (VERIFIED / f"{pid}_t{tnum:05d}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(new[0].keys()))
            w.writeheader()
            w.writerows(new)
        print("  written")
    if not apply:
        print("\ndry run; pass --apply to write")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().apply)
