"""Deterministic rebuild of US20190160090A1 table 7 (patent TABLE 6).

`corpus_completeness.py` flagged this table with a defect no value-level check
can see: the extraction took the compound identifier from the wrong column. The
patent prints

    ISIS NO | Start Site | Stop Site | Sequence | Chemistry | % Inhibition | ...

and the extraction recorded the Stop Site as the Compound ID, so ISIS 568637
(sequence CGCTGATTTGTCCGGG, 74% inhibition) was filed as compound "2061". Every
one of the 79 rows carries a wrong identifier, and the table was nevertheless
marked `verified`. It is the only table of the four patents where this happens
(the check that found it scans for a column of 5-7 digit integers sitting to the
left of the column the extraction actually used as the identifier).

The identifiers are the join key to the release, so the consequence is not a
cosmetic one: none of these 79 measurements can match anything, and they were
being scored as coverage failures charged to the release.

The rebuild is a direct column read rather than a re-prompt, for the reason
recorded in the benchmark notes: on multi-column patent grids a fixed parse is
both more accurate and auditable, where the model repeatedly mis-melts. Assay
context comes from the paragraph introducing the table, which states HepG2 at
20,000 cells/well, electroporation, 1000 nM, ~24 hours, results as percent
inhibition.

Usage:
    uv run python analyses/validation/fix_agt_t7.py --dry-run
    uv run python analyses/validation/fix_agt_t7.py --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from html import unescape
from pathlib import Path

SCRAPE = Path.home() / "dphil" / "tox-patent-scrape"
RUN = SCRAPE / "verify" / "data" / "runs" / "US20190160090A1_t00007.json"
CSV = SCRAPE / "verify" / "data" / "verified" / "US20190160090A1_t00007.csv"
XML = next((SCRAPE / "data" / "step2").glob("US20190160090A1-*.XML"))

# column layout of the printed grid, read once from the header block
COL_ISIS, COL_SEQ, COL_CHEM, COL_PCT = 0, 3, 4, 5

CONTEXT = {
    "cell_line": "HepG2",
    "cell_line_species": "human",
    "dosage_nm": "1000",
    "target_RNA": "AGT",
    "cells_per_well": "20000",
    "transfection_method": "Electroporation",
    "treatment_period_hrs": "24",
    "measurement_note": "inhibition",
}
FIELDS = ["Compound ID", "sequence", "sugar_motif", "backbone_motif", "cell_line",
          "cell_line_species", "dosage_nm", "target_RNA", "Inhibition_pct",
          "cells_per_well", "transfection_method", "treatment_period_hrs",
          "measurement_note"]


def grid(xml, num):
    m = re.search(r'<tables[^>]*num="0*%d"[^>]*>([\s\S]*?)</tables>' % num, xml)
    rows = []
    for rm in re.finditer(r"<row>([\s\S]*?)</row>", m.group(1)):
        cells = []
        for c in re.findall(r"<entry[^>]*>([\s\S]*?)</entry>|<entry[^>]*/>", rm.group(1)):
            t = unescape(re.sub(r"<[^>]+>", "", c or ""))
            t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
            cells.append(re.sub(r"\s+", " ", t).strip())
        rows.append(cells)
    return rows


def build():
    g = grid(XML.read_text(errors="ignore"), 7)
    out = []
    for cells in g:
        if len(cells) < 9 or not re.fullmatch(r"\d{6}", cells[COL_ISIS]):
            continue
        pct = cells[COL_PCT].strip()
        if not re.fullmatch(r"[+-]?\d*\.?\d+", pct.replace(",", "")):
            continue
        out.append(dict(CONTEXT, **{
            "Compound ID": cells[COL_ISIS],
            "sequence": re.sub(r"[^ACGTU]", "", cells[COL_SEQ].upper()),
            "sugar_motif": cells[COL_CHEM].strip(),
            "backbone_motif": "",
            "Inhibition_pct": pct,
        }))
    return out


def main(apply):
    d = json.loads(RUN.read_text())
    old = d.get("rows") or []
    new = build()

    print(f"old rows {len(old)}  ->  new rows {len(new)}")
    old_ids = [str(r.get("Compound ID")) for r in old]
    new_ids = [r["Compound ID"] for r in new]
    print(f"identifiers changed: {sum(a != b for a, b in zip(old_ids, new_ids))}"
          f"/{min(len(old), len(new))}")
    # the measurements themselves should be untouched; only the key changes
    ov = [str(r.get("Inhibition_pct")) for r in old]
    nv = [r["Inhibition_pct"] for r in new]
    print(f"inhibition values changed: {sum(a != b for a, b in zip(ov, nv))}"
          f"/{min(len(ov), len(nv))}")
    for a, b in list(zip(old, new))[:3]:
        print(f'  {a.get("Compound ID"):>8s} -> {b["Compound ID"]}   '
              f'{b["sequence"]}  {b["Inhibition_pct"]}%')
    if not apply:
        print("\ndry run; pass --apply to write")
        return

    shutil.copy(RUN, RUN.with_suffix(".json.prefix"))
    d["rows"] = new
    d["status"] = "verified"
    d["comment"] = ("rebuilt deterministically from the patent XML: the previous "
                    "extraction read the Stop Site column as the Compound ID "
                    "(analyses/validation/fix_agt_t7.py)")
    RUN.write_text(json.dumps(d, indent=1))

    import csv
    with CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(new)
    print(f"\nwrote {RUN}\nwrote {CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().apply)
