"""Remove the fabricated dose arm from US20240026353A1 table 52 (patent TABLE 41).

The table prints four concentrations and an IC50:

    Compound ID | 78.125 nM | 313.5 nM | 1250.0 nM | 5000.0 nM | IC50 (uM)

The extraction produced five measurements per compound. The fifth is the IC50
column read as though it were another concentration arm, converted with the
same `100 - x` rule the %-control columns need, and filed under a dose of
2000 nM that the table never prints. For compound 1249164 the printed IC50 is
`<0.1`, which surfaced as an inhibition of 99.9% at a concentration that does
not exist.

Nineteen of the 95 rows are affected, one per compound. No other table of the
four patents extracts a dose the patent does not print, which is the check that
found this (`corpus_completeness.py`, dose-header agreement). The convention
elsewhere in the corpus is that IC50 columns are not captured as measurements,
so the fix is to drop the arm rather than to relabel it.

Usage:
    uv run python analyses/validation/fix_scn2a_t52.py --dry-run
    uv run python analyses/validation/fix_scn2a_t52.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

SCRAPE = Path.home() / "dphil" / "tox-patent-scrape"
RUN = SCRAPE / "verify" / "data" / "runs" / "US20240026353A1_t00052.json"
CSV_PATH = SCRAPE / "verify" / "data" / "verified" / "US20240026353A1_t00052.csv"

PRINTED_DOSES = {"78.125", "313.5", "1250.0", "5000.0"}
FABRICATED = "2000.0"


def main(apply):
    d = json.loads(RUN.read_text())
    rows = d.get("rows") or []
    keep = [r for r in rows if str(r.get("dosage_nm")).strip() != FABRICATED]
    drop = [r for r in rows if str(r.get("dosage_nm")).strip() == FABRICATED]

    print(f"rows {len(rows)} -> {len(keep)}  (dropping {len(drop)})")
    kept_doses = sorted({str(r.get("dosage_nm")).strip() for r in keep})
    print(f"remaining doses: {kept_doses}")
    assert set(kept_doses) == PRINTED_DOSES, "kept a dose the patent does not print"
    for r in drop[:3]:
        print(f'  drop {r.get("Compound ID")}  {r.get("dosage_nm")} nM  '
              f'{r.get("Inhibition_pct")}%')
    if not apply:
        print("\ndry run; pass --apply to write")
        return

    shutil.copy(RUN, RUN.with_suffix(".json.preic50"))
    d["rows"] = keep
    d["comment"] = ("dropped the IC50 column, which had been extracted as a "
                    "fabricated 2000 nM dose arm "
                    "(analyses/validation/fix_scn2a_t52.py)")
    RUN.write_text(json.dumps(d, indent=1))
    if keep:
        with CSV_PATH.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
            w.writeheader()
            w.writerows(keep)
    print(f"\nwrote {RUN}\nwrote {CSV_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().apply)
