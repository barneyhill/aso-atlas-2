"""Normalise cell-line species labels left as the placeholder "other".

Adjudicating the release's `cell_line_species` against this corpus found the
disagreement running the opposite way to the one expected: of 863 disagreements,
861 were this corpus being wrong and the release being right. Two clusters:

  631 rows  cell_line = LLC-MK2, species recorded as "other"
            (US20160251655A1 tables 21, 55-61). LLC-MK2 is a rhesus monkey
            kidney line; the patent's own heading calls them "rhesus LLC-MK2
            cells".

   72 rows  cell_line = "primary hepatocytes", species "other"
            (US20190160090A1 table 49). The patent states these are
            "cryopreserved individual male cynomolgus monkey primary
            hepatocytes".

Both are normalisation failures rather than misreadings: the value was extracted,
then mapped to a placeholder instead of a species. The field is not cosmetic -
`analyses/logic/models/oligoai_build_csv.py` filters the OligoAI training set to
`cell_line_species == "human"`, so a mislabelled row silently changes what the
model trains on.

Only rows whose species is currently a placeholder are touched, and only where
the cell line identifies the species unambiguously. Rows already carrying a
species are left alone, so this cannot overwrite a correct value.

Usage:
    uv run python analyses/validation/fix_species_labels.py --dry-run
    uv run python analyses/validation/fix_species_labels.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

SCRAPE = Path.home() / "dphil" / "tox-patent-scrape"
RUNS = SCRAPE / "verify" / "data" / "runs"
VERIFIED = SCRAPE / "verify" / "data" / "verified"

PLACEHOLDER = {"", "other", "na", "n/a", "none", "null", "unknown"}

# (patent, cell line as written) -> species, with the patent text that settles it
RULES = {
    ("US20160251655A1", "llc-mk2"): (
        "rhesus monkey",
        'patent heading: "Antisense Inhibition of C9ORF72 ... in LLC-MK2 Cells"; '
        'body: "Cultured rhesus LLC-MK2 cells"'),
    ("US20190160090A1", "primary hepatocytes"): (
        "monkey",
        'patent: "cryopreserved individual male cynomolgus monkey primary hepatocytes"'),
}


def main(apply):
    changed = Counter()
    touched = []
    for f in sorted(RUNS.glob("*.json")):
        pid = f.stem.split("_t")[0]
        d = json.loads(f.read_text())
        if d.get("klass") != "inhibition":
            continue
        rows = d.get("rows") or []
        n = 0
        for r in rows:
            line = str(r.get("cell_line") or "").strip().lower()
            sp = str(r.get("cell_line_species") or "").strip().lower()
            rule = RULES.get((pid, line))
            if rule and sp in PLACEHOLDER:
                r["cell_line_species"] = rule[0]
                n += 1
        if n:
            changed[(pid, f.stem.split("_t")[1])] = n
            touched.append((f, d, rows, n))

    total = sum(changed.values())
    print(f"{total} rows across {len(touched)} tables")
    for (pid, t), n in sorted(changed.items()):
        print(f"  {pid} t{int(t)}: {n}")
    for (pid, line), (species, why) in RULES.items():
        print(f'\n  {line} -> {species}\n    {why}')
    if not apply:
        print("\ndry run; pass --apply to write")
        return

    for f, d, rows, _n in touched:
        shutil.copy(f, f.with_suffix(".json.prespecies2"))
        d["rows"] = rows
        d["comment"] = ((d.get("comment") or "") +
                        " | cell_line_species placeholder resolved from the cell "
                        "line (analyses/validation/fix_species_labels.py)").strip(" |")
        f.write_text(json.dumps(d, indent=1))
        csv_path = VERIFIED / f"{f.stem}.csv"
        if csv_path.exists() and rows:
            with csv_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\nwrote {len(touched)} tables")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().apply)
