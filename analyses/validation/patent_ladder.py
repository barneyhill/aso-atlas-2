"""Reconcile the patent counts quoted at different pipeline stages (rebuttal C1).

Reviewer Tw5Q W1/Q1 and AC #1 both flag "1,125 patents" in the Methods against
"606 patents" on the dataset card. They count different stages, and this script
establishes the whole ladder from the source list to the released files so
neither number has to be asserted.

Each stage is a strict subset of the one above it; the script fails if that
nesting ever breaks, which is what would signal a real inconsistency rather than
a definitional one.

Requires the extraction repo alongside this one (same dependency as
corpus_completeness.py).

    uv run python analyses/validation/patent_ladder.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SCRAPE = Path.home() / "dphil" / "tox-patent-scrape" / "data"
APPS_CSV = SCRAPE / "ionis-isis-patent-apps.csv"
XML_DIR = SCRAPE / "step2"

RAW = ROOT / "data" / "oligostack" / "raw"
PROCESSED = ROOT / "data" / "oligostack" / "processed"
OUT_JSON = ROOT / "data" / "results" / "patent_ladder.json"

PROCESSED_FILES = [
    "in_vitro_inhibition_processed.parquet",
    "dose_response_processed.parquet",
    "hepatictoxicity_processed.parquet",
    "neurotoxicity_processed.parquet",
]


def normalise(display_key: str) -> str:
    """'US 2004/0220395 A1' -> 'US20040220395A1', matching the XML filenames."""
    return re.sub(r"[\s/]", "", str(display_key))


def main() -> None:
    if not APPS_CSV.exists():
        raise SystemExit(f"Missing {APPS_CSV} — the extraction repo is not alongside this one")

    apps = pd.read_csv(APPS_CSV, low_memory=False)
    listed = {normalise(k) for k in apps["Display Key"].dropna()}

    xml_files = [p for p in XML_DIR.iterdir() if p.is_file()]
    downloaded = {p.stem.split("-")[0] for p in xml_files}
    # The Methods' 1,125 is the set entering table extraction, i.e. those XMLs
    # that actually contain a table element. It is NOT the count retrieved.
    with_table = {
        p.stem.split("-")[0]
        for p in xml_files
        if "<table" in p.read_text(errors="ignore").lower()
    }

    extracted: set[str] = set()
    for path in sorted(RAW.glob("*.csv")):
        col = pd.read_csv(path, usecols=["USPTO ID"], low_memory=False)["USPTO ID"]
        extracted |= set(col.dropna().astype(str))

    cleaned: set[str] = set()
    released: set[str] = set()
    for filename in PROCESSED_FILES:
        table = pq.read_table(
            PROCESSED / filename, columns=["USPTO ID", "HELM Annotation"]
        ).to_pandas()
        cleaned |= set(table["USPTO ID"])
        released |= set(table[table["HELM Annotation"].notna()]["USPTO ID"])

    ladder = [
        ("listed_applications", listed,
         "Ionis/Isis A1 applications in the source Lens export"),
        ("xmls_downloaded", downloaded,
         "USPTO full-text XMLs successfully retrieved"),
        ("xmls_with_a_table", with_table,
         "XMLs containing at least one <table> element; the figure quoted in Methods"),
        ("patents_with_extracted_rows", extracted,
         "At least one row extracted into a raw collation file"),
        ("patents_after_cleaning", cleaned,
         "At least one measurement surviving QC, HGNC mapping and deduplication"),
        ("patents_in_release", released,
         "At least one measurement with a resolvable HELM annotation; the card's count"),
    ]

    # Every stage must be a subset of the previous one. A stray patent appearing
    # downstream would mean the stages are not actually nested, which is the only
    # version of this that would be a genuine error rather than a definition.
    strays = {}
    for (name, current, _), (prev_name, previous, _) in zip(ladder[1:], ladder[:-1]):
        extra = current - previous
        if extra:
            strays[f"{name}_not_in_{prev_name}"] = sorted(extra)[:20]

    result = {
        "stages": [
            {"name": name, "patents": len(members), "description": description}
            for name, members, description in ladder
        ],
        "drops": {
            f"{prev_name}_to_{name}": len(previous) - len(current)
            for (name, current, _), (prev_name, previous, _)
            in zip(ladder[1:], ladder[:-1])
        },
        "strays": strays,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    width = max(len(name) for name, _, _ in ladder)
    for name, members, description in ladder:
        print(f"  {name:<{width}}  {len(members):>6,}   {description}")
    print("\n  drops: " + ", ".join(f"{k} {v:,}" for k, v in result["drops"].items()))
    print(f"\nWrote {OUT_JSON}")

    if strays:
        raise SystemExit(
            "Pipeline stages are not nested; these patents appear downstream of a "
            f"stage that excludes them: {strays}"
        )


if __name__ == "__main__":
    main()
