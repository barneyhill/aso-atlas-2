"""
Generate validation audit spreadsheet for ASO Atlas 2.0 LLM extraction pipeline.

Samples 100 rows from 100 distinct patents, stratified equally across 4 datasets
(25 each), and produces a CSV for manual comparison against patent source XMLs.

Usage:
    uv run python analyses/validation/generate_audit.py [--seed 2026] [--output data/validation_audit.csv]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "oligostack" / "processed"
PATENT_DIRS = [
    ROOT / "pipeline" / "interface" / "static" / "patents" / d
    for d in ("IONIS", "IONIS2", "scn2a", "test")
]

DATASETS = [
    {
        "name": "neurotoxicity",
        "file": "neurotoxicity_processed.parquet",
        "n_sample": 25,
        "fields": [
            "FOB_score", "dosage_ug", "species",
            "species_strain", "latency_time_hours", "administration_method",
            "tolerability_score_type",
        ],
    },
    {
        "name": "hepatotoxicity",
        "file": "hepatictoxicity_processed.parquet",
        "n_sample": 25,
        "fields": [
            "ALT", "dosage_mg_per_kg", "species", "species_strain",
            "num_doses", "dosing_period_days", "measurement_source",
            "adminstration_method", "target_RNA",
        ],
    },
    {
        "name": "dose_response",
        "file": "dose_response_processed.parquet",
        "n_sample": 25,
        "fields": [
            "Inhibition_pct", "dosage_nm", "cell_line",
            "target_RNA", "cell_line_species", "transfection_method",
            "treatment_period_hrs",
        ],
    },
    {
        "name": "in_vitro_inhibition",
        "file": "in_vitro_inhibition_processed.parquet",
        "n_sample": 25,
        "fields": [
            "Inhibition_pct", "dosage_nm", "cell_line",
            "target_RNA", "cell_line_species", "transfection_method",
            "treatment_period_hrs",
        ],
    },
]


def find_xml(patent_id: str, table_num: str) -> tuple[str, str]:
    """Find the XML and context files, handling padded/unpadded table numbers."""
    candidates = [str(table_num)]
    stripped = str(table_num).lstrip("0") or "0"
    if stripped != str(table_num):
        candidates.append(stripped)
    padded = str(table_num).zfill(5)
    if padded != str(table_num):
        candidates.append(padded)

    for d in PATENT_DIRS:
        for tn in candidates:
            xml = d / f"{patent_id}_table_{tn}.xml"
            if xml.exists():
                ctx = d / f"{patent_id}_table_{tn}_context.txt"
                return str(xml), str(ctx) if ctx.exists() else ""
    return "", ""


def helm_to_sequence(helm: str) -> tuple[str, int]:
    bases = re.findall(r"\(([^)]+)\)", helm)
    seq = "".join(b.replace("[5meC]", "C") for b in bases)
    return seq, len(bases)


def sample_dataset(
    config: dict,
    rng: np.random.Generator,
    used_patents: set[str],
) -> list[dict]:
    df = pd.read_parquet(PROCESSED_DIR / config["file"])
    available = sorted(set(df["USPTO ID"].unique()) - used_patents)

    n = min(config["n_sample"], len(available))
    if n < config["n_sample"]:
        print(
            f"  WARNING: {config['name']} only has {len(available)} available patents "
            f"(requested {config['n_sample']})"
        )

    selected_patents = rng.choice(available, size=n, replace=False)

    rows = []
    for patent_id in selected_patents:
        patent_rows = df[df["USPTO ID"] == patent_id]
        idx = rng.integers(len(patent_rows))
        row = patent_rows.iloc[idx]

        xml_path, ctx_path = find_xml(patent_id, str(row["Table Number"]))

        audit_row = {
            "dataset": config["name"],
            "USPTO_ID": patent_id,
            "Table_Number": row["Table Number"],
            "Compound_ID": row["Compound ID"],
            "xml_filepath": xml_path,
            "context_filepath": ctx_path,
            "xml_found": bool(xml_path),
            "patent_url": f"https://patents.google.com/patent/{patent_id}",
        }

        helm = row.get("HELM Annotation", "")
        audit_row["extracted_HELM_Annotation"] = helm if pd.notna(helm) else ""
        if pd.notna(helm) and "(" in str(helm):
            seq, seq_len = helm_to_sequence(str(helm))
            audit_row["extracted_sequence"] = seq
            audit_row["seq_len"] = seq_len
        else:
            audit_row["extracted_sequence"] = ""
            audit_row["seq_len"] = ""

        for field in config["fields"]:
            val = row.get(field, "")
            if isinstance(val, (list, np.ndarray)):
                audit_row[f"extracted_{field}"] = str(val)
            elif pd.notna(val):
                audit_row[f"extracted_{field}"] = val
            else:
                audit_row[f"extracted_{field}"] = ""

        rows.append(audit_row)

    used_patents.update(selected_patents)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=str, default="data/validation_audit.csv"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    used_patents: set[str] = set()
    all_rows: list[dict] = []

    for config in DATASETS:
        print(f"Sampling {config['n_sample']} from {config['name']}...")
        rows = sample_dataset(config, rng, used_patents)
        all_rows.extend(rows)
        print(f"  Sampled {len(rows)} rows from {len(rows)} patents")

    df = pd.DataFrame(all_rows)
    df.insert(0, "audit_id", range(1, len(df) + 1))
    df["verdict_numerical"] = ""
    df["verdict_sequence"] = ""
    df["verdict_chemistry"] = ""
    df["notes"] = ""

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    xml_found = df["xml_found"].sum()
    print(f"\nWrote {len(df)} audit rows to {out_path}")
    print(f"  Unique patents: {df['USPTO_ID'].nunique()}")
    print(f"  XML files found: {xml_found}/{len(df)}")
    for name in df["dataset"].unique():
        n = (df["dataset"] == name).sum()
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
