"""Prepare ASO Atlas 2.0 for HuggingFace release.

Reads the 4 canonical parquets, validates schemas, standardises column order,
and writes release-ready parquets + dataset card to aso-atlas-2-release/.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROCESSED = Path("data/oligostack/processed")
RELEASE = Path("aso-atlas-2-release")

CANONICAL_FILES = {
    "in_vitro_inhibition": "in_vitro_inhibition_processed.parquet",
    "dose_response": "dose_response_processed.parquet",
    "hepatotoxicity": "hepatictoxicity_processed.parquet",
    "neurotoxicity": "neurotoxicity_processed.parquet",
}

INVITRO_COL_ORDER = [
    "USPTO ID", "Table Number", "Compound ID", "HELM Annotation",
    "cell_line", "dosage_nm", "target_RNA", "Inhibition_pct",
    "cells_per_well", "cell_line_species", "transfection_method",
    "treatment_period_hrs", "ccle_cell_line_name", "ccle_model_id",
    "ccle_oncotree_lineage", "ccle_oncotree_disease", "cell_line_mapping_source",
]


def validate_table(name: str, table: pa.Table) -> None:
    for field in table.schema:
        if field.name in ("USPTO ID", "Compound ID", "HELM Annotation"):
            col = table.column(field.name)
            null_count = col.null_count
            if null_count > 0:
                raise ValueError(f"{name}: {field.name} has {null_count} nulls")

    if table.num_rows == 0:
        raise ValueError(f"{name}: empty table")

    print(f"  {name}: {table.num_rows:,} rows, {table.num_columns} cols — OK")


def main() -> None:
    RELEASE.mkdir(exist_ok=True)

    print("Reading and filtering...")
    tables = {}
    for name, filename in CANONICAL_FILES.items():
        path = PROCESSED / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        table = pq.read_table(path)
        # Drop rows without HELM — no chemistry, no use for ML
        helm_not_null = pc.is_valid(table.column("HELM Annotation"))
        n_before = table.num_rows
        table = table.filter(helm_not_null)
        if table.num_rows < n_before:
            print(f"  {name}: dropped {n_before - table.num_rows:,} rows with null HELM")
        tables[name] = table

    print("\nValidating...")
    for name, table in tables.items():
        validate_table(name, table)

    # Standardise in_vitro and dose_response column order
    for name in ("in_vitro_inhibition", "dose_response"):
        tables[name] = tables[name].select(INVITRO_COL_ORDER)

    print("\nWriting release parquets...")
    for name, table in tables.items():
        out = RELEASE / f"{name}.parquet"
        pq.write_table(table, out)
        print(f"  {out} ({out.stat().st_size / 1024:.0f} KB)")

    # Summary statistics for the dataset card
    print("\n=== Release Summary ===")
    total_rows = sum(t.num_rows for t in tables.values())
    print(f"Total rows: {total_rows:,}")
    for name, table in tables.items():
        compounds = table.column("Compound ID").to_pylist()
        patents = table.column("USPTO ID").to_pylist()
        print(f"  {name}: {table.num_rows:,} rows, "
              f"{len(set(compounds)):,} compounds, "
              f"{len(set(patents)):,} patents")

    all_compounds = set()
    all_patents = set()
    for table in tables.values():
        all_compounds.update(table.column("Compound ID").to_pylist())
        all_patents.update(table.column("USPTO ID").to_pylist())
    print(f"Unique compounds: {len(all_compounds):,}")
    print(f"Unique patents: {len(all_patents):,}")

    print("\nDone. Now run: uv run python aso-atlas-2-release/upload_hf_release.py")


if __name__ == "__main__":
    main()
