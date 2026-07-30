"""Prepare ASO Atlas 2.0 for HuggingFace release.

Single source of truth for the release. Reads the 4 canonical parquets and emits:

  aso-atlas-2-release/
    <config>.parquet                          complete endpoint datasets
    release_manifest.json                     every quotable count
    croissant.json                            Croissant 1.1 core + NeurIPS RAI
                                              metadata, with pinned files/hashes
    README.md                                 dataset card generated from the manifest

The separate folds/ directory contains only the exact endpoint-specific fold
assignments used for the submitted paper. No additional recommended partition
is generated.

    uv run python aso-atlas-2-release/prepare_hf_release.py
"""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analyses.utils.compounds import has_measurement

PROCESSED = Path("data/oligostack/processed")
RELEASE = Path("aso-atlas-2-release")
CARD_TEMPLATE = RELEASE / "card_template.md"

RELEASE_VERSION = "neurips-rebuttal"
# One identifier shared by the dataset card, the Croissant metadata, and the git
# tag in *both* repositories, so the AC's "which snapshot is this?" has one answer.
RELEASE_TAG = RELEASE_VERSION
CODE_REPO = "https://github.com/barneyhill/aso-atlas-2"
HF_REPO = "https://huggingface.co/datasets/barneyhill/aso-atlas-2"
DATASET_DOI = "https://doi.org/10.57967/hf/8687"
# Initial public availability of the dataset. Version-specific identity is
# carried separately by version and the pinned release URLs.
DATE_PUBLISHED = "2026-04-30"
MANUSCRIPT_REF = "neurips-submission"
MANUSCRIPT_COMMIT = "ff973af"

CANONICAL_FILES = {
    "in_vitro_inhibition": "in_vitro_inhibition_processed.parquet",
    "dose_response": "dose_response_processed.parquet",
    "hepatotoxicity": "hepatictoxicity_processed.parquet",
    "neurotoxicity": "neurotoxicity_processed.parquet",
}

CONFIG_DESCRIPTIONS = {
    "in_vitro_inhibition": "Single-dose in vitro mRNA inhibition measurements",
    "dose_response": "Multi-dose in vitro inhibition measurements for dose-response curve fitting",
    "hepatotoxicity": "In vivo hepatic toxicity biomarker measurements (ALT, AST, ALB, BUN, CREA, TBIL, PC ratio)",
    "neurotoxicity": "In vivo neurotoxicity functional observation battery (FOB) scores",
}

INVITRO_COL_ORDER = [
    "USPTO ID", "Table Number", "Compound ID", "HELM Annotation",
    "model_eligible",
    "cell_line", "dosage_nm", "target_RNA", "Inhibition_pct",
    "cells_per_well", "cell_line_species", "transfection_method",
    "treatment_period_hrs", "ccle_cell_line_name", "ccle_model_id",
    "ccle_oncotree_lineage", "ccle_oncotree_disease", "cell_line_mapping_source",
]

BIOMARKER_COLS = ["ALB", "ALT", "AST", "BUN", "CREA", "TBIL", "PC_ratio"]

PAPER_FOLD_FILES = [
    "in_vitro_inhibition_folds.csv.gz",
    "potency_folds.csv.gz",
    "mouse_hepatic_folds.csv.gz",
    "rat_hepatic_folds.csv.gz",
    "mouse_neuro_folds.csv.gz",
    "rat_neuro_folds.csv.gz",
    "oligoai_folds.csv.gz",
]

CROISSANT_CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "annotation": "cr:annotation",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "description": {"@container": "@language"},
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "excludes": "cr:excludes",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "readLines": "cr:readLines",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sdVersion": "cr:sdVersion",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "unArchive": "cr:unArchive",
    "value": "cr:value",
    "name": {"@container": "@language"},
}


def validate_table(name: str, df: pd.DataFrame) -> None:
    for col in ("USPTO ID", "Compound ID"):
        n_null = df[col].isna().sum()
        if n_null:
            raise ValueError(f"{name}: {col} has {n_null} nulls")
    expected_eligibility = df["HELM Annotation"].notna()
    if not df["model_eligible"].equals(expected_eligibility):
        raise ValueError(f"{name}: model_eligible does not match HELM availability")
    if len(df) == 0:
        raise ValueError(f"{name}: empty table")


def readouts(name: str, df: pd.DataFrame) -> int:
    """Assay readouts, using the same unit definition as the paper.

    One non-null assay readout = one measurement; a hepatic row contributes up
    to 7 (one per biomarker channel).
    """
    if name == "hepatotoxicity":
        return int(
            df[BIOMARKER_COLS]
            .apply(lambda row: sum(has_measurement(row[c]) for c in BIOMARKER_COLS), axis=1)
            .sum()
        )
    if name == "neurotoxicity":
        return int(df["FOB_score"].apply(has_measurement).sum())
    return len(df)


def gene_symbol_map() -> pd.Series:
    """Return the transcript-to-gene mapping used by the submitted manuscript."""
    invitro = pd.read_parquet(
        PROCESSED / "in_vitro_inhibition_processed_with_genomic_data.parquet",
        columns=["target_RNA", "gene_symbol"],
    )
    dose_response = pd.read_parquet(
        PROCESSED / "dose_response_with_genomic.parquet",
        columns=["target_RNA", "gene_symbol"],
    )
    return (
        pd.concat([invitro, dose_response])
        .dropna(subset=["gene_symbol"])
        .drop_duplicates("target_RNA")
        .set_index("target_RNA")["gene_symbol"]
    )


def named_target_genes(df: pd.DataFrame, gene_map: pd.Series) -> set[str]:
    """Resolve non-null target identifiers to the manuscript's named-gene unit."""
    targets = df["target_RNA"].dropna().unique()
    return {str(gene_map.get(target, target)) for target in targets}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _croissant_type(arrow_type: pa.DataType) -> tuple[str, bool]:
    is_array = (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    )
    value_type = arrow_type.value_type if is_array else arrow_type
    if pa.types.is_boolean(value_type):
        return "sc:Boolean", is_array
    if pa.types.is_integer(value_type):
        return "sc:Integer", is_array
    if pa.types.is_floating(value_type) or pa.types.is_decimal(value_type):
        return "sc:Float", is_array
    return "sc:Text", is_array


def build_croissant(manifest: dict) -> dict:
    """Build the complete Croissant file required by NeurIPS 2026 E&D.

    Hugging Face generates core Croissant metadata from the card, but NeurIPS
    explicitly requires authors to add minimal Responsible AI and provenance
    fields. This checked-in file is the canonical version submitted to
    OpenReview; every downloadable URL is pinned to the release tag.
    """
    primary_paths = [Path(f"{config}.parquet") for config in CANONICAL_FILES]
    fold_paths = [Path("folds") / name for name in PAPER_FOLD_FILES]
    data_paths = primary_paths + fold_paths

    distribution = []
    for rel in data_paths:
        path = RELEASE / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing Croissant resource: {path}")
        is_parquet = path.suffix == ".parquet"
        distribution.append({
            "@type": "cr:FileObject",
            "@id": rel.as_posix(),
            "name": rel.name,
            "description": (
                "Version-pinned assay records."
                if is_parquet else
                "Exact endpoint-specific paper benchmark fold assignments."
            ),
            "contentUrl": (
                f"{HF_REPO}/resolve/{RELEASE_TAG}/{rel.as_posix()}?download=true"
            ),
            "encodingFormat": "application/x-parquet" if is_parquet else "application/gzip",
            "contentSize": f"{path.stat().st_size} B",
            "sha256": _sha256(path),
        })

    record_sets = []
    for rel in primary_paths:
        path = RELEASE / rel
        config = rel.stem
        record_id = config
        fields = []
        for arrow_field in pq.read_schema(path):
            data_type, is_array = _croissant_type(arrow_field.type)
            field_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", arrow_field.name).strip("_")
            field = {
                "@type": "cr:Field",
                "@id": f"{record_id}/{field_id}",
                "name": arrow_field.name,
                "dataType": data_type,
                "source": {
                    "fileObject": {"@id": rel.as_posix()},
                    "extract": {"column": arrow_field.name},
                },
            }
            if is_array:
                field["isArray"] = True
            fields.append(field)
        record_sets.append({
            "@type": "cr:RecordSet",
            "@id": record_id,
            "name": config,
            "description": (
                f"{manifest['configs'][config]['rows']:,} {config} assay records."
            ),
            "field": fields,
        })

    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": manifest["name"],
        "description": (
            "A versioned benchmark dataset for evaluating antisense oligonucleotide "
            "prediction across the preclinical pipeline, with in vitro inhibition, "
            "dose-response, hepatotoxicity and neurotoxicity assay records. "
            f"{manifest['linkage_definition']}"
        ),
        "conformsTo": [
            "http://mlcommons.org/croissant/1.1",
            "http://mlcommons.org/croissant/RAI/1.0",
        ],
        "url": f"{HF_REPO}/tree/{RELEASE_TAG}",
        "sameAs": DATASET_DOI,
        "version": manifest["version"],
        "datePublished": DATE_PUBLISHED,
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "creator": [
            {"@type": "sc:Person", "name": "Barney Hill"},
            {"@type": "sc:Person", "name": "Nicola Whiffin"},
            {"@type": "sc:Person", "name": "Carlo Rinaldi"},
            {"@type": "sc:Person", "name": "Stephan J. Sanders"},
        ],
        "keywords": [
            "antisense oligonucleotides", "drug discovery", "toxicity",
            "dose response", "benchmarking", "HELM",
        ],
        "citeAs": (
            "@dataset{hill_aso_atlas_2_2026, author={Hill, Barney and Whiffin, "
            "Nicola and Rinaldi, Carlo and Sanders, Stephan J.}, title={ASO Atlas "
            f"2.0}}, year={{2026}}, version={{{RELEASE_VERSION}}}, "
            "doi={10.57967/hf/8687}}"
        ),
        "isLiveDataset": False,
        "rai:dataLimitations": [
            "The measurements come from patent disclosures by one organisation and "
            "mainly represent 2'-MOE gapmers; other chemistries, organisations and "
            "experimental protocols are underrepresented.",
            "Patent reporting is selective, protocols are heterogeneous and the "
            "measurements have not been independently replicated across laboratories.",
            "LLM-assisted extraction can introduce errors despite automated checks and "
            "a manually verified audit. This dataset is not recommended for clinical "
            "decision-making, human safety prediction, dose selection or legal conclusions."
            ,
            "Rows without a resolvable HELM annotation are retained to preserve the complete "
            "assay corpus but cannot be used as chemistry-aware model inputs; model_eligible "
            "identifies the HELM-resolved benchmark subset."
        ],
        "rai:dataBiases": [
            "Single-originator and patent-selection bias may make reported pass rates and "
            "model performance unrepresentative of the wider ASO design space.",
            "Mouse and rat experiments dominate the in vivo records; monkey and dog data "
            "are sparse, and disclosed negative results may be incomplete.",
            "Cell-line, target-gene and assay-protocol coverage follows the source patents "
            "rather than a population-balanced sampling design."
        ],
        "rai:personalSensitiveInformation": (
            "No human-subject records, patient-level health information or personally "
            "identifiable information are included. The release contains public patent "
            "assay records, compound structures, animal-study measurements and public "
            "CCLE cell-line identifiers only."
        ),
        "rai:dataUseCases": (
            "Established uses are research benchmarking and exploratory model development "
            "for patent-derived ASO in vitro efficacy and preclinical toxicity endpoints, "
            "using the supplied patent-grouped folds. The records represent reported assay "
            "outcomes, not independently replicated biological truth. Validity has not been "
            "established for clinical decisions, human toxicity, treatment selection, "
            "cross-chemistry generalisation or prospective laboratory deployment."
        ),
        "rai:dataSocialImpact": (
            "Potential benefits include more reproducible ASO model evaluation, lower "
            "screening costs and reduced animal use. Potential harms arise if extraction "
            "errors or the source patents' narrow chemistry and target distribution are "
            "treated as universally representative, which could misprioritise candidates "
            "or research areas. Version pinning, source identifiers, leakage-controlled "
            "folds, a validation audit and explicit non-clinical-use limitations mitigate "
            "but do not eliminate these risks."
        ),
        "rai:hasSyntheticData": False,
        "prov:wasDerivedFrom": [
            {"@id": "https://ppubs.uspto.gov/pubwebapp/"},
            {"@id": "https://depmap.org/portal/data_page/?tab=allData"},
        ],
        "prov:wasGeneratedBy": [
            {
                "@type": "prov:Activity",
                "prov:label": "USPTO patent-table extraction",
                "description": (
                    "A three-stage GPT-5-mini/GPT-5 pipeline located and transformed "
                    "tables from public USPTO full-text XML. Models called Python for "
                    "data transformations rather than directly generating assay values."
                ),
            },
            {
                "@type": "prov:Activity",
                "prov:label": "Curation, annotation and quality control",
                "description": (
                    "HELM chemistry, gene and cell-line names were standardised; CCLE "
                    "identifiers were joined from DepMap Public 25Q3 Model.csv (downloaded "
                    "2026-01-20; SHA-256 9dbb9de8805696c1345816ab07edd23fb4fd95e117739f3c5c3b1cf062c1233b); "
                    "implausible values and unusable chemistry were filtered; duplicate "
                    "measurements were resolved. Manual mappings and a verified "
                    "patent-table audit checked extraction quality."
                ),
            },
        ],
        "distribution": distribution,
        "recordSet": record_sets,
    }


def main() -> None:
    RELEASE.mkdir(exist_ok=True)

    print("Reading processed tables...")
    frames: dict[str, pd.DataFrame] = {}
    processed_readouts = 0
    gene_map = gene_symbol_map()
    processed_genes: set[str] = set()
    for name, filename in CANONICAL_FILES.items():
        path = PROCESSED / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pq.read_table(path).to_pandas()
        processed_readouts += readouts(name, df)
        processed_genes.update(named_target_genes(df, gene_map))
        # Preserve the complete assay corpus. HELM-resolved rows form the
        # chemistry-aware modelling subset used by the paper benchmarks.
        df["model_eligible"] = df["HELM Annotation"].notna()
        print(
            f"  {name}: {len(df):,} rows; "
            f"{int((~df['model_eligible']).sum()):,} without resolved HELM"
        )
        validate_table(name, df)
        frames[name] = df

    for name in ("in_vitro_inhibition", "dose_response"):
        frames[name] = frames[name][INVITRO_COL_ORDER]

    print("\nWriting release parquets...")
    manifest_configs = {}
    for name, df in frames.items():
        out = RELEASE / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
        print(f"  {out} ({len(df):,} rows, {out.stat().st_size / 1024:.0f} KB)")

        manifest_configs[name] = {
            "description": CONFIG_DESCRIPTIONS[name],
            "rows": len(df),
            "assay_readouts": readouts(name, df),
            "compounds": int(df["HELM Annotation"].nunique()),
            "model_eligible_rows": int(df["model_eligible"].sum()),
            "unresolved_helm_rows": int((~df["model_eligible"]).sum()),
            "model_eligible_assay_readouts": readouts(
                name, df[df["model_eligible"]]
            ),
            "unresolved_helm_readouts": readouts(
                name, df[~df["model_eligible"]]
            ),
            "patents": int(df["USPTO ID"].nunique()),
            "named_target_genes": len(named_target_genes(df, gene_map)),
            "model_eligible_patents": int(
                df.loc[df["model_eligible"], "USPTO ID"].nunique()
            ),
            "model_eligible_named_target_genes": len(
                named_target_genes(df[df["model_eligible"]], gene_map)
            ),
        }

        # Remove the abandoned generated global-split layout from local staging.
        split_dir = RELEASE / name
        if split_dir.exists():
            shutil.rmtree(split_dir)
            print(f"  removed obsolete split directory {split_dir}")

    all_helms = set()
    all_patents = set()
    all_genes: set[str] = set()
    model_patents = set()
    model_genes: set[str] = set()
    for name, df in frames.items():
        all_helms.update(df["HELM Annotation"].dropna())
        all_patents.update(df["USPTO ID"])
        all_genes.update(named_target_genes(df, gene_map))
        eligible = df[df["model_eligible"]]
        model_patents.update(eligible["USPTO ID"])
        model_genes.update(named_target_genes(eligible, gene_map))

    source_id_sets = {
        name: set(df["Compound ID"].dropna().unique())
        for name, df in frames.items()
    }
    all_source_ids = set().union(*source_id_sets.values())
    source_id_membership = {
        source_id: sum(source_id in ids for ids in source_id_sets.values())
        for source_id in all_source_ids
    }
    cross_category_linkage = {
        "identity_key": "Compound ID",
        "total_source_ids": len(all_source_ids),
        "source_ids_in_multiple_categories": sum(
            count >= 2 for count in source_id_membership.values()
        ),
    }
    cross_category_linkage["pct_in_multiple_categories"] = round(
        100
        * cross_category_linkage["source_ids_in_multiple_categories"]
        / cross_category_linkage["total_source_ids"],
        1,
    )

    totals = {
        "released_assay_readouts": sum(c["assay_readouts"] for c in manifest_configs.values()),
        "compounds": len(all_helms),
        "patents": len(all_patents),
        "named_target_genes": len(all_genes),
        "model_eligible_assay_readouts": sum(
            c["model_eligible_assay_readouts"] for c in manifest_configs.values()
        ),
        "unresolved_helm_readouts": sum(
            c["unresolved_helm_readouts"] for c in manifest_configs.values()
        ),
        "model_eligible_patents": len(model_patents),
        "model_eligible_named_target_genes": len(model_genes),
    }

    manifest = {
        "name": "ASO Atlas 2.0",
        "version": RELEASE_VERSION,
        "release_tag": RELEASE_TAG,
        "code_repo": CODE_REPO,
        "manuscript_commit": MANUSCRIPT_COMMIT,
        "manuscript_ref": MANUSCRIPT_REF,
        "unit_definition": (
            "A row is one assay record. A measurement (assay readout) is one non-null "
            "assay value: in vitro inhibition and dose-response rows carry one readout "
            "each, a hepatotoxicity row carries up to 7 (one per biomarker channel), and "
            "a neurotoxicity row carries one FOB readout."
        ),
        "gene_definition": (
            "A named target gene is one unique non-null target_RNA after resolving "
            "transcript identifiers to gene symbols with the same genomic annotation "
            "mapping used by the submitted manuscript."
        ),
        "compound_definition": (
            "A unique HELM compound is one unique non-null HELM Annotation. Rows without "
            "a resolvable HELM remain in the assay corpus but are excluded from "
            "unique-HELM-compound and model-benchmark counts."
        ),
        "linkage_definition": (
            "Cross-category linkage retains the submitted source-identifier unit: a "
            "distinct Compound ID reported in at least two of the four assay categories. "
            f"By this definition, {cross_category_linkage['source_ids_in_multiple_categories']:,} "
            f"of {cross_category_linkage['total_source_ids']:,} source Compound IDs "
            f"({cross_category_linkage['pct_in_multiple_categories']:.1f}%) are linked."
        ),
        "model_eligibility_definition": (
            "model_eligible is true exactly when HELM Annotation is non-null. These rows "
            "form the chemistry-aware modelling subset; false rows remain valid assay "
            "readouts for corpus and phenotype-frequency analyses."
        ),
        "paper_benchmark_folds": {
            "directory": "folds/",
            "role": (
                "Exact endpoint-specific five-fold assignments used for the submitted "
                "paper's benchmark results."
            ),
            "oligogym_endpoints": [
                "in_vitro_inhibition", "potency", "mouse_hepatic",
                "rat_hepatic", "mouse_neuro", "rat_neuro",
            ],
            "oligoai_file": "folds/oligoai_folds.csv.gz",
            "validation": (
                "OligoGym patent-to-fold membership is checked against groups_test in "
                "data/results/oligogym_benchmark.json; OligoAI membership is read from "
                "the five frozen oligoai_train_fold CSVs."
            ),
        },
        "totals": totals,
        "cross_category_linkage": cross_category_linkage,
        "configs": manifest_configs,
    }

    manifest["paper_cross_reference"] = {
        "paper_assay_readouts": processed_readouts,
        "note": (
            "The manuscript and release count assay readouts over every processed row. "
            "Rows whose HELM could not be resolved remain public and are marked "
            "model_eligible=false; chemistry-aware model analyses use the resolved "
            "subset, while cross-category linkage uses the submitted source Compound ID unit."
        ),
        "delta": processed_readouts - totals["released_assay_readouts"],
        "paper_named_target_genes": len(processed_genes),
        "released_named_target_genes": len(all_genes),
        "excluded_named_target_genes": sorted(processed_genes - all_genes),
    }

    (RELEASE / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {RELEASE / 'release_manifest.json'}")

    croissant = build_croissant(manifest)
    (RELEASE / "croissant.json").write_text(json.dumps(croissant, indent=2) + "\n")
    print(f"Wrote {RELEASE / 'croissant.json'} (Croissant 1.1 + NeurIPS RAI)")

    (RELEASE / "README.md").write_text(render_card(manifest))
    print(f"Wrote {RELEASE / 'README.md'} (generated — do not hand-edit)")

    print("\n=== Release Summary ===")
    print(f"Assay readouts:    {totals['released_assay_readouts']:,}")
    print(f"Unique compounds:  {totals['compounds']:,}")
    print(f"Unique patents:    {totals['patents']:,}")
    print(f"Named genes:       {totals['named_target_genes']:,}")
    print(f"Model readouts:    {totals['model_eligible_assay_readouts']:,}")
    print(f"Unresolved HELM:   {totals['unresolved_helm_readouts']:,} readouts")
    print("\nDone. Now run: uv run python aso-atlas-2-release/upload_hf_release.py")


def render_card(m: dict) -> str:
    """Fill card_template.md from the manifest.

    Tokens are <<NAME>> rather than str.format placeholders because the card
    contains literal HELM braces.
    """
    cfg_yaml = []
    for name, c in m["configs"].items():
        cfg_yaml.append(f"  - config_name: {name}")
        cfg_yaml.append(f"    description: {c['description']}")
        cfg_yaml.append("    data_files:")
        cfg_yaml.append("      - split: full")
        cfg_yaml.append(f"        path: {name}.parquet")

    table = [
        "| Config | Readouts | Model-eligible readouts | Unique HELM compounds | Named target genes |",
        "|--------|---------:|------------------------:|----------------------:|-------------------:|",
    ]
    for name, c in m["configs"].items():
        table.append(
            f"| `{name}` | {c['assay_readouts']:,} | "
            f"{c['model_eligible_assay_readouts']:,} | {c['compounds']:,} | "
            f"{c['named_target_genes']:,} |"
        )
    table.append(
        f"| **Total** | **{m['totals']['released_assay_readouts']:,}** | "
        f"**{m['totals']['model_eligible_assay_readouts']:,}** | "
        f"**{m['totals']['compounds']:,}** | "
        f"**{m['totals']['named_target_genes']:,}** |"
    )

    t = m["totals"]
    xref = m.get("paper_cross_reference")
    xref_line = ""
    if xref:
        xref_line = (
            f"The submitted manuscript and tagged `{m['version']}` snapshot both contain "
            f"{xref['paper_assay_readouts']:,} processed readouts. Of these, "
            f"{t['unresolved_helm_readouts']:,} lack a resolvable HELM annotation and are "
            f"retained with `model_eligible=false`; the remaining "
            f"{t['model_eligible_assay_readouts']:,} readouts form the chemistry-aware "
            "modelling subset. The manuscript's "
            "1,125 count refers to table-bearing source documents entering extraction; "
            f"{t['patents']:,} USPTO filings contribute data to the complete release and "
            f"{t['model_eligible_patents']:,} contribute to its HELM-resolved subset."
        )

    tokens = {
        "<<CONFIGS_YAML>>": "\n".join(cfg_yaml),
        "<<VERSION>>": m["version"],
        "<<RELEASE_TAG>>": m["release_tag"],
        "<<CODE_REPO>>": m["code_repo"],
        "<<MANUSCRIPT_COMMIT>>": m["manuscript_commit"],
        "<<MANUSCRIPT_REF>>": m["manuscript_ref"],
        "<<SUMMARY_TABLE>>": "\n".join(table),
        "<<PAPER_XREF>>": xref_line,
        "<<TOTAL_READOUTS>>": f"{t['released_assay_readouts']:,}",
        "<<TOTAL_COMPOUNDS>>": f"{t['compounds']:,}",
        "<<TOTAL_PATENTS>>": f"{t['patents']:,}",
        "<<TOTAL_GENES>>": f"{t['named_target_genes']:,}",
        "<<MODEL_READOUTS>>": f"{t['model_eligible_assay_readouts']:,}",
        "<<UNRESOLVED_READOUTS>>": f"{t['unresolved_helm_readouts']:,}",
        "<<GENE_DEFINITION>>": m["gene_definition"],
        "<<COMPOUND_DEFINITION>>": m["compound_definition"],
        "<<LINKAGE_DEFINITION>>": m["linkage_definition"],
        "<<MODEL_ELIGIBILITY_DEFINITION>>": m["model_eligibility_definition"],
    }

    card = CARD_TEMPLATE.read_text()
    for token, value in tokens.items():
        card = card.replace(token, value)

    unfilled = [line for line in card.splitlines() if "<<" in line and ">>" in line]
    if unfilled:
        raise ValueError(f"card_template.md has unfilled tokens: {unfilled}")
    return card


if __name__ == "__main__":
    main()
