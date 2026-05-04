---
license: cc-by-4.0
language:
  - en
pretty_name: ASO Atlas 2.0
size_categories:
  - 100K<n<1M
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - biology
  - chemistry
  - drug-discovery
  - antisense-oligonucleotides
  - toxicity
  - benchmarking
configs:
  - config_name: in_vitro_inhibition
    data_files: in_vitro_inhibition.parquet
    description: Single-dose in vitro mRNA inhibition measurements
  - config_name: dose_response
    data_files: dose_response.parquet
    description: Multi-dose in vitro inhibition measurements for dose-response curve fitting
  - config_name: hepatotoxicity
    data_files: hepatotoxicity.parquet
    description: In vivo hepatic toxicity biomarker measurements (ALT, AST, ALB, BUN, CREA, TBIL, PC ratio)
  - config_name: neurotoxicity
    data_files: neurotoxicity.parquet
    description: In vivo neurotoxicity functional observation battery (FOB) scores
---

# ASO Atlas 2.0

A benchmark dataset for evaluating antisense oligonucleotide (ASO) prediction across the preclinical pipeline.

## Dataset Summary

ASO Atlas 2.0 contains 283,239 measurements across 168,395 unique ASO compounds extracted from 606 USPTO patent filings via an LLM-based extraction pipeline. Each compound's chemistry is encoded in [HELM notation](https://piscienceofhope.github.io/HELMNotationToolkit/), enabling direct use as model input.

The dataset spans four preclinical assay types that mirror the sequential screening pipeline used in ASO drug development:

| Config | Rows | Compounds | Description |
|--------|------|-----------|-------------|
| `in_vitro_inhibition` | 174,297 | 163,688 | Single-dose mRNA knockdown (% inhibition) |
| `dose_response` | 99,719 | 16,814 | Multi-dose inhibition for IC50 fitting |
| `hepatotoxicity` | 4,323 | 2,037 | Liver toxicity biomarkers (ALT, AST, etc.) |
| `neurotoxicity` | 4,900 | 3,316 | Functional observation battery scores |

## Usage

```python
from datasets import load_dataset

# Load a single config
ds = load_dataset("barneyhill/aso-atlas-2", "in_vitro_inhibition")

# Load all configs
configs = ["in_vitro_inhibition", "dose_response", "hepatotoxicity", "neurotoxicity"]
atlas = {c: load_dataset("barneyhill/aso-atlas-2", c) for c in configs}
```

## Data Fields

### Common fields (all configs)

| Field | Type | Description |
|-------|------|-------------|
| `USPTO ID` | string | Source patent identifier (e.g., `US20150011212`) |
| `Table Number` | int | Table number within the patent |
| `Compound ID` | int | Compound identifier within the patent |
| `HELM Annotation` | string | Full chemical structure in HELM notation |
| `target_RNA` | string | Target gene/transcript name |

### in_vitro_inhibition / dose_response

| Field | Type | Description |
|-------|------|-------------|
| `cell_line` | string | Cell line used (e.g., HepG2, A549) |
| `dosage_nm` | float | Dose in nanomolar |
| `Inhibition_pct` | float | mRNA inhibition percentage (−1000 to 100) |
| `cells_per_well` | float | Cells seeded per well |
| `cell_line_species` | string | Species of origin (human, mouse, etc.) |
| `transfection_method` | string | Delivery method (electroporation, lipofection, etc.) |
| `treatment_period_hrs` | float | Treatment duration in hours |
| `ccle_cell_line_name` | string | Matched CCLE cell line name |
| `ccle_model_id` | string | DepMap model ID |
| `ccle_oncotree_lineage` | string | Oncotree tissue lineage |
| `ccle_oncotree_disease` | string | Oncotree disease classification |
| `cell_line_mapping_source` | string | How the CCLE match was determined |

### hepatotoxicity

| Field | Type | Description |
|-------|------|-------------|
| `species` | string | Test species (mouse, rat, monkey, dog) |
| `species_strain` | string | Strain (e.g., CD-1 mice, Sprague-Dawley rats) |
| `dosage_mg_per_kg` | float | Dose per administration (mg/kg) |
| `num_doses` | float | Number of doses administered |
| `dosing_period_days` | float | Duration of dosing period |
| `adminstration_method` | string | Route (subcutaneous, intraperitoneal) |
| `measurement_source` | string | Sample type (plasma, urine) |
| `ALT` | list[float] | Alanine aminotransferase values (IU/L) |
| `AST` | list[float] | Aspartate aminotransferase values (IU/L) |
| `ALB` | list[float] | Albumin values (g/dL) |
| `BUN` | list[float] | Blood urea nitrogen values (mg/dL) |
| `CREA` | list[float] | Creatinine values (mg/dL) |
| `TBIL` | list[float] | Total bilirubin values (mg/dL) |
| `PC_ratio` | list[float] | Protein/creatinine ratio |

### neurotoxicity

| Field | Type | Description |
|-------|------|-------------|
| `species` | string | Test species |
| `species_strain` | string | Strain |
| `dosage_ug` | float | Dose in micrograms |
| `FOB_score` | list[float] | Functional observation battery scores (0–7) |
| `latency_time_hours` | float | Time from dosing to observation |
| `administration_method` | string | Route (typically intracerebroventricular) |
| `tolerability_score_type` | string | Scoring system used |

## HELM Chemistry Encoding

HELM (Hierarchical Editing Language for Macromolecules) encodes full oligonucleotide chemistry including sugar modifications, backbone linkages, and base modifications.

**Example:** `RNA1{[moe](C)[sp].[moe](T)[sp].d(A)[sp].d(G)[sp].[cet](A)}$$$$`

| Sugar | Chemistry | Backbone | Chemistry |
|-------|-----------|----------|-----------|
| `d` | 2'-deoxy (DNA) | `[sp]` | Phosphorothioate |
| `[moe]` | 2'-O-methoxyethyl | `.` | Phosphodiester |
| `[cet]` | Constrained ethyl | `[am]` | Phosphoramidate |
| `[lna]` | Locked nucleic acid | | |
| `[fR]` | 2'-fluoro | | |
| `[m]` | 2'-O-methyl | | |

## Dataset Creation

### Source

All data was extracted from publicly available USPTO patent filings using an LLM-based text mining pipeline. Patents were identified via keyword search for antisense oligonucleotide-related terms, and data tables within each patent were parsed to extract compound structures, assay conditions, and measured endpoints.

### Processing

1. HELM quality filtering: removed sequences with uncertain chemistry (`?`), length <11 nucleotides, missing DNA gap, homopolymers, or naked DNA
2. Gene name standardisation via manual curated mappings
3. Cell line standardisation and CCLE (Cancer Cell Line Encyclopedia) enrichment
4. Inhibition values filtered to [−1000%, 100%]
5. Hepatic biomarker values filtered to physiologically plausible ranges
6. FOB scores validated to 0–7 range
7. Deduplication by compound and measurement

### No canonical train/test split

This dataset ships raw measurements without a predefined split. The accompanying paper uses patent-level GroupKFold cross-validation to prevent data leakage (compounds from the same patent always stay in the same fold). We recommend the same strategy.

## Considerations

### Biases

- **Patent bias**: Data comes exclusively from pharmaceutical patent filings, which over-represent compound classes pursued by industry (predominantly 2'-MOE gapmers). Academic and early-stage chemistries may be underrepresented.
- **Species bias**: Mouse and rat data dominate; monkey and dog data are sparse.
- **Reporting bias**: Patents report results selectively. Negative results may be underrepresented.

### Limitations

- **No independent validation**: All measurements originate from patent applicants' own experiments. Cross-laboratory reproducibility is unknown.
- **Extraction errors**: LLM-based extraction may introduce errors. A validation audit is included in the accompanying code repository.
- **Heterogeneous protocols**: Assay conditions (cell lines, doses, timepoints) vary across patents and are not standardised.

### Ethical considerations

No human subjects data. All data derives from public USPTO patent documents and public biological databases (CCLE). No personally identifiable information is present.```
