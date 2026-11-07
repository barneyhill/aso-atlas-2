---
license: cc-by-4.0
language:
  - en
pretty_name: ASO Atlas 2.0
version: 2.0
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
    description: Single-dose in vitro mRNA inhibition measurements
    data_files:
      - split: full
        path: in_vitro_inhibition.parquet
  - config_name: dose_response
    description: Multi-dose in vitro inhibition measurements for dose-response curve fitting
    data_files:
      - split: full
        path: dose_response.parquet
  - config_name: hepatotoxicity
    description: In vivo hepatic toxicity biomarker measurements (ALT, AST, ALB, BUN, CREA, TBIL, PC ratio)
    data_files:
      - split: full
        path: hepatotoxicity.parquet
  - config_name: neurotoxicity
    description: In vivo neurotoxicity functional observation battery (FOB) scores
    data_files:
      - split: full
        path: neurotoxicity.parquet
---

# ASO Atlas 2.0

A dataset of patent-reported antisense oligonucleotide (ASO) experiments for studying
in vitro activity, dose response, hepatotoxicity and neurotoxicity across the preclinical
pipeline. It is designed for research benchmarking, exploratory modelling and analysis
of the published ASO assay landscape.

## At a glance

- **295,007 assay readouts** from **606 USPTO patent filings**
- **165,782 unique, chemistry-resolved ASOs** represented in HELM notation
- **430 named target genes**
- Four ready-to-load configurations covering in vitro efficacy and in vivo safety
- Patent-grouped cross-validation folds for leakage-controlled benchmarking

A named target gene is one unique non-null target_RNA after resolving transcript identifiers to gene symbols with the release's genomic annotation mapping.

## Quick start

```python
from datasets import load_dataset

dataset = load_dataset(
    "barneyhill/aso-atlas-2",
    "in_vitro_inhibition",
    split="full",
)
print(dataset[0])
```

Replace `in_vitro_inhibition` with `dose_response`, `hepatotoxicity` or
`neurotoxicity` to load another assay type.

## What's included

Each configuration is a separate table because the experimental endpoints and schemas
differ. Together they cover four stages commonly encountered in preclinical ASO
evaluation:

| Config | Readouts | Model-eligible readouts | Unique HELM compounds | Named target genes |
|--------|---------:|------------------------:|----------------------:|-------------------:|
| `in_vitro_inhibition` | 174,459 | 173,286 | 161,434 | 389 |
| `dose_response` | 99,821 | 97,236 | 16,375 | 130 |
| `hepatotoxicity` | 15,831 | 15,427 | 1,881 | 42 |
| `neurotoxicity` | 4,896 | 4,892 | 3,312 | 15 |
| **Total** | **295,007** | **290,841** | **165,782** | **430** |

Of the 295,007 readouts, 290,841 have a resolved HELM annotation.
The remaining 4,166 retain a reported assay outcome but cannot be used
directly by chemistry-aware models.

## Benchmark folds

The [`folds/`](folds/) directory provides frozen, endpoint-specific five-fold assignments
for reproducible evaluation with OligoGym and OligoAI. The splits group observations by
source patent to reduce train-test leakage.

Each model-ready endpoint was constructed separately before patent-level `GroupKFold` was
applied. A patent's fold number can therefore differ between endpoints and between
OligoGym and OligoAI. In each file, rows assigned to fold `k` are the test set for run
`k`; the other four folds form the training set. See [`folds/README.md`](folds/README.md)
for schemas and fold sizes.

## Versioning and reproducibility

This card describes **ASO Atlas 2.0**. The data tables, generated counts and machine-readable
metadata are released together. The source and release-building code is available in the
[ASO Atlas 2 repository](https://github.com/barneyhill/aso-atlas-2), and [`croissant.json`](croissant.json) provides
Croissant 1.1 core + RAI metadata. The dataset has a persistent Hugging Face DOI:
[`10.57967/hf/8687`](https://doi.org/10.57967/hf/8687).

## Data Fields

### Common fields (all configs)

| Field | Type | Description |
|-------|------|-------------|
| `USPTO ID` | string | Source patent identifier (e.g., `US20150011212`) |
| `Table Number` | int | Table number within the patent |
| `Compound ID` | int | Compound identifier within the patent |
| `HELM Annotation` | string, nullable | Full chemical structure in HELM notation; null where chemistry could not be resolved |
| `model_eligible` | bool | True when a resolved HELM annotation permits chemistry-aware model input |
| `target_RNA` | string | Target gene/transcript name |

### in_vitro_inhibition / dose_response

| Field | Type | Description |
|-------|------|-------------|
| `cell_line` | string | Cell line used (e.g., HepG2, A549) |
| `dosage_nm` | float | Dose in nanomolar |
| `Inhibition_pct` | float | mRNA inhibition percentage (-1000 to 100) |
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
| `FOB_score` | list[float] | Functional observation battery scores (0-7) |
| `latency_time_hours` | float | Time from dosing to observation |
| `administration_method` | string | Route (typically intracerebroventricular) |
| `tolerability_score_type` | string | Scoring system used |

## HELM Chemistry Encoding

HELM (Hierarchical Editing Language for Macromolecules) encodes full oligonucleotide chemistry including sugar modifications, backbone linkages, and base modifications.

**Example:** `RNA1{{[moe](C)[sp].[moe](T)[sp].d(A)[sp].d(G)[sp].[cet](A)}}$$$$`

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

Cell-line enrichment uses `Model.csv` from **DepMap Public 25Q3**, downloaded on
20 January 2026 (source SHA-256
`9dbb9de8805696c1345816ab07edd23fb4fd95e117739f3c5c3b1cf062c1233b`). Only
cell-line names, model identifiers and OncoTree fields derived from that lookup are
included here; no patient-level source fields are released.

### Source terms and attribution

- USPTO patent full-text XML is governed by the [USPTO website terms](https://www.uspto.gov/terms-use-uspto-websites). Patent text and drawings are typically not copyright-restricted, subject to the exceptions stated there; source patent identifiers are retained on every record.

- DepMap Public 25Q3 is attributed to Broad DepMap and its source release terms. Broad-generated DepMap data is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). ASO Atlas republishes only derived cell-line mapping fields.

- The processed ASO Atlas 2.0 release is distributed under CC BY 4.0. Dataset publication does not grant permission to practise inventions claimed by active patents.

### Processing

1. HELM quality filtering: removed sequences with uncertain chemistry (`?`), length <11 nucleotides, missing DNA gap, homopolymers, or naked DNA
2. Gene name standardisation via manual curated mappings
3. Cell line standardisation and CCLE (Cancer Cell Line Encyclopedia) enrichment
4. Inhibition values filtered to [-1000%, 100%]
5. Hepatic biomarker values filtered to physiologically plausible ranges
6. FOB scores validated to 0-7 range
7. Deduplication by compound and measurement
8. Rows without a resolvable HELM annotation are retained and marked `model_eligible=false`

## Intended uses

The established uses are research benchmarking and exploratory model development for
patent-derived ASO in vitro efficacy and preclinical toxicity endpoints. The supplied
patent-grouped folds support reproducible, leakage-controlled evaluation. The measurements
represent outcomes reported in patents, not independently replicated biological truth.

This dataset is **not validated for clinical decision-making, prediction of human safety,
dose selection, treatment selection, prospective laboratory deployment, legal conclusions,
or generalisation to chemistries and organisations absent from the source patents**.

## Considerations

### Biases

- **Patent bias**: Data comes exclusively from pharmaceutical patent filings, which over-represent compound classes pursued by industry (predominantly 2'-MOE gapmers). Academic and early-stage chemistries may be underrepresented.
- **Single-originator bias**: All filings come from one organisation, so quantities estimated from this corpus (pass rates, enrichment factors) bound to its chemistry and target choices rather than to ASO development in general.
- **Species bias**: Mouse and rat data dominate; monkey and dog data are sparse.
- **Reporting bias**: Patents report results selectively. Negative results may be underrepresented.

### Limitations

- **No independent validation**: All measurements originate from patent applicants' own experiments. Cross-laboratory reproducibility is unknown.
- **Extraction errors**: LLM-based extraction may introduce errors. A validation audit is included in the accompanying code repository.
- **Heterogeneous protocols**: Assay conditions (cell lines, doses, timepoints) vary across patents and are not standardised.
- **Unresolved chemistry**: 4,166 assay readouts lack a resolvable HELM annotation and cannot be used directly in chemistry-aware models.

### Ethical considerations

No human-subject records, patient-level health information or personally identifiable
information are included. The release contains public patent assay records, compound
structures, animal-study measurements and public CCLE cell-line identifiers. It contains
no synthetic assay records: LLMs transformed source tables but were not permitted to
generate measurement values directly.

### Social impact

Potential benefits include more reproducible ASO model evaluation, lower screening costs
and reduced animal use. Potential harms arise if extraction errors or the source patents'
narrow chemistry and target distribution are treated as universally representative; this
could misprioritise candidates or research areas. Version pinning, source identifiers,
leakage-controlled folds, a validation audit and the non-clinical-use limits above reduce
but do not eliminate those risks.
