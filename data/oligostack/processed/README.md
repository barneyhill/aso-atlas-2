# OligoStack Processed Data

## hepatictoxicity_processed.parquet

Hepatic toxicity data for antisense oligonucleotides (ASOs), cleaned and processed from USPTO patent data.

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `Compound ID` | string | Unique identifier for each ASO compound |
| `USPTO ID` | string | Source patent identifier |
| `Table Number` | string | Table number within the patent |
| `HELM Annotation` | string | Chemistry encoding in HELM notation (see below) |
| `target_RNA` | string | Target RNA transcript (27.5% complete) |
| `species` | string | Test species: mouse, rat, monkey, dog |
| `species_strain` | string | Specific strain (e.g., CD-1 mice, Sprague-Dawley rats) |
| `dosage_mg_per_kg` | float | Dose per administration (mg/kg) |
| `num_doses` | int | Number of doses administered |
| `dosing_period_days` | float | Duration of dosing period |
| `adminstration_method` | string | Route: subcutaneous, intraperitoneal |
| `measurement_source` | string | Sample type: plasma, urine |
| `ALT` | array | Alanine aminotransferase values (IU/L) |
| `AST` | array | Aspartate aminotransferase values (IU/L) |
| `ALB` | array | Albumin values (g/dL) |
| `BUN` | array | Blood urea nitrogen values (mg/dL) |
| `CREA` | array | Creatinine values (mg/dL) |
| `TBIL` | array | Total bilirubin values (mg/dL) |
| `PC_ratio` | array | Protein/creatinine ratio |

### HELM Chemistry Encoding

HELM (Hierarchical Editing Language for Macromolecules) encodes oligonucleotide chemistry.

**Format example:**
```
RNA1{[moe](C)[sp].[moe](T)[sp].d(A)[sp].d(G)[sp].[cet](A)}$$$$
```

**Sugar modifications:**
| Notation | Chemistry |
|----------|-----------|
| `d` | 2'-deoxy (DNA) |
| `[moe]` | 2'-O-methoxyethyl (MOE) |
| `[cet]` | Constrained ethyl (cEt) |
| `[lna]` | Locked nucleic acid (LNA) |
| `[fR]` | 2'-fluoro |
| `[m]` | 2'-O-methyl |

**Backbone linkages:**
| Notation | Chemistry |
|----------|-----------|
| `[sp]` | Phosphorothioate (PS) |
| `.` | Phosphodiester (PO) |
| `[am]` | Phosphoramidate (PMO) |

**Base modifications:**
| Notation | Chemistry |
|----------|-----------|
| `5meC`, `5MeC` | 5-methylcytosine |

### Data Processing

Applied in `analyses/clean.py`:
1. Parsed dosage values and standardized units to mg/kg
2. Converted mg/kg/wk to mg/kg per dose
3. Filtered to dosage <= 10,000 mg/kg
4. Filtered to administration methods: intraperitoneal, subcutaneous
5. Filtered to measurement sources: plasma, urine
6. Standardized species names
7. Applied biomarker valid ranges (ALT/AST: 10-50,000 IU/L, ALB: 1-100 g/dL)
8. Collapsed duplicate records by grouping columns

### Statistics

- **Records:** 4,801
- **Unique compounds:** 2,350
- **Species distribution:** mouse (3,323), rat (1,342), monkey (130), dog (6)
- **HELM coverage:** 97.5% (120 records missing)
