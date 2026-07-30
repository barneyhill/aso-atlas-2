# ASO Atlas 2.0

Code and data for **"ASO Atlas 2.0: Evaluating antisense oligonucleotide prediction across the preclinical pipeline"** (Hill, Whiffin, Rinaldi, Sanders).

Submitted to NeurIPS 2026.

## Dataset

ASO Atlas 2.0 is the largest public multi-endpoint ASO preclinical dataset, extracted from Ionis Pharmaceuticals' USPTO patent filings. The complete release spans 430 target genes and four assay categories; rows with resolved chemistry are annotated with HELM strings:

| Assay | Readouts | Unique HELM compounds |
|---|--:|--:|
| In vitro inhibition | 174,459 | 161,434 |
| Dose-response (IC50) | 99,821 | 16,375 |
| Hepatotoxicity (7 biomarkers) | 15,831 | 1,881 |
| Neurotoxicity (FOB scores) | 4,896 | 3,312 |

The extraction pipeline began with 1,125 table-bearing source documents. The downloadable
`neurips-rebuttal` snapshot contains records contributed by 606 patents. Across the four assay
categories, 15,339 of 168,537 distinct source `Compound ID` values (9.1%) occur in at
least two categories. This source-ID linkage statistic is separate from the 165,782
unique resolved HELM structures reported above.

The dataset is hosted on Hugging Face: [huggingface.co/datasets/barneyhill/aso-atlas-2](https://huggingface.co/datasets/barneyhill/aso-atlas-2)

The manuscript and tagged `neurips-rebuttal` download both contain all 295,007 processed
readouts across 430 named target genes. Of these, 290,841 readouts
have resolved HELM chemistry and are marked `model_eligible=true`; 4,166 retain their
assay outcomes with unresolved chemistry. Exact endpoint-specific paper folds are included
for the model-eligible subset. See the versioned dataset card and DOI
[10.57967/hf/8687](https://doi.org/10.57967/hf/8687).

The models used in the primary analysis (OligoAI + XGBoost toxicity) are deployed at [sitlabs.org/oligoai](https://sitlabs.org/oligoai).

## Repository structure

```
analyses/
├── logic/
│   ├── clean.py                  # Raw CSVs → processed parquets (§3)
│   ├── pipeline.py               # Pipeline attrition & cost model (§4.1)
│   ├── enrichment.py             # Enrichment factor computation (§4.1)
│   ├── ef_table.py               # Strategy leaderboard, Table 1 (§5.2)
│   ├── species_transfer.py       # Cross-species transfer analysis (§6)
│   ├── cv_audit.py               # GroupKFold leakage checks (§5)
│   └── models/
│       ├── hepatotox.py          # Hagedorn 2013 hepatotoxicity replication (§5.1)
│       ├── neurotox.py           # Hagedorn 2022 neurotoxicity replication (§5.1)
│       ├── oligogym_benchmark.py # OligoGym 6-model benchmark (§5.1)
│       ├── oligoai_efs.py        # OligoAI enrichment factors (§5.2)
│       └── oligoai_build_csv.py  # Per-fold training CSVs for OligoAI (§5.1)
├── plotting/                     # One script per figure → typst/plots/
├── utils/
│   ├── helm.py                   # HELM parser & feature extraction (§3.1)
│   ├── compounds.py              # Compound-level utilities
│   └── oligogym_adapter.py       # Bridges project data to OligoGym (§5.1)
└── export_paper_numbers.py       # Key numbers → typst/data/paper_numbers.json

patent_collate/                   # LLM-powered extraction pipeline (§3.1)
├── cli.py                        # Three-stage CLI: XML → JSON → HELM → schema
└── src/                          # Prompts, client, similarity dedup

data/
├── oligostack/
│   ├── raw/                      # LLM-extracted CSVs from USPTO patents (§3.1)
│   └── processed/                # Cleaned, deduplicated parquets (§3)
├── results/                      # JSON outputs from analysis pipeline
└── models/                       # Trained model checkpoints (§5)

typst/
├── main.typ                      # Manuscript source
├── appendix.typ
├── paper2_preclinical.bib
├── data/paper_numbers.json       # Auto-generated; no hardcoded values in text
└── plots/                        # Auto-generated figures (SVG)

tests/                            # Correctness tests (pytest)
```

## Reproducing results

### Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Typst](https://typst.app/) (manuscript compilation)
- [just](https://github.com/casey/just) (command runner)

### Setup

```bash
uv sync --frozen
```

The committed `.python-version` and `uv.lock` pin the interpreter and complete Python
environment used for the release.

### Reproduce from the released processed snapshot

```bash
just reproduce
```

This reruns the lightweight fitted baselines and cost model, exports the paper ledger,
regenerates the figures, and compiles `typst/main.pdf`. It starts from the versioned
processed parquets and frozen heavyweight benchmark outputs included in the code release;
it does not require an OpenAI key or the raw LLM-extraction intermediates.

To rebuild the processed parquets from local raw extraction outputs instead, run
`just build`. Those large intermediates are not part of the Hugging Face data product;
the extraction workflow that creates them from USPTO XML is documented under
`patent_collate/`.

### Individual steps

| Command | What it does | Time |
|---|---|---|
| `just reproduce` | Reproduce the paper from versioned processed data/results | ~2 min |
| `just build` | Full rebuild from locally available raw extraction outputs | depends on inputs |
| `just analysis` | Clean raw data + fit Hagedorn models + run pipeline | ~80s |
| `just export` | Export paper numbers to JSON | <5s |
| `just plots` | Generate all figures | ~30s |
| `just compile` | Compile manuscript (`typst/main.typ`) | ~1s |
| `just test` | Run test suite | <10s |
| `just aso-atlas-2-release` | Rebuild and audit the exact HF/Croissant bundle | ~5s |
| `just oligogym` | Run OligoGym benchmark (CPU-heavy) | ~2-4h |

### Key pipeline stages

1. **`just analysis`** processes locally available raw patent CSVs into cleaned parquets, fits the Hagedorn hepatotoxicity and neurotoxicity replication models, and runs the pipeline cost model.
2. **`just export`** reads JSON results and writes `paper_numbers.json`, which the manuscript reads so that no numbers are hardcoded.
3. **`just plots`** generates all figures as SVGs into `typst/plots/`.
4. **`just compile`** compiles the Typst manuscript to PDF.

## Tests

```bash
just test
```

The versioned release contract checks dataset counts, paper reconciliation, split leakage,
the explicit Hugging Face allowlist, file hashes, and the NeurIPS Croissant core and RAI
fields. Additional validation-audit tests are packaged with the corresponding code-release
evidence rather than the Hugging Face dataset.

## License

CC BY 4.0
