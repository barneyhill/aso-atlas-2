# ASO Atlas 2.0

Code and data for **"ASO Atlas 2.0: Evaluating antisense oligonucleotide prediction across the preclinical pipeline"** (Hill, Whiffin, Rinaldi, Sanders).

Submitted to NeurIPS 2026.

## Dataset

ASO Atlas 2.0 is the largest public multi-endpoint ASO preclinical dataset, extracted from Ionis Pharmaceuticals' USPTO patent filings and annotated with HELM chemical-structure strings. It spans 430 target genes and four assay categories:

| Assay | Measurements | Unique ASOs |
|---|--:|--:|
| In vitro inhibition | 174,459 | 163,802 |
| Dose-response (IC50) | 99,821 | 16,834 |
| Hepatotoxicity (7 biomarkers) | 15,831 | 2,051 |
| Neurotoxicity (FOB scores) | 4,896 | 3,316 |

The dataset is hosted on Hugging Face: [huggingface.co/datasets/barneyhill/aso-atlas-2](https://huggingface.co/datasets/barneyhill/aso-atlas-2)

The best-performing models (OligoAI + CatBoost toxicity) are deployed at [sitlabs.org/oligoai](https://sitlabs.org/oligoai).

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
uv sync
```

### Build everything

```bash
just build    # clean → models → pipeline → export → plots → compile (~2 min)
```

This runs the full pipeline end-to-end and compiles the manuscript to `typst/main.pdf`.

### Individual steps

| Command | What it does | Time |
|---|---|---|
| `just analysis` | Clean data + fit Hagedorn models + run pipeline | ~80s |
| `just export` | Export paper numbers to JSON | <5s |
| `just plots` | Generate all figures | ~30s |
| `just compile` | Compile manuscript (`typst/main.typ`) | ~1s |
| `just test` | Run test suite | <10s |
| `just oligogym` | Run OligoGym benchmark (CPU-heavy) | ~2-4h |

### Key pipeline stages

1. **`just analysis`** processes raw patent CSVs into cleaned parquets, fits the Hagedorn hepatotoxicity and neurotoxicity replication models, and runs the pipeline cost model.
2. **`just export`** reads JSON results and writes `paper_numbers.json`, which the manuscript reads so that no numbers are hardcoded.
3. **`just plots`** generates all figures as SVGs into `typst/plots/`.
4. **`just compile`** compiles the Typst manuscript to PDF.

## Tests

```bash
just test
```

Covers HELM parsing, enrichment factor computation, OligoAI evaluation, and consistency of exported paper numbers.

## License

CC BY 4.0
