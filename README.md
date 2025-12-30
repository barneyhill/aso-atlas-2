# Paper 2: Modified Oligonucleotide Analysis

This repository contains tools and analyses for studying modified oligonucleotides, particularly antisense oligonucleotides (ASOs).

## Structure

```
.
├── analyses/           # Analysis scripts
├── data/               # Datasets (hepatotoxicity data with HELM sequences)
├── modxna/             # HELM → AMBER conversion pipeline
├── papers/             # Reference papers
└── plots/              # Generated figures
```

## modxna: HELM to AMBER Conversion

The `modxna/` directory contains an automated pipeline to convert HELM-encoded oligonucleotide sequences into AMBER-ready molecular dynamics simulation files.

### Quick Start

```bash
# Install dependencies (one-time)
cd modxna && ./scripts/install.sh

# Convert HELM to AMBER files
source scripts/setup_env.sh
uv run python -m modxna.cli "RNA1{[moe](C)[sp].[moe](T)[sp].d(A)}$$$$" -o output/

# Generated files:
# - output/aso.parm7      (AMBER topology)
# - output/aso.crd        (AMBER coordinates)
# - output/md_inputs/     (Minimization/equilibration scripts)
```

See `modxna/README.md` for detailed documentation.

## Data

The `data/oligostack/processed/` directory contains hepatotoxicity data with HELM-encoded ASO sequences. See `data/oligostack/processed/README.md` for column descriptions.
