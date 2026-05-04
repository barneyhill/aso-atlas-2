# patent-collate

LLM-powered pipeline for extracting structured experimental data from USPTO patent XML tables. Used to build [ASO Atlas 2.0](https://huggingface.co/datasets/barneyhill/aso-atlas-2), a multi-endpoint antisense oligonucleotide preclinical dataset spanning 255K+ measurements.

## Overview

Three-stage extraction system where an LLM generates bespoke Python scripts for each patent table:

1. **XML to JSON** -- Generates per-table extraction functions that parse compound IDs and experimental data from XML
2. **HELM Annotation** -- Generates chemistry annotation functions that produce nucleotide-level [HELM](https://pistoiaalliance.atlassian.net/wiki/spaces/PUB/pages/14614561/HELM+Notation) structure strings
3. **Schema Collation** -- Generates mapping functions that extract measurements matching a user-defined schema

Each generated script is validated by executing it in a sandboxed subprocess. Failed scripts are automatically regenerated (up to 5 attempts). All API responses are cached with `diskcache` for reproducibility and cost savings.

## Requirements

- Python >= 3.11
- OpenAI API key (GPT-5 / GPT-5-mini)

## Installation

```bash
pip install -e .
# or
uv pip install -e .
```

## Configuration

```bash
export OPENAI_API_KEY=sk-...
```

Or pass `--api-key` directly, or create `~/.config/patent-collate/config.yaml`:

```yaml
openai: sk-...
```

## Usage

### Full pipeline

```bash
python cli.py run-all \
  --input-dir /path/to/patent_xmls/ \
  --schema schema.json \
  --output-dir results/
```

### Individual stages

```bash
# Step 1: Extract tables to JSON
python cli.py step1 --input-dir /path/to/patent_xmls/ --output step1.json

# Step 2: Annotate HELM chemistry
python cli.py step2 --table-data step1.json --output step2.json

# Step 3: Collate to schema
python cli.py step3 --table-data step1.json --schema schema.json --output step3.json
```

### Options

```
--model             LLM model (default: gpt-5-mini)
--service-tier      API tier: flex, default, none (default: flex)
--max-workers       Parallel threads (default: 50)
--max-attempts      Retries per table (default: 5)
--debug             Enable file logging
--cache-dir         Override cache directory
```

## Schema Format

Define a JSON file mapping field names to descriptions:

```json
{
  "compound_id": "string",
  "inhibition_pct": "Percent inhibition, float 0-100. Calculate as 100 - relative expression if needed.",
  "cell_line": "Cell line name, string.",
  "dosage_nm": "Dosage in nanomolar (nM), float."
}
```

The LLM uses these descriptions to interpret heterogeneous table formats and generate appropriate mapping logic.

## Input Format

The pipeline expects a directory of USPTO patent XML files (`.xml` or `.xml.gz`). Each file should contain a single `<table>` element. Tables are automatically deduplicated by content hash and sequence similarity before processing.

## Cost

Processing ASO Atlas 2.0 (35,871 tables across 1,125 patents) cost approximately $500 using `gpt-5-mini` with flex pricing. Use `--model gpt-5-mini --service-tier flex` for lowest cost.
