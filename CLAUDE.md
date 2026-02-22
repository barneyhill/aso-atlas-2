# CLAUDE.md

## Package manager

Use `uv` for everything (`uv run`, `uv sync`, `uv pip install`).

## After making changes

| What changed | Command to run |
|---|---|
| `analyses/plotting/*.py` | `just plots && just compile` |
| `analyses/export_paper_numbers.py` | `just export && just compile` |
| `analyses/logic/*.py` | `just analysis && just test` |
| `typst/**/*.typ` | `just compile` |
| Any plotting + export together | `just build` |

- `just build` = `analysis` + `export` + `plots` + `compile` (full rebuild)
- `just analysis` = `clean` + `hagerdorn` + `pipeline` (~80s)
- `just hagerdorn` = run Hagerdorn hepatotox + neurotox models (~50s)
- `just export` = export paper numbers to JSON (assumes data already built)
- `just plots` = generate all figures (assumes data already built)
- `just test` = `uv run pytest tests/ -x -q`
- `just compile` = `typst compile typst/main.typ` (~1s)
- Do **not** run heavy training (`analyses/05_oligoai2/train.py`) unless explicitly asked.

## Project layout

- `analyses/logic/clean.py` — cleans raw CSVs → processed parquets (run via `just analysis`)
- `analyses/logic/models/hepatotox.py` — Hagerdorn 2013 hepatotoxicity model replication
- `analyses/logic/models/neurotox.py` — Hagerdorn 2022 neurotoxicity model replication
- `analyses/logic/pipeline.py` — pipeline attrition + cost analysis with Hagerdorn enrichment
- `analyses/plotting/` — reads processed data, writes `typst/plots/` and `typst/tables/`
- `analyses/export_paper_numbers.py` — exports key numbers to `typst/data/paper_numbers.json`
- `typst/main.typ` — manuscript (reads `paper_numbers.json`, no hardcoded values)
- `tests/` — correctness tests

### Legacy (to be ported)

- `analyses/01_preprocess/` — preprocessing scripts
- `analyses/02_summarise/` — summarisation scripts
- `analyses/03_analyse/` — analysis scripts
- `analyses/05_oligoai2/` — multi-task ASO model (OligoAI2) [archived]
- `analyses/hypotheses/` — hypothesis testing

## Tests

```bash
just test
```
