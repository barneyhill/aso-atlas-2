# Autonomous Hypothesis Testing

When the user proposes a hypothesis about ASO hepatotoxicity, execute the full pipeline automatically without pausing for confirmation.

## Workflow Overview

```
User proposes hypothesis
    ↓
1. Setup: Create directory structure
    ↓
2. Literature: Search PubMed for prior work
    ↓
3. Analysis: Write & run self-contained script
    ↓
4. Report: Write README with full results
    ↓
5. Registry: Update REGISTRY.md
```

---

## Step 1: Setup

1. Read `REGISTRY.md` to determine next ID (e.g., if last is 001, next is 002)
2. Create directory: `analyses/hypotheses/NNN_<short_slug>/`
   - Use lowercase, underscores, descriptive name
   - Example: `002_gc_content`, `003_protein_binding_motifs`
3. Create `figures/` subdirectory

---

## Step 2: Literature Search

Search for relevant prior work using WebSearch:
- Query: `"ASO hepatotoxicity" OR "antisense oligonucleotide liver toxicity" [hypothesis topic]`
- Query: `"phosphorothioate" hepatotoxicity [specific mechanism]`
- PubMed: `site:pubmed.ncbi.nlm.nih.gov [query]`

Document 3-5 relevant papers in the README:
- Citation (authors, journal, year)
- Key finding relevant to hypothesis
- DOI or PMID link

---

## Step 3: Analysis

### Data Access
```python
import pandas as pd
df = pd.read_parquet('data/oligostack/processed/hepatictoxicity_processed.parquet')
```

### Key Columns
| Column | Description |
|--------|-------------|
| `HELM Annotation` | ASO chemistry in HELM format |
| `ALT` | Alanine aminotransferase (IU/L) - **primary toxicity marker** |
| `AST` | Aspartate aminotransferase (IU/L) |
| `species` | mouse, rat, monkey, dog |
| `adminstration_method` | subcutaneous, intraperitoneal, intravenous |
| `Compound ID` | Unique ASO identifier |
| `dosage_mg_per_kg` | Dose level |

### Default Filters
```python
filtered = df[
    (df['species'] == 'mouse') &
    (df['adminstration_method'] == 'subcutaneous') &
    df['HELM Annotation'].notna() &
    df['ALT'].notna()
]
```

### ALT Handling
ALT values are stored as numpy arrays (multiple measurements). Flatten to mean:
```python
import numpy as np

def flatten_alt(val):
    if isinstance(val, np.ndarray):
        return float(np.mean(val)) if len(val) > 0 else np.nan
    return float(val) if val is not None else np.nan

filtered['ALT_mean'] = filtered['ALT'].apply(flatten_alt)
```

### HELM Parsing
Normalize double braces and parse:
```python
import re

def normalize_helm(helm):
    return helm.replace('{{', '{').replace('}}', '}')

def parse_helm_to_sequence(helm):
    """Extract base sequence from HELM annotation."""
    helm = normalize_helm(helm)
    match = re.search(r'\{(.+?)\}', helm)
    if not match:
        return None

    nucleotides = match.group(1).split('.')
    bases = []
    for nuc in nucleotides:
        # Extract base from patterns like [moe](A)[sp] or d(G)[sp]
        base_match = re.search(r'\(([^)]+)\)', nuc)
        if base_match:
            base = base_match.group(1)
            # Normalize 5meC to C
            if base.lower() in ('5mec', '5-mec'):
                base = 'C'
            bases.append(base[0] if len(base) > 1 else base)
    return ''.join(bases)
```

### Standard Statistical Tests
```python
from scipy import stats

# 1. Spearman correlation
rho, p = stats.spearmanr(metric_values, alt_values)

# 2. Mann-Whitney U (for threshold comparisons)
high_group = alt_values[metric >= threshold]
low_group = alt_values[metric < threshold]
U, p = stats.mannwhitneyu(high_group, low_group, alternative='two-sided')
```

### Visualization Template
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
# ... boxplot or scatter ...
ax.set_ylabel('ALT (IU/L)')
ax.set_yscale('log')
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='~3x ULN')
plt.savefig('figures/main_result.png', dpi=150)
```

### Script Structure
Each `analysis.py` should be **fully self-contained**:
- All imports at top
- All helper functions defined in file (copy from examples, don't import)
- Clear `main()` function
- Print results to stdout AND save figures

---

## Step 4: Report

Create `README.md` in the hypothesis directory with this structure:

```markdown
# Hypothesis NNN: [Title]

## Hypothesis
[Clear statement of the hypothesis being tested]

## Rationale
[Why this hypothesis is plausible - biological mechanism, prior observations]

## Literature
- Author et al. (Year). Title. *Journal*. [DOI](link)
  - Key finding: ...
- ...

## Methods
- Dataset: N unique ASOs (mouse, subcutaneous)
- Metric: [What was calculated]
- Statistics: Spearman correlation, Mann-Whitney U

## Results

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Spearman | ρ = X.XX | X.XXe-XX | ... |
| Mann-Whitney (≥3 vs <3) | U = XXX | X.XXe-XX | ... |

![Main Result](figures/main_result.png)

## Conclusion
**[Supported / Not Supported / Inconclusive]**

[Brief interpretation and implications]
```

---

## Step 5: Update Registry

Add a row to `REGISTRY.md`:

```markdown
| NNN | [Hypothesis summary] | Complete | [Result] | [Key stat] | [Direction] | YYYY-MM-DD |
```

Result options:
- `Supported` - hypothesis confirmed
- `Not supported` - hypothesis rejected
- `Inconclusive` - insufficient data or unclear signal
- `Opposite` - significant effect in opposite direction

---

## Example Workflow

User: "Test whether GC content in the gap region correlates with hepatotoxicity"

Claude:
1. Creates `analyses/hypotheses/002_gc_content_gap/figures/`
2. Searches PubMed for "ASO GC content hepatotoxicity"
3. Writes `analysis.py` to:
   - Extract gap region from HELM
   - Calculate GC%
   - Correlate with ALT
4. Runs analysis, generates boxplot
5. Writes README with results
6. Updates REGISTRY.md

---

## Notes

- Always aggregate to mean ALT per Compound ID before statistics
- Use log scale for ALT visualizations (wide range of values)
- Report both correlation AND threshold-based tests
- If hypothesis suggests direction (higher X → higher ALT), test one-tailed if appropriate
- Document unexpected findings - they often lead to new hypotheses
