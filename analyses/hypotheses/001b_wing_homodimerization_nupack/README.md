# Hypothesis 001b: Wing Homodimerization (NUPACK Thermodynamic Analysis)

## Summary

**Status**: Hypothesis NOT supported (confirms original 001 finding)

This analysis revisits the wing homodimerization hypothesis using NUPACK for thermodynamically rigorous free energy (ΔG) calculations instead of simple contiguous base-pair counting.

**Key finding**: ASOs with more stable homodimers (more negative ΔG) show **lower** hepatotoxicity (ALT), consistent with the original hypothesis 001 results.

## Background

The original hypothesis 001 proposed that ASOs with higher self-complementarity form homodimers more readily, potentially leading to:
- Altered pharmacokinetics
- Changed protein binding profiles
- Toxic aggregates in hepatocytes

The original analysis used max contiguous base pairs as a proxy for dimerization propensity. This analysis uses NUPACK to compute actual thermodynamic stability (ΔG) of homodimer complexes.

## Methods

### NUPACK Thermodynamic Calculations

NUPACK 4.0.2.0 was used to compute equilibrium thermodynamics:
- **Model**: DNA parameters, 37°C, 150 mM Na+, 1 mM Mg2+
- **Metrics computed**:
  - `homodimer_dG`: Free energy of full ASO homodimer (kcal/mol)
  - `monomer_dG`: Free energy of intramolecular folding
  - `ddG_dimerization`: Dimerization propensity (homodimer_dG - 2×monomer_dG)
  - Wing-specific homodimer ΔG values

### Statistical Tests

- Spearman correlation (ΔG vs ALT)
- Mann-Whitney U test (median split)
- Linear regression (log ALT)

### Limitations

- NUPACK uses standard DNA parameters; MOE sugars and PS backbone alter real thermodynamics
- In vitro equilibrium model vs in vivo kinetics
- Intracellular ionic conditions vary by compartment

## Results

### Dataset
- **N = 784** unique ASOs (mouse, subcutaneous administration)
- **N = 755** with valid NUPACK calculations

### Key Statistical Results

| Metric | Spearman ρ | p-value | R² | Interpretation |
|--------|-----------|---------|-----|----------------|
| **Full ASO Homodimer ΔG** | **+0.316** | **5.5e-19** | 0.079 | More stable dimers → lower ALT |
| ΔΔG Dimerization | +0.247 | 5.9e-12 | 0.056 | Higher dimerization propensity → lower ALT |
| 5' Wing Homodimer ΔG | +0.109 | 1.8e-02 | 0.019 | Weak effect |
| 3' Wing Homodimer ΔG | +0.268 | 5.5e-10 | 0.066 | Moderate effect |
| Wing5-Wing3 Heterodimer ΔG | +0.191 | 3.5e-07 | 0.036 | Moderate effect |

**Note**: Positive ρ indicates that as ΔG increases (becomes less negative, i.e., less stable dimer), ALT increases. Equivalently, more stable dimers correlate with lower ALT.

### Mann-Whitney U Results (Full ASO Homodimer ΔG)

| Group | N | Median ALT (IU/L) |
|-------|---|-------------------|
| Stable dimers (ΔG ≤ -5.20) | 378 | 64.5 |
| Unstable dimers (ΔG > -5.20) | 377 | 140.0 |

**U = 47,660, p = 3.4e-15** - Highly significant difference.

### Method Comparison

Correlation between NUPACK ΔG and original contiguous BP metric:
- **Spearman ρ = -0.799** (p < 1e-168)

This strong negative correlation confirms that both methods capture the same underlying phenomenon (self-complementarity), with the negative sign reflecting that higher contiguous BP → more negative ΔG.

The original contiguous BP metric showed ρ = -0.174 with ALT (negative correlation = higher BP → lower ALT), which is consistent with the NUPACK finding (positive ρ because higher ΔG values are less stable).

## Conclusion

The NUPACK thermodynamic analysis **confirms and strengthens** the original hypothesis 001 finding:

**ASOs with higher self-complementarity (more stable homodimers) exhibit LOWER hepatotoxicity.**

This is the **opposite** of what the original biological hypothesis predicted. Possible explanations:
1. **Sequestration**: Dimerization may sequester ASOs from toxic protein interactions
2. **Reduced off-target binding**: Self-complementary sequences may have lower affinity for cellular targets
3. **Design selection**: High-complementarity sequences may correlate with other protective features

The thermodynamic approach provides stronger mechanistic grounding for this finding, with effect sizes (ρ = 0.316) exceeding the original structural metric (ρ = -0.174 for contiguous BP).

## Files

- `analysis.py` - Main analysis script
- `nupack_metrics.py` - NUPACK calculation functions
- `nupack_metrics.csv` - Computed metrics for all ASOs
- `figures/` - Visualizations

## Figures

- `nupack_summary.png` - Multi-panel summary of all metrics
- `homodimer-dG_vs_alt.png` - Scatter plot with regression
- `homodimer-dG_boxplot.png` - Boxplot by quartiles
- `method_comparison.png` - NUPACK vs contiguous BP correlation
