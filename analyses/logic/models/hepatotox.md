# Hagedorn 2013 -- Hepatotoxicity prediction from sequence

> Hagedorn PH et al. "Hepatotoxic potential of therapeutic oligonucleotides
> can be predicted from their sequence and modification pattern."
> *Nucleic Acid Ther.* 23(5):302-10, 2013.
> [DOI: 10.1089/nat.2013.0436](https://doi.org/10.1089/nat.2013.0436)

## Animal model

- **Species/strain:** C57BL/6J and NMRI female mice (Taconic, Denmark).
- **Dosing:** 5 IV injections of 15 mg/kg on days 0, 3, 7, 10, 14
  (75 mg/kg cumulative). Validation doses ranged 25-125 mg/kg total.
- **Sacrifice:** Day 16.

## Biomarkers

- **Primary:** Serum alanine aminotransferase (ALT), measured by enzymatic
  assay (Horiba ABX Diagnostics, 96-well format). Reported as fold-change
  relative to saline control mean.
- **Secondary:** Histopathology (25 categories scored 0-3), liver/body weight
  ratio, liver qRT-PCR for target mRNA knockdown.

## Upper limit of normal (ULN)

From the distribution of ALT in all saline-treated mice across all studies:
- Median *m* = 1.0, median absolute deviation sigma = 0.24.
- ULN = *m* + 3*sigma* = **1.7** (fold-change units).

## Classification scheme

| Class   | Criterion             |
|---------|-----------------------|
| Low-tox | ALT < 2x ULN         |
| High-tox| ALT > 5x ULN         |

Oligonucleotides between 2x and 5x ULN were excluded from training.

## Feature engineering: dinucleotide representation

Oligonucleotides were LNA gapmers with a central DNA gap (>=7 nt) flanked by
2-3 LNA nucleotides per end. Each ASO was encoded as a 64-dimensional vector
of all possible dinucleotide counts:

- 4 DNA nucleotides (a, c, t, g) + 4 LNA variants (A, C, T, G) = 8 monomer
  types, giving (8)^2 = **64 dinucleotide features**.
- Each dinucleotide feature counts overlapping pairs of adjacent nucleotides,
  preserving both base identity and sugar modification.

## Random forest classifier

| Parameter            | Value                |
|----------------------|----------------------|
| Algorithm            | Random forest (unpruned trees) |
| Number of trees      | 5,000                |
| Features per split   | 8 (of 64)           |
| Split criterion      | Gini impurity        |
| Prediction rule      | Majority vote        |
| Validation           | Out-of-bag (OOB) + stratified 10-fold CV (Levenshtein distance strata) |

## Training / validation split

| Set          | N    | Composition            |
|--------------|------|------------------------|
| Training     | 206  | 97 low-tox, 109 high-tox (75 mg/kg screen) |
| Validation   | 23   | 13 high-tox (25-60 mg/kg), 10 low-tox (100-125 mg/kg); 10 different mRNA targets, 8 novel |

No ASO from the validation set appeared in the training set.

## Reported performance

| Metric      | OOB  | CV (1-2 edits) | CV (3-5 edits) | CV (6-8 edits) | Validation |
|-------------|------|----------------|----------------|----------------|------------|
| Accuracy    | 80%  | 87%            | 80%            | 69%            | 74%        |
| Sensitivity | 83%  | 90%            | 77%            | 77%            | 80%        |
| Specificity | 76%  | 83%            | 82%            | 63%            | 69%        |

All p < 0.05 (Fisher's exact test). Stratified CV bins are by Levenshtein
edit distance between ASO sequences. No AUC was reported.

## Key findings

- 11 of the 17 most important dinucleotides (by mean decrease in Gini
  impurity) overlapped with univariate association tests.
- Some dinucleotides increase and others decrease hepatotoxic potential;
  overall prediction depends on a complex interplay of all dinucleotides.
- The classifier can guide redesign of high-tox ASOs to reduce their
  hepatotoxic potential while preserving target engagement.

## Divergences from Hagedorn 2013

| Aspect | Paper | Our implementation | Rationale |
|--------|-------|--------------------|-----------|
| Chemistry | LNA gapmers | DNA/MOE/cEt (LNA excluded) | Tests generalisation to MOE/cEt |
| Biomarkers | ALT only | ALT, AST, TBIL | Extends to all available hepatic biomarkers |
| ALT thresholds | 5×/2× ULN | 3×/1× ULN | Captures more signal in MOE/cEt data |
| Species filter | C57BL/6J only | All mouse strains | Strain-level data unavailable in OligoStack |
| class_weight | None | `'balanced'` | Handles class imbalance |
