# Hagedorn 2022 -- Acute neurotoxicity prediction from sequence

> Hagedorn PH et al. "Acute neurotoxicity of antisense oligonucleotides after
> intracerebroventricular injection into mouse brain can be predicted from
> sequence features." *Nucleic Acid Ther.* 32(3):151-162, 2022.
> [DOI: 10.1089/nat.2021.0071](https://doi.org/10.1089/nat.2021.0071)

## Animal model

- **Species/strain:** C57BL/6J female mice (Jackson Laboratories), 20-30 g.
- **Route:** Intracerebroventricular (ICV) freehand injection with Hamilton
  microsyringe (27-30 ga needle, polyethylene guard at 2.5 mm depth).
- **Dose:** Single bolus of **100 ug** in 5 uL saline, injected over 20-30 s
  into the right (or left) lateral ventricle.
- **Observation:** Behavioural assessment for 1 h post-injection (acute effects
  wane by 2-4 h, no reappearance at 24 h-7 days).
- **Group size:** 4-6 mice per ASO.

## Functional observational battery (FOB) scoring

Five neurobehavioural categories, each scored 0-4:

1. Hyperactivity
2. Decreased activity and arousal
3. Motor dysfunction / ataxia
4. Abnormal posture and breathing
5. Tremor / convulsions

**Total score:** 0 (no side effects) to 20 (severe, requiring euthanasia).
Per-mouse scores were averaged per treatment group.

### Tolerability classes

| Class    | FOB score range | Fraction |
|----------|-----------------|----------|
| Mild     | < 4             | ~60%     |
| Moderate | 4-7             | }        |
| Marked   | 7-18            | } ~40%   |
| Severe   | > 18            |          |

Only ASOs with mild scores (<= 4) were considered suitable for further
development; the remaining ~40% were excluded.

## Calcium oscillation cellular assay

- **Cell type:** Primary cortical neurons from Sprague-Dawley rat embryos
  (E19), cultured 11 days in 384-well plates (25,000 cells/well).
- **Assay:** FLIPR-based measurement of spontaneous calcium oscillations
  using fluo-4 AM calcium indicator.
- **Conditions:** 25 uM ASO in Hank's Balanced Salt Solution + 2 mM CaCl2
  + 10 mM HEPES, **with 1 mM MgCl2** (AMPA-dependent, NMDA-blocked).
- **Readout:** Spike amplitude and frequency. A scoring system counted 1-s
  bins where signal exceeded 50% of mean control amplitude; converted to
  percent of control.
- **Correlation with FOB:** Spearman rho significant between calcium
  oscillation scores and in vivo FOB scores (p < 0.001).

The calcium oscillation assay was used to screen 1,645 additional ASOs (not
dosed in mice) as the training set for the linear model.

## Feature engineering: 5 sequence features

The model uses a weighted linear combination of five features:

| Feature          | Description                                        | Effect on toxicity |
|------------------|----------------------------------------------------|-------------------|
| n_G (G count)    | Number of guanine nucleotides                      | Increases         |
| n_A (A count)    | Number of adenine nucleotides                      | Decreases         |
| n_T (T count)    | Number of thymine nucleotides                      | Decreases (weak)  |
| n_C (C count)    | Number of cytosine nucleotides                     | Decreases (weak)  |
| G-free 3' stretch| Length of contiguous non-G stretch from 3' end     | Decreases         |

The G-free stretch from the 5' end was *not* informative. ASO length and
number of LNA modifications did not directly explain calcium oscillation
scores.

## Linear model

```
score = b0 + b_G * n_G + b_A * n_A + b_T * n_T + b_C * n_C + b_gfree * g_free_3p
```

- Parameters estimated by least-squares (QR decomposition via R's `lm()`).
- All terms significant at p < 4e-10 (F-statistic).
- Dinucleotide or trinucleotide expansions did not improve fit and risked
  overfitting; they were not pursued.

## Training / test / validation split

| Set         | N       | Target      | Usage                        |
|-------------|---------|-------------|------------------------------|
| Training    | 1,645   | MAPT pre-mRNA | Calcium oscillation scores (cells only) |
| Test        | 148     | MAPT pre-mRNA | FOB scores in mice           |
| Validation  | 19      | STC1 pre-mRNA | FOB scores in mice (independent target) |

All ASOs were LNA gapmers with full phosphorothioate (PS) backbones.

## Classification threshold

A receiver operating characteristic analysis on the test set identified an
optimal cutoff at a **calculated score of 70**:
- ASOs with score > 70 classified as acceptable (mild FOB).
- ASOs with score <= 70 classified as potentially neurotoxic.

## Reported performance

| Metric      | Test set (N=148) | Validation set (N=19) |
|-------------|------------------|-----------------------|
| Accuracy    | 82%              | 89%                   |
| AUC (ROC)   | 87%              | --                    |
| Specificity / NPV | 91%       | --                    |

For comparison, the measured calcium oscillation scores achieved an AUC of
75% on the test set (vs 87% for the model), suggesting the model captures
the underlying signal better than raw measurements.

## Key findings

- ~40% of LNA PS ASOs injected ICV caused moderate-to-severe acute
  neurobehavioural effects within minutes of dosing.
- Guanine content (especially near the 3' end) is the strongest predictor of
  acute neurotoxicity; adenine content is protective.
- Sugar modifications (LNA vs DNA) did not differentially contribute to
  acute neurotoxic potential.
- A simple 5-feature linear model outperforms the cellular calcium oscillation
  assay for predicting in vivo FOB scores.
- Negative control ASOs designed with zero G nucleotides all had FOB < 1.
- Mechanism is linked to AMPA-dependent (not NMDA-dependent) spontaneous
  calcium oscillation reduction in neurons.

## Divergences from Hagedorn 2022

| Aspect | Paper | Our implementation | Rationale |
|--------|-------|--------------------|-----------|
| Chemistry | LNA, full PS | DNA/MOE/cEt (LNA excluded) | Tests generalisation |
| Dose | 100 μg ICV | 700 μg ICV | 100 μg yields ~21 rows; 700 μg gives 2,697 |
| Binary labels | FOB ≤ 4 vs > 4 | FOB ≤ 1 vs ≥ 3 | Cleaner signal; excludes boundary |
| Model type | OLS → ROC threshold | Logistic regression | Direct binary classification |
| Additional models | Linear only | Linear + 4 RF | Benchmark extension for pipeline |
| Validation | Fixed train/test/val | GroupKFold by target | Different data structure |
