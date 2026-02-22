#import "@preview/clean-math-paper:0.2.5": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#let R = json("data/paper_numbers.json")
#let comma(n) = {
  let s = str(n)
  let out = ""
  let len = s.len()
  for (i, c) in s.codepoints().enumerate() {
    if i > 0 and calc.rem(len - i, 3) == 0 { out += "," }
    out += c
  }
  out
}

// Modify some arguments, which can be overwritten in the template call
#page-args.insert("numbering", "1/1")
#text-args-title.insert("size", 2em)
#text-args-title.insert("fill", black)
#text-args-authors.insert("size", 12pt)

#show: template.with(
title: "OligoStack: a public dataset and cost-based benchmark for antisense oligonucleotide preclinical screening",
authors: (
(name: "Barney Hill", affiliation-id: "1,3,5,*"),
(name: "Nicola Whiffin", affiliation-id: "1,3,4"),
(name: "Stephan J. Sanders", affiliation-id: "1,5,6,7"),
(name: "Carlo Rinaldi", affiliation-id: "1,5,*"),
),
affiliations: (
(id: "1", name: "Department of Paediatrics, University of Oxford, OX3 7TY Oxford, United Kingdom"),
(id: "3", name: "Big Data Institute, University of Oxford, Oxford, UK"),
(id: "4", name: "Broad Center for Mendelian Genomics, Program in Medical and Population Genetics, Broad Institute of MIT and Harvard, Cambridge, MA, USA"),
(id: "5", name: "Institute of Developmental and Regenerative Medicine (IDRM), IMS-Tetsuya Nakamura Building, Old Road Campus, OX3 7TY Oxford, United Kingdom"),
(id: "6", name: "New York Genome Center, New York, NY 10013, USA"),
(id: "7", name: "Department of Psychiatry and Behavioral Sciences, UCSF Weill Institute for Neurosciences, University of California, San Francisco, San Francisco, CA 94178, USA"),
(id: "*", name: "Correspondence to barney.hill@merton.ox.ac.uk, carlo.rinaldi@idrm.ox.ac.uk"),
),
date: date,
link-color: rgb("#008002"),
abstract: [
Antisense oligonucleotides (ASOs) are a maturing therapeutic modality, yet preclinical development remains costly due to sequential in vitro and in vivo screening with high attrition rates. Progress in computational ASO design has been hampered by the absence of a standardised public dataset linking chemical structure to multiple preclinical endpoints, and by the lack of evaluation frameworks that translate predictive accuracy into practical cost savings. Here we introduce *OligoStack*, the largest public multi-endpoint ASO preclinical dataset, comprising #comma(R.in_vitro.n_measurements) in vitro efficacy measurements, #comma(R.dose_response.n_measurements) dose-response curves, #comma(R.hepatic.n_records) hepatotoxicity records, and #comma(R.neuro.n_records) neurotoxicity records across #comma(R.genes.n_unique) target genes --- all extracted from USPTO patent filings using a language-model-powered pipeline. We propose a cost-based benchmarking framework that evaluates classifiers by their ability to enrich the candidate pool at expensive in vivo stages, quantifying savings in terms of the number of animals and dollars required to yield a development candidate. As a baseline benchmark, we replicate the two published Hagedorn sequence-based models (hepatotoxicity, 2013; neurotoxicity, 2022) on OligoStack using GroupKFold cross-validation by target gene. OligoStack, the benchmarking framework, and all baseline code are publicly available to accelerate computational ASO research.
],
)

= Introduction

Antisense oligonucleotides (ASOs) that recruit RNase H to degrade target mRNA have emerged as a clinically validated therapeutic modality, with several approved drugs targeting conditions from spinal muscular atrophy to hereditary transthyretin amyloidosis. The 2'-MOE gapmer design --- flanking modified nucleotides surrounding a central DNA gap --- has proven particularly successful in the central nervous system, where intrathecal delivery achieves sustained target knockdown.

Despite this clinical success, the vast majority of genetically amenable conditions remain undeveloped. A key barrier is the cost of preclinical screening: identifying a single development candidate typically requires synthesising and testing thousands of ASO sequences through a sequential pipeline of in vitro efficacy assays, dose-response characterisation, and multi-species in vivo toxicity studies. Each successive stage is more expensive and lower-throughput than the last, with in vivo studies in rodents and primates dominating total programme costs.

Several computational approaches have demonstrated that oligonucleotide sequence features carry predictive information about ASO properties. Hagedorn _et al._ (2013) showed that dinucleotide composition predicts hepatotoxicity in locked nucleic acid (LNA) gapmers using random forest classifiers. Hagedorn _et al._ (2022) extended this approach to neurotoxicity prediction using a logistic regression model based on five sequence features. More recently, Gehrmann _et al._ (2025) introduced OligoAI, a deep learning model trained on a proprietary database of 188,521 ASO sequences that achieves 3.14$times$ enrichment for in vitro efficacy prediction. However, these models were developed on proprietary datasets and there is no common benchmark for comparing their contributions across different pipeline stages.

A critical gap remains: there is no standardised public dataset linking ASO chemical structure to multiple preclinical endpoints, and no evaluation framework that translates model accuracy into practical screening cost savings. Without these, it is difficult to compare methods fairly or to quantify the real-world value of computational pre-screening.

In this work, we make three contributions:

+ *OligoStack* --- the largest public multi-endpoint ASO preclinical dataset, extracted from USPTO patent filings and annotated with HELM chemical-structure strings. The dataset spans #comma(R.genes.n_unique) target genes across four assay types.

+ A *cost-based benchmarking framework* that evaluates classifiers by their enrichment of the candidate pool at expensive pipeline stages, translating accuracy into dollar savings and animal reduction.

+ A *baseline benchmark* replicating the Hagedorn hepatotoxicity (2013) and neurotoxicity (2022) models on OligoStack with proper GroupKFold cross-validation by target gene, establishing reference performance for future methods.

= Results

== OligoStack Dataset

#figure(
  image("plots/fig1/fig1.svg", width: 100%),
  caption: [Overview of the OligoStack extraction pipeline and resulting dataset. *(A)* Three-stage pipeline converting USPTO patent tables into structured preclinical ASO datasets with HELM annotations. *(B)* Flow of compounds through in vitro hit screening, dose-response characterisation, and in vivo toxicity assessment; box heights are proportional to measurement counts, flow widths to the fraction of shared compounds. *(C)* Distribution of measurements across target genes; each coloured segment represents a major gene, with remaining genes grouped as "Other".],
) <fig1>

OligoStack comprises four linked assay categories extracted from USPTO patent filings: #comma(R.in_vitro.n_measurements) single-concentration in vitro inhibition measurements across #comma(R.in_vitro.n_asos) ASOs, #comma(R.dose_response.n_measurements) multi-dose response measurements across #comma(R.dose_response.n_asos) ASOs, #comma(R.hepatic.n_records) hepatorenal toxicity records across #comma(R.hepatic.n_asos) ASOs with #R.hepatic.n_biomarker_channels biomarker channels, and #comma(R.neuro.n_records) neurotoxicity records across #comma(R.neuro.n_asos) ASOs (@fig1). All compounds are annotated with HELM chemical-structure strings enabling extraction of chemistry-aware features. Target gene identity was resolved for #R.neuro.gene_coverage_pct% of neurotoxicity records through compound-level cross-referencing and patent-level majority voting.

== The Preclinical Screening Pipeline

#figure(
  image("plots/fig2/fig2.svg", width: 100%),
  caption: [The ASO preclinical screening pipeline. *(A)* Attrition funnel showing the number of ASO candidates entering each stage, per-stage pass rates, and cumulative costs. Box heights are proportional to log(ASO count). *(B)* Distribution of measurements at each pipeline stage with pass/fail thresholds (red dashed lines) and passing regions (green shading).],
) <fig2>

The ASO preclinical pipeline consists of seven sequential stages: in vitro efficacy (inhibition \>80%), in vitro potency (IC50 \<500 nM by electroporation), mouse liver toxicity (ALT \<100 IU/L), mouse neurotoxicity (FOB ≤1), rat liver toxicity, rat neurotoxicity, and monkey liver toxicity (@fig2). Back-calculation from OligoStack pass rates shows that producing a single development candidate requires screening approximately #comma(R.pipeline.baseline_n_initial) initial ASOs at an estimated total cost of \$#str(calc.round(R.pipeline.baseline_total_cost / 1000000, digits: 1))M. The majority of this cost is concentrated at the in vivo stages, where per-ASO costs range from \$15,000 (mouse) to \$100,000 (monkey).

== Clinical Benchmark: ION582

#figure(
  image("plots/fig3/fig3.svg", width: 100%),
  caption: [Benchmarking ION582 (zilganersen) against OligoStack distributions. *(A)* IC50 distribution in iCell GABANeurons under free uptake conditions with ION582 marked. *(B)* Mouse FOB score distribution (ICV, 700 µg, 3 h post-dose). *(C)* Rat FOB score distribution (IT, 3000 µg, 3 h post-dose).],
) <fig3>

To contextualise the dataset, we benchmarked ION582 (zilganersen), a clinical-stage ASO targeting _UBE3A-ATS_ for Angelman syndrome, against the OligoStack distributions (@fig3). This demonstrates that the dataset captures the range of preclinical properties relevant to clinical development.

== Toxicity Prediction

#figure(
  image("plots/fig4/fig4.svg", width: 100%),
  caption: [Hagedorn model replication on OligoStack with GroupKFold cross-validation by target gene. *Top row (A--C):* Hepatotoxicity (Hagedorn 2013). *(A)* Classification metrics for four RF model variants on ALT prediction. *(B)* ROC curve for the dinucleotide model. *(C)* Confusion matrix (accuracy = #str(R.hagerdorn_hepatotox.accuracy)). *Bottom row (D--F):* Neurotoxicity (Hagedorn 2022). *(D)* Classification metrics for five models (linear + 4 RF). *(E)* ROC curves for the linear and dinucleotide RF models. *(F)* Confusion matrix for the dinucleotide RF model (accuracy = #str(R.hagerdorn_neurotox.accuracy)).],
) <fig4>

We replicated the Hagedorn hepatotoxicity (2013) and neurotoxicity (2022) prediction approaches on OligoStack, adapting them from LNA to DNA/MOE/cET chemistry (@fig4). All models were evaluated with GroupKFold cross-validation (5 folds, grouped by target gene to prevent information leakage).

For hepatotoxicity, four random forest classifiers of increasing feature complexity were evaluated on ALT prediction. The dinucleotide model (288 features) achieved an accuracy of #str(R.hagerdorn_hepatotox.accuracy) with an AUC of #str(R.hagerdorn_hepatotox.auc) on #comma(R.hagerdorn_hepatotox.n) compounds across #comma(R.hagerdorn_hepatotox.n_groups) target gene groups (sensitivity #str(R.hagerdorn_hepatotox.sensitivity), specificity #str(R.hagerdorn_hepatotox.specificity)).

For neurotoxicity, we evaluated the original 5-feature logistic regression and four RF variants on mouse FOB scores (700 µg ICV; neurotoxic FOB ≥3 vs non-toxic FOB ≤1). The dinucleotide RF model achieved an accuracy of #str(R.hagerdorn_neurotox.accuracy) with an AUC of #str(R.hagerdorn_neurotox.auc) on #comma(R.hagerdorn_neurotox.n) compounds across #comma(R.hagerdorn_neurotox.n_groups) target gene groups.

== Cost Savings from Computational Pre-Screening

#figure(
  image("plots/fig5/fig5.svg", width: 100%),
  caption: [Cost savings from computational pre-screening. *(A)* Stacked bar chart comparing total pipeline costs by stage across four scenarios: baseline (no screening), Hagerdorn classifiers only (ALT + FOB), OligoAI only (in vitro efficacy), and all models combined. *(B)* Summary table of costs, savings, and enrichment factors.],
) <fig5>

To quantify the practical value of computational pre-screening, we compared three enrichment scenarios against the baseline pipeline (@fig5). The Hagerdorn classifiers (hepatotoxicity + neurotoxicity) enrich the candidate pool at the mouse ALT and mouse FOB stages, reducing cost by #str(R.pipeline.hagerdorn_savings_pct)% to \$#str(calc.round(R.pipeline.hagerdorn_total_cost / 1000000, digits: 2))M. OligoAI, a recently published deep learning model for in vitro efficacy prediction (Gehrmann _et al._, 2025), achieves a 3.14$times$ enrichment at the inhibition stage, reducing cost by #str(R.pipeline.oligoai_savings_pct)% to \$#str(calc.round(R.pipeline.oligoai_total_cost / 1000000, digits: 2))M. Combining all three enrichment stages yields a #str(R.pipeline.combined_savings_pct)% total reduction to \$#str(calc.round(R.pipeline.combined_total_cost / 1000000, digits: 2))M --- requiring only #comma(R.pipeline.combined_n_initial) initial ASOs instead of #comma(R.pipeline.baseline_n_initial).

= Discussion

We have introduced OligoStack, a public multi-endpoint ASO preclinical dataset, and a cost-based benchmarking framework for evaluating computational screening methods. Our replication of the Hagedorn hepatotoxicity and neurotoxicity models on OligoStack establishes baseline performance for future methods to improve upon.

The Hagedorn models achieve moderate classification accuracy on OligoStack, consistent with the original publications despite the shift from LNA to MOE/cET chemistry. The use of GroupKFold cross-validation by target gene provides a more realistic estimate of generalisation performance than random splits, since in practice new ASO programmes target novel genes not seen during model development.

Even modest enrichment at the expensive in vivo stages translates into meaningful cost savings, as demonstrated by our cost-based framework. When combined with OligoAI's in vitro enrichment (Gehrmann _et al._, 2025), the total cost reduction reaches #str(R.pipeline.combined_savings_pct)%. This highlights the importance of evaluating models not just by accuracy metrics but by their practical impact on screening economics, and suggests that stacking complementary models across pipeline stages can yield compounding benefits.

Several limitations should be noted. First, the Hagedorn models use only chemistry-derived sequence features and do not incorporate target gene context, target site thermodynamics, or off-target binding potential. Second, OligoStack is derived from Ionis Pharmaceuticals patents and therefore reflects the design space and chemical modifications used by a single organisation. Third, the binary classification approach (high vs low toxicity) discards intermediate cases and may oversimplify the dose-response relationship between sequence features and toxicity.

Future directions include: (i) deep learning models that operate directly on HELM strings rather than hand-crafted features; (ii) multi-task approaches that jointly predict across endpoints; (iii) target-aware models that incorporate gene-level information; and (iv) regression rather than classification to capture the full spectrum of toxicity outcomes.

= Methods

== OligoStack Extraction Pipeline

OligoStack is a three-stage language-model-powered pipeline that converts unstructured tables in USPTO patent XML files into flat, structured preclinical ASO datasets annotated with HELM chemical-structure strings.

*Stage 1 — Table extraction.* USPTO patent XML documents filed after 2001 by ISIS Pharmaceuticals (now Ionis Pharmaceuticals) were retrieved via the USPTO bulk-data API and split into individual table files, each paired with the five paragraphs of prose immediately preceding the table. Tables were deduplicated in two phases: exact MD5 hash matching, followed by pairwise sequence similarity (Python `SequenceMatcher`, 90% threshold) with length-based pruning; connected components were resolved by depth-first search to select a single canonical file per group, reducing 35,871 raw tables from 1,125 Ionis Pharmaceuticals patents to 8,435 canonical tables. For each canonical table, GPT-5-mini generated a bespoke Python extraction function tailored to that table's XML layout. Generated scripts were executed in a sandboxed subprocess with a restricted import whitelist (`json`, `re`, `xml.etree.ElementTree`, `math`) and a ten-second timeout. An agentic self-repair loop allowed the model to observe execution output or errors and regenerate corrected code, for up to five attempts per table.

*Stage 2 — HELM annotation.* For each canonical table the model received the surrounding patent prose and a table preview, then generated a function that constructs Hierarchical Editing Language for Macromolecules (HELM) strings nucleotide-by-nucleotide from the chemistry description. To avoid hallucinated annotations, the function was required to return null when explicit modification data could not be found in the patent text. All HELM strings were validated by a rule-based checker that enforces correct sugar tokens (`[moe]`, `d`, `r`, `m`, `[cet]`, `[fR]`, `[lna]`), canonical bases (`A`, `C`, `G`, `T`, `U`, `[5meC]`), backbone tokens (`[sp]`, `[am]`, `.`), balanced bracket syntax, and terminal-nucleotide constraints.

*Stage 3 — Schema-driven collation.* Each assay category was extracted in a separate pass by supplying a schema dictionary that maps desired column names to natural-language descriptions. GPT-5 generated a mapping function per table that translates table-specific field names to the target schema, performs unit conversions, and unpivots multi-measurement rows into separate records. The same sandboxed execution and self-repair loop as Stage~1 were applied. HELM annotations were merged into each output record by compound identifier, following canonical-link chains across duplicate tables. Four schemas were defined, one per assay category: (i)~in vitro inhibition --- percent inhibition, cell line, dosage in nM, transfection method, treatment period, and target gene; (ii)~dose response --- the same fields with dosage in the original unit (nM or~$mu$M); (iii)~hepatorenal toxicity --- seven serum biomarkers (ALB, ALT, AST, BUN, CREA, TBIL, protein/creatinine ratio), dosage, species, strain, number of doses, dosing period, measurement source, and administration route; and (iv)~neurotoxicity --- FOB score, dosage, species, strain, latency, administration method, and score type.

== Data Collection and Preprocessing

All preclinical ASO data were extracted from USPTO patent filings using the OligoStack pipeline. Raw tables were parsed into four assay categories: in vitro hit screening (single-concentration percent inhibition), in vitro dose response (multi-dose percent inhibition), in vivo hepatorenal toxicity (serum biomarkers), and in vivo neurological toxicity (functional observational battery scores). Each category underwent a standardised cleaning pipeline described below.

Five HELM-level quality filters were applied uniformly across all four assay categories before any assay-specific cleaning. Compounds whose HELM annotation contained uncertain sugar or backbone tokens (indicated by `?` placeholders in the HELM string) were removed, as were sequences with ten or fewer nucleotides (likely extraction artefacts), uniformly modified compounds with no DNA gap (steric-blocking ASOs outside the scope of gapmer-focused analysis), homopolymer sequences comprising a single repeated nucleotide (extraction artefacts from poly-T or poly-A placeholder annotations), and naked DNA oligonucleotides lacking any sugar modifications (unmodified sequences where the extraction model could not identify the modification pattern).

*In vitro inhibition.* Inhibition values outside the range \[--1000%, 100%\] were discarded as extraction artefacts. Cell line names and species labels were standardised to canonical forms (e.g.~"A-431" $arrow$ "A431", "cynomolgus" $arrow$ "monkey"). Duplicate measurements --- defined as identical compound ID and inhibition value --- were collapsed, retaining the most recent patent. This yielded #comma(R.in_vitro.n_measurements) measurements across #comma(R.in_vitro.n_asos) ASOs.

*Dose response.* Dosages reported in $mu$M were converted to nM. The same inhibition range filter and species standardisation were applied. Rows with non-positive dosages were removed. Cell line names were harmonised using a lookup table of 150+ known aliases. Deduplication on compound, dosage, and inhibition yielded #comma(R.dose_response.n_measurements) measurements across #comma(R.dose_response.n_asos) ASOs.

*Hepatorenal toxicity.* Dosage strings were parsed into numeric values and units. Weekly dosing rates (mg/kg/wk) were converted to per-dose equivalents (mg/kg) using the reported number of doses and dosing period. Only subcutaneous and intraperitoneal administration routes with plasma or urine measurement sources were retained. Extreme dosages ($>$ 10,000 mg/kg) were excluded. Biomarker values were range-filtered (ALT and AST: 10--50,000 IU/L; ALB: 1--100 g/dL). Rows sharing the same compound, species, dosing regimen, and administration method were collapsed, aggregating biomarker replicates into lists. Species and strain names were standardised. The final dataset contained #comma(R.hepatic.n_records) collapsed records across #comma(R.hepatic.n_asos) ASOs, with #R.hepatic.n_biomarker_channels biomarker channels (ALB, ALT, AST, BUN, CREA, TBIL, protein/creatinine ratio).

*Neurological toxicity.* FOB (functional observational battery) score strings were parsed into numeric lists, retaining only values in the valid range \[0, 7\]. Species and strain names were standardised (e.g.~"C57/B16 mice" $arrow$ "C57BL/6 mice"). Since the raw neurotoxicity data lacked target gene annotations, compound-to-gene mappings were inferred in two steps: first by direct lookup against the in vitro and dose response datasets, then by patent-level majority vote for remaining compounds (assigning the most frequent target gene within each USPTO patent). This resolved target gene identity for #R.neuro.gene_coverage_pct% of neurotoxicity records. Deduplication on HELM annotation, species, administration method, score type, and dosage yielded #comma(R.neuro.n_records) records across #comma(R.neuro.n_asos) ASOs.

*Gene symbol mapping.* Target RNA names from the patent text were mapped to canonical HGNC gene symbols using the Ensembl REST API, supplemented by a manually curated dictionary of 300+ aliases (e.g.~"Tau" $arrow$ #emph[MAPT], "PKK" $arrow$ #emph[PRKDC], "K-Ras" $arrow$ #emph[KRAS]). After merging synonyms, the combined dataset spans #comma(R.genes.n_unique) unique target genes and #comma(R.genes.n_total_measurements) total measurements.

== Hagedorn Hepatotoxicity Model (2013)

We replicated the Hagedorn _et al._ (2013) approach for predicting ASO hepatotoxicity from sequence composition. The original method uses random forest classifiers trained on dinucleotide features of LNA gapmers to classify compounds as high or low hepatotoxicity based on serum ALT levels.

*Feature extraction.* Four feature sets of increasing complexity were evaluated: (i) a baseline model with 5 features (presence of chemical modification, MOE/cET/DNA nucleotide counts, and sequence length); (ii) a counts model with 15 features (sugar-base combination counts plus phosphorothioate linkage count); (iii) the Hagedorn dinucleotide model with 288 features (all pairwise dinucleotide transitions across 12 nucleotide types and 2 linkage types); and (iv) a position-specific model with 480 features (nucleotide identity at each position from both ends). Dosing covariates (mg/kg, number of doses, dosing period, administration route) were included as additional features.

*Label assignment.* For each biomarker, the upper limit of normal (ULN) was estimated as median + 3 $times$ MAD from the data distribution. Compounds with mean ALT > 3 $times$ ULN were labelled "high" toxicity; those with ALT < 1 $times$ ULN were labelled "low". Intermediate compounds were excluded from classification.

*Model training.* Random forest classifiers (1000 trees, max 8 features per split, balanced class weights) were trained with GroupKFold cross-validation (5 folds, grouped by target gene). This prevents information leakage from compounds targeting the same gene appearing in both training and test sets.

== Hagedorn Neurotoxicity Model (2022)

We replicated the Hagedorn _et al._ (2022) approach for predicting ASO neurotoxicity from sequence features, evaluated on mouse FOB scores from ICV administration at 700 µg.

*Linear model.* The original 5-feature logistic regression model uses: G nucleotide count, A nucleotide count, G-free stretch from the 3' end, G-free stretch from the 5' end, and total sequence length. Balanced class weights were applied.

*RF models.* The same four random forest feature sets used for hepatotoxicity were also evaluated for neurotoxicity, without dosing covariates (all neurotoxicity data used the same ICV 700 µg protocol).

*Label assignment.* Compounds with mean FOB ≥ 3 were labelled "neurotoxic"; those with mean FOB ≤ 1 were labelled "non-toxic". Intermediate scores were excluded.

== Cross-Validation Procedure

All models were evaluated using GroupKFold cross-validation with 5 folds, where groups were defined by target gene. This ensures that all compounds targeting the same gene appear exclusively in either the training or test set within each fold, preventing inflated performance estimates from gene-level information leakage. Group assignment prioritised direct target gene annotations, supplemented by compound-level cross-referencing to the in vitro dataset and USPTO patent ID as a fallback proxy.

== Cost-Based Evaluation Framework

To translate classification accuracy into practical value, we developed a cost-based evaluation framework. For each pipeline stage, per-ASO costs were estimated from industry benchmarks: \$500 (in vitro efficacy), \$2,000 (dose-response), \$15,000 (mouse hepatotoxicity), \$20,000 (mouse neurotoxicity), \$25,000 (rat hepatotoxicity), \$30,000 (rat neurotoxicity), and \$100,000 (monkey hepatotoxicity).

The *enrichment factor* at a given stage is defined as the ratio of the pass rate among classifier-selected compounds to the base pass rate: $"EF" = P("pass" bar.v "selected") / P("pass")$, where "selected" denotes compounds predicted as low-toxicity (P(high) \< 0.5). An enrichment factor greater than one indicates that the classifier concentrates passing compounds among its predictions.

Pipeline costs are back-calculated by determining the number of initial ASOs needed to yield one development candidate, given the pass rates at each stage. With classifier pre-screening, the effective pass rates at enriched stages increase by the enrichment factor, reducing the required number of initial candidates and the associated costs at all downstream stages.

= Code Availability

// Link to code repository if applicable

#bibliography("zotero.bib")
