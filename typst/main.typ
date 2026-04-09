#import "@preview/bloated-neurips:0.7.0": neurips2025

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

#let affls = (
  oxford-paeds: (
    department: "Department of Paediatrics",
    institution: "University of Oxford",
    location: "Oxford",
    country: "United Kingdom",
  ),
  bdi: (
    department: "Big Data Institute",
    institution: "University of Oxford",
    location: "Oxford",
    country: "United Kingdom",
  ),
  broad: (
    department: "Broad Center for Mendelian Genomics, Program in Medical and Population Genetics",
    institution: "Broad Institute of MIT and Harvard",
    location: "Cambridge, MA",
    country: "USA",
  ),
  idrm: (
    department: "Institute of Developmental and Regenerative Medicine (IDRM)",
    institution: "University of Oxford",
    location: "Oxford",
    country: "United Kingdom",
  ),
  nygc: (
    institution: "New York Genome Center",
    location: "New York, NY",
    country: "USA",
  ),
  ucsf: (
    department: "Department of Psychiatry and Behavioral Sciences, UCSF Weill Institute for Neurosciences",
    institution: "University of California, San Francisco",
    location: "San Francisco, CA",
    country: "USA",
  ),
)

#let authors = (
  (name: "Barney Hill", affl: ("oxford-paeds", "bdi", "idrm"), email: "barney.hill@merton.ox.ac.uk"),
  (name: "Nicola Whiffin", affl: ("oxford-paeds", "bdi", "broad")),
  (name: "Stephan J. Sanders", affl: ("oxford-paeds", "idrm", "nygc", "ucsf")),
  (name: "Carlo Rinaldi", affl: ("oxford-paeds", "idrm"), email: "carlo.rinaldi@idrm.ox.ac.uk"),
)

#show: neurips2025.with(
  title: [ASO Atlas 2.0: a cost-based benchmark for antisense oligonucleotide preclinical screening],
  authors: (authors, affls),
  keywords: ("Antisense Oligonucleotides", "Drug Discovery", "Toxicity Prediction", "Benchmarking"),
  abstract: [
    Preclinical drug development is a sequential, high-attrition process in which candidates must clear in vitro efficacy and dose-response screens before advancing to costly in vivo toxicity studies, and failure at any late stage wastes the cost of every preceding experiment. Despite growing interest in machine-learning-guided drug design, no public benchmark is structured around the sequential preclinical pipeline, linking the same candidates from in vitro efficacy through in vivo toxicity with an evaluation framework that reflects stage-gated decision-making. This gap has left the field without a realistic evaluation setting. Models are trained and tested on isolated endpoints, with no framework for quantifying whether predictive accuracy translates into practical savings in animals, time, or cost. Here we introduce ASO Atlas 2.0, the first such multi-endpoint preclinical dataset, built around antisense oligonucleotides (ASOs), short synthetic nucleic acids that silence disease-related genes. ASO Atlas 2.0 comprises #comma(R.in_vitro.n_measurements) in vitro efficacy measurements, #comma(R.dose_response.n_measurements) dose-response curves, #comma(R.hepatic.n_records) hepatotoxicity records, and #comma(R.neuro.n_records) neurotoxicity records across #comma(R.genes.n_unique) target genes, all extracted from Ionis Pharmaceuticals patent filings using an agentic language-model workflow. We propose a cost-based benchmarking framework that evaluates classifiers by their ability to enrich the candidate pool at expensive in vivo stages, quantifying savings in terms of the number of animals and dollars required to yield a development candidate. ASO Atlas 2.0, the benchmarking framework, and all baseline code are publicly available to accelerate computational ASO research.
  ],
  bibliography: bibliography("zotero.bib"),
  appendix: [
    = Supplementary Figures

    #figure(
      image("plots/fig3/fig3.svg", width: 100%),
      caption: [Clinical benchmark: ION582 (Zilganersen) in context of ASO Atlas 2.0 distributions. Grey bars show all targets; blue bars highlight _UBE3A-ATS_-targeting ASOs; the red arrow marks ION582. *(A)* IC50 distribution under gymnosis/free-uptake conditions. *(B)* Mouse bFOB score distribution (ICV, 700 µg, 3 h post-dose). *(C)* Rat mFOB score distribution (IT, 3,000 µg, 3 h post-dose).],
    ) <figS1>

    #figure(
      image("plots/supp_mouse_rat_alt/mouse_vs_rat_alt.svg", width: 70%),
      caption: [Cross-species hepatotoxicity concordance. Mean ALT per compound in mouse vs rat (log-log), with Spearman correlation shown.],
    ) <figS2>
  ],
  accepted: false,
)

= Introduction

Antisense oligonucleotides (ASOs) that recruit RNase H to degrade target mRNA have emerged as a clinically validated therapeutic modality, with several approved drugs targeting conditions from spinal muscular atrophy to hereditary transthyretin amyloidosis. The 2'-MOE gapmer design --- flanking modified nucleotides surrounding a central DNA gap --- has proven particularly successful in the central nervous system, where intrathecal delivery achieves sustained target knockdown.

Despite this clinical success, the vast majority of genetically amenable conditions remain undeveloped. A key barrier is the cost of preclinical screening: identifying a single development candidate typically requires synthesising and testing thousands of ASO sequences through a sequential pipeline of in vitro efficacy assays, dose-response characterisation, and multi-species in vivo toxicity studies. Each successive stage is more expensive and lower-throughput than the last, with in vivo studies in rodents and primates dominating total programme costs.

Several computational approaches have demonstrated that oligonucleotide sequence features carry predictive information about ASO properties. @hagedorn_hepatotoxic_2013 showed that dinucleotide composition predicts hepatotoxicity in locked nucleic acid (LNA) gapmers using random forest classifiers. @hagedorn_acute_2022 extended this approach to neurotoxicity prediction using a logistic regression model based on five sequence features. More recently, @hill_accurately_2025 introduced OligoAI, a deep learning model trained on a proprietary database of 188,521 ASO sequences that achieves 3.14$times$ enrichment for in vitro efficacy prediction. However, these models were developed on proprietary datasets and there is no common benchmark for comparing their contributions across different pipeline stages.

A critical gap remains: there is no standardised public dataset linking ASO chemical structure to multiple preclinical endpoints, and no evaluation framework that translates model accuracy into practical screening cost savings. Without these, it is difficult to compare methods fairly or to quantify the real-world value of computational pre-screening.

In this work, we make three contributions:

+ *ASO Atlas 2.0* --- the largest public multi-endpoint ASO preclinical dataset, extracted from USPTO patent filings and annotated with HELM chemical-structure strings. The dataset spans #comma(R.genes.n_unique) target genes across four assay types.

+ A *cost-based benchmarking framework* that evaluates classifiers by their enrichment of the candidate pool at expensive pipeline stages, translating accuracy into dollar savings and animal reduction.

+ A *baseline benchmark* replicating the Hagedorn hepatotoxicity @hagedorn_hepatotoxic_2013 and neurotoxicity @hagedorn_acute_2022 models on ASO Atlas 2.0 with proper GroupKFold cross-validation by target gene, establishing reference performance for future methods.

= Results

== ASO Atlas 2.0 Dataset

#figure(
  image("plots/fig1/fig1.svg", width: 100%),
  caption: [Overview of the ASO Atlas 2.0 extraction pipeline and resulting dataset. *(A)* Three-stage pipeline converting USPTO patent tables into structured preclinical ASO datasets with HELM annotations. *(B)* Distribution of measurements across target genes; each coloured segment represents a major gene, with remaining genes grouped as "Other". *(C)* Flow of compounds through in vitro hit screening, dose-response characterisation, and in vivo toxicity assessment; box heights are proportional to measurement counts, flow widths to the fraction of shared compounds.],
) <fig1>

ASO Atlas 2.0 comprises four linked assay categories extracted from USPTO patent filings: #comma(R.in_vitro.n_measurements) single-concentration in vitro inhibition measurements across #comma(R.in_vitro.n_asos) ASOs, #comma(R.dose_response.n_measurements) multi-dose response measurements across #comma(R.dose_response.n_asos) ASOs, #comma(R.hepatic.n_records) hepatorenal toxicity records across #comma(R.hepatic.n_asos) ASOs with #R.hepatic.n_biomarker_channels biomarker channels, and #comma(R.neuro.n_records) neurotoxicity records across #comma(R.neuro.n_asos) ASOs (@fig1). All compounds are annotated with HELM chemical-structure strings enabling extraction of chemistry-aware features. Target gene identity was resolved for #R.neuro.gene_coverage_pct% of neurotoxicity records through compound-level cross-referencing and patent-level majority voting.

== The Preclinical Screening Pipeline

#figure(
  image("plots/fig2/fig2.svg", width: 100%),
  caption: [The ASO preclinical screening pipeline. *(A)* Distribution of measurements at each pipeline stage with pass/fail thresholds (red dashed lines); grey shading marks the failing region. *(B)* Attrition funnel showing the number of ASO candidates entering each stage, per-stage pass rates, and cumulative costs. Box heights are proportional to log(ASO count).],
) <fig2>

The ASO preclinical pipeline consists of seven sequential stages: in vitro efficacy (inhibition \>80%), in vitro potency (IC50 \<500 nM by electroporation), mouse liver toxicity (ALT \<2$times$ULN), mouse neurotoxicity (bFOB ≤1), rat liver toxicity (ALT \<2$times$ULN), rat neurotoxicity (mFOB ≤1), and monkey liver toxicity (@fig2). Back-calculation from ASO Atlas 2.0 pass rates shows that producing a single development candidate requires screening approximately #comma(R.pipeline.baseline_n_initial) initial ASOs at an estimated total cost of \$#str(calc.round(R.pipeline.baseline_total_cost / 1000000, digits: 1))M. The majority of this cost is concentrated at the in vivo stages, where per-ASO costs range from \$15,000 (mouse) to \$100,000 (monkey).

== Toxicity Prediction

#figure(
  image("plots/fig4/fig4.svg", width: 100%),
  caption: [OligoAI-tox: Hagedorn's dinucleotide RF trained on ASO Atlas 2.0 with GroupKFold cross-validation by target gene. *Top row:* Hepatotoxicity (mouse ALT, rat ALT). *Bottom row:* Neurotoxicity (mouse bFOB, rat mFOB). Each panel pair shows ROC curves and confusion matrices. For neurotoxicity, the dashed grey line shows the Hagedorn et al. 2022 baseline (5 fixed coefficients). Mouse ALT: accuracy = #str(R.oligoai_tox_hepatotox.accuracy), AUC = #str(R.oligoai_tox_hepatotox.auc). Mouse bFOB: accuracy = #str(R.oligoai_tox_neurotox.accuracy), AUC = #str(R.oligoai_tox_neurotox.auc).],
) <fig3>

We trained OligoAI-tox, Hagedorn's dinucleotide random forest architecture applied to ASO Atlas 2.0, adapting it from LNA to DNA/MOE/cET chemistry (@fig3). We trained four explicit models: mouse ALT, mouse bFOB, rat ALT, and rat mFOB, each evaluated with GroupKFold cross-validation (5 folds, grouped by target gene).

For hepatotoxicity, the mouse ALT model (128 dinucleotide features, 5,000 trees) achieved an accuracy of #str(R.oligoai_tox_hepatotox.accuracy) with an AUC of #str(R.oligoai_tox_hepatotox.auc) (sensitivity #str(R.oligoai_tox_hepatotox.sensitivity), specificity #str(R.oligoai_tox_hepatotox.specificity)) on #comma(R.oligoai_tox_hepatotox.n) compounds across #comma(R.oligoai_tox_hepatotox.n_groups) target gene groups. The rat ALT model achieved an AUC of #str(R.rat_hepatotox.auc) on #comma(R.rat_hepatotox.n) compounds.

For neurotoxicity, we compared OligoAI-tox against the Hagedorn et al. 2022 baseline (5 fixed coefficients from the original publication). On mouse bFOB scores (700 µg ICV; neurotoxic bFOB $>$ 1 vs non-toxic bFOB $<=$ 1), OligoAI-tox achieved an accuracy of #str(R.oligoai_tox_neurotox.accuracy) with an AUC of #str(R.oligoai_tox_neurotox.auc) on #comma(R.oligoai_tox_neurotox.n) compounds across #comma(R.oligoai_tox_neurotox.n_groups) target gene groups, compared to an AUC of #str(R.hagedorn_linear_neurotox.auc) for the Hagedorn et al. 2022 model. The rat mFOB model achieved an AUC of #str(R.rat_neurotox.auc) on #comma(R.rat_neurotox.n) compounds.

== OligoGym Model Benchmark

#figure(
  include "data/oligogym_benchmark.typ",
  caption: [OligoGym benchmark: Spearman $rho$ (mean $plus.minus$ s.d. across folds) for 9 single-task model architectures and one multi-task Transformer on four ASO toxicity regression tasks. Single-task models use the best featurizer and hyperparameters per model. The multi-task Transformer shares a single encoder across all endpoints with in vitro inhibition as an auxiliary task. GroupKFold cross-validation by target gene (5 folds). Bold indicates best model per dataset.],
) <tbl_benchmark>

To contextualise OligoAI-tox within the broader landscape of oligonucleotide property prediction methods, we benchmarked all model architectures from OligoGym on our four toxicity datasets (@tbl_benchmark). Models were trained in regression mode (predicting continuous biomarker values) using OligoGym's OneHotEncoder and KMersCounts featurizers, with GroupKFold cross-validation matching our evaluation protocol.

Neurotoxicity prediction proved substantially more tractable than hepatotoxicity across all model classes. For mouse FOB, the best single-task models achieved Spearman $rho$ $approx$ 0.72, while mouse ALT prediction remained challenging ($rho$ $approx$ 0.35). Gradient-boosted methods (CatBoost, XGBoost) performed best for hepatotoxicity, while the Transformer excelled at neurotoxicity.

We also trained a multi-task Transformer with a shared encoder across all four vivo endpoints and in vitro inhibition as an auxiliary task, using round-robin batching to equalise gradient updates across tasks. Despite this balancing, the multi-task model underperformed single-task baselines across all endpoints, with the largest gap for neurotoxicity ($rho$ = 0.34 vs 0.72). This negative transfer suggests that the sequence features predictive of in vitro knockdown and in vivo toxicity are largely distinct, and that naively sharing representations across these tasks degrades specialisation. This finding highlights the value of the multi-endpoint benchmark: future methods should explore task-specific architectures or more selective parameter sharing to exploit cross-endpoint signal without suffering interference.

== Cost Savings from Computational Pre-Screening

#figure(
  image("plots/fig6/fig6.svg", width: 100%),
  caption: [Cost savings from computational pre-screening. Stacked bar chart comparing total pipeline costs by stage across four scenarios: baseline (no screening), OligoAI only (in vitro efficacy), OligoAI-tox only (ALT + bFOB), and all models combined.],
) <fig4>

To quantify the practical value of computational pre-screening, we compared three enrichment scenarios against the baseline pipeline (@fig4). OligoAI-tox (hepatotoxicity + neurotoxicity) enriches the candidate pool at the mouse ALT and mouse bFOB stages, reducing cost by #str(R.pipeline.oligoai_tox_savings_pct)% to \$#str(calc.round(R.pipeline.oligoai_tox_total_cost / 1000000, digits: 2))M. OligoAI, a recently published deep learning model for in vitro efficacy prediction @hill_accurately_2025, achieves a 3.14$times$ enrichment at the inhibition stage, reducing cost by #str(R.pipeline.oligoai_savings_pct)% to \$#str(calc.round(R.pipeline.oligoai_total_cost / 1000000, digits: 2))M. Combining all enrichment stages yields a #str(R.pipeline.combined_savings_pct)% total reduction to \$#str(calc.round(R.pipeline.combined_total_cost / 1000000, digits: 2))M --- requiring only #comma(R.pipeline.combined_n_initial) initial ASOs instead of #comma(R.pipeline.baseline_n_initial).

== Cross-Species Concordance and Sequence Motifs

#figure(
  image("plots/fig5/fig5.svg", width: 100%),
  caption: [Selection bias, cross-assay and cross-biomarker correlations, base composition, and cross-species concordance. *(A)* Selection-bias KDEs showing biomarker distributions for compounds that do (blue) vs don't (grey) advance to the next pipeline stage. *(B)* Cross-assay Spearman $rho$: pairwise correlations between per-compound metrics across pipeline stages (BH-corrected; n.s. = not significant). *(C)* Cross-biomarker Spearman $rho$: correlations between mouse hepatotoxicity biomarkers (Bonferroni-corrected). *(D)* Nucleotide base $times$ biomarker Spearman $rho$: base composition vs toxicity biomarkers (BH-corrected). *(E)* Cross-species bFOB concordance: mouse bFOB vs rat mFOB score heatmap (integer-rounded); cell values are compound counts.],
) <fig5>

To assess whether mouse-based screening translates to rat outcomes, we compared per-compound mean biomarker values for compounds tested in both species (@fig5). For hepatotoxicity biomarkers, Spearman correlations were modest but statistically significant: ALT ($rho$ = #str(R.cross_species_hepatotox.ALT.spearman_rho), p < 10#super[--10], n = #str(R.cross_species_hepatotox.ALT.n_shared)), AST ($rho$ = #str(R.cross_species_hepatotox.AST.spearman_rho), p < 10#super[--8], n = #str(R.cross_species_hepatotox.AST.n_shared)), and TBIL ($rho$ = #str(R.cross_species_hepatotox.TBIL.spearman_rho), p < 10#super[--5], n = #str(R.cross_species_hepatotox.TBIL.n_shared)). Binary concordance rates (agreement on high vs low classification using species-specific ULN thresholds) were high: #str(R.cross_species_hepatotox.ALT.concordance_rate) for ALT, #str(R.cross_species_hepatotox.AST.concordance_rate) for AST, and #str(R.cross_species_hepatotox.TBIL.concordance_rate) for TBIL --- though this largely reflects the dominance of concordant low-toxicity compounds.

Neurotoxicity showed stronger cross-species concordance. Mouse bFOB and rat mFOB scores (700 $mu$g ICV vs 3,000 $mu$g IT respectively) were well correlated ($rho$ = #str(R.cross_species_neurotox.FOB.spearman_rho), p $approx$ 0, n = #comma(R.cross_species_neurotox.FOB.n_shared)), with a binary concordance rate of #str(R.cross_species_neurotox.FOB.concordance_rate) across #str(R.cross_species_neurotox.FOB.concordance_n) classifiable compounds.

Within mouse, the hepatotoxicity biomarkers show a structured correlation pattern (@fig5 C). ALT and AST are strongly correlated ($rho$ = 0.92), consistent with their shared hepatocellular origin, while TBIL is moderately correlated with both ($rho$ $approx$ 0.4). The renal markers BUN and CREA form a separate cluster ($rho$ = 0.36), largely independent of the liver enzymes. This structure suggests that ALT and AST carry largely redundant information for toxicity classification, while TBIL and renal biomarkers may offer complementary signal.

= Discussion

We have introduced ASO Atlas 2.0, a public multi-endpoint ASO preclinical dataset, and a cost-based benchmarking framework for evaluating computational screening methods. OligoAI-tox --- Hagedorn's dinucleotide random forest trained on ASO Atlas 2.0 --- establishes baseline performance for future methods to improve upon.

OligoAI-tox achieves moderate classification accuracy on ASO Atlas 2.0, consistent with the original Hagedorn publications despite the shift from LNA to MOE/cET chemistry. The use of GroupKFold cross-validation by target gene provides a more realistic estimate of generalisation performance than random splits, since in practice new ASO programmes target novel genes not seen during model development.

The cross-species concordance analysis provides empirical support for mouse-based screening as a proxy for rat outcomes, particularly for neurotoxicity where bFOB and mFOB scores are strongly correlated between species. This supports the pipeline design assumption that mouse-stage enrichment carries forward to later species.

Even modest enrichment at the expensive in vivo stages translates into meaningful cost savings, as demonstrated by our cost-based framework. When combined with OligoAI's in vitro enrichment @hill_accurately_2025, the total cost reduction reaches #str(R.pipeline.combined_savings_pct)%. This highlights the importance of evaluating models not just by accuracy metrics but by their practical impact on screening economics, and suggests that stacking complementary models across pipeline stages can yield compounding benefits.

Several limitations should be noted. First, OligoAI-tox uses only chemistry-derived sequence features and does not incorporate target gene context, target site thermodynamics, or off-target binding potential. Second, ASO Atlas 2.0 is derived from Ionis Pharmaceuticals patents and therefore reflects the design space and chemical modifications used by a single organisation. Third, the binary classification approach (high vs low toxicity) discards intermediate cases and may oversimplify the dose-response relationship between sequence features and toxicity.

Future directions include: (i) deep learning models that operate directly on HELM strings rather than hand-crafted features; (ii) multi-task approaches that jointly predict across endpoints; (iii) target-aware models that incorporate gene-level information; and (iv) regression rather than classification to capture the full spectrum of toxicity outcomes.

= Methods

== ASO Atlas 2.0 Extraction Pipeline

ASO Atlas 2.0 is a three-stage language-model-powered pipeline that converts unstructured tables in USPTO patent XML files into flat, structured preclinical ASO datasets annotated with HELM chemical-structure strings.

*Stage 1 — Table extraction.* USPTO patent XML documents filed after 2001 by ISIS Pharmaceuticals (now Ionis Pharmaceuticals) were retrieved via the USPTO bulk-data API and split into individual table files, each paired with the five paragraphs of prose immediately preceding the table. Tables were deduplicated in two phases: exact MD5 hash matching, followed by pairwise sequence similarity (Python `SequenceMatcher`, 90% threshold) with length-based pruning; connected components were resolved by depth-first search to select a single canonical file per group, reducing 35,871 raw tables from 1,125 Ionis Pharmaceuticals patents to 8,435 canonical tables. For each canonical table, GPT-5-mini generated a bespoke Python extraction function tailored to that table's XML layout. Generated scripts were executed in a sandboxed subprocess with a restricted import whitelist (`json`, `re`, `xml.etree.ElementTree`, `math`) and a ten-second timeout. An agentic self-repair loop allowed the model to observe execution output or errors and regenerate corrected code, for up to five attempts per table.

*Stage 2 — HELM annotation.* For each canonical table the model received the surrounding patent prose and a table preview, then generated a function that constructs Hierarchical Editing Language for Macromolecules (HELM) strings nucleotide-by-nucleotide from the chemistry description. To avoid hallucinated annotations, the function was required to return null when explicit modification data could not be found in the patent text. All HELM strings were validated by a rule-based checker that enforces correct sugar tokens (`[moe]`, `d`, `r`, `m`, `[cet]`, `[fR]`, `[lna]`), canonical bases (`A`, `C`, `G`, `T`, `U`, `[5meC]`), backbone tokens (`[sp]`, `[am]`, `.`), balanced bracket syntax, and terminal-nucleotide constraints.

*Stage 3 — Schema-driven collation.* Each assay category was extracted in a separate pass by supplying a schema dictionary that maps desired column names to natural-language descriptions. GPT-5 generated a mapping function per table that translates table-specific field names to the target schema, performs unit conversions, and unpivots multi-measurement rows into separate records. The same sandboxed execution and self-repair loop as Stage~1 were applied. HELM annotations were merged into each output record by compound identifier, following canonical-link chains across duplicate tables. Four schemas were defined, one per assay category: (i)~in vitro inhibition --- percent inhibition, cell line, dosage in nM, transfection method, treatment period, and target gene; (ii)~dose response --- the same fields with dosage in the original unit (nM or~$mu$M); (iii)~hepatorenal toxicity --- seven serum biomarkers (ALB, ALT, AST, BUN, CREA, TBIL, protein/creatinine ratio), dosage, species, strain, number of doses, dosing period, measurement source, and administration route; and (iv)~neurotoxicity --- FOB score (bFOB or mFOB), dosage, species, strain, latency, administration method, and score type.

== Data Collection and Preprocessing

All preclinical ASO data were extracted from USPTO patent filings using the ASO Atlas 2.0 pipeline. Raw tables were parsed into four assay categories: in vitro hit screening (single-concentration percent inhibition), in vitro dose response (multi-dose percent inhibition), in vivo hepatorenal toxicity (serum biomarkers), and in vivo neurological toxicity (functional observational battery scores). Each category underwent a standardised cleaning pipeline described below.

Five HELM-level quality filters were applied uniformly across all four assay categories before any assay-specific cleaning. Compounds whose HELM annotation contained uncertain sugar or backbone tokens (indicated by `?` placeholders in the HELM string) were removed, as were sequences with ten or fewer nucleotides (likely extraction artefacts), uniformly modified compounds with no DNA gap (steric-blocking ASOs outside the scope of gapmer-focused analysis), homopolymer sequences comprising a single repeated nucleotide (extraction artefacts from poly-T or poly-A placeholder annotations), and naked DNA oligonucleotides lacking any sugar modifications (unmodified sequences where the extraction model could not identify the modification pattern).

*In vitro inhibition.* Inhibition values outside the range \[--1000%, 100%\] were discarded as extraction artefacts. Cell line names and species labels were standardised to canonical forms (e.g.~"A-431" $arrow$ "A431", "cynomolgus" $arrow$ "monkey"). Duplicate measurements --- defined as identical compound ID and inhibition value --- were collapsed, retaining the most recent patent. This yielded #comma(R.in_vitro.n_measurements) measurements across #comma(R.in_vitro.n_asos) ASOs.

*Dose response.* Dosages reported in $mu$M were converted to nM. The same inhibition range filter and species standardisation were applied. Rows with non-positive dosages were removed. Cell line names were harmonised using a lookup table of 150+ known aliases. Deduplication on compound, dosage, and inhibition yielded #comma(R.dose_response.n_measurements) measurements across #comma(R.dose_response.n_asos) ASOs.

*Hepatorenal toxicity.* Dosage strings were parsed into numeric values and units. Weekly dosing rates (mg/kg/wk) were converted to per-dose equivalents (mg/kg) using the reported number of doses and dosing period. Only subcutaneous and intraperitoneal administration routes with plasma or urine measurement sources were retained. Extreme dosages ($>$ 10,000 mg/kg) were excluded. Biomarker values were range-filtered (ALT and AST: 10--50,000 IU/L; ALB: 1--100 g/dL). Rows sharing the same compound, species, dosing regimen, and administration method were collapsed, aggregating biomarker replicates into lists. Species and strain names were standardised. The final dataset contained #comma(R.hepatic.n_records) collapsed records across #comma(R.hepatic.n_asos) ASOs, with #R.hepatic.n_biomarker_channels biomarker channels (ALB, ALT, AST, BUN, CREA, TBIL, protein/creatinine ratio).

*Neurological toxicity.* Tolerability score strings (bFOB for mice, mFOB for rats) were parsed into numeric lists, retaining only values in the valid range \[0, 7\].

ASO Atlas 2.0 contains two complementary tolerability scoring systems developed at Ionis Pharmaceuticals. In each system, seven criteria are each scored 0 (met) or 1 (not met), and summed into a tolerability score ranging from 0 (no signs) to 7 (all criteria failed). The _behavioural FOB_ (bFOB) assesses consciousness and motor responsiveness 3 h after intracerebroventricular (ICV) delivery of 700 µg in C57BL/6 mice: bright/alert/responsive, standing or hunched without stimuli, any movement without stimuli, forward movement after lifting, any movement after lifting, response to tail pinch, and regular breathing. The _motor FOB_ (mFOB) assesses segmental motor function 3 h after intrathecal (IT) delivery of 3,000 µg in Sprague-Dawley rats: movement of the tail, posterior posture, hind limbs, hind paws, forepaws, anterior posture, and head. In the processed dataset, bFOB accounts for 2,928 mouse records and mFOB for 1,804 rat records. Species and strain names were standardised (e.g.~"C57/B16 mice" $arrow$ "C57BL/6 mice"). Since the raw neurotoxicity data lacked target gene annotations, compound-to-gene mappings were inferred in two steps: first by direct lookup against the in vitro and dose response datasets, then by patent-level majority vote for remaining compounds (assigning the most frequent target gene within each USPTO patent). This resolved target gene identity for #R.neuro.gene_coverage_pct% of neurotoxicity records. Deduplication on HELM annotation, species, administration method, score type, and dosage yielded #comma(R.neuro.n_records) records across #comma(R.neuro.n_asos) ASOs.

*Gene symbol mapping.* Target RNA names from the patent text were mapped to canonical HGNC gene symbols using the Ensembl REST API, supplemented by a manually curated dictionary of 300+ aliases (e.g.~"Tau" $arrow$ #emph[MAPT], "PKK" $arrow$ #emph[PRKDC], "K-Ras" $arrow$ #emph[KRAS]). After merging synonyms, the combined dataset spans #comma(R.genes.n_unique) unique target genes and #comma(R.genes.n_total_measurements) total measurements.

== Hagedorn Hepatotoxicity Model (2013)

We replicated the @hagedorn_hepatotoxic_2013 approach for predicting ASO hepatotoxicity from sequence composition. The original method uses random forest classifiers trained on dinucleotide features of LNA gapmers to classify compounds as high or low hepatotoxicity based on serum ALT levels.

*Feature extraction.* Four feature sets of increasing complexity were evaluated: (i) a baseline model with 5 features (presence of chemical modification, MOE/cET/DNA nucleotide counts, and sequence length); (ii) a counts model with 15 features (sugar-base combination counts plus phosphorothioate linkage count); (iii) the Hagedorn dinucleotide model with 288 features (all pairwise dinucleotide transitions across 12 nucleotide types and 2 linkage types); and (iv) a position-specific model with 480 features (nucleotide identity at each position from both ends). Dosing covariates (mg/kg, number of doses, dosing period, administration route) were included as additional features.

*Label assignment.* Species-specific upper limits of normal (ULN) were taken from published reference intervals as the mean of male and female 97.5th percentiles: 75 IU/L for C57BL/6J mice @otto_clinical_2016, 39 IU/L for Sprague-Dawley rats @he_sex-specific_2017, and 103 IU/L for adult cynomolgus macaques @bakker_reference_2023. The pipeline pass threshold was set at 2$times$ULN (150 IU/L mouse, 78 IU/L rat, 206 IU/L monkey). For classifier training, compounds with mean ALT $≥$ 2$times$ULN were labelled "high" toxicity; those with ALT < 2$times$ULN were labelled "low".

*Model training.* Random forest classifiers (1000 trees, max 8 features per split, balanced class weights) were trained with GroupKFold cross-validation (5 folds, grouped by target gene). This prevents information leakage from compounds targeting the same gene appearing in both training and test sets.

== Hagedorn Neurotoxicity Model (2022)

We replicated the @hagedorn_acute_2022 approach for predicting ASO neurotoxicity from sequence features, evaluated on mouse bFOB scores from ICV administration at 700 µg.

*Linear model.* The original 5-feature logistic regression model uses: G nucleotide count, A nucleotide count, G-free stretch from the 3' end, G-free stretch from the 5' end, and total sequence length. Balanced class weights were applied.

*RF models.* The same four random forest feature sets used for hepatotoxicity were also evaluated for neurotoxicity, without dosing covariates (all neurotoxicity data used the same ICV 700 µg protocol).

*Label assignment.* Compounds with mean bFOB > 1 were labelled "neurotoxic"; those with mean bFOB ≤ 1 were labelled "non-toxic".

== Cross-Validation Procedure

All models were evaluated using GroupKFold cross-validation with 5 folds, where groups were defined by target gene. This ensures that all compounds targeting the same gene appear exclusively in either the training or test set within each fold, preventing inflated performance estimates from gene-level information leakage. Group assignment prioritised direct target gene annotations, supplemented by compound-level cross-referencing to the in vitro dataset and USPTO patent ID as a fallback proxy.

== Cost-Based Evaluation Framework

To translate classification accuracy into practical value, we developed a cost-based evaluation framework. For each pipeline stage, per-ASO costs were estimated from industry benchmarks: \$500 (in vitro efficacy), \$2,000 (dose-response), \$15,000 (mouse hepatotoxicity), \$20,000 (mouse neurotoxicity), \$25,000 (rat hepatotoxicity), \$30,000 (rat neurotoxicity), and \$100,000 (monkey hepatotoxicity).

The *enrichment factor* at a given stage is defined as the ratio of the pass rate among classifier-selected compounds to the base pass rate: $"EF" = P("pass" bar.v "selected") / P("pass")$, where "selected" denotes the top 25% compounds with lowest predicted toxicity risk (lowest P(high)). An enrichment factor greater than one indicates that the classifier concentrates passing compounds among its predictions.

Pipeline costs are back-calculated by determining the number of initial ASOs needed to yield one development candidate, given the pass rates at each stage. With classifier pre-screening, the effective pass rates at enriched stages increase by the enrichment factor, reducing the required number of initial candidates and the associated costs at all downstream stages.

= Code Availability

// Link to code repository if applicable


