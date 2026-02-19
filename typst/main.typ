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
title: "Jointly predicting RNase H-mediated gapmer potency and tolerability to reduce preclinical screening costs",
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
abstract: [Your abstract goes here.],
)

= Introduction

- Relative clinical success for 2-MOE's in CNS
- Despite this vast majority of amenable conditions remain undeveloped bc of cost constraints.
- ASOs require "thick" preclinical pipelines due to failure rates
- 

= Results

== Fig 1: Dataset Overview

_Message: "We assembled the largest public multi-endpoint ASO preclinical dataset, linking in vitro efficacy to in vivo toxicity for thousands of compounds."_

- *Panel A — Sankey diagram.* Data flow from in vitro inhibition → dose response → hepatic tox / neuro tox. Box heights ∝ measurements, flow widths ∝ compound overlap.
- *Panel B — Gene circle.* Donut chart of measurements per target gene aggregated across all four data types. Major genes shown individually; remaining grouped as "Other".

#figure(
  image("plots/fig1/fig1.svg", width: 100%),
  caption: [Overview of the OligoStack extraction pipeline and resulting dataset. *(A)* Three-stage pipeline converting USPTO patent tables into structured preclinical ASO datasets with HELM annotations. *(B)* Flow of compounds through in vitro hit screening, dose-response characterisation, and in vivo toxicity assessment; box heights are proportional to measurement counts, flow widths to the fraction of shared compounds. *(C)* Distribution of measurements across target genes; each coloured segment represents a major gene, with remaining genes grouped as "Other".],
) <fig1>

== Fig 2: The Pipeline Problem

_Message: "Sequential screening requires \~2,000 initial candidates at \~\$3.5M to yield 3 development candidates. Most cost is concentrated at expensive in vivo stages where attrition is highest."_

- *Panel A — Pipeline attrition funnel.* 7 stages (inhibition \>80%, IC50 \<500nM, mouse ALT \<100, mouse FOB ≤1, rat ALT \<100, rat FOB ≤1, monkey ALT \<100). Shows ASO counts, pass rates, and per-stage costs.
- *Panel B — Stage measurement distributions.* 8-panel histograms with pass/fail thresholds (red dashed lines + green pass regions). Gives intuition for how stringent each cutoff is.
- *Status:* Both panels exist (`pipeline_attrition.svg`, `pipeline_distributions.svg`). Need composition into a single figure.

#figure(
  image("plots/fig2/fig2.svg", width: 100%),
  caption: [The ASO preclinical screening pipeline. *(A)* Attrition funnel showing the number of ASO candidates entering each stage, per-stage pass rates, and cumulative costs. Box heights are proportional to log(ASO count). *(B)* Distribution of measurements at each pipeline stage with pass/fail thresholds (red dashed lines) and passing regions (green shading).],
) <fig2>

== Fig 3: Model Overview

_Message: "OligoAI2 jointly predicts ASO potency and tolerability from chemical structure alone, using a shared learned representation across in vitro and in vivo endpoints."_

- *Panel A — High-level architecture schematic.* Input: ASO chemical structure (HELM notation) + experimental covariates (dose, species, transfection). Middle: "Learned ASO representation" (black box). Output: 4 predicted endpoints (% inhibition, ALT, AST, FOB score). Annotate: trained on 249K in vitro + 27K in vivo measurements.
- *Panel B — Training strategy.* Simple 2-phase diagram: Phase 1 warms up on abundant in vitro data, Phase 2 jointly fine-tunes with scarce in vivo data. Emphasise the data imbalance problem this solves.
- *Status:* Does not exist. Best created as a clean vector schematic (Figma/draw.io). Keep deliberately simple — detailed architecture goes in Methods text.

== Fig 4: Results and Pipeline Impact

_Message: "OligoAI2 accurately predicts toxicity endpoints and, when used as a pre-screen, enriches the candidate pool 2–3× at toxicity stages — translating to significant cost savings."_

- *Panel A — Predicted vs. actual scatter plots.* One sub-panel per task (inhibition, ALT, AST, FOB). Density-coloured points, Pearson _r_ annotated. Pass/fail thresholds marked as dashed lines (80% inhibition, 100 IU/L ALT, FOB ≤1). Test set, de-normalised to original units.
- *Panel B — Enrichment bar chart.* Grouped bars: base rate vs. top-10% pass rate for each endpoint, enrichment factor annotated (Inhibition 1.3×, ALT 2.1×, FOB 3.2×). Bridges from "model is accurate" to "model improves screening".
- *Panel C — Cost comparison.* Stacked bar chart: Baseline vs. OligoAI2-guided pipeline costs, broken down by stage. Shows where savings concentrate (in vivo stages).
- *Status:* Panel C exists (`pipeline_cost_comparison.svg`). Panels A and B need creation via `analyses/plotting/plot_figure4.py`.

#figure(
  image("plots/fig4/fig4.svg", width: 100%),
  caption: [OligoAI2 enrichment and pipeline cost impact. *(B)* Grouped bars comparing the base pass rate (grey) with the top-10% model-selected pass rate (coloured) at each endpoint; enrichment factors annotated above. *(C)* Stacked bar chart of total pipeline costs by stage for the baseline screening pipeline versus the OligoAI2-guided pipeline.],
) <fig4>

= Discussion

// Interpret your findings

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

*Gene symbol mapping.* Target RNA names from the patent text were mapped to canonical HGNC gene symbols using the Ensembl REST API, supplemented by a manually curated dictionary of 300+ aliases (e.g.~"Tau" $arrow$ _MAPT_, "PKK" $arrow$ _PRKDC_, "K-Ras" $arrow$ _KRAS_). After merging synonyms, the combined dataset spans #comma(R.genes.n_unique) unique target genes and #comma(R.genes.n_total_measurements) total measurements.

#figure(
  image("plots/fig5/fig5.svg", width: 100%),
  caption: [UMAP projection of OligoAI2 learned ASO representations, coloured by chemistry type *(left)* and GC content *(right)*. Each point represents a unique HELM-annotated compound. The model's bottleneck layer separates the two dominant gapmer designs --- 5-10-5 MOE with full phosphorothioate backbone (blue) and mixed PS/PO backbone (orange) --- as well as 3-10-3 cEt gapmers (red). Within each chemistry cluster, GC content varies smoothly, indicating that the embedding captures both chemical modification patterns and sequence composition.],
) <fig5>

== Problem Formulation

// Describe your approach

= Code Availability

// Link to code repository if applicable

#bibliography("zotero.bib")
