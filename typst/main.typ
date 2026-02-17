#import "@preview/clean-math-paper:0.2.5": *

#let date = datetime.today().display("[month repr:long] [day], [year]")

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

- *Panel A — Sankey diagram.* Data flow from in vitro inhibition (156K measurements, 8.5K ASOs) → dose response (65K, 4.5K) → hepatic tox (4.8K, 2.4K) / neuro tox (1K, 300). Box heights ∝ measurements, flow widths ∝ compound overlap.
- *Panel B — Summary statistics.* Measurement counts, unique ASOs, species, source (USPTO patents). Small inset table or Typst annotation alongside Panel A.
- *Status:* Panel A exists. Panel B needs minor addition.

#figure(
  image("plots/fig1_flowchart.svg", width: 100%),
  caption: [Overview of preclinical ASO data extracted from USPTO patents, showing the flow of compounds through in vitro hit screening, dose-response characterisation, and in vivo toxicity assessment. Box heights are proportional to the number of measurements; flow widths are proportional to the fraction of shared compounds.],
) <fig1>

== Fig 2: The Pipeline Problem

_Message: "Sequential screening requires \~2,000 initial candidates at \~\$3.5M to yield 3 development candidates. Most cost is concentrated at expensive in vivo stages where attrition is highest."_

- *Panel A — Pipeline attrition funnel.* 7 stages (inhibition \>80%, IC50 \<500nM, mouse ALT \<100, mouse FOB ≤1, rat ALT \<100, rat FOB ≤1, monkey ALT \<100). Shows ASO counts, pass rates, and per-stage costs.
- *Panel B — Stage measurement distributions.* 8-panel histograms with pass/fail thresholds (red dashed lines + green pass regions). Gives intuition for how stringent each cutoff is.
- *Status:* Both panels exist (`pipeline_attrition.svg`, `pipeline_distributions.svg`). Need composition into a single figure.

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

= Discussion

// Interpret your findings

= Methods

== Problem Formulation

// Describe your approach

= Code Availability

// Link to code repository if applicable

#bibliography("zotero.bib")
