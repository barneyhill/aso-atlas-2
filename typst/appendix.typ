#import "utils.typ": R, comma, mstd

#set figure(numbering: n => "A" + str(n))
#counter(figure.where(kind: image)).update(0)
#counter(figure.where(kind: table)).update(0)

#counter(heading).update(1)
#set heading(numbering: "A.1.1")

== Collation Framework <secS_collation>

USPTO patent XML documents filed after 2001 by Ionis Pharmaceuticals were retrieved via the USPTO bulk-data API and split into individual table files, each paired with the five paragraphs of prose immediately preceding the table. Tables were deduplicated in two phases: exact MD5 hash matching, followed by pairwise sequence similarity (Python `SequenceMatcher`, 90% threshold) with length-based pruning; connected components were resolved by depth-first search to select a single canonical file per group, reducing 35,871 raw tables from 1,125 Ionis Pharmaceuticals patents to 8,435 canonical tables.

For each canonical table, GPT-5-mini generated a bespoke Python extraction function tailored to that table's XML layout. Generated scripts were executed in a sandboxed subprocess with a restricted import whitelist (`json`, `re`, `xml.etree.ElementTree`, `math`) and a ten-second timeout. An agentic self-repair loop allowed the model to observe execution output or errors and regenerate corrected code, for up to five attempts per table.

For HELM annotation, the model received the five paragraphs immediately preceding the table in the patent XML, the table XML itself, and five sampled compound entries from Stage 1, then generated a function that constructs HELM strings nucleotide-by-nucleotide from the chemistry description. To avoid hallucinated annotations, the function was required to return null when explicit modification data could not be found in the patent text. All HELM strings were validated by a rule-based checker that enforced correct sugar tokens (`[moe]`, `d`, `r`, `m`, `[cet]`, `[fR]`, `[lna]`), canonical bases (`A`, `C`, `G`, `T`, `U`, `[5meC]`), backbone tokens (`[sp]`, `[am]`, `.`), balanced bracket syntax, and terminal-nucleotide constraints.

For schema-based collation, each assay category was extracted in a separate pass by supplying a schema dictionary that mapped desired column names to natural-language descriptions. GPT-5 generated a mapping function per table that translated table-specific field names to the target schema, performed unit conversions, and unpivoted multi-measurement rows into separate rows. The same sandboxed execution and self-repair loop were applied. HELM annotations were merged into each output row by compound identifier, following canonical-link chains across duplicate tables. Four schemas were defined, one per assay category: (i)~in vitro inhibition: percent inhibition, cell line, dosage in nM, transfection method, treatment period, and target gene; (ii)~dose response: the same fields with dosage in the original unit (nM or~$mu$M); (iii)~hepatorenal toxicity: seven serum biomarkers (ALB, ALT, AST, BUN, CREA, TBIL, protein/creatinine ratio), dosage, species, strain, number of doses, dosing period, measurement source, and administration route; and (iv)~neurotoxicity: FOB score (bFOB or mFOB), dosage, species, strain, latency, administration method, and score type (@secS_schema).

== Preprocessing and Quality Control <secS_preprocessing>

Raw tables were parsed into four assay categories: in vitro hit screening (single-concentration percent inhibition), in vitro dose response (multi-dose percent inhibition), in vivo hepatorenal toxicity (serum biomarkers), and in vivo neurotoxicity (functional observational battery scores). Each category underwent a standardised cleaning pipeline described below.

Five HELM-level quality filters were applied uniformly across all four assay categories before any assay-specific cleaning. Compounds whose HELM annotation contained uncertain sugar or backbone tokens (indicated by `?` placeholders in the HELM string) were removed, as were sequences with ten or fewer nucleotides (likely extraction artefacts), uniformly modified compounds with no DNA gap (steric-blocking ASOs outside the scope of gapmer-focused analysis), homopolymer sequences comprising a single repeated nucleotide (extraction artefacts from poly-T or poly-A placeholder annotations), and naked DNA oligonucleotides lacking any sugar modifications (unmodified sequences where the extraction model could not identify the modification pattern).

*In vitro inhibition.* Inhibition values outside the range \[--1000%, 100%\] were discarded as extraction artefacts. Cell line names and species labels were standardised to canonical forms (e.g.~"A-431" $arrow$ "A431", "cynomolgus" $arrow$ "monkey"). Duplicate measurements (defined as identical compound ID and inhibition value) were collapsed, retaining the most recent patent. This yielded #comma(R.in_vitro.n_measurements) measurements across #comma(R.in_vitro.n_asos) ASOs.

*Dose response.* Dosages reported in $mu$M were converted to nM. The same inhibition range filter and species standardisation were applied. Rows with non-positive dosages were removed. Cell line names were harmonised using a lookup table of 121 known aliases. Deduplication on compound, dosage, and inhibition yielded #comma(R.dose_response.n_measurements) measurements across #comma(R.dose_response.n_asos) ASOs.

*Hepatorenal toxicity.* Dosage strings were parsed into numeric values and units. Weekly dosing rates (mg/kg/wk) were converted to per-dose equivalents (mg/kg) using the reported number of doses and dosing period. Only subcutaneous and intraperitoneal administration routes with plasma or urine measurement sources were retained. Extreme dosages ($>$ 10,000 mg/kg) were excluded. Biomarker values were range-filtered (ALT and AST: 10--50,000 IU/L; ALB: 1--100 g/dL). Rows sharing the same compound, species, dosing regimen, and administration method were collapsed, aggregating biomarker replicates into lists. Species and strain names were standardised. The final dataset contained #comma(R.hepatic.n_records) observations across #comma(R.hepatic.n_asos) ASOs, with #R.hepatic.n_biomarker_channels biomarker channels (ALB, ALT, AST, BUN, CREA, TBIL, protein/creatinine ratio).

*Neurological toxicity.* Tolerability score strings (bFOB for mice, mFOB for rats) were parsed into numeric lists, retaining only values in the valid range \[0, 7\].

ASO Atlas 2.0 incorporated two complementary tolerability scoring systems developed at Ionis Pharmaceuticals @freierCompoundsMethodsReducing2022. In each system, seven criteria are each scored 0 (met) or 1 (not met), and summed into a tolerability score ranging from 0 (no signs) to 7 (all criteria failed). The _behavioural FOB_ (bFOB) assesses consciousness and motor responsiveness 3 h after intracerebroventricular (ICV) delivery of 700 $mu$g in C57BL/6 mice: bright/alert/responsive, standing or hunched without stimuli, any movement without stimuli, forward movement after lifting, any movement after lifting, response to tail pinch, and regular breathing. The _motor FOB_ (mFOB) assesses segmental motor function 3 h after intrathecal (IT) delivery of 3,000 $mu$g in Sprague-Dawley rats: movement of the tail, posterior posture, hind limbs, hind paws, forepaws, anterior posture, and head. In the processed dataset, bFOB accounts for #comma(R.neuro.n_mouse) mouse observations and mFOB for #comma(R.neuro.n_rat) rat observations. Species and strain names were standardised (e.g.~"C57/B16 mice" $arrow$ "C57BL/6 mice"). Since the raw neurotoxicity data lacked target gene annotations, compound-to-gene mappings were inferred in two steps: first by direct lookup against the in vitro and dose response datasets, then by patent-level majority vote for remaining compounds (assigning the most frequent target gene within each USPTO patent). This resolved target gene identity for #R.neuro.gene_coverage_pct% of neurotoxicity observations. Deduplication on HELM annotation, species, administration method, score type, and dosage yielded #comma(R.neuro.n_records) observations across #comma(R.neuro.n_asos) ASOs.

*Gene symbol mapping.* Target RNA names from the patent text were mapped to canonical HGNC gene symbols using the Ensembl REST API, supplemented by a manually curated dictionary of 300+ aliases (e.g.~"Tau" $arrow$ #emph[MAPT], "PKK" $arrow$ #emph[PRKDC], "K-Ras" $arrow$ #emph[KRAS]). After merging synonyms, the combined dataset spans #comma(R.genes.n_unique) unique target genes and #comma(R.genes.n_total_measurements) total measurements.

== Dataset Characterisation

=== Cross-Species Concordance and Biomarker Structure <secS_concordance>

To assess whether mouse-based screening translates to rat outcomes, we compared per-compound mean biomarker values for compounds tested in both species (@fig_concordance). For hepatotoxicity biomarkers, cross-species Spearman correlations were modest but statistically significant: ALT ($rho$ = #str(R.cross_species_hepatotox.ALT.spearman_rho), p < 10#super[--10], n = #str(R.cross_species_hepatotox.ALT.n_shared)), AST ($rho$ = #str(R.cross_species_hepatotox.AST.spearman_rho), p < 10#super[--8], n = #str(R.cross_species_hepatotox.AST.n_shared)), and TBIL ($rho$ = #str(R.cross_species_hepatotox.TBIL.spearman_rho), p < 10#super[--5], n = #str(R.cross_species_hepatotox.TBIL.n_shared)).

Neurotoxicity showed stronger cross-species concordance. Mouse bFOB and rat mFOB scores (700 $mu$g ICV vs 3,000 $mu$g intrathecal (IT) respectively) were well correlated ($rho$ = #str(R.cross_species_neurotox.FOB.spearman_rho), p $approx$ 0, n = #comma(R.cross_species_neurotox.FOB.n_shared)), with a binary concordance rate of #str(R.cross_species_neurotox.FOB.concordance_rate) across #str(R.cross_species_neurotox.FOB.concordance_n) classifiable compounds.

Within mouse, the hepatotoxicity biomarkers showed a correlation structure consistent with known clinical biochemistry @gianniniLiverEnzymeAlteration2005 (@fig_concordance C). ALT and AST were strongly correlated ($rho$ = 0.92), as expected given that both are cytoplasmic aminotransferases released upon hepatocyte damage. TBIL was moderately correlated with both ($rho$ $approx$ 0.4), reflecting its partially distinct mechanistic origin. The renal markers BUN and CREA formed a separate cluster ($rho$ = 0.36), largely independent of the liver enzymes. This structure confirms that ALT and AST carry largely redundant information for toxicity classification, while TBIL and renal biomarkers may offer complementary signal.

Base composition analysis (@fig_concordance D) revealed that guanine content was strongly positively correlated with neurotoxicity scores in both species (mouse bFOB $rho$ = #str(R.base_biomarker.G_mouse_bfob.rho), rat mFOB $rho$ = #str(R.base_biomarker.G_rat_mfob.rho)). For hepatic biomarkers, cytosine content was positively correlated with rat ALT ($rho$ = #str(R.base_biomarker.C_rat_alt.rho)) but not mouse ALT ($rho$ = #str(R.base_biomarker.C_mouse_alt.rho), n.s.), consistent with the modest cross-species ALT concordance ($rho$ = #str(R.cross_species_hepatotox.ALT.spearman_rho)) reported in @fig_concordance B. @fig_concordance E visualises the cross-species neurotoxicity concordance as an integer-rounded score heatmap, showing strong diagonal concentration consistent with the quantitative correlation reported above.

#figure(
  image("plots/fig_concordance/fig_concordance.svg", width: 100%),
  caption: [Selection bias, cross-assay and cross-species correlations, biomarker structure, and base composition. *(A)* Selection-bias KDEs showing biomarker distributions for compounds that do (blue) vs don't (grey) advance to the next pipeline stage. *(B)* Cross-assay and cross-species Spearman $rho$: pairwise correlations between per-compound metrics across pipeline stages and species (BH-corrected; n.s. = not significant). *(C)* Cross-biomarker Spearman $rho$: correlations between mouse hepatotoxicity biomarkers (Bonferroni-corrected). *(D)* Nucleotide base $times$ biomarker Spearman $rho$: base composition vs toxicity biomarkers (BH-corrected). *(E)* Cross-species bFOB concordance: mouse bFOB vs rat mFOB score heatmap (integer-rounded); cell values are compound counts.],
) <fig_concordance>

=== Clinical Validation

#figure(
  image("plots/fig_clinical/fig_clinical.svg", width: 100%),
  caption: [Clinical validation: ION582 (ACCATTTTGACCTTCTTAGC, 5-10-5 MOE gapmer), a Phase 3 candidate for Angelman syndrome, in context of ASO Atlas 2.0 distributions. Grey bars show all targets; blue bars highlight _UBE3A-ATS_-targeting ASOs; the red arrow marks ION582. *(A)* IC50 distribution under gymnosis/free-uptake conditions. *(B)* Mouse bFOB score distribution (ICV, 700 $mu$g, 3 h post-dose). *(C)* Rat mFOB score distribution (IT, 3,000 $mu$g, 3 h post-dose).],
) <fig_clinical>

=== Cross-Species Transfer and Baseline Models

#figure(
  image("plots/fig_species_transfer/species_transfer.svg", width: 100%),
  caption: [Cross-species transfer of sequence-based toxicity prediction (Random Forest, 5-fold GroupKFold by patent on the test species). Each panel compares baseline (train and test on the same species) against cross-species transfer (train on one species, test on the other). Spearman $rho$ values are shown per panel.],
) <fig_species_transfer>

#figure(
  include "data/spearman_table.typ",
  caption: [Spearman $rho$ (mean $plus.minus$ s.d. across 5 folds) for each model and endpoint. GroupKFold cross-validation grouped by patent with HELM-level deduplication. Bold indicates best model per endpoint.],
) <tbl_spearman>

== Model and Evaluation Details

=== OligoAI Training Details <secS_oligoai>

Our previous efficacy model, OligoAI @hillAccuratelyModellingRNase2025, a RiNALMo-based sequence+chemistry model for ASO inhibition, was trained on the combined ASO Atlas 2.0 in vitro inhibition and dose-response tables (#comma(R.oligoai.efficacy.n) efficacy rows from #R.oligoai.efficacy.n_tables patent tables). Chemistry was restricted to DNA/MOE/cET backbones with PO/PS linkages (covering 99.8% of parsable rows); human cell lines were retained and target transcripts resolved to GRCh38 via Ensembl release 110 with a $plus.minus$50 nt flanking context. Rows were deduplicated at the (HELM, cell line, target RNA, dosage) level with mean inhibition. Potency was evaluated as rank correlation of log IC#sub[50] from 4PL fits to dose-response curves (#R.oligoai.potency.n_valid of #R.oligoai.potency.n_candidates curves converged). Each fold was trained on a single NVIDIA RTX A6000 GPU (RunPod) for approximately 5 GPU-hours (\~25 GPU-hours total across 5 folds). Cross-validation procedure and enrichment results are reported in the main text.

=== Cost Sensitivity and Impact

#figure(
  image("plots/fig_cost_sensitivity/fig_cost_sensitivity.svg", width: 100%),
  caption: [Sensitivity of combined pipeline savings to per-stage cost assumptions. All seven stage costs are drawn independently from $U[0.5 times, 1.5 times]$ their nominal values (10,000 Monte Carlo samples). Dashed line marks the median.],
) <fig_cost_sensitivity>

#figure(
  image("plots/fig_animal_reduction/fig_animal_reduction.svg", width: 60%),
  caption: [Animal usage reduction under CatBoost pre-screening. Stacked bars show the number of animals required at each in vivo pipeline stage (assuming 4 animals per study). CatBoost pre-screening reduces the number of compounds entering animal studies, cutting total animal usage from the baseline.],
) <fig_animal_reduction>

== Extraction Schema Reference <secS_schema>

@tbl_schema_invitro, @tbl_schema_hepatic, and @tbl_schema_neuro list the column definitions supplied to the Stage~3 LLM extraction pass for each assay category. Each schema was provided as a dictionary mapping column names to natural-language descriptions; the LLM generated a per-table mapping function that translated heterogeneous patent table formats into these target columns, performing unit conversions as specified. The dose-response schema is identical to in vitro inhibition except that dosage is extracted in the original reported unit (nM or $mu$M) rather than normalised.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Field*], [*Type*], [*Description*]),
    table.hline(),
    [`Inhibition_pct`], [float], [Percent reduction in target RNA vs untreated control (UTC).],
    [`dosage_nm`], [float], [ASO concentration in nM.],
    [`target_RNA`], [string], [mRNA target name, without suffixes (e.g.~"HBV" not "HBV mRNA").],
    [`cell_line`], [string], [Cell line identifier (e.g.~HCT116, A431).],
    [`cell_line_species`], [string], [Species of the cell line (human, mouse, etc.).],
    [`transfection_method`], [string], [Electroporation, Gymnosis, or Lipofection.],
    [`treatment_period_hrs`], [float], [Hours between transfection and inhibition measurement.],
    [`cells_per_well`], [float], [Number of cells seeded per well.],
    table.hline(),
  ),
  caption: [Stage~3 extraction schema for in vitro inhibition and dose response. Dose response uses the same fields but extracts dosage in the reported unit (nM or $mu$M) rather than normalising to nM.],
) <tbl_schema_invitro>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Field*], [*Type*], [*Description*]),
    table.hline(),
    [`ALT`], [float], [Alanine aminotransferase, ALT (IU/L).],
    [`AST`], [float], [Aspartate aminotransferase, AST (IU/L).],
    [`ALB`], [float], [Albumin, ALB (g/dL).],
    [`BUN`], [float], [Blood urea nitrogen (mg/dL).],
    [`CREA`], [float], [Creatinine (mg/dL).],
    [`TBIL`], [float], [Total bilirubin (mg/dL). Aliases: bilirubin, BIL.],
    [`PC_ratio`], [float], [Protein/creatinine ratio (unitless).],
    [`dosage`], [string], [Dose as value + unit (e.g.~"30 mg/kg").],
    [`species`], [string], [Mouse, rat, or monkey.],
    [`species_strain`], [string], [Animal strain (e.g.~C57BL/6 mice).],
    [`num_doses`], [int], [Number of doses administered before measurement.],
    [`dosing_period_days`], [float], [Days between first administration and measurement.],
    [`measurement_source`], [string], [Plasma, urine, or other.],
    [`administration_method`], [string], [Route of administration (e.g.~subcutaneous).],
    [`target_RNA`], [string], [mRNA target name.],
    table.hline(),
  ),
  caption: [Stage~3 extraction schema for hepatorenal toxicity. Only rows with matching units are retained; non-convertible values are excluded by the extraction function.],
) <tbl_schema_hepatic>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Field*], [*Type*], [*Description*]),
    table.hline(),
    [`FOB_score`], [list[float]], [Functional observational battery score (0--7). bFOB for mouse, mFOB for rat.],
    [`dosage_ug`], [float], [Dosage in $mu$g. Converted from mg where reported ($1 "mg" = 1000 mu g$).],
    [`species`], [string], [Mouse or Rat.],
    [`species_strain`], [string], [Animal strain (e.g.~C57BL/6 mice, Sprague-Dawley rats).],
    [`latency_time_hours`], [float], [Time between administration and measurement (hours).],
    [`administration_method`], [string], [ICV (intracerebroventricular) or IT (intrathecal).],
    [`tolerability_score_type`], [string], [Behavioural (bFOB) or Body parts (mFOB).],
    table.hline(),
  ),
  caption: [Stage~3 extraction schema for *neurotoxicity*.],
) <tbl_schema_neuro>
