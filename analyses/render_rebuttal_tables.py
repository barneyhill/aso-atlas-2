"""Render data-backed Markdown tables into bounded rebuttal blocks."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_RESULTS = ROOT / "data/results/threshold_sweep.json"
CONDITIONAL_RESULTS = ROOT / "data/results/s1_conditional_pipeline.json"
AC_UNCERTAINTY_RESULTS = ROOT / "data/results/ac_uncertainty.json"
RELEASE_MANIFEST = ROOT / "aso-atlas-2-release/release_manifest.json"
CANONICAL_AUDIT_RESULTS = ROOT / "data/results/canonical_source_audit.json"
LINEAGE_RECALL_RESULTS = ROOT / "data/results/lineage_rescued_recall.json"
NORMALIZATION_RESULTS = ROOT / "data/results/normalization_assessment.json"
ERROR_ADJUDICATION_RESULTS = ROOT / "data/results/canonical_error_adjudication.json"
DOSE_SCALE_RESULTS = ROOT / "data/results/dose_scale_assessment.json"
RESPONSE = ROOT / "typst/response.md"
THRESHOLD_BLOCK = "threshold-sensitivity"
CONDITIONAL_BLOCK = "conditional-enrichment"
MODEL_CHOICE_BLOCK = "model-choice-sensitivity"
RELEASE_RECONCILIATION_BLOCK = "release-reconciliation"
RELEASE_SUMMARY_BLOCK = "release-summary"
VALIDATION_SUMMARY_BLOCK = "validation-summary"
VALIDATION_AUDIT_BLOCK = "validation-audit"


def _sequence(points, key, formatter):
    values = []
    for point in points:
        value = point.get(key)
        values.append("—" if value is None else formatter(value))
    return " / ".join(values)


def render_threshold_sensitivity(data: dict) -> str:
    """Render the AC-requested gate, pass-rate, EF and savings table."""
    lines = [
        ">",
        "> | Stage and gates (in order) | Baseline pass rate (%) | Selected pass rate (%) | EF | Projected savings (%) |",
        "> | --- | ---: | ---: | ---: | ---: |",
    ]
    omitted = []
    for stage in data["tornado"]:
        points = stage["points"]
        if all(point.get("model_selected_pass_rate") is None for point in points):
            omitted.append(stage["display_name"])
            continue
        gates = " / ".join(point["threshold_display"] for point in points)
        baseline = _sequence(points, "baseline_pass_rate", lambda x: f"{100 * x:.1f}")
        selected = _sequence(
            points, "model_selected_pass_rate", lambda x: f"{100 * x:.1f}",
        )
        enrichment = _sequence(points, "enrichment_factor", lambda x: f"{x:.2f}")
        savings = _sequence(points, "expected_savings_pct", lambda x: f"{x:.1f}")
        lines.append(
            f"> | {stage['display_name']} ({gates}) | {baseline} | {selected} | "
            f"{enrichment} | {savings} |"
        )
    if omitted:
        names = ", ".join(omitted)
        lines.extend([
            ">",
            f"> {names} is omitted because no model is deployed there, so EF is undefined; "
            "its pass rate and projected savings are unchanged across the swept gates.",
        ])
    lines.append(">")
    return "\n".join(lines)


def render_conditional_enrichment(data: dict) -> str:
    """Render marginal-versus-conditional enrichment for all corrected links."""
    lines = [
        ">",
        "> | Linked transition | Linked n | Baseline pass: marginal → conditional (%) | Model-selected pass: marginal → gate-matched (%) | Gate-matched n | Conditional EF / marginal EF |",
        "> | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    pairs = sorted(data["pairs"].values(), key=lambda pair: pair["idx"][0])
    for pair in pairs:
        baseline = (
            f"{100 * pair['q_rat_marginal_in_shared']:.1f} → "
            f"{100 * pair['q_rat_given_mouse_pass']:.1f}"
        )
        model = (
            f"{100 * pair['model_precision_marginal']:.1f} → "
            f"{100 * pair['model_precision_gate_matched']:.1f}"
        )
        lines.append(
            f"> | {pair['display_name']} | {pair['n_shared']:,} | {baseline} | {model} | "
            f"{pair['n_gate_matched']:,} | "
            f"{pair['conditional_to_marginal_ef_ratio']:.2f}× |"
        )
    lines.extend([
        ">",
        "> The final column is the model-precision uplift divided by the baseline-pass-rate "
        "uplift. Values below 1 mean marginal multiplication overstates downstream enrichment; "
        "values above 1 mean conditioning strengthens it.",
        ">",
    ])
    return "\n".join(lines)


def render_model_choice_sensitivity(data: dict) -> str:
    """Render the marginal expected-count sensitivity across in-vivo models."""
    corrected = data["canonical"]["corrected_expected_savings_pct"]
    lines = [
        ">",
        "> | In vivo model | Marginal expected-count savings (%) |",
        "> | --- | ---: |",
    ]
    for row in data["model_choice_sensitivity"]:
        lines.append(
            f"> | {row['model']} | {row['marginal_expected_savings_pct']:.1f} |"
        )
    lines.extend([
        ">",
        "> This is a model-choice sensitivity under the marginal pipeline. The linked-compound "
        "conditional correction was estimated only for XGBoost, the model used in the combined "
        f"cost result, so these values do not replace the corrected {corrected:.1f}% primary "
        "projection.",
        ">",
    ])
    return "\n".join(lines)


def render_release_reconciliation(data: dict) -> str:
    """Render the complete release tuple and HELM-resolved modelling subset."""
    totals = data["totals"]
    lines = [
        ">",
        "> | Release | Readouts | Unique HELM compounds | Named target genes | Contributing patents |",
        "> | --- | ---: | ---: | ---: | ---: |",
        f"> | `{data['release_tag']}` | **{totals['released_assay_readouts']:,}** | "
        f"**{totals['compounds']:,}** | "
        f"**{totals['named_target_genes']:,}** | **{totals['patents']:,}** |",
        ">",
        f"> All {totals['released_assay_readouts']:,} readouts are public. "
        f"{totals['model_eligible_assay_readouts']:,} have resolved HELM chemistry and "
        f"{totals['unresolved_helm_readouts']:,} are retained with `model_eligible=false`. "
        f"Unique-compound, linkage and benchmark counts use the resolved subset, which "
        f"covers {totals['model_eligible_patents']:,} patents and "
        f"{totals['model_eligible_named_target_genes']:,} named genes.",
        ">",
    ]
    return "\n".join(lines)


def render_release_summary(data: dict) -> str:
    """Render the AC-facing authoritative release tuple."""
    totals = data["totals"]
    return (
        f"> **1. Resource reconciliation.** The canonical `{data['release_tag']}` release contains "
        f"**{totals['released_assay_readouts']:,} readouts, "
        f"{totals['compounds']:,} unique HELM compounds, "
        f"{totals['named_target_genes']:,} named target genes and "
        f"{totals['patents']:,} contributing patents**. Of these, "
        f"**{totals['model_eligible_assay_readouts']:,} readouts have resolved HELM; "
        f"the remaining {totals['unresolved_helm_readouts']:,} are retained with "
        "`model_eligible=false`**."
    )


def _estimate_cell(result: dict, ci_key: str = "ci95") -> str:
    """Format a data-backed numerator, denominator, estimate and 95% interval."""
    numerator = result.get("k", result.get("recovered"))
    denominator = result.get("n", result.get("gold_readouts"))
    pct = result.get("pct", result.get("recall_pct"))
    if ci_key in result:
        ci = result[ci_key]
        low = ci.get("ci_lo", ci.get("low_pct"))
        high = ci.get("ci_hi", ci.get("high_pct"))
    else:
        low = result["ci_lo"]
        high = result["ci_hi"]
    return (
        f"**{numerator:,}/{denominator:,} "
        f"({pct:.2f}% [{low:.2f}, {high:.2f}])**"
    )


def _extraction_correctness(canonical: dict) -> dict:
    """Expose the canonical audit's outcome-correctness fields in table-row form."""
    return {
        "k": canonical["fully_correct_measurements"],
        "n": canonical["scoped_atlas_readouts"],
        "pct": canonical["fully_correct_measurement_precision_pct"],
        "ci95": canonical["precision_ci95"],
    }


def render_validation_summary(
    canonical: dict,
    lineage: dict,
    normalization: dict,
) -> str:
    """Render the AC-facing validation summary with distinct estimands."""
    extraction = _extraction_correctness(canonical)
    recovery = lineage["overall"]
    chemistry = normalization["sequence_and_chemistry"]
    sequence = chemistry["sequence"]
    sugar = chemistry["sugar_motif_where_stated"]
    linkage = chemistry["linkage_motif_where_stated"]
    table_mapping = canonical["table_mapping"]
    return "\n".join([
        "> **2. Extraction validation.** We replaced the 100-row spot check with an exhaustive audit of",
        f"> **{table_mapping['verified_measurement_tables']:,} manually verified measurement tables",
        f"> ({canonical['gold_readouts_scored']:,} readouts)**. Outcome-extraction correctness among",
        f"> Atlas outputs in the frozen canonical-table scope is {_estimate_cell(extraction)}.",
        f"> Separately, following every verified source measurement through the actual table- and",
        f"> row-deduplication lineage recovers {_estimate_cell(recovery)}. At compound level,",
        f"> normalization correctness is {_estimate_cell(sequence)} for sequence,",
        f"> {_estimate_cell(sugar)} for sugar motif where the patent states it, and",
        f"> {_estimate_cell(linkage)} for linkage motif where stated. These are distinct",
        "> denominators and estimands, so we report them separately rather than pooling them.",
        "> All intervals apply only to this four-patent audit, not the full corpus.",
    ])


def _adjudication_text(issue: dict) -> str:
    """Reassemble adjudication text split by legacy CSV commas in the JSON export."""
    fragments = [issue["adjudication"], *issue.get("null", [])]
    return ",".join(fragments)


def _recall_cell(result: dict) -> str:
    """Format a stratified recovery estimate and interval."""
    ci = result["ci95"]
    return (
        f"{result['recall_pct']:.2f}% "
        f"[{ci['low_pct']:.2f}, {ci['high_pct']:.2f}]"
    )


def render_validation_audit(
    canonical: dict,
    lineage: dict,
    normalization: dict,
    adjudication: dict,
    dose_scale: dict,
) -> str:
    """Render validation estimands, denominators and concrete audited failure modes."""
    chemistry = normalization["sequence_and_chemistry"]
    rows = [
        (
            "Outcome-extraction correctness",
            "Atlas readouts in frozen canonical-table scope",
            _extraction_correctness(canonical),
        ),
        (
            "Lineage-aware recovery / coverage",
            "All manually verified source readouts",
            lineage["overall"],
        ),
        (
            "Sequence normalization",
            "Audited unique compounds",
            chemistry["sequence"],
        ),
        (
            "Sugar-motif normalization",
            "Compounds whose patent states the sugar motif",
            chemistry["sugar_motif_where_stated"],
        ),
        (
            "Linkage-motif normalization",
            "Compounds whose patent states the linkage motif",
            chemistry["linkage_motif_where_stated"],
        ),
    ]
    lines = [
        ">",
        "> | Quantity | Denominator | Correct or recovered; estimate [95% CI] |",
        "> | --- | --- | ---: |",
    ]
    for label, denominator, result in rows:
        lines.append(f"> | {label} | {denominator} | {_estimate_cell(result)} |")

    issues_by_id = {issue["issue_id"]: issue for issue in adjudication["issues"]}
    example_ids = ("AGT_INH_T17", "AGT_HEP_T41")
    missing = set(example_ids) - issues_by_id.keys()
    if missing:
        raise ValueError(f"missing validation examples: {sorted(missing)}")
    lines.extend([
        ">",
        "> The first row measures correctness among scoped Atlas outputs; the second measures",
        "> recovery among all verified source readouts after following actual deduplication",
        "> lineage. Normalization is compound-level and is scored only where the relevant source",
        "> feature is available. A single pooled “validation accuracy” would conflate these",
        "> denominators.",
        ">",
        "> Stratified recovery from the same lineage-aware audit:",
        ">",
        "> | Stratum | Recall [95% CI] |",
        "> | --- | ---: |",
    ])
    strata = [
        ("Endpoint: inhibition", lineage["by_endpoint"]["inhibition"]),
        ("Endpoint: hepatorenal", lineage["by_endpoint"]["hepatotox"]),
        ("Endpoint: FOB", lineage["by_endpoint"]["neurotox"]),
        ("Table format: dose series", lineage["by_table_format"]["dose-series"]),
        ("Table format: single dose", lineage["by_table_format"]["single-dose"]),
        ("Layout: multi-column", lineage["by_table_layout"]["multi-column melt"]),
        (
            "Layout: single measurement column",
            lineage["by_table_layout"]["single measurement column"],
        ),
        ("Chemistry: 5-10-5 MOE", lineage["by_chemistry_class"]["5-10-5 MOE gapmer"]),
        (
            "Chemistry: other-wing MOE",
            lineage["by_chemistry_class"]["MOE gapmer (other wing)"],
        ),
        ("Chemistry: cEt", lineage["by_chemistry_class"]["cEt gapmer"]),
        ("Chemistry: mixed MOE/cEt", lineage["by_chemistry_class"]["mixed MOE/cEt"]),
    ]
    for label, result in strata:
        lines.append(f"> | {label} | {_recall_cell(result)} |")

    orientation = normalization["percent_control_orientation"]
    dose_ci = dose_scale["thousand_fold_error_rate_ci95"]
    orientation_ci = orientation["unflipped_error_rate_ci95"]
    losses = lineage["losses_by_reason"]
    lines.extend([
        ">",
        "> | Unit-normalization check | Errors / informative pairs; rate [95% CI] |",
        "> | --- | ---: |",
        f"> | 1,000-fold dose-scale error | {dose_scale['thousand_fold_errors']:,}/"
        f"{dose_scale['both_state_dose']:,}; {dose_scale['thousand_fold_error_rate_pct']:.0f}% "
        f"[{dose_ci['low_pct']:.0f}, {dose_ci['high_pct']:.2f}] |",
        f"> | Unflipped `% control` value | {orientation['confirmed_unflipped_errors']:,}/"
        f"{orientation['orientation_informative_pairs']:,}; "
        f"{orientation['unflipped_error_rate_pct']:.0f}% "
        f"[{orientation_ci['low_pct']:.0f}, {orientation_ci['high_pct']:.2f}] |",
        ">",
        f"> The {sum(losses.values()):,} unrecovered readouts comprise "
        f"{losses['no_raw_collation_lineage']:,} with no raw collation lineage, "
        f"{losses['raw_row_found_but_content_differs']:,} whose raw-row content differs, "
        f"{losses['eligible_group_not_released']:,} in eligible groups absent from the release, "
        f"{losses['filtered_before_deduplication']:,} filtered before deduplication, and "
        f"{losses['content_not_preserved_in_winner']:,} overwritten without preserving the outcome.",
        ">",
        "> Consequential audited failure examples:",
        ">",
    ])
    for issue_id in example_ids:
        issue = issues_by_id[issue_id]
        lines.append(
            f"> - **{issue['charge_to'].removeprefix('Atlas ').capitalize()} "
            f"({issue['table_refs']}):** {_adjudication_text(issue)}"
        )
    lines.append(">")
    return "\n".join(lines)


def generated_markers(name: str) -> tuple[str, str]:
    return (
        f"<!-- BEGIN GENERATED: {name} -->",
        f"<!-- END GENERATED: {name} -->",
    )


def extract_generated_block(text: str, name: str) -> str:
    start, end = generated_markers(name)
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected exactly one generated block named {name!r}")
    return text.split(start, 1)[1].split(end, 1)[0].strip("\n")


def replace_generated_block(text: str, name: str, rendered: str) -> str:
    start, end = generated_markers(name)
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected exactly one generated block named {name!r}")
    before, remainder = text.split(start, 1)
    _current, after = remainder.split(end, 1)
    return f"{before}{start}\n{rendered.rstrip()}\n{end}{after}"


def main() -> None:
    threshold_data = json.loads(THRESHOLD_RESULTS.read_text())
    conditional_data = json.loads(CONDITIONAL_RESULTS.read_text())
    ac_uncertainty_data = json.loads(AC_UNCERTAINTY_RESULTS.read_text())
    release_manifest = json.loads(RELEASE_MANIFEST.read_text())
    canonical_audit = json.loads(CANONICAL_AUDIT_RESULTS.read_text())
    lineage_recall = json.loads(LINEAGE_RECALL_RESULTS.read_text())
    normalization = json.loads(NORMALIZATION_RESULTS.read_text())
    error_adjudication = json.loads(ERROR_ADJUDICATION_RESULTS.read_text())
    dose_scale = json.loads(DOSE_SCALE_RESULTS.read_text())
    response = replace_generated_block(
        RESPONSE.read_text(),
        THRESHOLD_BLOCK,
        render_threshold_sensitivity(threshold_data),
    )
    response = replace_generated_block(
        response,
        CONDITIONAL_BLOCK,
        render_conditional_enrichment(conditional_data),
    )
    response = replace_generated_block(
        response,
        MODEL_CHOICE_BLOCK,
        render_model_choice_sensitivity(ac_uncertainty_data),
    )
    response = replace_generated_block(
        response,
        RELEASE_RECONCILIATION_BLOCK,
        render_release_reconciliation(release_manifest),
    )
    response = replace_generated_block(
        response,
        RELEASE_SUMMARY_BLOCK,
        render_release_summary(release_manifest),
    )
    response = replace_generated_block(
        response,
        VALIDATION_SUMMARY_BLOCK,
        render_validation_summary(canonical_audit, lineage_recall, normalization),
    )
    response = replace_generated_block(
        response,
        VALIDATION_AUDIT_BLOCK,
        render_validation_audit(
            canonical_audit,
            lineage_recall,
            normalization,
            error_adjudication,
            dose_scale,
        ),
    )
    RESPONSE.write_text(response)
    print(f"Updated generated rebuttal tables in {RESPONSE}")


if __name__ == "__main__":
    main()
