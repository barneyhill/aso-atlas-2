# Frozen/live table-status reconciliation

Three measurement tables in the frozen 311-table audit carry `verified` in the
production map but `review` in later verifier run files. This evidence bundle
resolves whether their frozen outcomes remain supported by the patent sources.

| Patent and XML table | Printed table | Measurements | Disposition |
|---|---|---:|---|
| US20210038631A1:1 (IRF4) | Table 1 | 69 | Retain; source values and sequences agree |
| US20240026353A1:16 (SCN2A) | Table 5 | 78 | Retain; source values and sequences agree |
| US20240026353A1:18 (SCN2A) | Table 7 | 78 | Retain; source values and sequences agree |

Source XML table numbers and printed captions differ in the SCN2A patent. The
comparison retains dagger-marked compound IDs and excludes continuation rows
containing only additional target sites. Negative inhibition values remain valid:
inhibition is `100 - percent control`, without clipping.

All 225 source measurements agree exactly with the frozen Gold ledger. All 225
live sequences and outcomes agree with the independently read source cells.
Dose, cell line and target identity agree as well. No frozen outcome, audit
denominator or release measurement was changed.

The live files contain extraction-run messages and July 26 extraction timestamps.
The verifier's `extract` and `rerun-with-comment` handlers unconditionally write
`status: review` and remove the signed-off CSV. Their only listed problem here is
the informational single-dose/export label. No status-transition log survives:
rerunning explains how this state can arise, but the exact historical action
cannot be asserted. The frozen map's label does not itself record a sign-off; the sign-off for these tables is recorded in the completed review below.

The `.xml` files freeze the actual table and relevant preceding source paragraphs.
The `.json` files freeze live rows and provenance metadata without executing their
extraction scripts. `manifest.json` records hashes of the original files and each
snapshot. Source and live paths are provenance only; the check needs no external
workspace. Run `just audit-source-status` to regenerate
`data/results/source_status_review.json`; it is also part of `just audit`.

This original reconciliation resolved retention in the outcome audit. The archived
run snapshots retain their original review states and blank chemistry/treatment
fields. The subsequent source review below completes these fields separately.

The remaining 12 live review runs are IRF4 XML Tables 111–114, 119–122,
125–126 and 129–130. None occurs in the frozen Gold ledger. Their source captions
state reduction of **cell proliferation** in KMS11, H929, U2932, TMD8, Karpas299
or SupM2 cells. These are not measurements of target-RNA knockdown, and are outside
the efficacy endpoint definition used by this audit. Their table XML is frozen
here, and the check verifies their absence from the audit and the proliferation
caption. Their current live classification as `inhibition` should not silently
bring them into a later target-knockdown audit.

## Completed chemistry and context review (2026-10-11)

The outstanding tables were cross-referenced directly against the source. Codex,
an AI agent, drafted the chemistry and treatment fields from the actual source
cells and explicit preceding patent paragraphs, and Barney Hill then verified
them manually (sign-off recorded 2026-11-07). The original evidence snapshots
above remain intact.

| XML table | Source paragraphs | Sugar motif | Backbone motif | Cytosines | Treatment |
|---|---|---|---|---|---|
| IRF4:1 | 0401–0403 | `eeeeeddddddddddeeeee` | `sssssssssssssssssss` | All 5-methyl | Approximately 24 h |
| SCN2A:16 | 0620–0622 | `eeeeeddddddddddeeeee` | `soooossssssssssooss` | All 5-methyl | Approximately 24 h |
| SCN2A:18 | 0620–0622 | `eeeeeddddddddddeeeee` | `soooossssssssssooss` | All 5-methyl | Approximately 24 h |

All 225 sequences are 20 nt; the sugar patterns have 20 positions and the linkage
patterns have 19. The source explicitly assigns these patterns to the relevant
tables, rather than requiring inference from the Atlas output. Outcome values,
compound IDs, dose, cell line, cell density, transfection method, species and
target were cross-checked. Approximately 24 hours is stored as `24`, with the
approximation retained in each measurement note. Derived HELM encodes every
cytosine as `[5meC]` and was parsed back to check sequence, sugars and linkages.

[completed/review.json](completed/review.json) preserves the source paragraphs,
review attribution, hashes and all 12 exclusion decisions. Completed run JSONs
and CSVs are in `completed/`; [the combined CSV](completed/gold_measurements_with_chemistry.csv)
contains the 225 reviewed measurements with sequences, chemistry and derived HELM.
It is a completion subset, not the whole four-patent Gold corpus.

The three live verifier runs were set to `verified` and their signed-off CSVs
restored. The 12 proliferation runs were set to `excluded` and classified as
`skip`; four stale verified CSVs were removed from active exports. All affected
files were backed up under the verifier's
`data/_backup_20261011-114152_source_review/` directory. Existing annotations and
scripts are retained in the run files. Review metadata records Codex as `draftedBy`, Barney Hill as `reviewer`
and `humanVerified: true`.

`just audit-source-status` checks the immutable original snapshots and the
source-supported completion. Its result includes
`subsequent_chemistry_context_completion`; the original snapshot status fields
continue to describe the historical evidence. The frozen measurement ledger and
paper results were not regenerated by this completion.
