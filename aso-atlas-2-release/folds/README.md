# ASO Atlas 2.0 — paper cross-validation folds

These are the exact endpoint-specific 5-fold assignments used for the submitted
paper's benchmark results. They are the reproduction partitions requested by the
reviewers; they are distinct from the release's optional global canonical split.

The paper constructed each benchmark endpoint independently after its own filtering
and aggregation, then applied patent-level `GroupKFold` with HELM-level deduplication.
Consequently each endpoint has its own patent-to-fold assignment. The assignments
below are validated against the test-group membership frozen in the submitted run's
result file, rather than assumed from the shared grouping rule.

The six OligoGym endpoint CSVs contain:
- `group` — patent id (the CV group)
- `model_input_helm` — OligoGym adapter representation used by the benchmark
- `fold`  — test fold in {1..5}; for fold k, test = rows with fold==k, train = the rest

`model_input_helm` is deliberately not described as the raw release HELM: OligoGym's
adapter normalises that representation before modelling. Use the corresponding
adapter in `analyses/utils/oligogym_adapter.py`, or join on its `x` output.

`oligoai_folds.csv.gz` separately maps each full `helm_annotation` to its exact
OligoAI test fold. It is derived from the five frozen training CSVs used by the
submitted runs; OligoAI and OligoGym must not be assumed to share fold numbers.

| OligoGym endpoint | Unique model inputs | Patents | Per-fold sizes |
|---|--:|--:|---|
| in_vitro_inhibition | 134388 | 204 | 26842, 26785, 26968, 26889, 26904 |
| potency | 7892 | 128 | 1579, 1579, 1578, 1578, 1578 |
| mouse_hepatic | 1778 | 69 | 371, 322, 290, 374, 421 |
| rat_hepatic | 488 | 36 | 94, 86, 84, 96, 128 |
| mouse_neuro | 2696 | 23 | 538, 541, 539, 540, 538 |
| rat_neuro | 1774 | 17 | 358, 343, 343, 373, 357 |

OligoAI: 117,780 unique HELMs; test rows per fold: 36003, 36007, 36006, 36008, 36002.
