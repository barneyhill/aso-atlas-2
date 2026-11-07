# ASO Atlas 2.0 — benchmark cross-validation folds

These are frozen endpoint-specific 5-fold assignments for reproducible OligoGym and
OligoAI evaluation. They group observations by source patent to reduce train-test
leakage.

Each benchmark endpoint was constructed independently after its own filtering and
aggregation, then patent-level `GroupKFold` was applied with HELM-level deduplication.
Consequently each endpoint has its own patent-to-fold assignment. The assignments
below are validated against the frozen test-group membership rather than inferred from
the shared grouping rule.

The six OligoGym endpoint CSVs contain:
- `group` — patent id (the CV group)
- `model_input_helm` — OligoGym adapter representation used by the benchmark
- `fold`  — test fold in {1..5}; for fold k, test = rows with fold==k, train = the rest

`model_input_helm` is deliberately not described as the raw release HELM: OligoGym's
adapter normalises that representation before modelling. Use the corresponding
adapter in `analyses/utils/oligogym_adapter.py`, or join on its `x` output.

`oligoai_folds.csv.gz` separately maps each full `helm_annotation` to its frozen
OligoAI test fold. OligoAI and OligoGym must not be assumed to share fold numbers.

| OligoGym endpoint | Unique model inputs | Patents | Per-fold sizes |
|---|--:|--:|---|
| in_vitro_inhibition | 134388 | 204 | 26842, 26785, 26968, 26889, 26904 |
| potency | 7892 | 128 | 1579, 1579, 1578, 1578, 1578 |
| mouse_hepatic | 1778 | 69 | 371, 322, 290, 374, 421 |
| rat_hepatic | 488 | 36 | 94, 86, 84, 96, 128 |
| mouse_neuro | 2696 | 23 | 538, 541, 539, 540, 538 |
| rat_neuro | 1774 | 17 | 358, 343, 343, 373, 357 |

OligoAI: 117,780 unique HELMs; test rows per fold: 36003, 36007, 36006, 36008, 36002.
