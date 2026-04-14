# Hypothesis Registry

Central tracking of all hypotheses tested against the ASO hepatotoxicity dataset.

| ID | Hypothesis | Status | Result | Key Stat | Direction | Date |
|----|------------|--------|--------|----------|-----------|------|
| 001 | Wing self-complementarity promotes hepatotoxicity via homodimerization | Complete | Not supported | ρ=-0.25, p=2.8e-13 | Opposite (↓ALT) | 2025-01-04 |
| 001b | Wing homodimerization (NUPACK thermodynamic) | Complete | Confirms 001 | ρ=+0.32, p=5.5e-19 | Stable dimers ↓ALT | 2026-01-08 |
| 002 | Sequence motifs (3-mers) drive hepatotoxicity | Complete | Supported | 34/64 sig (FDR<0.05) | Mixed | 2025-01-04 |
| 003 | RDKit molecular descriptors predict hepatotoxicity (ALT > 100) | Complete | Partial | Acc=0.61, F1=0.60 | Mixed | 2026-01-08 |
| 004 | IC50 potency correlates with ALT toxicity | Complete | Supported | ρ=-0.16, p<0.001 | Higher potency → higher tox | 2026-01-08 |
| 005 | Kinetic model (Pedersen 2014) predicts IC50 from binding affinity | Complete | Not supported | ρ≈0, p>0.5 | No correlation | 2026-01-15 |
| 006 | mRNA half-life/expression affects knockdown efficacy | Complete | Partial | ρ=+0.21, p=0.001 | Longer half-life → higher knockdown | 2026-01-21 |
| 008 | 5' positional bias in ASO efficacy (co-transcriptional accessibility) | Complete | Opposite | ρ=+0.088, p=2.6e-233 | 3' ASOs more effective | 2026-02-13 |
| 009 | OligoAI-tox predictions on N1C known-toxicity control ASOs | Complete | Inconclusive | Mouse 6/10, Rat 7/10 (5/5 sens) | Rat model catches all positives | 2026-04-10 |
