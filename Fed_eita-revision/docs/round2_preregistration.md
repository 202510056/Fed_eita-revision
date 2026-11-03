# Second-round experiments: pre-registration

Written and committed **before** `src/fed_eita_round2_experiments.py` was run. It fixes what is
measured, how, and how the results will be reported, so that no choice depends on the outcome.

## Fixed constraints

- The method is unchanged: preprocessing, partitions, label noise, XGBoost settings, trust
  (Eqs. 12–16), scores and fusion weights (Eqs. 8–11) are imported from the published
  `src/fed_eita_revision_additions.py`. No file in `src/`, `main/`, `supplementary/` or
  `verification/` is modified.
- Same ten seeds (42, 7, 13, 99, 2024, 1, 3, 5, 11, 17), same evaluation subsets.
- No hyper-parameter, weight or threshold is selected on any of these results.
- Every experiment below is reported in full, whatever its outcome.

## Experiments

| ID | Request | Setting | Compared rules |
|---|---|---|---|
| E1 | Reviewer 1, point 1 | Zero-day setting (three families withheld, α_D = 0.5); symmetric noise on clients 3 and 4 at r ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80}, plus the published moderate setting (0.15/0.12) | Fed-EITA, uniform, FedAvg, exclusion (E2) |
| E2 | Reviewer 1, point 8 | As E1; ensemble of clients 1, 2, 5 only, Mahalanobis statistics from these clients only (oracle reference) | — |
| E3a | Reviewer 1, point 7; Reviewer 2, comment 3 | Closed set, TON-IoT, α_D = 0.5, moderate noise; published evaluation subsets restricted to test flows with no exact duplicate among the client-training flows on the 15 selected features | Fed-EITA, FedAvg, uniform, drift-weighted, exclusion |
| E3b | as E3a | Zero-day setting, moderate noise; (i) ID duplicates removed, (ii) ID and held-out duplicates removed | Fed-EITA fusion |
| E4 | Reviewer 2, comment 2 | Zero-day setting, moderate noise, trust-weighted aggregation; every non-empty subset of {entropy, energy, Mahalanobis}, weights of Eq. (11) renormalized to the retained terms; MSP as reference | each subset vs the full fusion |

## Outcomes

- E1: per noise level, the number of seeds in which clients 3 and 4 receive the two lowest
  trust weights; minimum and maximum trust weight; accuracy, macro-F1, AUROC, FPR@95.
- E2–E4: accuracy, macro-F1 (closed set) and AUROC, AUPR, FPR@95 (zero-day).
- Mean and **sample** standard deviation (ddof = 1) over the ten seeds.

## Statistics

Paired over seeds: mean difference, 95% paired-bootstrap CI (10^4 resamples), Cohen's d_z,
paired t-test and exact Wilcoxon signed-rank test, Holm correction within each family:

- E1, one family per noise level: Fed-EITA vs uniform, FedAvg and exclusion, on accuracy and AUROC (6 tests).
- E3a: Fed-EITA vs FedAvg, uniform, drift-weighted and exclusion on accuracy, duplicates removed (4 tests).
- E4: each subset and MSP vs the full fusion, on AUROC and FPR@95 (14 tests).

A difference is called significant only when its Holm-adjusted p-value is below 0.05; the test is named.

## Integrity checks (the run stops if I1–I4 fail)

- I1: the moderate and 0.40/0.40 Fed-EITA rows of E1 reproduce `supplementary/results/add_A_robust_per_seed.csv` exactly.
- I2: E3a reproduces `main/results/r2_aggregation_per_seed.csv` (TON-IoT, α_D = 0.5) and `main/results/r2_trust_weights_per_seed.csv` exactly.
- I3: the re-derived partitions reproduce the FedAvg weights of the published `setup()`.
- I4: the recomputed full fusion equals the published fused score.
- I5 (warning only): duplicate shares compared with the manuscript (29.9% and 32.3%).

## Amendment 1 (before any experimental result was used)

The first execution (Google Colab, Python 3.13, pinned library versions) stopped at integrity
check I1: the reproduced Fed-EITA rows of the published moderate-noise setting had identical
accuracies (predictions) and trust weights equal to the printed precision, but AUROC/FPR@95 values that differed from the published files by
at most 2.4 × 10⁻⁴ percentage points. This is floating-point round-off of the platform
(Python version, BLAS, CPU), not a change of method or data. No experimental output of that
execution was written or used.

The reproduction checks I1 and I2 are therefore stated with explicit tolerances: accuracy must
be identical (tolerance 10⁻⁶ points, below the 0.005-point effect of a single changed
prediction), AUROC and FPR@95 within 10⁻³ percentage points, trust weights within 10⁻⁶. The
observed maximum deviations are written to the run log. I3 and I4 remain exact (10⁻⁹), since
they compare quantities computed in the same run. Nothing else changes.
