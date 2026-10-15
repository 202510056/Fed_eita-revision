# Reviewer traceability

Maps each major reviewer concern to the code, the output files, the response table and the conclusion the evidence supports. The authoritative source is the response letter (Response to Reviewers, revised submission). Comment numbers follow the letter: Reviewer 1 = 1.x, Reviewer 2 = 2.x, Reviewer 3 = 3.x. Table numbers are the response-letter tables R1–R13. Their LaTeX labels are given in `main/README.md` and `supplementary/README.md`.

Paths: `M/` = `main/results/`, `S/` = `supplementary/results/`, `E` = `src/fed_eita_revision_experiments.py`, `A` = `src/fed_eita_revision_additions.py`.

## Federated mechanism and baselines

| Concern | Implementation | Output | Table | Conclusion supported | Remaining limitation |
|---|---|---|---|---|---|
| 1.1 / 2.1 What is exchanged; valid FL algorithm for XGBoost | `E: federated_setup`, `agg`, `model_bytes`; `A: setup` | `M/r3_communication_per_seed.csv` | R8 | One-shot, prediction-level federated ensemble: each client uploads one booster; the server weights client predictions. The 20 "rounds" are trust-update iterations | No parameter-level federated training; the reviewer may see an ensemble as a narrower form of FL |
| 1.2 / 2.2 Baselines incompatible with XGBoost | `E: federated_setup` (FedAvg, uniform, drift); `A: aggregate`, `setup` (Median, Trimmed mean, Krum, Multi-Krum) | `M/r2_aggregation_per_seed.csv`, `S/add_A_robust_per_seed.csv` | R3, R4, R10 | Only rules exactly defined for tree ensembles are compared, each with an equation, on the same client models | No shared-neural-architecture baseline and no federated gradient-boosting baseline |
| 2.9 Nearly identical baseline results | all per-seed CSVs; `run_log.txt` | `M/`, `S/` | R3–R7 | Earlier baselines were post-hoc transforms of the same boosters; they were removed, and every value is now reported per seed | — |

## Trust and threat model

| Concern | Implementation | Output | Table | Conclusion supported | Remaining limitation |
|---|---|---|---|---|---|
| 1.3 / 2.7 Trust may penalize clients that observe attacks | `E: federated_setup` (anomaly rate on the server split); `src/fed_eita_trust_ablation.py` | `M/r2_trust_weights_per_seed.csv`, `S/ablation_per_seed.csv` | R5, R13 | AR is measured on the server's reference data, so local attack exposure does not enter it. Trust identifies the noisy clients (AR 0.998 / 0.733 vs. 0.093–0.110). Identification does not raise accuracy over uniform averaging, the anomaly rate alone over-penalizes, and the memory has no effect | No experiment with an honest client under a real attack outbreak; AR is a reliability proxy under label noise, not an integrity check |
| 1.4 / 2.8 Threat model vs. experiments; robust aggregation | `E: inject_noise`; `A: main` (experiment A) | `S/add_A_robust_per_seed.csv`, `S/add_A_stats.csv` | R10 | 2 of 5 clients with symmetric label noise (15%/12%, severe 40%/40%). Fed-EITA matches Median and Trimmed mean and is more accurate than Krum and Multi-Krum under severe noise (not significant). Bulyan is not applicable with 5 clients | Targeted, adaptive and colluding attacks not evaluated |

## Zero-day protocol and scores

| Concern | Implementation | Output | Table | Conclusion supported | Remaining limitation |
|---|---|---|---|---|---|
| 2.5 OOD protocol without contamination | `src/leakfree_prep.py: prepare`; `E` R1 block; `A: matrices` | `M/r1_ood_per_seed.csv`, `S/add_B_rotation_per_seed.csv` | R1, R2, R11, R12 | Held-out families never enter fitting, normalization or calibration; preprocessing is refitted per rotation; results are robust to the correction (R12) | Simulated zero-day attacks (known benchmark families withheld) |
| 2.6 Score normalization and fusion | `E: H_norm`, `E_margin_raw`, `Mahal`, `MinMaxFromVal`; `src/verify_sufficient_statistics.py` | `M/r1_ood_per_seed.csv`, `S/verify_sufficient_statistics.txt` | R1 | Normalizers fitted on in-distribution validation scores; β = (0.42, 0.28, 0.30) fixed a priori; energy from raw margins, T = 1; federated Mahalanobis estimate exact (deviation ≤ 1.2 × 10⁻¹⁴) | The fusion is below entropy alone (−0.91 AUROC) |
| 2.10 Complete OOD results | `E` R1 block | `M/r1_ood_per_seed.csv`, `M/r1_ood_per_class_per_seed.csv` | R1, R2, R6 | AUROC, AUPR, FPR@95 per score, per family, with paired statistics | — |
| 3.5 / 3.6 Inconsistent figure values; "Fed-EITA-v2" | `src/make_tables.py` | `main/tables/`, `supplementary/tables/` | all | Every table is generated from one run and one model definition | — |

## Data, leakage and replication

| Concern | Implementation | Output | Table | Conclusion supported | Remaining limitation |
|---|---|---|---|---|---|
| 2.12 Preprocessing and partition leakage | `src/leakfree_prep.py` (`split_rows`, `encode_scale`, `prepare`, `TON_DROP`) | `M/all_results.json` (selected features) | R9, R12 | Encoding, scaling and feature selection fitted on client-training rows only; identifiers excluded; split sizes and seeds reported | Random stratified split, not temporal or device-disjoint; duplicate flows not removed |
| 1.5 / 2.11 / 3.8 Edge-IIoT | `src/leakfree_prep.py: read_edge`; `E: run_agg("Edge-IIoT", ...)` | `M/r2_aggregation_per_seed.csv` | R3, R4, R6 | Independent replication with its own preprocessing and a full baseline comparison; no advantage (Fed-EITA 90.00% vs. FedAvg 90.12%, not significant) | 60,000-flow stratified subset |

## Statistics

| Concern | Implementation | Output | Table | Conclusion supported | Remaining limitation |
|---|---|---|---|---|---|
| 1.6 / 2.13 Statistical analysis | `src/make_tables.py: paired`, `add_holm` | `main/tables/statistics_main.csv`, `supplementary/tables/statistics_robust.csv`, `supplementary/tables/statistics_ablation.csv` | R6, R7 | Ten seeds; paired bootstrap CI, d_z, t-test and Wilcoxon, Holm within families. Significant: Fed-EITA vs. FedAvg at α_D = 1.0 (t-test) and 0.3 (Wilcoxon); uniform above Fed-EITA at α_D = 0.5 (−0.04 points) | Ten seeds give limited power for small differences |

## Cost and figures

| Concern | Implementation | Output | Table | Conclusion supported | Remaining limitation |
|---|---|---|---|---|---|
| 2.3 / 3.4 Convergence and communication | `E: model_bytes`, `run_agg(measure=True)` | `M/r3_communication_per_seed.csv` | R8 | No convergence claim (one-shot). Payload computed from serialized boosters: 0.66 MB per client on TON-IoT; Fed-EITA adds about 1.2% (trust scalars upper bound + sufficient statistics) | Computed, not measured on a network |
| 3.2 / 3.3 Identical or zero inference cost | `E: run_agg` (`infer_fedavg`, `infer_fedeita`, `timed`) | `M/r3_runtime_per_seed.csv` | R8 | Inference timed separately per method; the Fed-EITA timing covers the complete predictor: 26.45 ± 4.04 vs. 10.42 ± 1.28 ms per 1,000 flows (TON-IoT), 23.19 ± 1.93 vs. 8.18 ± 1.34 ms (Edge-IIoT) | Hardware-dependent; single machine |
| 3.9 "Remove noisy clients" ablation | `A: setup`, `src/fed_eita_trust_ablation.py` | `S/ablation_per_seed.csv` | R1, R3, R13 | The old row reflected lost class coverage; it is replaced by component ablations (scores in R1, aggregation in R3, trust signals in R13) | — |
| 3.7 Overclaiming in abstract | — | all | R1, R3, R6, R10 | Results reported as comparable, with effect sizes, including where Trimmed mean, uniform averaging and entropy are better | — |

## Reproducibility and reasoning layer

| Concern | Implementation | Output | Table | Conclusion supported | Remaining limitation |
|---|---|---|---|---|---|
| 2.15 / 3.10 Code and fitted parameters | whole repository; `M/all_results.json` | `verification/` | Table 5 (manuscript) | Public code, per-seed results, logs and table generator; parameters β, ω, λ = 0.30, τ = 75th percentile and T = 1 are reported and set in the code | Datasets must be obtained from their owners |
| 2.14 Reasoning layer | not part of the experiments | — | — | Presented as a deterministic rule-based interface; no usability claim | No analyst evaluation |
