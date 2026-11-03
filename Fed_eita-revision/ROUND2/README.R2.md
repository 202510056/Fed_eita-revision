# Fed-EITA: Uncertainty- and Trust-Aware Federated Intrusion Detection for Zero-Day IoT Threats

**Nour A. Kaymih, Yazan Adnan Mohd Ali, Haythem A. Bany Salameh** — Al Ain University

Code, per-seed results and table generator for the **revised manuscript** (submitted to *Cluster Computing*). Every number in the revised manuscript and in the response to the reviewers is produced by the code in `src/` from the result files in this repository.

---

## Overview

Fed-EITA addresses two problems that federated intrusion-detection studies usually treat separately: attacks never seen during training (zero-day attacks), and clients whose local data are unreliable. It is a **one-shot, prediction-level federated ensemble**:

1. each IoT client trains one XGBoost model on its own traffic and uploads it once;
2. the server scores every client on its own held-out reference (validation) split, using the client's validation accuracy and the share of highly uncertain predictions (anomaly rate), and turns these signals into trust weights;
3. at inference, the server combines the clients' class-probability vectors with the trust weights and flags possible zero-day traffic with a fused uncertainty score (normalized entropy, margin-based energy and Mahalanobis distance).

No model parameters are averaged and no training rounds are iterated. The method is therefore **not** iterative Federated Averaging; the "20 rounds" of the trust memory are trust-update iterations on fixed client models.

```
 client k (k = 1..5)                         server
 ┌──────────────────────┐   booster (once)   ┌──────────────────────────────────────────────┐
 │ local flows → XGBoost├───────────────────►│ trust_k from validation accuracy + anomaly   │
 └──────────────────────┘                    │ rate on the server-held reference split      │
                                             │                                              │
 test flow x ───────────────────────────────►│ P(x) = Σ_k w_k p_k(x)   (prediction level)   │
                                             │ S(x) = 0.42·a(x) + 0.28·E(x) + 0.30·M(x)     │
                                             │ → class prediction + zero-day score          │
                                             └──────────────────────────────────────────────┘
```

## What the results support

The evaluation was designed to test each claim, and the claims below are the ones that survived. All numbers are means over ten seeds (TON-IoT, three attack families held out, unless stated).

| Supported finding | Evidence |
|---|---|
| Trust identifies label-noisy clients without being told which they are | Noisy clients get anomaly rates 0.998 and 0.733 (clean: 0.093–0.110) and the two lowest weights (0.151, 0.172 vs. 0.225–0.227), although their validation accuracy is indistinguishable (Table R5) |
| Trust weighting reduces the instability of sample-size weighting (FedAvg) under moderately non-IID data (α_D = 0.3–0.5); at α_D = 0.1 both rules remain unstable | α_D = 0.3: 98.07 ± 0.43% vs. FedAvg 92.90 ± 5.57% (Wilcoxon, Holm-adjusted p = 0.012); α_D = 1.0: +0.20 points (t-test, Holm-adjusted p = 0.008) (Tables R3, R6) |
| Zero-day detection with the fused score | AUROC 89.39 ± 2.50%, FPR@95 38.19% with trust aggregation (Table R1) |
| More accurate than Krum-based aggregation under label noise | Severe noise: 98.83% vs. Krum 95.99% and Multi-Krum 98.22% (not significant after Holm correction) (Table R10) |
| Federated estimation of the Mahalanobis term is exact | Aggregated per-client sufficient statistics reproduce the pooled class means and covariances to machine precision (7.6 KB per client) |

## What the results do not support

These results are reported in the manuscript and are stated here with equal prominence:

- **Trust weighting does not universally outperform uniform averaging.** Closed-set accuracy is essentially that of uniform averaging (98.74% vs. 98.77% at α_D = 0.5; the difference is small but significant after Holm correction).
- **The full fusion does not outperform entropy alone** (89.39% vs. 90.31% AUROC), and Trimmed mean attains a slightly higher zero-day AUROC (89.56%). The entropy advantage (0.91 points) is significant by the Holm-adjusted Wilcoxon test (p = 0.049) but not by the Holm-adjusted paired t-test (p = 0.063); the trimmed-mean advantage is not significant.
- **The trust memory (λ) has no measurable benefit** under the one-shot protocol (Table R13), because client models do not change between trust iterations.
- **Most of the zero-day gain of trust over uniform averaging comes from the accuracy component** (accuracy-only trust: 89.44% AUROC vs. 89.39% for full trust; Table R13).
- **Edge-IIoT shows no advantage** over FedAvg (90.00% vs. 90.12%, not significant).
- **The full predictor costs extra inference time:** 26.45 ± 4.04 ms vs. 10.42 ± 1.28 ms per 1,000 flows on TON-IoT (23.19 vs. 8.18 ms on Edge-IIoT), i.e. about 0.016 ms more per flow (Table R8).

## Repository layout

```
Fed-EITA/
├── src/                         shared code (one implementation, used by both packages)
│   ├── leakfree_prep.py               training-only preprocessing (encoding, scaling, MI feature selection)
│   ├── fed_eita_revision_experiments.py   main experiment → main/results/
│   ├── fed_eita_revision_additions.py     robust aggregation + held-out-family rotation → supplementary/results/
│   ├── fed_eita_trust_ablation.py         trust ablation (Table R13) → supplementary/results/
│   ├── verify_sufficient_statistics.py    federated Mahalanobis check → supplementary/results/
│   └── make_tables.py                     regenerates all tables (R1–R13) from the result files
├── main/                        main research package: results, tables, README, run script
├── supplementary/               supplementary package: results, legacy runs, tables, README, run script
├── docs/                        methodology, experimental protocol, reproducibility, reviewer traceability
├── verification/                integrity checks (checksums, table regeneration, re-run comparison)
└── data/                        dataset instructions (datasets are not redistributed)
```

| Package | Experiments | Manuscript / response tables |
|---|---|---|
| [`main/`](main/README.md) | zero-day detection, aggregation comparison (4 non-IID levels + Edge-IIoT), trust behaviour, statistics, cost | R1, R3, R4, R5, R6, R8, R9 |
| [`supplementary/`](supplementary/README.md) | per-family detection, per-seed values, robust aggregation, rotation, preprocessing sensitivity, trust ablation, sufficient statistics | R2, R7, R10, R11, R12, R13 |

## Datasets

- **TON-IoT** network data (`Train_Test_Network`, 461,043 flows, 10 classes).
- **Edge-IIoTset** (`ML-EdgeIIoT-dataset`), stratified 60,000-flow subset (15 classes).

The datasets are not redistributed; see [`data/README.md`](data/README.md).

## Experimental protocol (summary)

Five simulated clients with a class-wise Dirichlet partition (α_D ∈ {0.1, 0.3, 0.5, 1.0}); clients 3 and 4 carry symmetric label noise (15%/12%, severe 40%/40%). Stratified 70/30 train/test split with a 10% server-held validation split. Encoding, scaling and feature selection are fitted on client-training rows only; in zero-day experiments ransomware, backdoor and injection are excluded from all fitting and calibration. Ten seeds: 42, 7, 13, 99, 2024, 1, 3, 5, 11, 17. Comparisons are paired by seed with bootstrap CIs, Cohen's d_z, t-tests and Wilcoxon tests, Holm-corrected within each family. Details: [`docs/experimental_protocol.md`](docs/experimental_protocol.md) and [`docs/methodology.md`](docs/methodology.md).

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Tables only (no datasets needed, seconds): regenerate every table from the published results
python src/make_tables.py
python verification/integrity_checks/check_tables.py      # confirms the regenerated tables equal the committed ones

# 2. Full re-run (datasets in data/, about 10 minutes on 4 vCPUs); results go to reproduced/ and are
#    compared automatically with the published results (timing columns excepted)
bash main/run_main.sh
bash supplementary/run_supplementary.sh
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for the environment, expected run times and the verification steps.

## Main results (TON-IoT, three attack families held out, 10 seeds)

**Where these numbers come from.** None of the values below is typed by hand. They are copied from `supplementary/tables/tables.json`, which `src/make_tables.py` generates from the per-seed result files (`supplementary/results/add_A_robust_per_seed.csv`). Those files were written by `src/fed_eita_revision_additions.py`; the console output of the run is in `supplementary/results/run_log_additions.txt` and that of the main experiment in `main/results/run_log.txt`. To check them yourself:

```bash
python verification/integrity_checks/verify_checksums.py   # published result files are unchanged
python verification/integrity_checks/check_tables.py       # tables regenerate exactly from the per-seed files
bash supplementary/run_supplementary.sh                     # full re-run (datasets required); compared automatically
```

An independent re-execution of both packages reproduced every non-timing result file exactly. Supplementary Table S6 of the paper is the mean and population standard deviation of `main/results/r1_ood_per_seed.csv`; the payload figures are serialized sizes, not network measurements.

| Aggregation | Accuracy, moderate noise (%) | Zero-day AUROC, moderate (%) | Accuracy, severe noise (%) | Zero-day AUROC, severe (%) |
|---|---|---|---|---|
| FedAvg | 97.12 ± 2.87 | 88.40 ± 1.90 | 96.74 ± 3.14 | 87.55 ± 2.19 |
| Uniform averaging | 98.95 ± 0.16 | 88.39 ± 3.58 | 98.89 ± 0.18 | 87.18 ± 4.36 |
| Coordinate-wise median | 98.87 ± 0.22 | 89.04 ± 2.50 | 98.73 ± 0.32 | 87.21 ± 4.20 |
| Trimmed mean (20%) | 98.96 ± 0.16 | 89.56 ± 2.26 | 98.91 ± 0.19 | 88.78 ± 2.52 |
| Krum | 98.41 ± 0.52 | 86.87 ± 3.20 | 95.99 ± 2.67 | 83.72 ± 3.37 |
| Multi-Krum | 98.72 ± 0.43 | 88.15 ± 1.40 | 98.22 ± 0.72 | 85.80 ± 3.61 |
| **Fed-EITA (trust)** | **98.90 ± 0.17** | **89.39 ± 2.50** | **98.83 ± 0.22** | **88.35 ± 2.62** |

Mean ± population standard deviation over ten seeds. All rules aggregate the same client models; only the aggregation differs. Median, Trimmed mean, Krum and Multi-Krum are prediction-level analogues of the corresponding parameter-level rules. Complete tables: `main/tables/` and `supplementary/tables/`.

## Limitations

- Federated clients are **simulated on a single machine**; there is no real network.
- **Communication costs are computed from serialized payload sizes**, not measured on a network.
- The threat model covers **label-noisy (unreliable) clients only**; targeted poisoning, backdoors, collusion and adaptive attacks were not evaluated, and the trust score is not a defence against them.
- No formal privacy guarantee: raw traffic stays local, but uploaded boosters and the per-class statistics of the Mahalanobis term may leak information about local data; neither differential privacy nor secure aggregation is implemented.
- The train/test split is stratified random, not temporal or device-disjoint, and duplicate flows were not removed. On the 15 selected features, 29.9% of the TON-IoT test flows (32.3% of the in-distribution test flows in the zero-day setting) and 3.0% of the Edge-IIoT test flows have an exact duplicate among the client-training flows, so closed-set accuracies are likely optimistic for unseen deployments. This is a feature-level overlap: on all raw columns, about 2.8% of the TON-IoT test flows are exact copies of a training flow. Both rates come from an independent audit of the data, not from the scripts in this repository, and no re-evaluation on the non-overlapping flows was performed.
- Client partitions are disjoint except for a small cross-client duplication: a client lacking a class receives a copy of one training sample of it (class completion, `ensure_all_classes`); in the TON-IoT runs at most 12 of the 290,457 training rows are placed in two clients.
- Label noise replaces 15%/12% (or 40%/40%) of the labels by uniformly drawn labels; because a drawn label can equal the original one, the fraction of labels that actually change is lower by the factor 1 − 1/C (13.5%/10.8% on TON-IoT).
- The operating threshold τ_zd and the rule-based explanation interface (Eqs. 19–20 of the manuscript) are not part of the released experiment code; no reported metric depends on them.
- The drift-weighted rule is labelled "Drift-weighted (FedProx-inspired)" in `main/results/r3_stats.csv` and `SUMMARY.md` for historical reasons; it is not FedProx; it is equivalent to one reweighting step of the simple γ-mean of Li et al. (2022), see `verification/README.md`. The published files are left unchanged so that their checksums remain valid.
- Edge-IIoT is an independent replication on a 60,000-flow subset, not a cross-dataset transfer.

## Verification

`verification/` contains SHA-256 checksums of every published result and table file, a script that regenerates all tables and compares them with the committed versions, a script that compares a re-run with the published results, and a check that the documented code changes are comment- and docstring-only. See [`verification/README.md`](verification/README.md).

## Citation

See `CITATION.cff`. The article reference will be added on publication.

## License

Code: MIT (see `LICENSE`). The datasets are subject to their own licences and are not redistributed.
