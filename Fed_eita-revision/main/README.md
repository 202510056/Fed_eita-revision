# Main research package

The experiments behind the principal claims of the revised manuscript: zero-day detection, the aggregation comparison across non-IID levels and on Edge-IIoT, client trust behaviour, the paired statistics and the computational and communication cost.

## Scientific objective

To test whether a one-shot, prediction-level federated ensemble of XGBoost clients can (i) detect held-out attack families with a fused uncertainty score and (ii) identify unreliable clients from their behaviour on a server-held reference split, and what this costs, compared with other aggregation rules applied to the **same** client models.

## Workflow (`src/fed_eita_revision_experiments.py`)

```
raw CSV ──► leakfree_prep.prepare(fit_rows = client-training rows)     encoding, scaling, MI top-15
        ──► split_rows: 70/30 train/test, 10% of train = server validation
        ──► dirichlet_split + inject_noise (clients 3, 4)                 per seed
        ──► 5 × XGBoost, trained once (one-shot)                          local training
        ──► federated_setup: trust from validation accuracy + anomaly rate (server side)
        ──► agg(): trust-weighted average of client probabilities        prediction level
        ──► fused score: 0.42·entropy + 0.28·energy(margins) + 0.30·Mahalanobis
        ──► metrics per seed  ──► CSV files in main/results/
```

The script runs three blocks:

| Block | What it measures | Output files |
|---|---|---|
| R1 | Zero-day detection; ransomware, backdoor and injection held out; every score under trust and FedAvg aggregation | `r1_ood_per_seed.csv`, `r1_ood_per_class_per_seed.csv`, summaries |
| R2 | Closed-set accuracy of FedAvg, uniform, drift-weighted and trust aggregation; TON-IoT at α_D ∈ {0.1, 0.3, 0.5, 1.0}, Edge-IIoT at 0.5; trust signals per client | `r2_aggregation_per_seed.csv`, `r2_trust_weights_per_seed.csv` |
| R3 | Paired statistics; measured runtime; computed payload | `r3_stats.csv`, `r3_runtime_per_seed.csv`, `r3_communication_per_seed.csv` |

`all_results.json` records the configuration and the selected features; `run_log.txt` is the console log of the published run (first line: `backend=xgboost | seeds=10 | LEAKFREE=True TRUST_ENERGY=margin`).

## Inputs and outputs

| | |
|---|---|
| Inputs | `TON_CSV`, `EDGE_CSV` (see `data/README.md`) |
| Published outputs | `main/results/` (never overwritten by the run script) |
| Tables | `main/tables/paper_tables.tex` (LaTeX), `tables.json` (plain values), `statistics_main.csv` (every paired comparison of this package) |

## Tables produced

| Response table | Label | Content |
|---|---|---|
| R1 | `tab:ood` | Zero-day AUROC, AUPR, FPR@95 per score |
| R3 | `tab:agg` | Accuracy of each aggregation rule per non-IID level and on Edge-IIoT |
| R4 | `tab:agg_full` | Precision, recall, F1, FPR, FNR at α_D = 0.5 |
| R5 | `tab:trust` | Validation accuracy, anomaly rate and trust weight per client |
| R6 | `tab:stats` | Paired statistics (its robust-aggregation rows use `supplementary/results/`) |
| R8 | `tab:cost` | Training, trust and inference time; payload |
| R9 | `tab:features` | Selected features and MI scores |

## Reproducing

```bash
python src/make_tables.py              # tables from the published results (no datasets needed)
bash main/run_main.sh                  # full re-run into reproduced/main/ (~3 min), then automatic comparison
```

The comparison requires every non-timing value to match the published files to 1e-9. Timing columns depend on the machine and are reported, not compared.

## What the results support

- Trust identifies the two label-noisy clients (anomaly rates 0.998 and 0.733 vs. 0.093–0.110; weights 0.151 and 0.172 vs. 0.225–0.227).
- Trust weighting is far more stable than sample-size weighting under non-IID data (α_D = 0.3: 98.07 ± 0.43% vs. 92.90 ± 5.57%) and significantly more accurate than FedAvg at α_D = 1.0.
- Zero-day AUROC of the fused score with trust aggregation: 89.39 ± 2.50%.

## What they do not support

- Accuracy equal to uniform averaging, not higher (98.74% vs. 98.77% at α_D = 0.5).
- The fused score is not better than entropy alone (89.39% vs. 90.31% AUROC).
- No advantage on Edge-IIoT (90.00% vs. FedAvg 90.12%, not significant).
- Inference with the complete predictor takes 26.45 ± 4.04 ms per 1,000 flows vs. 10.42 ± 1.28 ms for FedAvg (TON-IoT).

## Measured, computed and simulated quantities

| Quantity | Nature |
|---|---|
| Accuracy, AUROC, trust weights | computed from real model predictions on real data |
| Training, trust and inference times | measured wall-clock (`perf_counter`); inference = median of 5 timed runs on 1,000 test flows, mean ± std over seeds |
| Upload size | computed from the serialized booster size; plus a hypothetical upper-bound allowance for trust-related values (trust is computed on the server, so the protocol itself requires no trust messages) |
| Federated clients | simulated on one machine |

## Known limitations

See the Limitations section of the top-level README. Specific to this package: the drift-weighted rule is labelled "FedProx-inspired" in `r3_stats.csv` for historical reasons; it is not FedProx. The columns `Energy (prob., as implemented)` and `Fed-EITA fused (as implemented)` in the R1 files reproduce the probability-based energy of the original submission and are kept only to document that correction; no table or statistic uses them.
