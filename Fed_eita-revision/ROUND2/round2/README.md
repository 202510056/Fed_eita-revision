# Second-round revision experiments

Experiments requested in the second review round. They **import** the published pipeline
(`src/fed_eita_revision_additions.py`) unchanged and only add evaluations; no published file was modified.

| File | Content | Manuscript |
|---|---|---|
| `results/e1_trust_weights_per_seed.csv` | trust weights, validation accuracies and anomaly rates per client, noise sweep 0–80% | Appendix Table A7 |
| `results/e1e2_noise_sweep_per_seed.csv` | accuracy, macro-F1, AUROC, AUPR, FPR@95 of Fed-EITA, uniform, FedAvg and the oracle ensemble of clients 1, 2, 5 | Appendix Table A8, Section 5.2.3 |
| `results/e3a_dedup_closedset_per_seed.csv` | closed set, TON-IoT, α_D = 0.5, published test subsets with and without training duplicates | Appendix Table A9 |
| `results/e3b_dedup_zeroday_per_seed.csv` | zero-day setting with and without training duplicates | Appendix Table A10, Section 5.2.4 |
| `results/e4_fusion_ablation_per_seed.csv` | every subset of {entropy, energy, Mahalanobis} with renormalized Eq. (11) weights | Appendix Table A11, Section 5.2.4 |
| `results/round2_stats.csv` | paired statistics, Holm within the pre-registered families (`family_size` column) | Appendix, Sections 5.2.3–5.2.4 |
| `results/dataset_structure.csv`, `results/temporal_split_counts.csv` | capture windows and source hosts per class; class counts of a 70/30 split by time | Section 5.1.2 |
| `tables/paper_tables_round2.tex`, `tables/numbers_round2.json` | every revised table row (sample standard deviations) and every number quoted in the revised text | all revised tables |
| `console_round2.txt`, `results/run_log_round2.txt`, `checksums_round2.sha256` | complete console output of the run, its log and checksums | |
| `first_attempt/INTEGRITY_FAILED.txt` | log of the first execution, stopped by integrity check I1 (see Amendment 1 of `docs/round2_preregistration.md`); no result of it was used | |

Plan: `docs/round2_preregistration.md` (including Amendment 1). Scripts: `src/fed_eita_round2_experiments.py`,
`src/make_tables_round2.py`, `src/dataset_structure_check.py`.

```bash
TON_CSV=data/TON_IoT_Train_Test_Network.csv OUT_DIR=round2/results python src/fed_eita_round2_experiments.py   # ~16 min
TON_CSV=data/TON_IoT_Train_Test_Network.csv python src/dataset_structure_check.py
python src/make_tables_round2.py                                                                                # tables, seconds
```

The run stops before writing any result unless it reproduces the published Fed-EITA rows of
`supplementary/results/add_A_robust_per_seed.csv` (identical accuracy; AUROC and FPR@95 within 1e-3 points)
and the published closed-set accuracies and trust weights of `main/results/` (I1–I4). The published run used
Python 3.11; this run used Google Colab with Python 3.13 and the pinned library versions, and reproduced
accuracy exactly and AUROC/FPR@95 to within 2.4e-4 percentage points (floating-point round-off).
Tables of the second revision report **sample** standard deviations (ddof = 1); the published generator
`src/make_tables.py` (population standard deviations) is left unchanged for traceability.
