# Methodology as implemented

This document describes what the code computes. Equation numbers refer to the revised manuscript. Function names refer to `src/fed_eita_revision_experiments.py` unless stated otherwise.

## 1. Protocol: one-shot, prediction-level federated ensemble

- **Clients.** K = 5. Client k trains one XGBoost model (`xgb`, multi-class soft-probability objective, depth 6, learning rate 0.1, subsample and column-sample 0.8) on its own partition, once.
- **Exchanged information.** Each client uploads its serialized booster once (`model_bytes`). For the Mahalanobis term, a deployment would additionally send per-class counts, sums and second moments (7.6 KB per client on TON-IoT); the simulation computes the same estimate centrally, and `verify_sufficient_statistics.py` shows the two are identical.
- **Server.** Holds a validation split (10% of the training data) used only to compute trust and to fit score normalizers.
- **Deployed predictor.** P(x) = Σ_k w_k p_k(x) (`agg`), a weighted average of client class-probability vectors. No parameters are averaged, and there are no training rounds.

## 2. Uncertainty scores (Layer 3)

All scores are computed on the aggregated prediction (S_global, Eq. 17), and higher values indicate a more likely unseen attack.

| Score | Definition | Function |
|---|---|---|
| Entropy a(x) | −Σ_c P_c log₂ P_c / log₂ C, in [0, 1] | `H_norm` |
| Energy E(x), Eq. (9) | −T log Σ_c exp(g_c(x)/T), with T = 1 and g = trust-weighted raw XGBoost margins | `E_margin_raw`, `model_margins` |
| Mahalanobis M(x), Eq. (10) | min_c (x − μ_c)ᵀ Σ_c⁻¹ (x − μ_c); class statistics from a 1-in-30 subsample of client training rows | `Mahal` |
| MSP (baseline) | 1 − max_c P_c | `MSP_raw` |

**Normalization.** Each score is mapped to [0, 1] using the 1st and 99th percentiles of its values on the in-distribution validation split (`MinMaxFromVal`). Test and held-out data never enter the normalization.

**Fusion, Eq. (11).** S(x) = 0.7·(0.6·ã + 0.4·Ẽ) + 0.3·M̃, i.e. β = (0.42, 0.28, 0.30), fixed a priori without tuning.

**Legacy energy.** The original submission computed "energy" from normalized probabilities, which is constant (AUROC ≈ 50%). The columns `Energy (prob., as implemented)` and `Fed-EITA fused (as implemented)` reproduce it for documentation only.

## 3. Client trust (Layer 4)

Computed by the server in `federated_setup` (main experiment) and `setup` (`fed_eita_revision_additions.py`).

- **Validation accuracy** Acc_k of client k's model on the server split.
- **Anomaly rate, Eq. (12).** For client k, z_k = 0.6·ã + 0.4·Ẽ is computed from its own predictions on the server split, with energy from raw margins. τ is the 75th percentile of the same score for the uniform ensemble. AR_k is the share of validation rows with z_k ≥ τ. The Mahalanobis term is omitted here because it does not depend on the client model.
- **Participation** min(1, 0.5 + 0.5·r/R). It is identical for all clients in this protocol.
- **Consistency** max(0, 1 − 2·std of the client's mean uncertainty over the last five 100-row validation mini-batches).
- **Trust, Eq. (13):** θ_k = 0.40·Acc_k + 0.35·(1 − AR_k) + 0.15·participation + 0.10·consistency.
- **Memory, Eq. (14):** T_k ← λ·T_k + (1 − λ)·θ_k, with λ = 0.30, T initialized at 0.5, and R = 20 trust-update iterations. The weights are w_k = T_k / Σ_j T_j.

Because AR_k is measured on the server's reference data, a client that observes attacks in its own traffic is not penalized for that. AR_k measures the uncertainty of the client's model; it is a proxy for reliability under label noise, not an integrity check.

## 4. Aggregation rules compared

All rules act on the same client models and differ only in how they combine the client probability vectors.

| Rule | Weights / operation | Where |
|---|---|---|
| FedAvg (prediction level) | w_k ∝ n_k (partition size) | `federated_setup`, `setup` |
| Uniform averaging | w_k = 1/K (Fed-EITA without trust) | both |
| Drift-weighted | w_k ∝ exp(−10·mean((p_k − p̄)²)) on validation predictions | `federated_setup` |
| Coordinate-wise median | median over clients per class and sample, renormalized | `aggregate` (additions) |
| Trimmed mean (20%) | drop the highest and lowest client per coordinate, average the rest | `aggregate` (additions) |
| Krum / Multi-Krum (f = 2) | client(s) whose flattened validation predictions are closest to the others (sum of the n − f − 2 smallest squared distances) | `setup` (additions) |
| Fed-EITA | trust weights (Section 3) | both |

Median, Trimmed mean, Krum and Multi-Krum are prediction-level analogues of the parameter-level rules (Yin et al., 2018; Blanchard et al., 2017). FedProx, SCAFFOLD, FedNova, MOON and Ditto modify gradient-based training of a shared parametric model and do not apply to independently trained boosted trees; they are not compared.

## 5. Statistics

- **Pairing.** Comparisons are paired by seed: for each seed, all methods share partition, noise, client models and test subset.
- **Per comparison:** mean difference, 95% paired-bootstrap CI (10⁴ resamples, `default_rng(0)`), Cohen's d_z, paired t-test, Wilcoxon signed-rank test and Shapiro–Wilk test of the differences.
- **Multiplicity:** Holm correction within each comparison family (one dataset and α_D level, the OOD family, or one noise level). It is applied separately to the t-test and Wilcoxon p-values.
- **Std convention:** tables report the population standard deviation over seeds (ddof = 0); summary CSVs in the result folders use ddof = 1.
