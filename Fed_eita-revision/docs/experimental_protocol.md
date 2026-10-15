# Experimental protocol

## Data

| | TON-IoT | Edge-IIoT |
|---|---|---|
| File | `Train_Test_Network` | `ML-EdgeIIoT-dataset` |
| Rows used | 461,043 (all) | 60,000 (label-stratified subset, seed 42) |
| Classes | 10 | 15 |
| Train / validation / test | 290,457 / 32,273 / 138,313 | 70/30 split, 10% of train as validation |
| Excluded fields | timestamps, source/destination IPs, DNS query, DNS flags, SSL subject/issuer, HTTP URI and user agent, `weird_addl`, binary label (`TON_DROP`) | frame time, host IPs, DNS name, MQTT fields, HTTP fields, TCP options/payload, labels (`EDGE_DROP`) |

## Preprocessing (training-only, leakage-controlled) — `src/leakfree_prep.py`

1. **Split.** The stratified 70/30 train/test split and the 10% validation split depend only on the labels (`random_state = 42`). This is a random split, not a temporal or device-disjoint split, and duplicate flows are not removed.
2. **Encoding.** The categorical vocabulary is built from client-training rows; categories unseen there map to −1.
3. **Scaling.** Min-max scaling is fitted on client-training rows.
4. **Feature selection.** The top 15 features by mutual information are chosen on a 30,000-row sample of client-training rows.
5. **Zero-day experiments.** Steps 2–4 use in-distribution training rows only. In the rotation study they are refitted for every held-out configuration.

## Federated setting

- **Clients.** K = 5, with a class-wise Dirichlet partition (α_D = 0.5 by default; the non-IID sweep uses α_D ∈ {0.1, 0.3, 0.5, 1.0}).
- **Label noise.** Symmetric noise on clients 3 and 4: 15% / 12% (moderate, the main setting) or 40% / 40% (severe). Only labels are changed, from the start of training, using the experiment seed.
- **Trees per client.** 25 (closed-set, α_D = 0.5), 15 (non-IID sweep), 20 (zero-day), 12 (Edge-IIoT).
- **Seeds.** 42, 7, 13, 99, 2024, 1, 3, 5, 11, 17. Each seed fixes the partition, the noise, the models and the evaluation subsets.

## Zero-day (held-out family) evaluation

- **Main setting.** Ransomware, backdoor and injection are removed from training and validation and appear only in the test pool (`r1_*`, `add_A_*`).
- **Rotation.** Each family is held out alone, and all three together (`add_B_*`).
- **Evaluation subsets.** 20,000 in-distribution test flows and 5,000 held-out flows per seed.
- **Metrics.** AUROC, AUPR and FPR@95 (the false-positive rate on in-distribution flows at the threshold that detects 95% of held-out flows). Accuracy is reported on the known classes.
- These are simulated zero-day attacks: the held-out families are known benchmark classes withheld from training.

## Cost measurement — `run_agg(..., measure=True)`

- **Measured:** local training time (sum over clients) and trust/uncertainty computation time, with `perf_counter`.
- **Inference (measured):** time to process 1,000 test flows.
  - FedAvg: client probabilities and size-weighted average.
  - Fed-EITA: client probabilities and raw margins, trust-weighted aggregation, and the fused score (entropy, energy, Mahalanobis).
  - One-off fitting is excluded, and a 10-flow warm-up is run first. Each value is the median of 5 timed runs per seed; tables report mean ± std over seeds.
- **Payload (computed, not measured on a network):** serialized booster size per client. Fed-EITA adds three 8-byte trust scalars per client per trust iteration as an upper bound.

## Hardware and software of the published runs

Intel Xeon Platinum 8581C @ 2.30 GHz (4 vCPUs), 7.8 GiB RAM; Python 3.11.14, XGBoost 3.2.0, scikit-learn 1.8.0, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1.
