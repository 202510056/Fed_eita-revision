# Supplementary research package

Additional analyses that test the robustness and the limits of the main findings. All experiments use the shared code in `src/` and therefore the same preprocessing, splits, seeds, client models and scores as the main package.

## Contents

| Response table | Label | Analysis | Script | Result files |
|---|---|---|---|---|
| R2 | `tab:ood_family` | Zero-day detection per held-out family | `fed_eita_revision_experiments.py` | `main/results/r1_ood_per_class_per_seed.csv` |
| R7 | `tab:per_seed` | Per-seed accuracy and AUROC at α_D = 0.5 | `fed_eita_revision_experiments.py` | `main/results/r1_ood_per_seed.csv`, `r2_aggregation_per_seed.csv` |
| R10 | `tab:robust` | Robust aggregation: Median, Trimmed mean (20%), Krum, Multi-Krum vs. Fed-EITA, moderate and severe label noise | `fed_eita_revision_additions.py` (experiment A) | `add_A_robust_per_seed.csv` |
| R11 | `tab:rotation` | Rotating held-out family: each family held out alone, preprocessing refitted per configuration | `fed_eita_revision_additions.py` (experiment B) | `add_B_rotation_per_seed.csv` |
| R12 | `tab:leakfree` | Sensitivity to the corrections: first revision runs vs. final pipeline, same five seeds | — (comparison of existing runs) | `results_legacy/`, `add_A_robust_per_seed.csv` |
| R13 | `tab:ablation` | Trust ablation: uniform, accuracy only, anomaly rate only, accuracy + anomaly rate, full trust without memory, full trust, FedAvg | `fed_eita_trust_ablation.py` | `ablation_per_seed.csv`, `ablation_stats.csv` |
| — | — | Federated estimation of the Mahalanobis statistics (exact identity check, payload) | `verify_sufficient_statistics.py` | `verify_sufficient_statistics.txt` |

Tables R2 and R7 are computed from the main experiment's outputs and are listed here because they are supplementary material in the manuscript.

Statistics: `tables/statistics_robust.csv` (Table R10 comparisons) and `tables/statistics_ablation.csv` (Table R13 comparisons), paired over ten seeds with Holm correction within each noise level.

## Trust ablation (Table R13)

All configurations share the same client models and trust signals; only the mapping from signals to weights differs. The script re-uses the pipeline of `fed_eita_revision_additions.py` by import and stops with an error unless its full-trust rows reproduce `add_A_robust_per_seed.csv` exactly (`run_log_ablation.txt`: "INTEGRITY CHECK PASSED", maximum deviation 1.4e-14, i.e. floating-point rounding).

Findings (moderate noise, ten seeds):

- Uniform averaging is marginally more accurate than full trust (98.95% vs. 98.90%).
- The anomaly rate alone over-penalizes the noisy clients (weight 0.026) and lowers accuracy to 98.04%: under non-IID data their models still carry class knowledge other clients lack.
- Accuracy-only trust reaches 89.44% AUROC, full trust 89.39%: the zero-day gain over uniform averaging (88.39%) comes mainly from the accuracy component and is not significant after Holm correction.
- Full trust with and without memory give identical results: the memory (λ = 0.30) has no effect when client models are fixed.

## Robust aggregation (Table R10)

Coordinate-wise Median and Trimmed mean act on client probability vectors, Krum and Multi-Krum select clients by the distance between their validation predictions (f = 2). They are prediction-level analogues of the parameter-level rules (Yin et al., 2018; Blanchard et al., 2017); Bulyan is not applicable with five clients (it requires n ≥ 4f + 3). Fed-EITA matches Median and Trimmed mean and is more accurate than Krum and Multi-Krum under severe noise (not significant after correction).

## Reproducing

```bash
python src/make_tables.py                  # tables from the published results
bash supplementary/run_supplementary.sh    # re-run into reproduced/supplementary/ (~8 min), then automatic comparison
```

`results_legacy/` holds the first revision runs (five seeds, preprocessing fitted on the full dataset, probability-based energy in the trust score). They are used only for Table R12 and are not reported as results.

## Limitations

The robustness analyses cover label-noisy clients only; targeted poisoning, backdoors, collusion and adaptive attacks were not evaluated. The sufficient-statistics check uses five seeds; it verifies an algebraic identity, so additional seeds add no information.
