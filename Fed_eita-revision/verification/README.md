# Verification

Tools that let a reader check the published numbers without trusting the authors' description.

| Check | Command | Needs datasets | What it shows |
|---|---|---|---|
| Checksums | `python verification/integrity_checks/verify_checksums.py` | no | Every published result and table file is byte-identical to the version released with the manuscript (`checksums.sha256`, 55 files) |
| Table regeneration | `python verification/integrity_checks/check_tables.py` | no | Re-aggregating the per-seed results reproduces every table file exactly |
| Re-run comparison | `python verification/integrity_checks/compare_reproduction.py PUBLISHED RERUN` (called by the run scripts) | yes | A fresh run reproduces every non-timing value to 1e-9; timing columns are reported, not compared |
| Code equivalence | `python verification/integrity_checks/ast_equivalence.py OLD.py NEW.py` | no | Two versions of a script differ only in docstrings and comments |

## Evidence recorded for the published runs

- **Main experiment.** The corrected inference-timing run was compared with the previously published run. All non-timing results were identical: maximum difference 0 in `r1_ood_per_seed.csv`, `r1_ood_per_class_per_seed.csv`, `r2_aggregation_per_seed.csv`, `r2_trust_weights_per_seed.csv` and `r3_communication_per_seed.csv`.
- **Trust ablation.** Its full-trust rows reproduce the robust-aggregation experiment, computed by a separate script, with maximum deviation 1.4 × 10⁻¹⁴ (`supplementary/results/run_log_ablation.txt`).
- **Fusion score.** The main experiment's fused score equals the fused score computed independently in `fed_eita_revision_additions.py` for every seed (difference 0).
- **Sufficient statistics.** Aggregated per-client statistics reproduce the pooled Mahalanobis estimate: maximum deviation 1.1 × 10⁻¹⁴ for class means and 5.5 × 10⁻¹⁵ for covariances (`supplementary/results/verify_sufficient_statistics.txt`).
- **Repository assembly.** The results and code were reorganized into `main/` and `supplementary/`, with documentation added. `integrity_checks/assembly_checks.txt` records that every result file is byte-identical, that the experiment scripts changed only in docstrings and comments, and that all tables are unchanged.

## Logs

Each `run_log*.txt` starts with the configuration of the run: backend `xgboost`, number of seeds, `LEAKFREE=True`, `TRUST_ENERGY=margin`. Dataset sizes are given in the next lines (TON-IoT: 461,043 rows; Edge-IIoT: 60,000 rows).
