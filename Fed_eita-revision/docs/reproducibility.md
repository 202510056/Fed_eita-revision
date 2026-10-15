# Reproducibility

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned versions of the published runs
```

The XGBoost version matters for bit-identical predictions; the published runs used XGBoost 3.2.0 on CPU.

## Level 1 — verify the published numbers (no datasets, seconds)

```bash
python verification/integrity_checks/verify_checksums.py   # result and table files unchanged since publication
python verification/integrity_checks/check_tables.py       # all tables regenerate exactly from the per-seed results
```

## Level 2 — re-run the experiments (datasets required, about 10 minutes on 4 vCPUs)

Place the datasets as described in `data/README.md`, then run from the repository root:

```bash
bash main/run_main.sh                     # ~3 min
bash supplementary/run_supplementary.sh   # ~7 min
```

Both scripts write to `reproduced/` and never overwrite the published results. They then call `verification/integrity_checks/compare_reproduction.py`, which requires every non-timing value to equal the published value to 1e-9. On the original hardware and software the published results were reproduced exactly (see `verification/README.md`). Wall-clock times differ between machines and are reported but not compared.

## Order and dependencies

| Step | Script | Depends on |
|---|---|---|
| 1 | `fed_eita_revision_experiments.py` | datasets |
| 2 | `fed_eita_revision_additions.py` | TON-IoT |
| 3 | `fed_eita_trust_ablation.py` | step 2 in the same `OUT_DIR` (integrity check against `add_A_robust_per_seed.csv`) |
| 4 | `verify_sufficient_statistics.py` | TON-IoT |
| 5 | `make_tables.py` | result folders only |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `TON_CSV`, `EDGE_CSV` | `data/...csv` | dataset paths |
| `OUT_DIR` | `results_revision` / `results` | output folder (the run scripts set `reproduced/...`) |
| `LEAKFREE` | `1` | `0` reproduces the first revision runs (full-data preprocessing) |
| `TRUST_ENERGY` | `margin` | `prob` reproduces the legacy probability-based energy in the trust score |
| `FAST` / `QUICK` | `0` | five seeds / small functional check (not used for any reported number) |

`LEAKFREE=0 TRUST_ENERGY=prob FAST=1` reproduces `supplementary/results_legacy/`. The run scripts `main/run_main.sh` and `supplementary/run_supplementary.sh` pin `LEAKFREE=1 TRUST_ENERGY=margin FAST=0 QUICK=0`, so values set elsewhere in the shell cannot change a re-run, and they stop with an error if a required dataset file is missing.

## Determinism

All randomness is seeded: the data split (`random_state = 42`), partitions, noise, XGBoost (`random_state = seed`), evaluation subsets (`seed + 500` / `seed + 900`) and bootstrap (`default_rng(0)`). The ablation script recomputes the Fed-EITA rows of the robust-aggregation experiment independently and stops unless they match; in the published run the maximum deviation was 1.4 × 10⁻¹⁴.
