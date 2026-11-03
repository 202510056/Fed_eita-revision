# Columns of the published result files that the manuscript does not use

`main/results/r1_ood_per_seed.csv`, `r1_ood_summary.csv`, `r1_ood_per_class*.csv` contain two score rows
that are **not** used by any table or statement of the manuscript:

| Row (`score`) | Meaning | Used in the manuscript? |
|---|---|---|
| `Energy (prob., as implemented)` | the probability-based "energy" of the original submission, computed from normalized probabilities | No. Kept only to document the correction of the energy score (Appendix Table A3 compares the first revision runs before and after the correction). |
| `Fed-EITA fused (as implemented)` | Eq. (11) with that probability-based energy | No (same reason). |
| `Energy (margin, Eq. 9)` | energy from raw XGBoost margins, Eq. (9) | **Yes**: "Energy E(x)" in Tables 12, A1, A2, A6. |
| `Fed-EITA fused (margin energy)` | Eq. (11) with the margin energy | **Yes**: "Fed-EITA fusion" in all tables. |

The result files are protected by SHA-256 checksums (`verification/integrity_checks/checksums.sha256`), so the
legacy rows are labelled here instead of being removed from the files. `src/make_tables.py` and
`src/make_tables_round2.py` read only the margin-energy rows.
