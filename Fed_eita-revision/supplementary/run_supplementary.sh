#!/usr/bin/env bash
# Re-runs the supplementary experiments (Tables R10, R11, R13 and the sufficient-statistics
# check) into reproduced/supplementary/ and compares them with supplementary/results/.
# The published files are never overwritten.
# Usage (from the repository root):
#   TON_CSV=data/TON_IoT_Train_Test_Network.csv bash supplementary/run_supplementary.sh
set -euo pipefail
: "${TON_CSV:=data/TON_IoT_Train_Test_Network.csv}"
export TON_CSV OUT_DIR=reproduced/supplementary
# Pin the configuration of the published runs, so that variables already set in the shell
# (for example FAST=1 or LEAKFREE=0) cannot silently change the experiment.
export LEAKFREE=1 TRUST_ENERGY=margin FAST=0 QUICK=0
[ -f "$TON_CSV" ] || { echo "ERROR: TON-IoT file not found: $TON_CSV (see data/README.md)" >&2; exit 1; }
mkdir -p "$OUT_DIR"
python src/fed_eita_revision_additions.py   2>&1 | tee "$OUT_DIR/run_log_additions.txt"   # ~3.5 min
python src/fed_eita_trust_ablation.py       2>&1 | tee "$OUT_DIR/run_log_ablation.txt"    # ~3 min; must reproduce add_A rows exactly
python src/verify_sufficient_statistics.py  2>&1 | tee "$OUT_DIR/run_log_verify.txt"      # < 1 min
python verification/integrity_checks/compare_reproduction.py supplementary/results "$OUT_DIR"
