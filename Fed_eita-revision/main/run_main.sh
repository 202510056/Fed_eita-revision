#!/usr/bin/env bash
# Re-runs the main experiment (Tables R1, R3-R9) into reproduced/main/ and compares it with
# the published results in main/results/. The published files are never overwritten.
# Usage (from the repository root):
#   TON_CSV=data/TON_IoT_Train_Test_Network.csv EDGE_CSV=data/ML-EdgeIIoT-dataset.csv bash main/run_main.sh
set -euo pipefail
: "${TON_CSV:=data/TON_IoT_Train_Test_Network.csv}"; : "${EDGE_CSV:=data/ML-EdgeIIoT-dataset.csv}"
export TON_CSV EDGE_CSV OUT_DIR=reproduced/main
# Pin the configuration of the published runs, so that variables already set in the shell
# (for example FAST=1 or LEAKFREE=0) cannot silently change the experiment.
export LEAKFREE=1 TRUST_ENERGY=margin FAST=0 QUICK=0
[ -f "$TON_CSV" ] || { echo "ERROR: TON-IoT file not found: $TON_CSV (see data/README.md)" >&2; exit 1; }
[ -f "$EDGE_CSV" ] || { echo "ERROR: Edge-IIoT file not found: $EDGE_CSV (see data/README.md)" >&2; exit 1; }
mkdir -p "$OUT_DIR"
python src/fed_eita_revision_experiments.py 2>&1 | tee "$OUT_DIR/run_log.txt"        # ~3 min on 4 vCPUs
python verification/integrity_checks/compare_reproduction.py main/results "$OUT_DIR"
