"""Compares a re-run with the published results.

    python verification/integrity_checks/compare_reproduction.py PUBLISHED_DIR RERUN_DIR

Every CSV present in both folders is compared column by column. Wall-clock timing columns
are reported but not required to match, because they depend on hardware and load; every
other value must agree to 1e-9. Exit code 0 means all non-timing results were reproduced."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

TIMING = {"local_training_s", "trust_and_uncertainty_s", "infer_fedavg_ms_per_1k", "infer_fedeita_ms_per_1k"}
pub, new = Path(sys.argv[1]), Path(sys.argv[2])
ok, n = True, 0
for f in sorted(pub.glob("*.csv")):
    g = new / f.name
    if not g.exists():
        print(f"  not produced by this run: {f.name}"); continue
    a, b = pd.read_csv(f), pd.read_csv(g); n += 1
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        print(f"  MISMATCH {f.name}: shape or columns differ {a.shape} vs {b.shape}"); ok = False; continue
    num = [c for c in a.select_dtypes("number").columns if c not in TIMING]
    txt = [c for c in a.columns if c not in a.select_dtypes("number").columns]
    dev = float(np.nanmax(np.abs(a[num].values - b[num].values))) if num else 0.0
    same_txt = bool(a[txt].astype(object).fillna("<NA>").astype(str).equals(b[txt].astype(object).fillna("<NA>").astype(str))) if txt else True
    timing = sorted(TIMING & set(a.columns))
    print(f"  {f.name:<34} max |diff| = {dev:.1e}  text identical = {same_txt}" + (f"  (timing columns not compared: {', '.join(timing)})" if timing else ""))
    ok &= dev < 1e-9 and same_txt
print(f"\n{n} files compared. " + ("ALL NON-TIMING RESULTS REPRODUCED EXACTLY" if ok and n else "DIFFERENCES FOUND"))
sys.exit(0 if ok and n else 1)
