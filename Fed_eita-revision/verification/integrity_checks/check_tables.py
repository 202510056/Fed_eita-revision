"""Regenerates all tables in a temporary copy and checks that they equal the committed tables.

    python verification/integrity_checks/check_tables.py

No experiment is run; this only re-aggregates the published per-seed results."""
import shutil, subprocess, sys, tempfile, filecmp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    shutil.copytree(ROOT / "src", t / "src")
    for pkg in ["main", "supplementary"]:
        for sub in ["results", "results_legacy"]:
            if (ROOT / pkg / sub).exists(): shutil.copytree(ROOT / pkg / sub, t / pkg / sub)
    subprocess.run([sys.executable, str(t / "src" / "make_tables.py")], check=True, capture_output=True)
    ok = True
    for pkg in ["main", "supplementary"]:
        for f in sorted((ROOT / pkg / "tables").iterdir()):
            same = filecmp.cmp(f, t / pkg / "tables" / f.name, shallow=False)
            ok &= same
            print(f"  {pkg}/tables/{f.name:<26} {'identical' if same else 'DIFFERENT'}")
print("TABLES REPRODUCED EXACTLY" if ok else "TABLE DIFFERENCES FOUND"); sys.exit(0 if ok else 1)
