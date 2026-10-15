"""Verifies the SHA-256 checksums of all published result and table files.

    python verification/integrity_checks/verify_checksums.py

checksums.sha256 was written when the repository was assembled; a mismatch means a
published number was changed after that point."""
import hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
bad = 0
for line in (Path(__file__).parent / "checksums.sha256").read_text().splitlines():
    digest, rel = line.split(maxsplit=1)
    p = ROOT / rel
    ok = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
    bad += not ok
    if not ok: print(f"  MISMATCH or missing: {rel}")
print("ALL CHECKSUMS MATCH" if not bad else f"{bad} FILE(S) DIFFER"); sys.exit(0 if not bad else 1)
