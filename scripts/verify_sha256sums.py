#!/usr/bin/env python3
"""Verify hashes/SHA256SUMS against git blob bytes (or LF-normalized working tree)."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = ROOT / "hashes" / "SHA256SUMS"
TEXT_SUFFIXES = {".md", ".json", ".csv", ".py", ".yml", ".yaml", ".tex", ".bib", ".txt", ".cff", ".example"}


def bytes_for(rel: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        data = (ROOT / rel).read_bytes()
        p = ROOT / rel
        if p.suffix.lower() in TEXT_SUFFIXES or p.name in {".gitattributes", ".gitignore"}:
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return data


def main() -> int:
    if not SUMS.exists():
        print("missing", SUMS)
        return 2
    ok = bad = 0
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, rel = parts[0], parts[1]
        got = hashlib.sha256(bytes_for(rel)).hexdigest()
        if got == digest:
            ok += 1
        else:
            bad += 1
            print("BAD", rel)
    print(f"ok={ok} bad={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
