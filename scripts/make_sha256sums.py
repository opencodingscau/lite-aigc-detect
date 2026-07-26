#!/usr/bin/env python3
"""Aggregate SHA256SUMS for the public freeze surface (manifests + key JSONs).

Does NOT hash raw images. Checkpoints are listed if present under checkpoints/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hashes" / "SHA256SUMS"

GLOBS = [
    "freeze/freeze_package.json",
    "freeze/frozen_numbers.json",
    "freeze/SHA256_MANIFESTS.json",
    "freeze/TABLES.md",
    "latency_batch1/summary.json",
    "external_refs/summary.json",
    "external_refs/univfd_report.json",
    "external_refs/npr_report.json",
    "reproduced_tables/*.csv",
    "reproduced_tables/verification_report.json",
    "docs/model_architectures.md",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files: list[Path] = []
    for pattern in GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    # also include freeze/SHA256_MANIFESTS remote paths as already recorded
    lines = []
    for p in files:
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        lines.append(f"{digest(p)}  {rel}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} files)")


if __name__ == "__main__":
    main()
