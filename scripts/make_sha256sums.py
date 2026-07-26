#!/usr/bin/env python3
"""Aggregate SHA256SUMS for the public freeze surface (manifests + key JSONs).

Hashes the **git blob bytes** (LF as stored in the repository) when the path is
tracked; otherwise hashes the working-tree file after normalizing text to LF.
This matches `.gitattributes` (`eol=lf`) and avoids Windows CRLF false failures.

Does NOT hash raw images. Checkpoints are listed if present under checkpoints/.
"""
from __future__ import annotations

import hashlib
import subprocess
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
    # verification_report.json excluded: timestamps / local git HEAD
    "docs/model_architectures.md",
    "checkpoints/README.md",
    ".gitattributes",
]

TEXT_SUFFIXES = {
    ".md",
    ".json",
    ".csv",
    ".py",
    ".yml",
    ".yaml",
    ".tex",
    ".bib",
    ".txt",
    ".cff",
    ".example",
}


def git_blob_bytes(rel: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def working_tree_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitattributes", ".gitignore"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def digest_for(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    data = git_blob_bytes(rel)
    if data is None:
        data = working_tree_bytes(path)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    files: list[Path] = []
    for pattern in GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    lines = []
    for p in files:
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        lines.append(f"{digest_for(p)}  {rel}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Always write LF
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"wrote {OUT} ({len(lines)} files)")


if __name__ == "__main__":
    main()
