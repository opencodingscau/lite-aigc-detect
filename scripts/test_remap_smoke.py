#!/usr/bin/env python3
"""Smoke-test remap_manifest_paths on a few lines (no full rewrite)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remap_manifest_paths.py"


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main() -> None:
    sample = {
        "path": "/PREVIOUS/DATA/DiffusionForensics/adm/bedroom/0_real/x.jpg",
        "label": 0,
        "source": "adm_bedroom_real",
    }
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "train.jsonl"
        src.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        out = td_path / "out"
        run(
            [
                sys.executable,
                str(SCRIPT),
                "--files",
                str(src),
                "--out-dir",
                str(out),
                "--old-prefix",
                "/PREVIOUS/DATA",
                "--new-prefix",
                "/data/aigc_datasets",
            ]
        )
        row = json.loads((out / "train.jsonl").read_text(encoding="utf-8").strip())
        assert row["path"] == "/data/aigc_datasets/DiffusionForensics/adm/bedroom/0_real/x.jpg", row

        out2 = td_path / "out2"
        run(
            [
                sys.executable,
                str(SCRIPT),
                "--files",
                str(src),
                "--out-dir",
                str(out2),
                "--old-prefix",
                "/PREVIOUS/DATA",
                "--new-prefix",
                "/data/aigc_datasets",
                "--relative",
            ]
        )
        row2 = json.loads((out2 / "train.jsonl").read_text(encoding="utf-8").strip())
        assert row2["path"] == "DiffusionForensics/adm/bedroom/0_real/x.jpg", row2

        out3 = td_path / "out3"
        run(
            [
                sys.executable,
                str(SCRIPT),
                "--files",
                str(src),
                "--out-dir",
                str(out3),
                "--map",
                "DiffusionForensics=/mnt/DF",
            ]
        )
        row3 = json.loads((out3 / "train.jsonl").read_text(encoding="utf-8").strip())
        assert row3["path"] == "/mnt/DF/adm/bedroom/0_real/x.jpg", row3

    print("remap smoke OK")


if __name__ == "__main__":
    main()
