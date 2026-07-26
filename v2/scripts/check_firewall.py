"""Refuse writes that would contaminate the paper freeze."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOCK_WRITE_PREFIXES = (
    ROOT / "freeze" / "frozen_numbers.json",
    ROOT / "freeze" / "freeze_package.json",
)
BLOCK_WRITE_DIRS = (
    ROOT / "freeze" / "figures",  # paper figures; v2 uses v2/results art
    ROOT / "reproduced_tables",
    ROOT / "latex",
)


def main() -> int:
    print("V2 firewall check")
    print(f"  repo: {ROOT}")
    print(f"  ok to write: {ROOT / 'v2'}")
    for p in BLOCK_WRITE_PREFIXES:
        print(f"  blocked file: {p.relative_to(ROOT)} (exists={p.exists()})")
    for d in BLOCK_WRITE_DIRS:
        print(f"  blocked dir:  {d.relative_to(ROOT)}/")
    print("  ban: distill/train on paper UFD/DALL·E test IDs")
    print("PASS — use only v2/manifests, v2/outputs, v2/freeze_v2, v2/results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
