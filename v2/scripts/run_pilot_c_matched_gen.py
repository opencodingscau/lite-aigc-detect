#!/usr/bin/env python3
"""Pilot C stub — matched modern-generator mini-set protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "pilot_c_matched_gen.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print("=== Pilot C: matched modern generators ===")
    if yaml and CFG.exists():
        cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
        print("protocol keys:", list(cfg.get("protocol", {}).keys()))
        print("groups:", list(cfg.get("generator_groups", {}).keys()))
    if args.dry_run:
        print("DRY-RUN: write pairing protocol under v2/matched_generators/ before generation.")
        return
    raise SystemExit("Generation pipeline not wired yet. Re-run with --dry-run.")


if __name__ == "__main__":
    main()
