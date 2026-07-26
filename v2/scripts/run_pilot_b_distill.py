#!/usr/bin/env python3
"""Pilot B stub — gated multi-teacher distillation."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "pilot_b_distill.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print("=== Pilot B: gated distillation ===")
    if yaml and CFG.exists():
        cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
        print("students:", cfg.get("students"))
        print("recipes:", cfg.get("recipes"))
        print("success_bars:", cfg.get("success_bars"))
    if args.dry_run:
        print("DRY-RUN: build v2/manifests/pilot_b/distill_pool.jsonl + dedup first.")
        return
    raise SystemExit("Distill loop not wired yet. Re-run with --dry-run.")


if __name__ == "__main__":
    main()
