#!/usr/bin/env python3
"""Pilot A stub — prints plan; training hooks come later."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "pilot_a_backbone.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print("=== Pilot A: compact backbone bake-off ===")
    if yaml and CFG.exists():
        cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
        for m in cfg.get("models", []):
            print(f"  - {m['display']} [{m['source']}]")
        print("metrics:", ", ".join(cfg.get("metrics_required", [])))
    else:
        print("  (install PyYAML to pretty-print config)")
    if args.dry_run:
        print("DRY-RUN: no training. Fill manifests under v2/manifests/pilot_a/ then implement train/eval.")
        return
    raise SystemExit("Training not wired yet. Re-run with --dry-run or implement trainer.")


if __name__ == "__main__":
    main()
