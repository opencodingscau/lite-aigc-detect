"""Create local v2 output folders (gitignored)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    ROOT / "outputs" / "pilot_a",
    ROOT / "outputs" / "pilot_b",
    ROOT / "outputs" / "pilot_c",
    ROOT / "manifests" / "pilot_a",
    ROOT / "manifests" / "pilot_b",
    ROOT / "manifests" / "pilot_c",
    ROOT / "distillation_pool" / "dedup",
    ROOT / "matched_generators" / "notes",
    ROOT / "freeze_v2",
]


def main() -> None:
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        keep = d / ".gitkeep"
        if not keep.exists() and d.name in {"pilot_a", "pilot_b", "pilot_c", "dedup", "notes", "freeze_v2"}:
            # manifests stay tracked via README; outputs only gitkeep at pilot roots if empty
            pass
        print("mkdir", d.relative_to(ROOT.parent) if False else d)
    print("workspace ready under v2/")


if __name__ == "__main__":
    main()
