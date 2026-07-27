#!/usr/bin/env python3
"""Pilot B step-0: build distill pool manifest from train+val (never test/UFD test)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Prefer remote paths when run on AutoDL; locally this is a template writer.
REMOTE_MAN = Path("/root/autodl-tmp/v2_exp/manifests")
OUT = Path("/root/autodl-tmp/v2_exp/manifests/pilot_b")
BLOCKLIST = [
    Path("/root/autodl-tmp/preflight/manifests/test.jsonl"),
    Path("/root/autodl-tmp/preflight/manifests/test_ood.jsonl"),
    Path("/root/autodl-tmp/outputs/ood_by_source/ufd_eval.jsonl"),
]


def load_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p = r.get("path", "")
        ids.add(p)
        ids.add(hashlib.sha256(p.encode()).hexdigest()[:16])
    return ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocked = set()
    for b in BLOCKLIST:
        blocked |= load_ids(b)
    rows = []
    for split in ("train.jsonl", "val.jsonl"):
        src = REMOTE_MAN / split
        for line in src.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            p = r["path"]
            if p in blocked:
                continue
            rows.append(
                {
                    "path": p,
                    "label": int(r["label"]),
                    "source": r.get("source", "unk"),
                    "split": split.replace(".jsonl", ""),
                    "sample_id": hashlib.sha256(p.encode()).hexdigest()[:16],
                }
            )
    out = OUT / "distill_pool.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "n": len(rows),
        "blocked_sources": [str(b) for b in BLOCKLIST],
        "note": "Pilot B pool = train+val minus any path overlapping paper test/OOD/UFD eval",
    }
    (OUT / "distill_pool_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
