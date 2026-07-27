#!/usr/bin/env python3
"""Export NPR + UnivFD soft labels on Pilot B distill pool (AutoDL)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Reuse loaders from paper eval script
sys.path.insert(0, "/root/autodl-tmp/v2_exp/lite_aigc")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lite_aigc"))

from eval_external_refs import load_npr, load_univfd  # noqa: E402


class PoolDS(Dataset):
    def __init__(self, jsonl: Path, transform):
        self.rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tf = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"]), r["path"], r.get("sample_id", ""), r.get("source", "unk")


@torch.no_grad()
def export_one(name, load_fn, pool, ext_root, out_path, batch, device):
    model, tf, score_fn, meta = load_fn(device, Path(ext_root))
    ds = PoolDS(pool, tf)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for x, y, paths, sids, srcs in loader:
            x = x.to(device)
            probs = score_fn(model, x).detach().float().cpu().tolist()
            if not isinstance(probs, list):
                probs = [float(probs)]
            for pth, lab, sid, src, pr in zip(paths, y.tolist(), sids, srcs, probs):
                f.write(
                    json.dumps(
                        {
                            "path": pth,
                            "sample_id": sid,
                            "label": int(lab),
                            "source": src,
                            "teacher": name,
                            "prob": float(pr),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    print(f"[{name}] wrote {n} -> {out_path} meta={meta}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/root/autodl-tmp/v2_exp/manifests/pilot_b/distill_pool.jsonl")
    ap.add_argument("--ext-root", default="/root/autodl-tmp/external")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/v2_exp/outputs/pilot_b/teacher_soft")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = Path(args.pool)
    assert pool.exists(), pool

    for name, fn in [("npr", load_npr), ("univfd", load_univfd)]:
        sys.path = [p for p in sys.path if "UniversalFakeDetect" not in p and "NPR-DeepfakeDetection" not in p]
        export_one(name, fn, pool, args.ext_root, out / f"{name}_soft.jsonl", args.batch, device)

    # merge keyed by path
    npr = {json.loads(l)["path"]: json.loads(l) for l in (out / "npr_soft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    ufd = {json.loads(l)["path"]: json.loads(l) for l in (out / "univfd_soft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    merged = out / "teachers_merged.jsonl"
    with merged.open("w", encoding="utf-8") as f:
        for pth, a in npr.items():
            b = ufd.get(pth)
            if b is None:
                continue
            f.write(
                json.dumps(
                    {
                        "path": pth,
                        "sample_id": a.get("sample_id"),
                        "label": a["label"],
                        "source": a.get("source"),
                        "npr_prob": a["prob"],
                        "univfd_prob": b["prob"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("merged", merged, flush=True)


if __name__ == "__main__":
    main()
