#!/usr/bin/env python3
"""Evaluate a ckpt on an arbitrary jsonl manifest (E8 Flux/SD3 table)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import JsonlImageDataset
from models import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    thr = None
    mp = Path(args.ckpt).parent / "metrics.json"
    if mp.exists():
        thr = json.loads(mp.read_text(encoding="utf-8")).get("val_threshold")

    ds = JsonlImageDataset(args.manifest, train=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
    probs, labels, sources = [], [], []
    for x, y, src in loader:
        p = torch.softmax(model(x.to(device)), 1)[:, 1]
        probs.extend(p.cpu().tolist())
        labels.extend(y.tolist())
        sources.extend(list(src))

    overall_auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else None
    pred = [1 if p >= (thr or 0.5) else 0 for p in probs]
    overall_acc = sum(int(a == b) for a, b in zip(pred, labels)) / max(len(labels), 1)

    by = defaultdict(lambda: {"y": [], "p": []})
    for p, y, s in zip(probs, labels, sources):
        by[s]["y"].append(y)
        by[s]["p"].append(p)

    # pair flux/sd35 with all reals for generator AUC
    real_y, real_p = [], []
    for s, b in by.items():
        if s.startswith("real_"):
            real_y.extend(b["y"])
            real_p.extend(b["p"])
    paired = {}
    for s, b in by.items():
        if s.startswith("real_"):
            continue
        y = real_y + b["y"]
        p = real_p + b["p"]
        if len(set(y)) > 1:
            paired[s] = {"auc": float(roc_auc_score(y, p)), "n_fake": len(b["y"]), "n_real": len(real_y)}

    out = {
        "model": args.model,
        "manifest": args.manifest,
        "val_threshold": thr,
        "overall": {"auc": overall_auc, "acc_at_thr": overall_acc, "n": len(labels)},
        "by_source": {
            s: {
                "n": len(b["y"]),
                "prob_mean": sum(b["p"]) / max(len(b["p"]), 1),
                "auc": float(roc_auc_score(b["y"], b["p"])) if len(set(b["y"])) > 1 else None,
            }
            for s, b in by.items()
        },
        "paired_vs_reals": paired,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
