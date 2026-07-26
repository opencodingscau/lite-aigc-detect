#!/usr/bin/env python3
"""E4b: per-source / per-generator OOD evaluation (+ build UFD eval jsonl)."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import JsonlImageDataset
from models import build_model


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs, labels, sources = [], [], []
    for x, y, src in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1)[:, 1]
        probs.extend(p.cpu().tolist())
        labels.extend(y.tolist())
        sources.extend(list(src))
    return probs, labels, sources


def auc_safe(y, p):
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def summarize_by_source(probs, labels, sources, thr: float | None):
    buckets = defaultdict(lambda: {"y": [], "p": []})
    for p, y, s in zip(probs, labels, sources):
        buckets[s]["y"].append(y)
        buckets[s]["p"].append(p)
    out = {}
    for s, b in sorted(buckets.items()):
        y, p = b["y"], b["p"]
        row = {
            "n": len(y),
            "pos_rate": sum(y) / max(len(y), 1),
            "prob_mean": sum(p) / max(len(p), 1),
            "auc": auc_safe(y, p),
        }
        if thr is not None:
            pred = [1 if v >= thr else 0 for v in p]
            row["acc_at_thr"] = sum(int(a == b) for a, b in zip(pred, y)) / max(len(y), 1)
            row["detect_rate_fake"] = (
                sum(1 for yy, pp in zip(y, p) if yy == 1 and pp >= thr) / max(sum(1 for yy in y if yy == 1), 1)
            )
        out[s] = row
    return out


def summarize_paired(probs, labels, sources, pairs: dict[str, tuple[str, str]], thr: float | None):
    """pairs: name -> (real_source, fake_source)"""
    by = defaultdict(lambda: {"y": [], "p": []})
    for p, y, s in zip(probs, labels, sources):
        by[s]["y"].append(y)
        by[s]["p"].append(p)
    out = {}
    for name, (rs, fs) in pairs.items():
        y = by[rs]["y"] + by[fs]["y"]
        p = by[rs]["p"] + by[fs]["p"]
        if not y:
            continue
        row = {"n": len(y), "auc": auc_safe(y, p), "real_n": len(by[rs]["y"]), "fake_n": len(by[fs]["y"])}
        if thr is not None and y:
            pred = [1 if v >= thr else 0 for v in p]
            row["acc_at_thr"] = sum(int(a == b) for a, b in zip(pred, y)) / len(y)
        out[name] = row
    return out


def build_ufd_jsonl(ufd_root: Path, out_path: Path, per_class: int = 400, seed: int = 42):
    rng = random.Random(seed)
    subsets = sorted([d for d in ufd_root.iterdir() if d.is_dir()])
    rows = []
    for d in subsets:
        for label, sub in ((0, "0_real"), (1, "1_fake")):
            folder = d / sub
            if not folder.exists():
                continue
            files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
            rng.shuffle(files)
            files = files[:per_class]
            for p in files:
                rows.append({"path": str(p), "label": label, "source": f"ufd_{d.name}"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), [d.name for d in subsets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest-root", default="manifests")
    ap.add_argument("--ufd-root", default="external/UniversalFakeDetect")
    ap.add_argument("--out", default="outputs/ood_by_source")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ufd-per-class", type=int, default=400)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.manifest_root)

    model = build_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    metrics_path = Path(args.ckpt).parent / "metrics.json"
    thr = None
    if metrics_path.exists():
        thr = json.loads(metrics_path.read_text(encoding="utf-8")).get("val_threshold")

    report = {"model": args.model, "ckpt": args.ckpt, "val_threshold": thr}

    # 1) existing test_ood by source + paired DF generators
    ood_ds = JsonlImageDataset(root / "test_ood.jsonl", train=False)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    probs, labels, sources = collect(model, ood_loader, device)
    report["test_ood_by_source"] = summarize_by_source(probs, labels, sources, thr)
    report["test_ood_paired"] = summarize_paired(
        probs,
        labels,
        sources,
        {
            "df_ldm_bedroom": ("ldm_bedroom_real", "ldm_bedroom_fake"),
            "df_stylegan_bedroom": ("stylegan_bedroom_real", "stylegan_bedroom_fake"),
            "df_projectedgan_bedroom": ("projectedgan_bedroom_real", "projectedgan_bedroom_fake"),
            "ufd_ldm100_in_ood": ("ufd_ldm100_real", "ufd_ldm100_fake"),
            "gangen_attgan": ("gangen_attgan_real", "gangen_attgan_fake"),
        },
        thr,
    )

    # 2) full UFD fine-grained
    ufd_jsonl = out / "ufd_eval.jsonl"
    n, names = build_ufd_jsonl(Path(args.ufd_root), ufd_jsonl, per_class=args.ufd_per_class)
    report["ufd_eval_n"] = n
    report["ufd_subsets"] = names
    ufd_ds = JsonlImageDataset(ufd_jsonl, train=False)
    ufd_loader = DataLoader(ufd_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    up, ul, us = collect(model, ufd_loader, device)
    report["ufd_by_subset"] = summarize_by_source(up, ul, us, thr)
    # each ufd_* source already mixes real+fake under same source name
    report["ufd_mean_auc"] = sum(v["auc"] for v in report["ufd_by_subset"].values() if v["auc"] is not None) / max(
        sum(1 for v in report["ufd_by_subset"].values() if v["auc"] is not None), 1
    )

    # 3) ID domain split (bedroom vs celebahq)
    id_ds = JsonlImageDataset(root / "test.jsonl", train=False)
    id_loader = DataLoader(id_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    ip, il, isrc = collect(model, id_loader, device)
    report["id_by_source"] = summarize_by_source(ip, il, isrc, thr)
    report["id_domain"] = summarize_paired(
        ip,
        il,
        isrc,
        {
            "bedroom_adm": ("adm_bedroom_real", "adm_bedroom_fake"),
            "celebahq_sdv2": ("sdv2_celebahq_real", "sdv2_celebahq_fake"),
        },
        thr,
    )

    out_path = out / f"{args.model}_ood_by_source.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Wrote", out_path, flush=True)
    print("UFD mean AUC", report["ufd_mean_auc"], flush=True)
    print("ID domains", json.dumps(report["id_domain"], indent=2), flush=True)
    print("DF paired", json.dumps(report["test_ood_paired"], indent=2), flush=True)
    print("UFD by subset", json.dumps(report["ufd_by_subset"], indent=2), flush=True)


if __name__ == "__main__":
    main()
