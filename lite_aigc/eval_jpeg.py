#!/usr/bin/env python3
"""E5: JPEG robustness Q=95/85/70 on ID (and optional OOD) splits."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import build_model


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class JpegJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, quality: int, size: int = 224):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.quality = quality
        self.tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        img = jpeg_compress(img, self.quality)
        return self.tf(img), int(r["label"]), r.get("source", "unk")


@torch.no_grad()
def evaluate(model, loader, device, threshold: float | None = None):
    model.eval()
    correct = total = 0
    probs, labels = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        p = torch.softmax(logits, 1)[:, 1]
        pred = logits.argmax(1) if threshold is None else (p >= threshold).long()
        correct += int((pred == y).sum())
        total += y.numel()
        probs.extend(p.cpu().tolist())
        labels.extend(y.cpu().tolist())
    auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    return {"acc": correct / max(total, 1), "auc": auc, "n": total}


def load_threshold(metrics_path: Path | None) -> float | None:
    if metrics_path is None or not metrics_path.exists():
        return None
    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    return d.get("val_threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest-root", default="manifests")
    ap.add_argument("--qualities", default="95,85,70")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--eval-ood", action="store_true")
    ap.add_argument("--metrics-json", default=None, help="optional metrics.json for val_threshold")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.manifest_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    thr = load_threshold(Path(args.metrics_json) if args.metrics_json else Path(args.ckpt).parent / "metrics.json")

    qualities = [int(q) for q in args.qualities.split(",") if q.strip()]
    result = {
        "model": args.model,
        "ckpt": args.ckpt,
        "val_threshold": thr,
        "jpeg": {},
    }

    # clean reference (no JPEG) on test
    from data import JsonlImageDataset

    clean = JsonlImageDataset(root / "test.jsonl", train=False, size=args.size)
    clean_loader = DataLoader(clean, batch_size=args.batch, shuffle=False, num_workers=4)
    result["clean_test"] = evaluate(model, clean_loader, device)
    if thr is not None:
        result["clean_test_at_thr"] = evaluate(model, clean_loader, device, threshold=thr)

    for q in qualities:
        ds = JpegJsonlDataset(root / "test.jsonl", quality=q, size=args.size)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
        row = {"test": evaluate(model, loader, device)}
        if thr is not None:
            row["test_at_thr"] = evaluate(model, loader, device, threshold=thr)
        if args.eval_ood:
            ood_ds = JpegJsonlDataset(root / "test_ood.jsonl", quality=q, size=args.size)
            ood_loader = DataLoader(ood_ds, batch_size=args.batch, shuffle=False, num_workers=4)
            row["ood"] = evaluate(model, ood_loader, device)
            if thr is not None:
                row["ood_at_thr"] = evaluate(model, ood_loader, device, threshold=thr)
        result["jpeg"][f"q{q}"] = row
        print(f"[JPEG Q={q}] test_auc={row['test']['auc']:.4f} test_acc={row['test']['acc']:.4f}", flush=True)

    out_path = out / f"{args.model}_jpeg.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Wrote", out_path, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
