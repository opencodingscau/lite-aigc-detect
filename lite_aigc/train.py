#!/usr/bin/env python3
"""Train / eval entry for Lite-AIGC-Detect formal experiments."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import JsonlImageDataset
from metrics import measure_flops, measure_fps, measure_params
from models import build_model


def evaluate(model, loader, device, threshold: float | None = None):
    model.eval()
    correct, total = 0, 0
    probs, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            p = torch.softmax(logits, 1)[:, 1]
            if threshold is None:
                pred = logits.argmax(1)
            else:
                pred = (p >= threshold).long()
            correct += int((pred == y).sum())
            total += y.numel()
            probs.extend(p.cpu().tolist())
            labels.extend(y.cpu().tolist())
    acc = correct / max(total, 1)
    auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    return {"acc": acc, "auc": auc, "n": total, "prob_mean": sum(probs) / max(len(probs), 1)}


def best_threshold_from_loader(model, loader, device) -> float:
    """Youden J on validation probabilities."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            p = torch.softmax(model(x), 1)[:, 1]
            probs.extend(p.cpu().tolist())
            labels.extend(y.tolist())
    best_t, best_j = 0.5, -1.0
    for i in range(1, 100):
        t = i / 100.0
        tp = fp = tn = fn = 0
        for p, y in zip(probs, labels):
            pred = 1 if p >= t else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--manifest-root", default="manifests")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="limit train samples for debug")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-ood", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.manifest_root)
    train_ds = JsonlImageDataset(root / "train.jsonl", train=True, size=args.size, limit=args.limit)
    val_ds = JsonlImageDataset(root / "val.jsonl", train=False, size=args.size)
    test_ds = JsonlImageDataset(root / "test.jsonl", train=False, size=args.size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = build_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best_auc, best_path = -1.0, out_dir / "best.pt"
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, crit, device)
        val = evaluate(model, val_loader, device)
        sched.step()
        row = {"epoch": ep, "loss": loss, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(
            f"[{args.model}] epoch {ep}/{args.epochs} loss={loss:.4f} "
            f"val_acc={val['acc']:.4f} val_auc={val['auc']:.4f}",
            flush=True,
        )
        if val["auc"] == val["auc"] and val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "val": val,
                    "args": vars(args),
                },
                best_path,
            )

    # load best and test (argmax + val-tuned threshold)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    thr = best_threshold_from_loader(model, val_loader, device)
    test = evaluate(model, test_loader, device)
    test_thr = evaluate(model, test_loader, device, threshold=thr)
    val_thr = evaluate(model, val_loader, device, threshold=thr)

    params = measure_params(model)
    flops = measure_flops(model, device, size=args.size)
    fps_gpu = measure_fps(model, device, size=args.size, batch=32)
    # CPU FPS optional (slower)
    fps_cpu = None
    try:
        model_cpu = build_model(args.model)
        model_cpu.load_state_dict(ckpt["model"])
        model_cpu.eval()
        fps_cpu = measure_fps(model_cpu, torch.device("cpu"), size=args.size, batch=8, warmup=5, iters=20)
    except Exception as e:  # noqa: BLE001
        fps_cpu = f"err:{e}"

    result = {
        "model": args.model,
        "best_val_auc": best_auc,
        "val_threshold": thr,
        "val_at_thr": val_thr,
        "test": test,
        "test_at_thr": test_thr,
        "params": params,
        "params_M": round(params / 1e6, 3),
        "flops": flops,
        "flops_G": round(flops / 1e9, 3) if isinstance(flops, float) else flops,
        "fps_gpu_bs32": round(fps_gpu, 2),
        "fps_cpu_bs8": round(fps_cpu, 2) if isinstance(fps_cpu, float) else fps_cpu,
        "train_sec": round(time.time() - t0, 2),
        "epochs": args.epochs,
        "train_n": len(train_ds),
        "history": history,
        "ckpt": str(best_path),
    }

    if args.eval_ood:
        ood_ds = JsonlImageDataset(root / "test_ood.jsonl", train=False, size=args.size)
        ood_loader = DataLoader(
            ood_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        result["ood"] = evaluate(model, ood_loader, device)
        result["ood_at_thr"] = evaluate(model, ood_loader, device, threshold=thr)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2), flush=True)
    print("Wrote", out_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
