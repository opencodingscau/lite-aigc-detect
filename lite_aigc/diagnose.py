#!/usr/bin/env python3
"""Diagnose LiteFreqNet prediction distribution and optimal thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from data import JsonlImageDataset
from models import build_model


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs, labels, preds = [], [], []
    logit_diffs = []
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, 1)[:, 1]
        pred = logits.argmax(1)
        probs.extend(p.cpu().tolist())
        labels.extend(y.tolist())
        preds.extend(pred.cpu().tolist())
        logit_diffs.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
    return (
        np.asarray(probs),
        np.asarray(labels),
        np.asarray(preds),
        np.asarray(logit_diffs),
    )


def best_threshold(probs, labels):
    """Maximize Youden's J = TPR - FPR on validation."""
    best_t, best_j, best_acc = 0.5, -1.0, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        acc = (tp + tn) / max(len(labels), 1)
        if j > best_j:
            best_j, best_t, best_acc = j, float(t), float(acc)
    return best_t, best_acc, best_j


def summarize(name, probs, labels, preds, diffs):
    auc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")
    acc_argmax = float(accuracy_score(labels, preds))
    return {
        "split": name,
        "n": int(len(labels)),
        "label_pos_rate": float(labels.mean()),
        "pred_pos_rate_argmax": float(preds.mean()),
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
        "prob_p10": float(np.percentile(probs, 10)),
        "prob_p50": float(np.percentile(probs, 50)),
        "prob_p90": float(np.percentile(probs, 90)),
        "logit_diff_mean": float(diffs.mean()),
        "auc": auc,
        "acc_argmax": acc_argmax,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lite_freq_net")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest-root", default="manifests")
    ap.add_argument("--out", default="outputs/diagnose")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.manifest_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    report = {"model": args.model, "ckpt": args.ckpt, "splits": {}}
    loaders = {}
    for split in ("val", "test", "test_ood"):
        path = root / f"{split}.jsonl"
        if not path.exists():
            continue
        ds = JsonlImageDataset(path, train=False)
        loaders[split] = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)

    val_probs = val_labels = None
    for split, loader in loaders.items():
        probs, labels, preds, diffs = collect(model, loader, device)
        s = summarize(split, probs, labels, preds, diffs)
        report["splits"][split] = s
        if split == "val":
            val_probs, val_labels = probs, labels
            thr, thr_acc, j = best_threshold(probs, labels)
            report["val_best_threshold"] = thr
            report["val_acc_at_best_thr"] = thr_acc
            report["val_youden_j"] = j
        print(json.dumps(s, ensure_ascii=False), flush=True)

    thr = report.get("val_best_threshold", 0.5)
    for split, loader in loaders.items():
        probs, labels, preds, diffs = collect(model, loader, device)
        pred_t = (probs >= thr).astype(int)
        report["splits"][split]["acc_at_val_thr"] = float(accuracy_score(labels, pred_t))
        report["splits"][split]["pred_pos_rate_at_val_thr"] = float(pred_t.mean())

    # band gates if present
    if hasattr(model, "freq") and model.freq is not None:
        g = torch.softmax(model.freq.band_logits.detach().cpu(), dim=0).tolist()
        report["band_gates_low_mid_high"] = g

    out_path = out / f"{args.model}_diagnose.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Wrote", out_path, flush=True)
    print(json.dumps({k: report[k] for k in report if k != "splits"}, indent=2), flush=True)
    print(json.dumps(report["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
