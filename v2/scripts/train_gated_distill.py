#!/usr/bin/env python3
"""Pilot B: confidence-gated multi-teacher distillation (student keeps paper preprocess)."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys

sys.path.insert(0, "/root/autodl-tmp/v2_exp/lite_aigc")
from models import build_model  # noqa: E402


def tf_train(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def tf_eval(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class DistillDS(Dataset):
    def __init__(self, merged_jsonl: Path, train: bool, size=224):
        self.rows = [json.loads(l) for l in merged_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tf = tf_train(size) if train else tf_eval(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        x = self.tf(img)
        y = int(r["label"])
        return x, y, float(r["npr_prob"]), float(r["univfd_prob"])


class EvalDS(Dataset):
    def __init__(self, jsonl: Path, size=224):
        self.rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tf = tf_eval(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"])


def gated_teacher_prob(npr, ufd, recipe: str, conf_thr=0.7, agree_eps=0.15):
    """Return (teacher_prob [B], gate_weight [B])."""
    if recipe == "npr_only":
        return npr, torch.ones_like(npr)
    if recipe == "univfd_only":
        return ufd, torch.ones_like(ufd)
    # gated_dual
    conf_n = torch.maximum(npr, 1 - npr)
    conf_u = torch.maximum(ufd, 1 - ufd)
    agree = (torch.abs(npr - ufd) <= agree_eps).float()
    both_conf = ((conf_n >= conf_thr) & (conf_u >= conf_thr)).float()
    gate = agree * both_conf
    # when gated: average teachers; else ignore KD (gate=0)
    tprob = 0.5 * (npr + ufd)
    return tprob, gate


def kd_bce(student_logits, teacher_prob, gate, T=2.0):
    # student fake-class prob from 2-logit CE head
    s_prob = torch.softmax(student_logits / T, dim=1)[:, 1]
    # soft BCE in prob space (stable for binary teachers)
    eps = 1e-6
    tp = teacher_prob.clamp(eps, 1 - eps)
    sp = s_prob.clamp(eps, 1 - eps)
    loss = -(tp * torch.log(sp) + (1 - tp) * torch.log(1 - sp))
    # scale by T^2 like classic KD
    loss = (T * T) * loss
    if gate is None:
        return loss.mean()
    denom = gate.sum().clamp_min(1.0)
    return (loss * gate).sum() / denom


@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1)[:, 1]
        probs.extend(p.cpu().tolist())
        labels.extend(y.tolist())
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--recipe", required=True, choices=["npr_only", "univfd_only", "gated_dual"])
    ap.add_argument("--merged", default="/root/autodl-tmp/v2_exp/outputs/pilot_b/teacher_soft/teachers_merged.jsonl")
    ap.add_argument("--manifest-root", default="/root/autodl-tmp/v2_exp/manifests")
    ap.add_argument("--out", default="/root/autodl-tmp/v2_exp/outputs/pilot_b")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--conf-thr", type=float, default=0.7)
    ap.add_argument("--agree-eps", type=float, default=0.15)
    ap.add_argument("--init-ckpt", default="", help="optional warm-start (e.g. paper LiteSSM-A)")
    ap.add_argument("--run-name", default="", help="optional output folder name override")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.run_name or f"{args.student}__{args.recipe}"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DistillDS(Path(args.merged), train=True)
    # hold out last 10% of merged rows as distill-val (by index, deterministic)
    n = len(train_ds.rows)
    cut = int(n * 0.9)
    val_rows = train_ds.rows[cut:]
    train_ds.rows = train_ds.rows[:cut]
    val_ds = DistillDS(Path(args.merged), train=False)
    val_ds.rows = val_rows

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # official ID / OOD eval
    man = Path(args.manifest_root)
    id_loader = DataLoader(EvalDS(man / "test.jsonl"), batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    ood_loader = DataLoader(EvalDS(man / "test_ood.jsonl"), batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.student).to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ck["model"], strict=True)
        print("warm-start", args.init_ckpt, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    ce = nn.CrossEntropyLoss()

    best = -1.0
    best_path = out_dir / "best.pt"
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n_seen, gate_sum = 0.0, 0, 0.0
        for x, y, npr, ufd in train_loader:
            x, y = x.to(device), y.to(device)
            npr, ufd = npr.to(device), ufd.to(device)
            logits = model(x)
            loss_ce = ce(logits, y)
            tprob, gate = gated_teacher_prob(npr, ufd, args.recipe, args.conf_thr, args.agree_eps)
            loss_kd = kd_bce(logits, tprob, gate if args.recipe == "gated_dual" else None)
            # for single-teacher recipes always apply KD; for gated_dual scale by mean gate
            if args.recipe == "gated_dual":
                w = args.kd_weight
            else:
                w = args.kd_weight
            loss = loss_ce + w * loss_kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = x.size(0)
            total += float(loss.item()) * bs
            n_seen += bs
            gate_sum += float(gate.sum().item())
        sched.step()
        # distill-val AUC on soft-heldout (uses labels)
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(device)
                p = torch.softmax(model(x), 1)[:, 1]
                probs.extend(p.cpu().tolist())
                labels.extend(y.tolist())
        val_auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
        row = {
            "epoch": ep,
            "loss": total / max(n_seen, 1),
            "val_auc": val_auc,
            "gate_frac": gate_sum / max(n_seen, 1),
        }
        hist.append(row)
        print(f"[{tag}] ep {ep}/{args.epochs} loss={row['loss']:.4f} val_auc={val_auc:.4f} gate_frac={row['gate_frac']:.3f}", flush=True)
        if val_auc == val_auc and val_auc > best:
            best = val_auc
            torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args), "val_auc": val_auc}, best_path)

    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model"])
    id_auc = eval_auc(model, id_loader, device)
    ood_auc = eval_auc(model, ood_loader, device)
    result = {
        "student": args.student,
        "recipe": args.recipe,
        "best_distill_val_auc": best,
        "id_test_auc": id_auc,
        "ood_pooled_auc": ood_auc,
        "train_sec": round(time.time() - t0, 2),
        "epochs": args.epochs,
        "kd_weight": args.kd_weight,
        "conf_thr": args.conf_thr,
        "agree_eps": args.agree_eps,
        "init_ckpt": args.init_ckpt,
        "ckpt": str(best_path),
        "history": hist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
