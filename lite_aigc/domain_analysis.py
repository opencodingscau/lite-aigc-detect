#!/usr/bin/env python3
"""E7: domain analysis figures (celebahq vs bedroom) + spectrum energy curves."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import JsonlImageDataset
from models import build_model


def radial_energy(gray: np.ndarray, bins: int = 32):
    """gray HxW float -> mean log-magnitude energy per radial bin."""
    spec = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(spec))
    h, w = mag.shape
    yy, xx = np.mgrid[-1 : 1 : complex(h), -1 : 1 : complex(w)]
    r = np.sqrt(xx * xx + yy * yy)
    edges = np.linspace(0, np.sqrt(2), bins + 1)
    energy = []
    for i in range(bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        energy.append(float(mag[m].mean()) if m.any() else 0.0)
    return np.asarray(energy)


def sample_paths(jsonl: Path, source_prefix: str, label: int, n: int, seed: int = 42):
    rng = random.Random(seed)
    rows = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["label"] == label and str(r.get("source", "")).startswith(source_prefix):
                rows.append(r["path"])
    rng.shuffle(rows)
    return rows[:n]


def mean_spectrum(paths, size=224):
    acc = None
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        arr = np.asarray(img).astype(np.float32) / 255.0
        gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        e = radial_energy(gray)
        acc = e if acc is None else acc + e
    return acc / max(len(paths), 1)


@torch.no_grad()
def domain_metrics(model, manifest, device, thr):
    model.eval()
    ds = JsonlImageDataset(manifest, train=False)
    # num_workers=0: avoid rare source/label misalign under multi-process on some images
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    buckets = defaultdict(lambda: {"y": [], "p": []})
    for x, y, src in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1)[:, 1].cpu().tolist()
        for pp, yy, s in zip(p, y.tolist(), src):
            if "bedroom" in s:
                dom = "bedroom"
            elif "celebahq" in s:
                dom = "celebahq"
            else:
                dom = "other"
            buckets[dom]["y"].append(yy)
            buckets[dom]["p"].append(pp)
    out = {}
    for dom, b in buckets.items():
        y, p = b["y"], b["p"]
        if len(set(y)) < 2:
            continue
        pred = [1 if v >= (thr or 0.5) else 0 for v in p]
        out[dom] = {
            "n": len(y),
            "auc": float(roc_auc_score(y, p)),
            "acc_at_thr": sum(int(a == b_) for a, b_ in zip(pred, y)) / len(y),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lite_freq_net_v2")
    ap.add_argument("--ckpt", default="checkpoints/lite_freq_net_v2/best.pt")
    ap.add_argument("--manifest-root", default="manifests")
    ap.add_argument("--out", default="outputs/domain_analysis")
    ap.add_argument("--n-spec", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.manifest_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    thr = None
    mp = Path(args.ckpt).parent / "metrics.json"
    if mp.exists():
        thr = json.loads(mp.read_text(encoding="utf-8")).get("val_threshold")

    # --- spectrum curves ---
    test = root / "test.jsonl"
    specs = {
        "bedroom_real": mean_spectrum(sample_paths(test, "adm_bedroom", 0, args.n_spec)),
        "bedroom_fake": mean_spectrum(sample_paths(test, "adm_bedroom", 1, args.n_spec)),
        "celebahq_real": mean_spectrum(sample_paths(test, "sdv2_celebahq", 0, args.n_spec)),
        "celebahq_fake": mean_spectrum(sample_paths(test, "sdv2_celebahq", 1, args.n_spec)),
    }
    x = np.linspace(0, 1, len(next(iter(specs.values()))))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    for ax, dom, real_k, fake_k in (
        (axes[0], "Bedroom (ADM)", "bedroom_real", "bedroom_fake"),
        (axes[1], "CelebA-HQ (SDv2)", "celebahq_real", "celebahq_fake"),
    ):
        ax.plot(x, specs[real_k], label="real", color="#2a6f97", lw=2)
        ax.plot(x, specs[fake_k], label="fake", color="#e76f51", lw=2)
        ax.fill_between(x, specs[real_k], specs[fake_k], color="#adb5bd", alpha=0.25)
        ax.set_title(dom)
        ax.set_xlabel("Normalized radial frequency")
        ax.set_ylabel("Mean log |FFT|")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    spec_path = out / "spectrum_domain_compare.png"
    fig.savefig(spec_path)
    plt.close(fig)

    # mid-band energy delta (bins ~0.3-0.7 of radial)
    def mid_energy(e):
        n = len(e)
        lo, hi = int(0.3 * n), int(0.7 * n)
        return float(e[lo:hi].mean())

    mid = {k: mid_energy(v) for k, v in specs.items()}

    # --- domain AUC bars for this model + optional baselines if metrics exist ---
    dom = domain_metrics(model, test, device, thr)

    fig2, ax = plt.subplots(figsize=(6, 4), dpi=140)
    domains = [d for d in ("bedroom", "celebahq") if d in dom]
    aucs = [dom[d]["auc"] for d in domains]
    colors = ["#2a6f97", "#e76f51"]
    bars = ax.bar(domains, aucs, color=colors[: len(domains)], width=0.55)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ID AUC")
    ax.set_title(f"Domain AUC — {args.model}")
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig2.tight_layout()
    bar_path = out / f"{args.model}_domain_auc.png"
    fig2.savefig(bar_path)
    plt.close(fig2)

    report = {
        "model": args.model,
        "domain_metrics": dom,
        "midband_energy": mid,
        "midband_delta": {
            "bedroom_fake_minus_real": mid["bedroom_fake"] - mid["bedroom_real"],
            "celebahq_fake_minus_real": mid["celebahq_fake"] - mid["celebahq_real"],
        },
        "figures": {"spectrum": str(spec_path), "domain_auc": str(bar_path)},
    }
    with open(out / f"{args.model}_domain_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
