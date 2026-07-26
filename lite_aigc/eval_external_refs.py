#!/usr/bin/env python3
"""Eval UnivFD + NPR on frozen manifests (inference-only, native preprocess).

Outputs JSON under outputs/external_refs/ (override with --out).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ManifestDataset(Dataset):
    def __init__(self, jsonl_path: Path, transform):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        # NPR: ensure even spatial dims after resize/crop
        x = self.transform(img)
        if x.shape[-1] % 2 == 1:
            x = x[..., :-1]
        if x.shape[-2] % 2 == 1:
            x = x[..., :-1, :]
        return x, int(r["label"]), r.get("source", "unk"), r["path"]


def auc_safe(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def summarize(probs, labels, sources):
    by = defaultdict(lambda: {"y": [], "p": []})
    for p, y, s in zip(probs, labels, sources):
        by[s]["y"].append(y)
        by[s]["p"].append(p)
    per = {}
    for s, b in sorted(by.items()):
        per[s] = {"n": len(b["y"]), "auc": auc_safe(b["y"], b["p"])}
    # UFD generators: sources like ufd_dalle, ufd_guided, ...
    ufd_keys = [k for k in per if k.startswith("ufd_")]
    ufd_aucs = [per[k]["auc"] for k in ufd_keys if per[k]["auc"] is not None]
    macro = float(np.mean(ufd_aucs)) if ufd_aucs else None
    worst = None
    worst_k = None
    if ufd_aucs:
        worst = float(min(ufd_aucs))
        worst_k = ufd_keys[int(np.argmin([per[k]["auc"] if per[k]["auc"] is not None else 9 for k in ufd_keys]))]
    pooled = auc_safe(labels, probs)
    return {
        "n": len(labels),
        "pooled_auc": pooled,
        "ufd_macro_auc": macro,
        "worst_generator": worst_k,
        "worst_auc": worst,
        "per_source": per,
    }


@torch.no_grad()
def collect_probs(model, loader, device, score_fn):
    model.eval()
    probs, labels, sources, paths = [], [], [], []
    for x, y, src, path in loader:
        x = x.to(device)
        s = score_fn(model, x)
        probs.extend(s.detach().cpu().tolist())
        labels.extend(y.tolist())
        sources.extend(list(src))
        paths.extend(list(path))
    return probs, labels, sources, paths


def domain_from_id(probs, labels, sources):
    # ID sources typically contain bedroom / celebahq
    buckets = {"bedroom": {"y": [], "p": []}, "celebahq": {"y": [], "p": []}}
    for p, y, s in zip(probs, labels, sources):
        sl = s.lower()
        if "bedroom" in sl:
            buckets["bedroom"]["y"].append(y)
            buckets["bedroom"]["p"].append(p)
        if "celeba" in sl or "celebahq" in sl:
            buckets["celebahq"]["y"].append(y)
            buckets["celebahq"]["p"].append(p)
    return {k: {"n": len(v["y"]), "auc": auc_safe(v["y"], v["p"])} for k, v in buckets.items()}


def load_univfd(device, ext_root: Path):
    repo = ext_root / "UniversalFakeDetect"
    sys.path.insert(0, str(repo))
    from models import get_model  # noqa: WPS433

    model = get_model("CLIP:ViT-L/14")
    ckpt = repo / "pretrained_weights" / "fc_weights.pth"
    state = torch.load(ckpt, map_location="cpu")
    model.fc.load_state_dict(state)
    model.eval().to(device)
    # Official validate.py: CenterCrop(224)+CLIP norm.
    # Add min-side pad/resize safeguard so smaller images do not crash CenterCrop.
    def _ensure_min(img: Image.Image, min_side: int = 224) -> Image.Image:
        w, h = img.size
        if w >= min_side and h >= min_side:
            return img
        scale = min_side / float(min(w, h))
        nw, nh = max(min_side, int(round(w * scale))), max(min_side, int(round(h * scale)))
        return img.resize((nw, nh), Image.BICUBIC)

    tf = T.Compose(
        [
            T.Lambda(lambda im: _ensure_min(im, 224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )

    def score_fn(m, x):
        return m(x).float().sigmoid().flatten()

    n_params = sum(p.numel() for p in model.parameters())
    return (
        model,
        tf,
        score_fn,
        {
            "ckpt": str(ckpt),
            "params": n_params,
            "preprocess": "official_centercrop224+CLIP_norm (min-side pad if needed)",
        },
    )


def load_npr(device, ext_root: Path):
    repo = ext_root / "NPR-DeepfakeDetection"
    sys.path.insert(0, str(repo))
    from networks.resnet import resnet50  # noqa: WPS433

    model = resnet50(num_classes=1)
    # Prefer official NPR.pth (repo root); fallback to ProGAN-4class
    ckpt = repo / "NPR.pth"
    if not ckpt.exists():
        ckpt = repo / "model_epoch_last_3090.pth"
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strip optional module. prefix
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    # NPR UniversalFakeDetect setting: no_resize=False, no_crop=True
    # base_options defaults: loadSize=256, cropSize=224 → Resize(256,256), no crop
    tf = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    def score_fn(m, x):
        return m(x).sigmoid().flatten()

    n_params = sum(p.numel() for p in model.parameters())
    return model, tf, score_fn, {"ckpt": str(ckpt), "params": n_params, "preprocess": "resize224_no_crop+ImageNet_norm+NPR"}


@torch.no_grad()
def measure_latency(model, device, size=224, warmup=50, iters=500, dtype=torch.float32):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device, dtype=dtype)
    if dtype == torch.float16:
        model = model.half()
        x = x.half()
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "batch": 1,
        "input_size": size,
        "warmup": warmup,
        "iters": iters,
        "dtype": str(dtype).replace("torch.", ""),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "scope": "model_only_synthetic_tensor",
    }


def eval_detector(name, load_fn, manifests, batch, device, out_root: Path, ext_root: Path):
    print(f"=== Loading {name} ===", flush=True)
    model, tf, score_fn, meta = load_fn(device, ext_root)
    report = {"detector": name, "meta": meta, "splits": {}}
    for split_name, path in manifests.items():
        print(f"[{name}] eval {split_name}: {path}", flush=True)
        ds = ManifestDataset(path, tf)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
        probs, labels, sources, paths = collect_probs(model, loader, device, score_fn)
        summ = summarize(probs, labels, sources)
        if split_name == "id_test":
            summ["domain"] = domain_from_id(probs, labels, sources)
            summ["id_auc"] = summ["pooled_auc"]
        report["splits"][split_name] = summ
        # save predictions
        pred_path = out_root / f"{name}_{split_name}_preds.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for pth, y, s, pr in zip(paths, labels, sources, probs):
                f.write(json.dumps({"path": pth, "label": y, "source": s, "prob": pr}) + "\n")
        report["splits"][split_name]["preds"] = str(pred_path)
        print(json.dumps({k: summ[k] for k in summ if k != "per_source"}, indent=2), flush=True)

    print(f"[{name}] measuring batch-1 latency...", flush=True)
    lat_size = 256 if name == "npr" else 224
    report["latency_batch1"] = measure_latency(model, device, size=lat_size)
    print(report["latency_batch1"], flush=True)
    out = out_root / f"{name}_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-root", default="external")
    ap.add_argument("--out", default="outputs/external_refs")
    ap.add_argument("--manifest-root", default="manifests")
    ap.add_argument("--ufd-eval", default="outputs/ood_by_source/ufd_eval.jsonl")
    ap.add_argument("--detectors", default="univfd,npr")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    ext_root = Path(args.ext_root)
    out_root = Path(args.out)
    manifest_root = Path(args.manifest_root)
    ufd_eval = Path(args.ufd_eval)

    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = {
        "id_test": manifest_root / "test.jsonl",
        "ufd_eval": ufd_eval,
    }
    for p in manifests.values():
        assert p.exists(), p

    loaders = {
        "univfd": load_univfd,
        "npr": load_npr,
    }
    summary = {}
    for name in [x.strip() for x in args.detectors.split(",") if x.strip()]:
        # isolate sys.path pollution between detectors
        sys.path = [p for p in sys.path if "UniversalFakeDetect" not in p and "NPR-DeepfakeDetection" not in p]
        summary[name] = eval_detector(name, loaders[name], manifests, args.batch, device, out_root, ext_root)

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
