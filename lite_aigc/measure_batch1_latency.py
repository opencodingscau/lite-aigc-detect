#!/usr/bin/env python3
"""Measure batch-1 p50/p95 latency for Panel-A compact models (model-only, FP32)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metrics import measure_flops, measure_fps, measure_params
from models import build_model

CKPTS = {
    "shufflenet_v2_x0_5": "checkpoints/shufflenet_v2_x0_5/best.pt",
    "mobilenet_v3_small": "checkpoints/mobilenet_v3_small/best.pt",
    "efficientnet_b0": "checkpoints/efficientnet_b0/best.pt",
    "lite_freq_net_v2": "checkpoints/lite_freq_net_v2/best.pt",
    "mobilemamba_lite": "checkpoints/mobilemamba_lite/best.pt",
    "mambapsa_cls": "checkpoints/mambapsa_cls/best.pt",
}


@torch.no_grad()
def latency_batch1(model, device, size=224, warmup=50, iters=500, dtype=torch.float32):
    model.eval()
    if dtype == torch.float16:
        model = model.half()
    x = torch.randn(1, 3, size, size, device=device, dtype=dtype)
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
        "warmup": warmup,
        "iters": iters,
        "dtype": str(dtype).replace("torch.", ""),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "scope": "model_only_synthetic_tensor_includes_fft_modules",
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/latency_batch1")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    rows = {}
    for name, ckpt in CKPTS.items():
        print(f"=== {name} ===", flush=True)
        model = build_model(name).to(device)
        if Path(ckpt).exists():
            state = torch.load(ckpt, map_location=device)
            # tolerate common wrappers
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        else:
            print(f"WARN missing ckpt {ckpt}; measuring architecture only", flush=True)

        lat = latency_batch1(model, device, warmup=args.warmup, iters=args.iters, dtype=dtype)
        # FPS@32 on same GPU for consistency (existing protocol)
        fps32 = measure_fps(model.float() if dtype == torch.float16 else model, device, batch=32, warmup=20, iters=50)
        params = measure_params(model)
        flops = measure_flops(model.float(), device)
        row = {
            "model": name,
            "ckpt": ckpt,
            "params": params,
            "params_M": round(params / 1e6, 3),
            "flops": flops,
            "flops_G": None if flops is None else round(flops / 1e9, 3),
            "batch1_latency": lat,
            "fps_bs32": fps32,
            "throughput_bs32_img_s": fps32,
        }
        rows[name] = row
        print(json.dumps(row, indent=2), flush=True)
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
