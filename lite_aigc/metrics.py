"""Efficiency metrics: params / FLOPs / FPS."""
from __future__ import annotations

import time

import torch


def measure_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, device, size=224):
    try:
        from thop import profile

        model.eval()
        x = torch.randn(1, 3, size, size, device=device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return float(flops)
    except Exception as e:  # noqa: BLE001
        return None


@torch.no_grad()
def measure_fps(model, device, size=224, batch=32, warmup=20, iters=50) -> float:
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return (batch * iters) / max(dt, 1e-8)
