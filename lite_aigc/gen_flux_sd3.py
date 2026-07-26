#!/usr/bin/env python3
"""Generate Flux.1-schnell and/or SD3.5-medium fake images for E8 small table."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image


PROMPTS = [
    "a realistic photo of a bedroom interior, natural lighting",
    "a photorealistic portrait of a young adult, studio lighting",
    "a realistic photo of a living room with a sofa and window",
    "a candid street photo of a person walking, daylight",
    "a realistic close-up photo of food on a wooden table",
    "a photorealistic landscape with mountains and a lake",
    "a realistic photo of an office desk with a laptop",
    "a photorealistic image of a cat sitting on a couch",
    "a realistic photo of a modern kitchen interior",
    "a photorealistic headshot of a middle-aged person",
]


def save_grid_meta(out_dir: Path, rows: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gen_flux(out_dir: Path, n: int, seed: int, steps: int = 4):
    from diffusers import FluxPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()
    rows = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        g = torch.Generator(device="cuda").manual_seed(seed + i)
        img = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=g,
            max_sequence_length=256,
        ).images[0]
        path = out_dir / f"flux_{i:04d}.png"
        img.save(path)
        rows.append({"path": str(path), "label": 1, "source": "flux_schnell", "prompt": prompt})
        print(f"[flux] {i+1}/{n} -> {path}", flush=True)
    save_grid_meta(out_dir, rows)
    del pipe
    torch.cuda.empty_cache()
    return rows


def gen_sd35(out_dir: Path, n: int, seed: int, steps: int = 28):
    from diffusers import StableDiffusion3Pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=dtype,
    )
    pipe = pipe.to("cuda")
    rows = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        g = torch.Generator(device="cuda").manual_seed(seed + i)
        img = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=4.5,
            generator=g,
        ).images[0]
        path = out_dir / f"sd35_{i:04d}.png"
        img.save(path)
        rows.append({"path": str(path), "label": 1, "source": "sd35_medium", "prompt": prompt})
        print(f"[sd35] {i+1}/{n} -> {path}", flush=True)
    save_grid_meta(out_dir, rows)
    del pipe
    torch.cuda.empty_cache()
    return rows


def sample_reals(data_root: Path, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    for rel in [
        "DiffusionForensics/sdv2/celebahq/0_real",
        "DiffusionForensics/adm/bedroom/0_real",
    ]:
        d = data_root / rel
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    candidates.append(p)
    rng.shuffle(candidates)
    rows = []
    for i, p in enumerate(candidates[:n]):
        src = "real_celebahq" if "celebahq" in str(p) else "real_bedroom"
        rows.append({"path": str(p), "label": 0, "source": src})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/flux_sd3_eval")
    ap.add_argument(
        "--data-root",
        default=os.environ.get("LITE_AIGC_DATA_ROOT", os.environ.get("LITE_AIGC_ROOT", ".")),
        help="Dataset root (override via LITE_AIGC_DATA_ROOT or LITE_AIGC_ROOT)",
    )
    ap.add_argument("--n-fake-each", type=int, default=150)
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backends", default="flux,sd35", help="comma: flux,sd35")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    fake_rows = []
    errors = []

    if "flux" in backends:
        try:
            fake_rows.extend(gen_flux(out / "flux", args.n_fake_each, args.seed))
        except Exception as e:  # noqa: BLE001
            errors.append(f"flux:{type(e).__name__}:{e}")
            print("FLUX failed:", e, flush=True)

    if "sd35" in backends:
        try:
            fake_rows.extend(gen_sd35(out / "sd35", args.n_fake_each, args.seed + 1000))
        except Exception as e:  # noqa: BLE001
            errors.append(f"sd35:{type(e).__name__}:{e}")
            print("SD3.5 failed:", e, flush=True)

    real_rows = sample_reals(Path(args.data_root), args.n_real, args.seed)
    # balance: if fewer fakes, trim reals
    n_fake = len(fake_rows)
    if n_fake == 0:
        raise SystemExit("No fake images generated: " + " | ".join(errors))
    real_rows = real_rows[:n_fake]

    all_rows = real_rows + fake_rows
    man = out / "eval.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps({k: r[k] for k in ("path", "label", "source")}, ensure_ascii=False) + "\n")

    summary = {
        "n_real": len(real_rows),
        "n_fake": n_fake,
        "sources": sorted({r["source"] for r in all_rows}),
        "errors": errors,
        "manifest": str(man),
    }
    with open(out / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
