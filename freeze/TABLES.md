# Frozen Tables (canonical display names)

> Source of truth for manuscript numbers: `freeze/frozen_numbers.json`.
> Primary OOD = **UFD Macro AUC**. OOD Pooled ≠ UFD Macro.
> Bootstrap B=1000, seed=42 for Panel A.
> Panel B = inference-only public checkpoints; **not** training-matched.
> Registry keys (`mobilemamba_lite` / `mambapsa_cls`) remain on disk; display = LiteSSM-A/B.

## Table 2. Overall Performance and Deployment Trade-Offs

### Panel A. Controlled lightweight models (frozen from-scratch protocol)

| Model | Arch | Params(M) | FLOPs(G) | B=1 p50 (ms) | B=1 p95 (ms) | B=32 thr (img/s) | ID AUC | UFD Macro | Worst-Gen |
|-------|------|----------:|---------:|-------------:|-------------:|-----------------:|-------:|----------:|----------:|
| LiteSSM-A | SSM | 1.74 | 0.705 | 144.4 | 171.5 | 226 | 0.946 | **0.718** | 0.504 |
| LiteSSM-B | SSM | 2.48 | 0.853 | 237.6 | 264.6 | 120 | 0.945 | 0.700 | 0.483 |
| EfficientNet-B0 | CNN | 4.01 | 0.414 | 6.64 | 8.95 | 4338 | 0.929 | 0.667 | 0.519 |
| LiteFreqNet v2 | CNN+FFT | 1.13 | 0.180 | 6.15 | 8.35 | 4747 | 0.906 | 0.645 | 0.451 |
| MobileNetV3-S | CNN | 1.52 | 0.061 | 4.30 | 5.61 | 7231 | 0.890 | 0.636 | 0.455 |
| ShuffleNet-x0.5 | CNN | 0.344 | 0.044 | 6.21 | 7.47 | 4817 | 0.880 | 0.674 | 0.467 |

Table note: RTX 4090D; PyTorch 2.3 / CUDA 12.1; FP32; warmup≥50 / 500 timed iters; `torch.cuda.synchronize()`; latency excludes image I/O and preprocess; throughput is whole-batch images/s at B=32 from the **same locked session**. Preliminary 367 img/s is superseded and not reported.

### Panel B. External pretrained reference detectors

| Detector | Pretrained | Params(M) | Batch-1 p50 (ms) | ID AUC | Bedroom | CelebA-HQ | UFD Macro | Worst-Gen |
|----------|------------|----------:|-----------------:|-------:|--------:|----------:|----------:|----------:|
| UnivFD (CLIP ViT-L/14) | CLIP + linear | 427.6 | 11.86 | 0.608 | 0.821 | 0.420 | **0.948** | 0.861 (guided) |
| NPR | Official NPR.pth | 1.44 | 2.89 | 0.947 | 0.995 | 0.980 | **0.976** | 0.837 (guided) |

### Panel B per-generator UFD AUC

| Detector | DALL·E | G10 | G27 | G50 | Guided | L100 | L200 | Lcfg | Macro |
|----------|------:|----:|----:|----:|-------:|-----:|-----:|-----:|------:|
| UnivFD | 0.969 | 0.942 | 0.956 | 0.962 | 0.861 | 0.989 | 0.992 | 0.916 | 0.948 |
| NPR | 0.990 | 0.995 | 0.995 | 0.996 | 0.837 | 0.999 | 0.998 | 0.999 | 0.976 |

## Narrative (locked)

> LiteSSM-A achieved the highest UFD Macro AUC among the controlled lightweight models while retaining a batch-32 throughput of 226 images/s. It is therefore selected as the preferred generalization–efficiency operating point, rather than the model with the lowest absolute latency.

## Experiment freeze status

Experimental work is permanently frozen. Remaining = writing / public repro / Gate C metadata.
