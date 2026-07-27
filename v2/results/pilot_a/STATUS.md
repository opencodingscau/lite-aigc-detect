# Pilot A — Wave-1 results (after reconnect)

**Status:** Wave-1 training completed before disconnect; UFD Macro eval completed 2026-07-27 after reconnect.

## Protocol
- Manifests: preflight bake-off (`train/val/test/test_ood`)
- Recipe: 15 epochs, bs64, AdamW 1e-4, seed 42, from-scratch
- Outputs: `/root/autodl-tmp/v2_exp/outputs/pilot_a/` (remote); local copies under this folder

## Summary table

| Model | Params (M) | ID test AUC | OOD pooled | **UFD Macro** | Worst gen | DALL·E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LiteSSM-A (`mobilemamba_lite`, ref) | 1.74* | 0.946* | 0.712* | **0.718** | 0.504 | 0.504 |
| RepViT-M0.9 | 4.72 | 0.915 | 0.667 | 0.697 | 0.469 | 0.469 |
| EfficientNetV2-S | 20.18 | 0.921 | 0.704 | 0.695 | **0.619** | **0.619** |
| MambaOut-proxy (ConvNeXt-T) | 27.82 | 0.905 | 0.617 | 0.614 | 0.471 | 0.472 |

\* Paper freeze numbers for LiteSSM-A ID/OOD/Params; Macro/Worst/DALL·E re-measured here on same UFD eval path.

## Decision signal (for two-week pilots)
- **Backbone swap alone does not beat LiteSSM-A Macro** (best challenger RepViT 0.697 vs 0.718).
- EfficientNetV2-S has **best DALL·E / Worst** among wave-1 (0.619) → useful distillation **student** candidate.
- Prefer next: **Pilot B gated distillation** (NPR / UnivFD / dual) with students LiteSSM-A + RepViT + EffV2-S.

## Files
- `*_metrics.json` — train/ID/OOD pooled
- `*_ood_by_source.json` — UFD Macro + per-subset
- `wave1_summary.json` — compact table

## Still TODO (Wave-2)
- Vendor EfficientViM / official MambaOut / EfficientVMamba / SHViT
- Batch-1 latency under same harness as paper
