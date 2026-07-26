# Seed sensitivity (P1) — completed minimal sweep

Primary manuscript tables remain **single-seed (42)** with bootstrap CIs = test-sample uncertainty only.
Appendix C reports a locked-recipe probe for LiteSSM-A and EfficientNet-B0 on seeds `{42,43,44}`.

## Results (from AutoDL RTX 4090D, 2026-07-27)

Source artifact: `formal/seed_sweep_results/summary.json` (also mirrored under `docs/seed_sweep_summary.json` if synced).

| Model | Metric | s42 | s43 | s44 | Mean±std (population) |
|-------|--------|-----|-----|-----|------------------------|
| LiteSSM-A | ID | 0.946 | 0.939 | 0.939 | 0.941±0.004 |
| LiteSSM-A | UFD Macro | 0.718 | 0.712 | 0.733 | 0.721±0.009 |
| EfficientNet-B0 | ID | 0.929 | 0.923 | 0.930 | 0.927±0.003 |
| EfficientNet-B0 | UFD Macro | 0.667 | 0.674 | 0.624 | 0.655±0.022 |

- LiteSSM-A Macro std ≤0.01 → cite as stability under the locked recipe (appendix only).
- EfficientNet-B0 Macro std ≈0.022 → keep single-seed primary ranking; do not overclaim CNN stability.
- Do **not** merge means into `freeze/frozen_numbers.json` primary rows.

## Protocol (executed)

- Manifests: frozen `3200/400/400/5600` (seed 42 splits).
- Recipe: 15 epochs, batch 64, AdamW lr 1e-4, CosineAnnealingLR.
- Seed 42: reused bake-off / baseline `best.pt`; seeds 43/44: full retrain.
- Metrics: ID test AUC from `metrics.json`; UFD Macro = `ufd_mean_auc` from `eval_by_source.py`.
