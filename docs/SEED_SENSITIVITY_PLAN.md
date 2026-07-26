# Seed sensitivity (P1) — protocol only until GPU run

Local machine has **no CUDA**; AutoDL password env `AUTODL_PASS` was not set in this session.
Do **not** invent multi-seed numbers. Main manuscript remains single-seed (seed 42) with bootstrap CIs = test-sample uncertainty only.

## Minimal planned sweep (when GPU available)

| Model | Seeds | Metrics |
|-------|-------|---------|
| LiteSSM-A (`mobilemamba_lite`) | 42, 43, 44 | ID AUC, UFD Macro |
| EfficientNet-B0 | 42, 43, 44 | ID AUC, UFD Macro |

- Same frozen manifests / recipe as main Panel A.
- Report mean±std in **Appendix only**; do not replace primary tables.
- If Macro std is small (e.g. ≤0.01), cite as stability evidence under the locked recipe.
- If large, keep single-seed limitation explicit.

## Remote command sketch

```bash
# on training host, from repo root with data mounted
for s in 42 43 44; do
  python -m lite_aigc.train --model mobilemamba_lite --seed $s --out outputs/seed_sweep/mobilemamba_lite_s$s
  python -m lite_aigc.train --model efficientnet_b0 --seed $s --out outputs/seed_sweep/efficientnet_b0_s$s
done
```

Exact flags must match the locked bake-off config used for `v1.0.0-paper.2`. After runs, add Appendix Table A3 and update `docs/BIB_AUDIT.md` / gates — do not silently merge into freeze primary numbers.
