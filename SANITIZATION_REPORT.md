# Sanitization report

Built from: `E:\sciencecre\aigc_datasets\formal`
Output: `E:\sciencecre\aigc_datasets\lite-aigc-detect`

## Excluded from public tree

- Root SSH/HF helpers: `_*.py`, `launch_*.py`, `*_remote.py`, `deploy_and_run.py`, …
- Raw images, `*.pt` / `*.pth` weights, `.env`, HF token files
- Working-only experiment scratch dirs not in the allowlist

## Path rewriting

- Former absolute cloud paths → portable relative keys (`checkpoints/...`, `manifests/...`, `outputs/...`)
- Hashes in `freeze/SHA256_MANIFESTS.json` preserved; **keys** rewritten
- Checkpoint **files** not renamed (only path strings in JSON)

## Residual scan hits (review)

Allowed mentions: documentation about excluding AutoDL helpers; literal patterns inside this builder if present.

(none requiring action)

## Cold-start check

```bash
cd lite-aigc-detect
python scripts/build_tables.py \
  --freeze-package freeze/freeze_package.json \
  --latency-summary latency_batch1/summary.json \
  --external-summary external_refs/summary.json \
  --output-dir reproduced_tables
```
Expected: `[PASS]`.
