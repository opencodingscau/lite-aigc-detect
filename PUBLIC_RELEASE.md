# Working-tree vs public tree

| Tree | Path | Purpose |
|------|------|---------|
| Working (private ops OK) | `E:\sciencecre\aigc_datasets\formal\` | Experiments, SSH helpers, full history |
| **Public release** | `E:\sciencecre\aigc_datasets\lite-aigc-detect\` | GitHub/Zenodo candidate |

Rebuild public tree:

```bash
cd formal
python scripts/build_public_release.py --out ../lite-aigc-detect --clean
cd ../lite-aigc-detect
python scripts/build_tables.py \
  --freeze-package freeze/freeze_package.json \
  --latency-summary latency_batch1/summary.json \
  --external-summary external_refs/summary.json \
  --output-dir reproduced_tables
```

Expected: `[PASS] max_abs_err=0`.

Do **not** push `formal/` with `launch_*.py` / `_ssh_*.py` / `.env`.
Push only `lite-aigc-detect/` after a final residual scan.
