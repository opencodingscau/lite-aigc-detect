# Manifests

Frozen JSONL manifests are released with the archival package (or under `manifests/` after remap).
Raw third-party images are **not** redistributed.

```bash
python scripts/remap_manifest_paths.py \
  --in-dir manifests_raw --out-dir manifests \
  --old-prefix /PREVIOUS/DATASET/ROOT \
  --new-prefix /LOCAL/DATASET/ROOT \
  --check-exists
```
