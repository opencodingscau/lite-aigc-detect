# Lite AIGC Detect

Protocol-locked comparison of **compact CNN**, **frequency-aware**, and **study-specific SSM** detectors for still-image real/fake classification.

**LiteSSM-A** denotes the compact baseline SSM classifier, whereas **LiteSSM-B** extends it with a BiViM-inspired bidirectional SelectiveSSM design. Both are study-specific pure-PyTorch implementations rather than official reproductions of an existing architecture.

## Name mapping (reproducibility only)

| Frozen experiment internal name | Paper formal name |
|---------------------------------|-------------------|
| `mobilemamba_lite` | **LiteSSM-A** |
| `mambapsa_cls` | **LiteSSM-B** |

Checkpoint filenames are **not** renamed. Display names live in `freeze/frozen_numbers.json`. See `docs/model_architectures.md` and `checkpoints/README.md`.

This repository is **not** a claim of universal / SOTA detection across heterogeneous training protocols.

## Cold-start table rebuild (no GPU)

```bash
# from repository root
python scripts/build_tables.py \
  --freeze-package freeze/freeze_package.json \
  --latency-summary latency_batch1/summary.json \
  --external-summary external_refs/summary.json \
  --output-dir reproduced_tables
```

Expected: `[PASS]` and `reproduced_tables/verification_report.json` with `all_pass: true`.

Remap manifests after relocating datasets:

```bash
python scripts/remap_manifest_paths.py \
  --in-dir manifests_raw --out-dir manifests \
  --old-prefix /PREVIOUS/DATASET/ROOT \
  --new-prefix /LOCAL/DATASET/ROOT \
  --check-exists
```

Smoke test:

```bash
python scripts/test_remap_smoke.py
```

Verify public file hashes (LF bytes as stored in git; see `.gitattributes`):

```bash
python scripts/verify_sha256sums.py

# Linux / macOS / Git Bash (after a correct LF checkout)
sha256sum -c hashes/SHA256SUMS

# regenerate the surface after intentional public-file edits
python scripts/make_sha256sums.py
```

`hashes/SHA256SUMS` records SHA256 of **repository LF bytes**. `scripts/verify_sha256sums.py` checks against `git` blobs (or LF-normalized working copies), so Windows CRLF checkouts do not produce false failures. A clean checkout with `.gitattributes` should also satisfy `sha256sum -c` without extra normalization.

## Locked efficiency (manuscript source)

| Model | B=1 p50 (ms) | B=32 thr (img/s) | UFD Macro |
|-------|-------------:|-----------------:|----------:|
| LiteSSM-A | 144.4 | 226 | 0.718 |
| LiteSSM-B | 237.6 | 120 | 0.700 |

Source: `freeze/frozen_numbers.json` (one RTX 4090D FP32 session). A preliminary 367 img/s figure is superseded and excluded from the manuscript.

## What this repo provides

| Artifact | Status |
|----------|--------|
| Training / eval code (`lite_aigc/`) | included |
| Architecture docs + figure | `docs/` |
| Locked freeze JSON + table rebuild | `freeze/`, `scripts/build_tables.py` |
| Panel A/B summaries | `latency_batch1/`, `external_refs/` |
| SHA256 surface | `hashes/SHA256SUMS` |
| Checkpoint layout + hashes | `checkpoints/README.md` (weights via release if license-cleared) |
| Raw third-party images | **not redistributed** |
| Zenodo DOI | **pending** — do not invent placeholders |

## Protocol (frozen)

- **Train sources:** DiffusionForensics ADM bedroom + SDv2 CelebA-HQ
- **Splits:** train 3200 / val 400 / test 400 / OOD 5600, seed `42`
- **Recipe:** 224², AdamW `1e-4`, CosineAnnealing, 15 epochs, `pretrained=false`, select by val AUC
- **Primary OOD metric:** **UFD Macro AUC**
- **JPEG claim scope:** evaluated Q70 only

## Citation

See `CITATION.cff`. Fill author ORCID and Zenodo DOI **after** the archival release exists.

## License

MIT for code in this repository (`LICENSE`). Upstream datasets and Panel B detectors keep their original licenses.

## Submission checklist

See `SUBMISSION_GATES.md` (Gates A/B/C).
