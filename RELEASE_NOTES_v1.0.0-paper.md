# Release v1.0.0-paper

**Manuscript (working title):** Lightweight AI-Generated Image Detection under Compression and Cross-Generator Shift: A Reproducible Study of CNN, Frequency-Aware, and State-Space Models

**Release commit:** `e275e44` (plus any tip commit updating `CITATION.cff` repository URL before tag)

This release reproduces the **frozen tables and figures** used in the Applied Sciences manuscript draft. It does **not** reproduce model training by default.

## Frozen headline metrics (Panel A)

| Model | UFD Macro | ID AUC | B=1 p50 (ms) | B=32 thr (img/s) |
|-------|----------:|-------:|-------------:|-----------------:|
| **LiteSSM-A** (`mobilemamba_lite`) | **0.718** | 0.946 | **144.4** | **226** |
| LiteSSM-B (`mambapsa_cls`) | 0.700 | 0.945 | 237.6 | 120 |

Canonical numeric source: `freeze/frozen_numbers.json`. Preliminary 367 img/s is superseded and excluded from the manuscript.

## Reproduce tables (no GPU)

```bash
git clone https://github.com/opencodingscau/lite-aigc-detect.git
cd lite-aigc-detect
git checkout v1.0.0-paper
conda env create -f environment.yml
conda activate lite-aigc-detect
python scripts/build_tables.py \
  --freeze-package freeze/freeze_package.json \
  --latency-summary latency_batch1/summary.json \
  --external-summary external_refs/summary.json \
  --output-dir reproduced_tables
python scripts/test_remap_smoke.py
python scripts/verify_sha256sums.py
```

Expected: `[PASS] max_abs_err=0`, Tables 2–5 CSV under `reproduced_tables/`, hash check `ok=15 bad=0`.

## Data & licensing

- **Raw third-party images are not redistributed.** Obtain DiffusionForensics / UniversalFakeDetect (and related) from their original providers under their licenses.
- Frozen **manifests / prediction summaries / hashes** are in this repository (`freeze/`, `latency_batch1/`, `external_refs/`, `hashes/SHA256SUMS`).
- **Panel A checkpoints** (LiteSSM-A/B, LiteFreqNet, compact CNNs): not bundled in git. See `checkpoints/README.md` for registry paths and SHA256. Publish to Zenodo only after confirming training-data terms allow derived weights.
- **Panel B (UnivFD, NPR):** do not rebundle; download from official releases and verify locally. See `checkpoints/README.md`.

## Known limitations

- Primary cross-generator metric is **UFD Macro AUC** (not OOD Pooled).
- DALL·E remains near chance for all Panel A models; bedroom domain is harder than CelebA-HQ.
- FLUX is appendix-only (unmatched subset); not used for architecture claims.
- JPEG claim scope in the manuscript is **Q70**.
- LiteSSM-A/B are study-specific pure-PyTorch classifiers, not official architecture releases.

## Checksums

- Public file surface: `hashes/SHA256SUMS`
- Checkpoint digests (when weights are obtained): `freeze/SHA256_MANIFESTS.json` and `checkpoints/README.md`

## Zenodo

- **Version DOI (cite this):** https://doi.org/10.5281/zenodo.21604045 (`v1.0.0-paper.2`; superseded by `v1.0.0-paper.3`)
- **Concept DOI:** https://doi.org/10.5281/zenodo.21604044

