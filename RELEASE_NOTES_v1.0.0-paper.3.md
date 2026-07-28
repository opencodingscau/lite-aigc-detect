# Release v1.0.0-paper.3

**Manuscript:** Compact AI-Generated Image Detection under Compression and Cross-Generator Shift

This release archives the Applied Sciences submission package after appendix diagnostics/ensembles and narrative freeze updates. It does **not** include the isolated `v2-experiments` training branch.

## What changed vs `v1.0.0-paper.2`

- Appendix D: training dynamics, Cohen's $\kappa$, threshold sweeps (frozen preds)
- Appendix E: Panel A equal-weight probability-mean ensembles (no Macro win) + exploratory M1/M1-v2 specialization table (not a main-table method)
- Discussion: three-layer DALL·E failure evidence (score overlap → error correlation → ensemble ineffective)
- Limitations / Future work: no modern-generator main-table method; pre-register next protocol
- Repro script: `scripts/run_ensemble_panel_a.py` → `docs/ensemble_panel_a.json`

## Frozen headline metrics (unchanged Panel A)

Canonical source remains `freeze/frozen_numbers.json` (LiteSSM-A UFD Macro **0.718**). Ensemble / M1-v2 numbers are appendix-only and do not alter the preferred operating point.

## Reproduce tables (no GPU)

```bash
git clone https://github.com/opencodingscau/lite-aigc-detect.git
cd lite-aigc-detect
git checkout v1.0.0-paper.3
conda env create -f environment.yml
conda activate lite-aigc-detect
python scripts/build_tables.py \
  --freeze-package freeze/freeze_package.json \
  --latency-summary latency_batch1/summary.json \
  --external-summary external_refs/summary.json \
  --output-dir reproduced_tables
python scripts/run_ensemble_panel_a.py   # optional; needs formal/_paper_assets preds locally
python scripts/verify_sha256sums.py
```

## Zenodo

- **Concept DOI (all versions):** https://doi.org/10.5281/zenodo.21604044
- **Version DOI:** back-filled in `CITATION.cff` / Data Availability after this GitHub Release is archived by Zenodo
