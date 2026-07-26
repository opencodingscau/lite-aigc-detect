# Reproducibility package checklist (pre-submission)

## Public release targets
- GitHub repository (code + configs + table scripts)
- Zenodo (or equivalent) archival DOI matching a release tag
- README with one-command train / eval / bootstrap / table rebuild

## Must include
- [ ] Training code (`lite_aigc/train.py` and model defs)
- [ ] Inference / eval code (`eval_by_source.py`, `eval_jpeg.py`, `eval_external_refs.py`, …)
- [ ] Bootstrap + table generation (`freeze_package_remote.py` / freeze scripts)
- [ ] Model configs and locked hyperparameters
- [ ] `requirements.txt` / environment lock
- [ ] License statement
- [ ] SHA256 of:
  - train / val / test / test_ood manifests
  - UFD eval manifest
  - JPEG eval inputs (if separate)
  - each reported checkpoint
  - frozen prediction CSV/JSONL shards

## Data Availability wording (use in manuscript)
See `latex/main.tex` `\dataavailability{...}` — do **not** write “available upon request” alone.

## After UnivFD+NPR and batch-1 latency finish
1. Merge Panel B numbers into Table 2 (two panels). ✅
2. Replace Figure 2 x-axis with batch-1 latency (ms/image). ✅ (`plot_figures.py`)
3. Re-freeze experiment package permanently. ✅ (no further model training)
4. Publish repo + DOI; paste URLs into Data Availability.
5. Strip remaining author/ORCID placeholders; clean Overleaf compile.

Figures: `freeze/figures/fig2_pareto_ufd_macro.png`, `fig3_generator_heatmap.png` (also copied to `latex/figures/`).
README skeleton: `README.md`, `requirements.txt`, `REPO_LAYOUT.md`.
