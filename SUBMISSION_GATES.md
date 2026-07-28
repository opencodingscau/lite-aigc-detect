# Submission gates (objective definition)

Aligned with the pre-submission plan: scientific evidence is largely frozen;
remaining work is **auditability**. Do not submit until all boxes below pass.

## Gate A — Public repro package usable by a third party

- [x] Public GitHub tree: https://github.com/opencodingscau/lite-aigc-detect
- [x] Absolute cloud paths sanitized in public JSON
- [x] `LICENSE` (MIT draft)
- [x] `CITATION.cff` (author + Zenodo DOI filled)
- [x] `environment.yml` + `requirements.txt`
- [x] `scripts/remap_manifest_paths.py`
- [x] `scripts/build_tables.py` cold-start rebuild (**PASS**, max_abs_err=0)
- [x] Fresh clone + conda env cold-start PASS (`725acdb`)
- [x] `.gitattributes` LF normalization for hash-stable text files
- [x] `checkpoints/README.md` distribution / license policy (weights not in git)
- [x] `scripts/make_sha256sums.py` → `hashes/SHA256SUMS`
- [x] Tag `v1.0.0-paper` / `v1.0.0-paper.2` / `v1.0.0-paper.3.1` + GitHub Release published
- [x] Zenodo version DOI **10.5281/zenodo.21643045** (concept **10.5281/zenodo.21604044**) back-filled
- [ ] Panel A checkpoints published only if dataset license allows derived weights
- [x] Data Availability contains real GitHub + Zenodo URLs

## Gate B — Method identity verifiable

- [x] Canonical names: **LiteSSM-A** / **LiteSSM-B** (registry mapping once in repro docs)
- [x] Naming follows real implementation (LiteSSM-B = bidirectional SelectiveSSM)
- [x] `docs/model_architectures.md` with stem/block/scan details
- [x] Architecture figure `fig_ssm_architecture.png`
- [x] LaTeX Method § with Table `tab:ssm-arch` + Fig `fig:ssm-arch`
- [x] Abstract / Results / Discussion / Conclusion / cover letter use LiteSSM-A/B
- [x] Efficiency locked to batch-1 session: **144.4 ms** / **226** img/s (367 superseded)
- [ ] Optional code registry aliases `litessm_a` / `litessm_b` (keep old keys for ckpt load)

## Gate C — Zero placeholders, compile, internal consistency

- [x] Real authors / affiliation / email / ORCID (single author: Kaihao Chen; co-authors TBD later)
- [x] Real CRediT contributions / Funding (none) / Acknowledgments
- [x] Removed fake `wang2021deepfake` (2103.XXXX) citation
- [x] Bibliography mechanical check + **literature expansion to 24 refs** (see `docs/BIB_AUDIT.md`; LAID/LOTA positioning table added)
- [x] Softened deployment/real-time wording; LiteSSM-A framed as operating point (not lowest latency)
- [x] Clean compile (local MiKTeX; 15-page PDF after literature rewrite)
- [x] PDF vs freeze numbers checklist (see `docs/PDF_NUMBER_CHECKLIST.md`)
- [x] Cover letter finalized (`COVER_LETTER.md`; single author; LAID/LOTA positioning noted)
- [x] Data Availability cites `v1.0.0-paper.3.1` + Zenodo version DOI `10.5281/zenodo.21643045` (concept DOI unchanged)

**Pre-submit reminder:** re-upload Overleaf `latex/` after this rewrite and spot-check Related Work + Refs.

## Locked efficiency (do not reopen)

| Claim | Locked value | Status |
|-------|-------------:|--------|
| LiteSSM-A B=1 p50 | **144.4 ms** | Manuscript source |
| LiteSSM-A B=1 p95 | **171.5 ms** | Manuscript source |
| LiteSSM-A B=32 thr | **226** img/s | Manuscript source |
| Preliminary 367 img/s | — | **Superseded**; provenance only in `frozen_numbers.json` |

## Suggested execution order

1. ~~`.gitattributes` LF fix~~ → next: license-clear Panel A weights decision.
2. Release candidate re-verify → tag `v1.0.0-paper`.
3. Zenodo DOI → back-fill.
4. Gate C author metadata + bib + Overleaf.
