# PDF vs freeze numbers checklist (Gate C)

Date: 2026-07-26  
Authority: `freeze/frozen_numbers.json`  
Scope: `latex/main.tex` + `latex/sections/*.tex` (source audit; PDF re-check after compile)

## Critical numbers

| Claim | Freeze | Manuscript source | Status |
|-------|--------|-------------------|--------|
| LiteSSM-A UFD Macro | 0.718 | Abstract; Results; Table overall/UFD | PASS |
| LiteSSM-B UFD Macro | 0.700 | Results; Discussion | PASS |
| LiteSSM-A ID AUC | 0.946 | Abstract; Tables | PASS |
| LiteSSM-A B1 p50 | 144.4 ms | Abstract; Results; Table | PASS |
| LiteSSM-A B32 thr | 226 img/s | Abstract; Results; Conclusion | PASS |
| LiteSSM-B OOD Pooled | 0.722 | Discussion/Conclusion as **OOD Pooled only** | PASS |
| LiteSSM-A OOD Pooled | 0.712 | Discussion (vs 0.722) | PASS |
| JPEG primary claim | Q70 | Methods; Results §JPEG; Discussion | PASS |
| UnivFD / NPR Macro | 0.948 / 0.976 | Abstract (pretrained panel) | PASS (source text) |

## Forbidden / boundary checks

| Check | Status |
|-------|--------|
| No superseded **367** img/s in manuscript `.tex` | PASS |
| No MobileMamba / MambaPSA / PSA display names | PASS |
| No TODO / Placeholder / TBD in manuscript sections | PASS |
| `0.722` never labeled as UFD Macro | PASS (OOD Pooled only) |
| FLUX scores not used for architecture claims | PASS (intro/methods/results point to Appendix) |
| Registry keys only for repro mapping | PASS (`03_methods.tex`) |

## Compile QC (local equivalent of Overleaf)

Local: MiKTeX 25.12, `pdflatex` ×2 from `latex/` → `main.pdf` (14 pages).

| Check | Status |
|-------|--------|
| `pdflatex` ×2 from `latex/` | PASS |
| No undefined references / citations | PASS (log clean of undef refs) |
| Template noise only | fancyhdr `\headheight` warnings (MDPI default) |
| All three figures present | PASS |
| Bibliography renders; no unused `sd35card` | PASS |

## Post-PDF checks (extracted text + link annotations)

- [x] Abstract / body contain 0.718, 144.4, 226, LiteSSM-A/B
- [x] Table numbers present in PDF text
- [x] No throughput **367** claim (one `367` hit is MDPI line number next to “real”, not FPS)
- [x] `0.722` only as OOD Pooled
- [x] FLUX deferred / appendix
- [x] Author: Kaihao Chen; email 13543148496; ORCID link `https://orcid.org/0009-0001-4945-6733`
- [x] Zenodo DOI links present

Overleaf: upload the same `latex/` tree, set main file `main.tex`, pdfLaTeX ×2 — should match this local build.
