# Bibliography audit (Gate C / literature expansion)

Date: 2026-07-28
Source: `latex/main.tex` `thebibliography`  
Target: **34** entries (was 11 → 24 → 26 → 34)

## Added for positioning (2024–2026)

| Key | Role | Venue |
|-----|------|-------|
| `zhu2023genimage` | Million-scale benchmark | NeurIPS 2023 / arXiv DOI |
| `liu2024fatformer` | Pretrained forgery-aware adaption | CVPR 2024 pp. 10770–10780 |
| `zhang2025vib` | VIB universal detection | CVPR 2025 pp. 23828–23837 |
| `cheng2025cospy` | Semantic+pixel fusion | CVPR 2025 pp. 13455–13465 |
| `zhong2025beyond` | Diffusion-as-denoiser features | CVPR 2025 pp. 8258–8268 |
| `chivaran2025laid` | Closest lightweight spatial/spectral benchmark | arXiv:2507.05162 |
| `wang2025lota` | Bit-plane efficient detector | ICCV 2025 pp. 17246–17255 |
| `zhu2024vim` | Bidirectional SSM vision backbone (LiteSSM-B inspiration) | ICML 2024 / PMLR 235 |
| `cinar2025pvism` | Related *Appl. Sci.* ViT detector | Appl. Sci. 15, 6429 |
| `lokner2024lightweight` | Related MDPI lightweight CNN | AI 5, 1575–1593 |
| `man2026mdp` | Related *Appl. Sci.* multi-domain Transformer | Appl. Sci. 16, 533 |
| `durall2020watch` | Spectral fingerprint classic | CVPR 2020 pp. 7890–7899 |
| `guillaro2025bfree` | Bias-controlled training pairs | CVPR 2025 pp. 18685–18694 |
| `liu2024vmamba` | Visual SSM backbone context | NeurIPS 2024 |

## Added for data provenance, metric definitions, and reconstruction context

| Key | Role | Venue |
|-----|------|-------|
| `ricker2024aeroblade` | Training-free latent-diffusion reconstruction context | CVPR 2024 pp. 9130--9140; arXiv DOI |
| `karras2018progan` | CelebA-HQ provenance | ICLR 2018 |
| `dhariwal2021adm` | ADM / guided-diffusion provenance | NeurIPS 2021 |
| `rombach2022ldm` | Latent Diffusion / Stable Diffusion provenance | CVPR 2022 pp. 10684--10695 |
| `radford2021clip` | CLIP provenance for UnivFD reference panel | ICML 2021 pp. 8748--8763 |
| `youden1950` | Validation threshold definition | *Cancer* 3, 32--35 |
| `cohen1960` | Cohen's $\kappa$ agreement statistic | *Educational and Psychological Measurement* 20, 37--46 |
| `kuncheva2003` | Ensemble diversity / double-fault context | *Machine Learning* 51, 181--207 |

Verified DOI backfills: `zhang2025vib`, `cheng2025cospy`, `zhong2025beyond`, `wang2025lota`, `guillaro2025bfree`, and `liu2024vmamba` (arXiv DOI; NeurIPS proceedings DOI not provided by the official page).

Target count: **34** `\bibitem` entries.


EfficientNet, MobileNetV3, ShuffleNetV2, UnivFD, CNNSpot, Frank freq., Mamba, NPR, DIRE, Corvi, FLUX.

## Notes

- No fake `XXXX` DOIs.
- MDPI cites are **topic-relevant** (2 *Appl. Sci.* + 1 *AI*), not citation stacking.
- LAID/LOTA are discussed with an explicit positioning table; scores are not merged across protocols.
