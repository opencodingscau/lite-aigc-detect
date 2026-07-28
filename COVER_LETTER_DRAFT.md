# SUPERSEDED — use `COVER_LETTER.md`

This draft is kept only for history. Authoritative cover letter: **`COVER_LETTER.md`**.

---

# Cover Letter Draft — Applied Sciences (MDPI)

**Subject:** Submission of manuscript — Lightweight AI-Generated Image Detection under Compression and Cross-Generator Shift

Dear Editors,

Please find enclosed our manuscript entitled:

**“Lightweight AI-Generated Image Detection under Compression and Cross-Generator Shift: A Reproducible Study of CNN, Frequency-Aware, and State-Space Models.”**

This work is an empirical, deployment-oriented study of compact detectors for **still-image** AI-generated content (not deepfake video). Under a locked no-pretraining protocol, we compare CNN baselines, a gated frequency-aware CNN, and two study-specific pure-PyTorch state-space classifiers (**LiteSSM-A** and **LiteSSM-B**), and we report Params/FLOPs, locked batch-1 latency and batch-32 throughput, JPEG Q70 behavior, content-domain splits, and per-generator UniversalFakeDetect results. An unmatched FLUX probe appears only in the appendix and is not used for architecture claims.

We believe the manuscript fits *Applied Sciences* because it prioritizes **reproducible measurement and practical operating-point selection** over claims of state-of-the-art accuracy. Our main recommendation is **LiteSSM-A** as the preferred generalization–efficiency operating point in the training-matched panel (UFD Macro 0.718; locked batch-32 throughput 226 images/s), while we explicitly document remaining boundaries (near-chance DALL·E AUCs; hard bedroom domain; non-universal frequency gains).

All quantitative tables are derived from a frozen prediction package with bootstrap confidence intervals. We clearly distinguish **UFD Macro AUC** (primary cross-generator metric) from **OOD Pooled AUC** to avoid ambiguous generalization claims. Code and frozen artifacts are available at https://github.com/opencodingscau/lite-aigc-detect (release `v1.0.0-paper.3`) and archived at Zenodo (https://doi.org/10.5281/zenodo.21604045).

This manuscript is original and not under consideration elsewhere. The author has approved the submission and declares no conflicts of interest related to this work.

Thank you for considering our submission.

Sincerely,  
Kaihao Chen  
College of Software Engineering, South China Agricultural University, Guangzhou, Guangdong, China  
ORCID: https://orcid.org/0009-0001-4945-6733  
13543148496@163.com

---

## Suggested highlights (optional MDPI box)

1. Protocol-matched comparison of CNN, frequency-aware, and SSM detectors without pretrained backbones.  
2. LiteSSM-A recommended as the preferred compact generalization–efficiency operating point (not lowest absolute latency).  
3. Honest failure boundaries: DALL·E ≈ chance; bedroom domain gap; frequency not universally helpful.  
4. Frozen metrics with bootstrap CIs; UFD Macro vs. OOD Pooled explicitly separated.
