# Cover Letter — Applied Sciences (MDPI)

**Subject:** Submission of manuscript — Compact AI-Generated Image Detection under Compression and Cross-Generator Shift

Dear Editors,

Please find enclosed the manuscript entitled:

**“Compact AI-Generated Image Detection under Compression and Cross-Generator Shift: A Reproducible Study of CNN, Frequency-Aware, and State-Space Models.”**

This work is a controlled, low-data empirical study of **compact** detectors for still-image AI-generated content (not deepfake video). Under a locked no-pretraining protocol, it compares CNN baselines, a gated frequency-aware CNN, and two study-specific pure-PyTorch state-space classifiers (**LiteSSM-A** and **LiteSSM-B**), and reports Params/FLOPs, locked batch-1 latency and batch-32 throughput, JPEG Q70 behavior, content-domain splits, and per-generator UniversalFakeDetect results. An unmatched FLUX probe appears only in the appendix and is not used for architecture claims.

The manuscript fits *Applied Sciences* because it prioritizes reproducible measurement and practical **operating-point selection** over claims of state-of-the-art accuracy. Relative to closely related lightweight benchmarks such as LAID and LOTA, the contribution is protocol-level: no Panel A pretraining, UFD Macro plus worst-generator reporting, CelebA-HQ/bedroom decomposition, and measured latency/throughput under one disclosed recipe. The main recommendation is **LiteSSM-A** as the preferred generalization–efficiency operating point in the training-matched panel (UFD Macro 0.718; locked batch-32 throughput 226 images/s), explicitly **not** as the lowest-latency or real-time model (batch-1 p50 144.4 ms). Remaining boundaries are documented (near-chance DALL·E AUCs for Panel A, while pretrained Panel B references remain strong; hard bedroom domain; non-universal frequency gains).

All quantitative tables are derived from a frozen prediction package with bootstrap confidence intervals. **UFD Macro AUC** (primary cross-generator metric) is distinguished from **OOD Pooled AUC**. Code and frozen artifacts are available at https://github.com/opencodingscau/lite-aigc-detect (release `v1.0.0-paper.2`) and archived at Zenodo (https://doi.org/10.5281/zenodo.21604045).

This manuscript is original and not under consideration elsewhere. The author has approved the submission and declares no conflicts of interest related to this work.

Thank you for considering this submission.

Sincerely,  
Kaihao Chen  
College of Software Engineering, South China Agricultural University, Guangzhou, Guangdong, China  
ORCID: https://orcid.org/0009-0001-4945-6733  
13543148496@163.com

---

## Suggested highlights (optional MDPI box)

1. Protocol-matched comparison of CNN, frequency-aware, and SSM detectors without pretrained backbones.  
2. Explicit positioning against LAID/LOTA and modern pretrained detectors without unfair score merging.  
3. LiteSSM-A recommended as the preferred compact generalization–efficiency operating point (not lowest absolute latency).  
4. Honest failure boundaries: Panel A DALL·E ≈ chance; bedroom domain gap; frequency not universally helpful.

## Graphical abstract (optional upload)

File: `latex/figures/fig_graphical_abstract.png` (also under `freeze/figures/`).
Summarizes locked protocol, operating-point plot, and main takeaway/boundaries for the MDPI graphical-abstract slot.
