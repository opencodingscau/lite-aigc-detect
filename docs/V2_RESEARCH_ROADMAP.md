# V2 / Follow-up Research Roadmap

**Status:** planning only. Does **not** modify the submitted paper freeze.

**Firewall (non-negotiable):**
- Keep `v1.0.0-paper.2` / Zenodo `10.5281/zenodo.21604045` numbers untouched.
- Do **not** edit `freeze/frozen_numbers.json`, paper manifests, or primary tables for new work.
- All v2 work: **new git branch**, **new manifests**, **new freeze version** (e.g. `freeze_v2/`), separate prediction shards.
- Never distill or train on current UFD / DALL·E **test** samples (even unlabeled).

---

## Priority order (adopted)

> Bias-controlled data protocol ＞ multi-cue gated distillation ＞ open-world generator updating ＞ stronger compact backbones + inference engineering.

Do **not** prioritize “find a better Mamba variant” as the lead story. Current evidence suggests bottlenecks are data bias, generator coverage, cue type, and pretraining—not backbone capacity alone.

| Rank | Track | Why |
| ---: | --- | --- |
| 1 | Matched, debiased modern-generator protocol | Fixes external validity + DALL·E wall |
| 2 | Multi-cue, confidence-gated distillation | Best shot at Macro/Worst under compact inference |
| 3 | Open-world compact detector updating | Clear real-world story; large method space |
| 4 | RepViT / SHViT / EfficientViM / MambaOut bake-off | Tests whether SSM is necessary; stronger students |
| 5 | Real transmission / degradation robustness | App value; pairs with consistency distillation |
| 6 | Foundation pretraining factorial study | High science value; larger scope |
| 7 | Kernel / TensorRT / INT8 | Necessary engineering; weak as sole paper |
| 8 | Video / temporal | Separate long-horizon project |
| 9 | Adversarial / watermark-aware | Later security line |

---

## Two-week screening pilots (do these first)

### Pilot A — Compact backbone bake-off (same protocol as possible, **new** freeze)
Models (exactly six):
1. LiteSSM-A (frozen checkpoint as reference only; re-eval under v2 manifests if needed)
2. EfficientViM
3. EfficientVMamba
4. MambaOut
5. RepViT
6. SHViT **or** EfficientViT

Report: UFD Macro, Worst-generator, DALL·E, Params/FLOPs, native PyTorch latency, and if feasible fused/ONNX/TensorRT + FP16.

### Pilot B — Distillation pilot (independent distillation pool)
Teachers: NPR alone; UnivFD alone; **confidence-gated dual teacher**.
Student: LiteSSM-A and/or RepViT / EfficientViM.
**No** plain \(L=\mathrm{CE}+\lambda\,\mathrm{KL}\) as the only recipe—prefer gated logits + optional feature / ranking consistency.

Distillation pool must be:
- extra reals + independently generated fakes;
- fully deduped vs final test manifests;
- teachers emit soft labels / features only.

### Pilot C — Small matched modern-generator val set
Content-/format-/size-/codec-matched real–fake pairs (B-Free-style bias control).  
Keep old generators (ADM, SDv2, Glide, DALL·E) as holdouts where license allows.

**Decision rule (after ~2 weeks):**
- Distillation clearly lifts DALL·E / Worst-generator → **Track P1: Bias-Controlled Multi-Cue Distillation**
- Backbone swap beats distillation → **Track P3: Is Mamba Necessary?**
- All static models keep collapsing on new generators → **Track P2: Open-World Detector Updating**

---

## Paper-shaped tracks (pick one after pilots)

### P1 — Bias-Controlled Multi-Cue Distillation
**Q:** After removing content/acquisition bias, can NPR + spectral + foundation cues compress into one compact student?

- Students: RepViT or EfficientViM (+ LiteSSM-A kept)
- Teachers: NPR; UnivFD/CLIP; optional spectral-tail / bit-plane (train-time only; STAL-style drop at inference)
- Data: content- and format-matched
- Losses: CE; confidence-gated logit KD; feature contrastive; cross-generator consistency; optional spectral aux
- Inference: student only

**Success bars (pre-register; do not aim at “0.85”):**
- UFD Macro \(+0.04\)–\(0.06\)
- Worst-generator AUC \(+0.08\)
- DALL·E AUC \(+0.10\)
- Params \(+{\le}10\%\); no teacher at inference; batch-1 latency \(+{\le}10\%\)

**Ablations:** no KD; NPR-only; CLIP-only; dual; no gating; unmatched vs matched data; LiteSSM vs RepViT/EfficientViM.

### P2 — Open-World Compact Detector Updating
**Q:** As generators arrive over time, how can a compact detector update cheaply without catastrophic forgetting?

Sequential protocol example: ADM/SDv2 → Glide/LDM → DALL·E → SDXL → FLUX → other modern.  
Budget per step: 5 / 20 / 100 labeled shots or unlabeled-only.

Baselines: no update; head-only; full FT; LoRA/adapter; replay; EWC; LiteUpdate; SSM adapter / state update.

Metrics: current-gen AUC; old-gen mean AUC; Average Forgetting; Worst-generator; update time; new params; replay storage.

### P3 — Is Mamba Necessary for Compact AIGC Detection?
**Q:** Is LiteSSM-A’s edge from SSM, or from width / local conv / training luck?

Strictly match Params, FLOPs, epochs, pretrain status, resolution, latency harness.  
Add CKA, freq sensitivity, error overlap (DALL·E / Bedroom), occlusion / spectral perturbation probes.

---

## Models / modules to try (v2 only)

**Compact backbones:** RepViT, SHViT, EfficientViT, MambaOut, EfficientViM, EfficientVMamba.

**Forensic cues / refs:** NPR (teacher / second modality), LOTA (bit-plane), SPAI (spectrum one-class), Secret Lies in Color (~1.4M color baseline), B-Free (paired debiased data), VIB-Net, CO-SPY, Beyond Generation (real-only).

**Public stress tests (prefer over ad-hoc social pipelines when possible):** CO-SPYBench, RRDataset; Community Forensics-style multi-generator coverage where licenses allow.

---

## Explicitly deferred

- Video / temporal SSM as v2 lead (too large a protocol reset).
- Adversarial + watermark joint as top-3 (separate security paper).
- Kernel-only latency paper without architecture / generalization questions.
- Any change to submitted LiteSSM-A ID **0.946**, Macro **0.718**, B1 **144.4 ms**, B32 **226** img/s, JPEG claim **Q70**, or Panel-B Macro/OOD wording.

---

## References (pointers used in planning)

- B-Free (CVPR 2025): bias-free paired generation for detection training.
- STAL (arXiv 2605.22751): spectral-tail auxiliary learning; drop frequency at inference.
- Cross-modal RGB↔NPR (CVPR 2026): multi-cue / mutual distillation.
- CO-SPY + CO-SPYBench (CVPR 2025); RRDataset (ICCV 2025).
- VIB-Net, FatFormer, CatAID: pretrained / bottleneck / adapter generalization lines.
- EfficientViM, EfficientVMamba, MambaOut, RepViT (CVPR).
- LiteUpdate (arXiv 2511.07192): lightweight detector updating.
- LOTA, SPAI, Secret Lies in Color: compact forensic-cue baselines.
