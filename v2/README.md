# Lite-AIGC Detect — V2 Experiments

**Branch:** `v2-experiments`  
**Paper freeze:** untouched (`freeze/`, `v1.0.0-paper.2`, Zenodo DOI).  
**Roadmap:** [`docs/V2_RESEARCH_ROADMAP.md`](../docs/V2_RESEARCH_ROADMAP.md)

## Priority (do not reorder casually)

1. Bias-controlled modern-generator protocol  
2. Multi-cue confidence-gated distillation  
3. Open-world generator updating  
4. Compact backbone bake-off (RepViT / MambaOut / EfficientViM / …)  
5. Transmission robustness → … (see roadmap)

## Two-week pilots (start here)

| ID | Name | Config | Status file |
| --- | --- | --- | --- |
| A | Compact backbone bake-off | `configs/pilot_a_backbone.yaml` | `results/pilot_a/STATUS.md` |
| B | Distillation pilot | `configs/pilot_b_distill.yaml` | `results/pilot_b/STATUS.md` |
| C | Matched modern-generator mini-set | `configs/pilot_c_matched_gen.yaml` | `results/pilot_c/STATUS.md` |

## Directory map

```
v2/
  README.md                 ← this file
  FIREWALL.md               ← hard rules vs paper freeze
  STATUS.md                 ← global pilot board
  configs/                  ← YAML for pilots
  manifests/                ← v2-only sample ID lists (no raw images)
  distillation_pool/        ← pool design + dedup logs (no paper test IDs)
  matched_generators/       ← bias-controlled protocol notes + manifests
  freeze_v2/                ← future locked numbers (empty until pilots finish)
  scripts/                  ← firewall check + pilot runners
  results/pilot_{a,b,c}/    ← metrics JSON / notes (not paper tables)
  outputs/                  ← local preds/ckpts (gitignored)
```

## Quick start

```bash
# always from repo root, on this branch
python v2/scripts/check_firewall.py

# create local output folders
python v2/scripts/init_workspace.py

# stubs (fill in training/eval later)
python v2/scripts/run_pilot_a_backbone.py --dry-run
python v2/scripts/run_pilot_b_distill.py --dry-run
python v2/scripts/run_pilot_c_matched_gen.py --dry-run
```

## Decision rule (after pilots)

- Distillation lifts DALL·E / Worst → **P1 Multi-Cue Distillation**  
- Backbone swap wins → **P3 Is Mamba Necessary?**  
- Static models keep collapsing → **P2 Open-World Updating**
