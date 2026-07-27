# V2 Global Status Board

Updated: 2026-07-27  
Branch: `v2-experiments`

| Pilot | Goal | Status | Owner notes |
| --- | --- | --- | --- |
| A Backbone bake-off | LiteSSM-A vs EfficientViM / EfficientVMamba / MambaOut / RepViT / SHViT|EfficientViT | **wave-1 DONE** | Macro: LiteSSM-A 0.718 > RepViT 0.697 ≈ EffV2-S 0.695 > ConvNeXt-T 0.614; EffV2 best DALL·E 0.619. Next: Pilot B |
| B Distillation | NPR / UnivFD / gated dual → compact student | **RUNNING on AutoDL** | softlabel export → 3 students × 3 recipes (+ LiteSSM scratch gated) |
| C Matched generators | Bias-controlled mini val (SDXL/FLUX + holdouts) | **scaffolded** | protocol draft only |

## Next actions (this week)

1. [ ] Build distillation-pool inventory (paths + licenses) without paper-test IDs  
2. [ ] Wire RepViT + MambaOut loaders in `v2/scripts/`  
3. [ ] Draft matched-pair generation script (content-matched, same size/codec)  
4. [ ] Run Pilot A dry-run metrics schema → `results/pilot_a/schema_example.json`

## Paper branch note

Layout / caption fixes for the submitted PDF stay on `main` if needed; **do not** merge v2 training code into the paper release tag.
