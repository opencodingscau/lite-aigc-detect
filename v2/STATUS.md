# V2 Global Status Board

Updated: 2026-07-27  
Branch: `v2-experiments`

| Pilot | Goal | Status | Owner notes |
| --- | --- | --- | --- |
| A Backbone bake-off | LiteSSM-A vs EfficientViM / EfficientVMamba / MambaOut / RepViT / SHViT|EfficientViT | **wave-1 running on AutoDL** | Wave-1: `repvit_m0_9`, `mambaout_proxy` (ConvNeXt-T), `efficientnet_v2_s` under `/root/autodl-tmp/v2_exp` |
| B Distillation | NPR / UnivFD / gated dual → compact student | **scaffolded** | need independent pool + dedup |
| C Matched generators | Bias-controlled mini val (SDXL/FLUX + holdouts) | **scaffolded** | protocol draft only |

## Next actions (this week)

1. [ ] Build distillation-pool inventory (paths + licenses) without paper-test IDs  
2. [ ] Wire RepViT + MambaOut loaders in `v2/scripts/`  
3. [ ] Draft matched-pair generation script (content-matched, same size/codec)  
4. [ ] Run Pilot A dry-run metrics schema → `results/pilot_a/schema_example.json`

## Paper branch note

Layout / caption fixes for the submitted PDF stay on `main` if needed; **do not** merge v2 training code into the paper release tag.
