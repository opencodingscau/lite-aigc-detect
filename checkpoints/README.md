# Checkpoints

Weights are **not** bundled in this Git tree. Place files under the paths below after obtaining them from the archival release (self-trained Panel A) or from **official upstream** sources (Panel B). Do not mirror third-party weights without permission.

## Panel A — study-trained (registry paths, do not rename files)

Frozen experiment registry keys stay on disk so hashes and loaders remain valid. Paper names are display-only.

| Paper name | Registry path | Table role | SHA256 (freeze) |
|------------|---------------|------------|-----------------|
| **LiteSSM-A** | `checkpoints/mobilemamba_lite/best.pt` | Panel A preferred operating point | `e939d33cc1136659719550bc0a84d5d03602e352768d922ec4b8ebdb470e6cae` |
| **LiteSSM-B** | `checkpoints/mambapsa_cls/best.pt` | Panel A SSM alternate | `b51f58f8a0de9048b90e5534844ee1f888a9a279c9724a5003d6e311567b0607` |
| LiteFreqNet v2 | `checkpoints/lite_freq_net_v2/best.pt` | Panel A frequency CNN | `f1ba7eb3e18d00dd1cf8fb64b4a4ea7ebb9a75521aa69f6a47ccac1ae3c88b2d` |
| EfficientNet-B0 | `checkpoints/efficientnet_b0/best.pt` | Panel A CNN | `aee1cc38ea2a926290e01dc434a8d15f022db1cdb1a0b7f25e90f49471149278` |
| MobileNetV3-S | `checkpoints/mobilenet_v3_small/best.pt` | Panel A CNN | `7268ea65ee82b7c1ccbc247dcc816ee2a18cdf168440f1927986fd18e1c9b60a` |
| ShuffleNet-x0.5 | `checkpoints/shufflenet_v2_x0_5/best.pt` | Panel A CNN | `fbedb56558b13a9bfe6501279b05a69ad3b62ac07fe3d365b818310e5a578867` |

Source of SHA256 values: `freeze/SHA256_MANIFESTS.json` (bytes match the frozen AutoDL bake-off `best.pt` files; only path keys were sanitized).

### Load / preprocess (Panel A)

- Task: binary still-image real (`0`) / fake (`1`)
- Input: RGB `224×224`, ImageNet mean/std (see `lite_aigc/data.py`)
- Seed / recipe: seed `42`, from-scratch, no ImageNet init (manuscript Methods)
- Example:

```bash
# after placing weights
python -c "import torch; from lite_aigc.models import build_model; \
m=build_model('mobilemamba_lite'); \
m.load_state_dict(torch.load('checkpoints/mobilemamba_lite/best.pt', map_location='cpu')); print('ok')"
```

(Adjust `build_model` import if running as scripts inside `lite_aigc/`.)

### License / redistribution (Panel A)

- Code in this repo: MIT (`LICENSE`).
- Self-trained weights are derived from third-party training images (DiffusionForensics subsets). **Release to Zenodo only after confirming dataset terms allow publishing derived weights.** Until that confirmation, treat Panel A weights as “available with the archival package if license-cleared,” not as freely rehostable by third parties.
- Do not redistribute raw training/test images with the weights.

## Panel B — external pretrained (link only; do not rebundle)

| Detector | Source | Expected local path after user download | Notes |
|----------|--------|-----------------------------------------|-------|
| UnivFD | Official UniversalFakeDetect release / CLIP ViT-L/14 linear probe weights | `external/UniversalFakeDetect/pretrained_weights/fc_weights.pth` | Use upstream license; verify SHA256 after download |
| NPR | Official NPR-DeepfakeDetection `NPR.pth` | `external/NPR-DeepfakeDetection/NPR.pth` | Unwrap `state['model']` and strip `module.` if needed (see `lite_aigc/eval_external_refs.py`) |

Provide only official download instructions and post-download checksums in the Zenodo/GitHub Release notes. **Do not** commit or re-upload UnivFD/NPR binaries into this repository.

## Recommended layout after download

```text
checkpoints/
├── README.md                 # this file
├── mobilemamba_lite/best.pt  # LiteSSM-A
├── mambapsa_cls/best.pt      # LiteSSM-B
├── lite_freq_net_v2/best.pt
├── efficientnet_b0/best.pt
├── mobilenet_v3_small/best.pt
└── shufflenet_v2_x0_5/best.pt
```

Optional: keep a release-side `checkpoints/SHA256SUMS` identical to the checkpoint rows in `freeze/SHA256_MANIFESTS.json`.

## What this repo reproduces without weights

`python scripts/build_tables.py` rebuilds Tables 2–5 from frozen JSON summaries and does **not** require checkpoints or a GPU.
