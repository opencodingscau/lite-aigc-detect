# Repository layout (public release)

```
lite-aigc-detect/
├── README.md
├── LICENSE
├── environment.yml / requirements.txt
├── lite_aigc/                 # train / eval / latency / external refs
├── scripts/                  # build_tables, remap, hashes, public builder
├── freeze/                   # freeze_package.json, TABLES, figures, hashes map
├── latency_batch1/           # locked efficiency summaries (paths sanitized)
├── external_refs/            # Panel B summaries (paths sanitized)
├── reproduced_tables/        # cold-start CSV + verification
├── docs/                     # LiteSSM-A/B architecture
├── latex/                    # MDPI manuscript sources
├── hashes/SHA256SUMS
├── checkpoints/              # empty placeholder; weights via Zenodo
└── manifests/                # empty placeholder; JSONL via release/remap
```

Private cloud SSH/HF launchers are **excluded** from this tree.
Canonical numbers: `freeze/frozen_numbers.json`.
