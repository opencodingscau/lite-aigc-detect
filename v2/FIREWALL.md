# V2 Firewall (paper isolation)

Violating any rule below contaminates the submitted manuscript. Treat as hard stops.

## Never modify (v1 paper)

- `freeze/frozen_numbers.json`
- `freeze/freeze_package.json`
- Paper manifests under `manifests/` used by release `v1.0.0-paper.2`
- Primary tables under `reproduced_tables/` that feed the paper
- Claimed numbers: LiteSSM-A ID **0.946**, Macro **0.718**, B1 **144.4 ms**, B32 **226** img/s; JPEG claim **Q70**; LiteSSM-B Macro **0.700** / OOD Pooled **0.722** (never as Macro)

## Always use for v2

- Branch: `v2-experiments` (or descendants)
- Manifests: `v2/manifests/**` only
- Numbers lock: `v2/freeze_v2/**` only
- Predictions / checkpoints: `v2/outputs/**` (gitignored)
- Results notes: `v2/results/**`

## Distillation / eval data bans

- Do **not** train or distill on current UFD **test** or DALL·E **test** sample IDs (even unlabeled).
- Distillation pool must be deduped against paper test manifests; keep a SHA256 list under `v2/distillation_pool/dedup/`.

## Allowed read-only references

- Loading paper checkpoints for **reference re-eval** under **new** v2 manifests is OK if outputs go to `v2/outputs/` and are labeled `reference_v1_ckpt`.
- Reading `freeze/frozen_numbers.json` for comparison tables in `v2/results/` is OK; never write back.
