# V2 manifests (sample IDs / relative paths only — never commit raw images)

Put pilot-specific JSONL here:

- `pilot_a/train.jsonl`, `val.jsonl`, `id_test.jsonl`, `ufd_eval.jsonl`
- `pilot_b/distill_pool.jsonl`
- `pilot_c/*.jsonl` for matched pairs

Recommended JSONL fields: `sample_id`, `path`, `label` (0/1), `generator`, `split`, `sha256`.

**Ban:** any `sample_id` that appears in paper UFD/DALL·E **test** manifests when building `distill_pool`.
