# Study-Specific SSM Architectures (Gate B)

**LiteSSM-A** denotes the compact baseline SSM classifier, whereas **LiteSSM-B** extends it with a BiViM-inspired bidirectional SelectiveSSM design. Both models are study-specific pure-PyTorch implementations rather than official reproductions of an existing architecture.

## Frozen registry → paper display names

| Frozen experiment internal name | Paper formal name |
|---------------------------------|-------------------|
| `mobilemamba_lite` | **LiteSSM-A** |
| `mambapsa_cls` | **LiteSSM-B** |

Do **not** rename checkpoint files. Loaders and table scripts may use `display_name` from `freeze/frozen_numbers.json`.

Manuscript, figures, tables, and READMEs use **LiteSSM-A/B** only. Registry keys appear only for reproducibility mapping.

Implementation: `lite_aigc/mamba_backbones.py`  
Canonical numbers: `freeze/frozen_numbers.json`  
Efficiency: trainable params via `sum(p.numel() for p in model.parameters() if p.requires_grad)`; FLOPs via `thop.profile` on `1×3×224×224`; batch-1 latency is **model-only** (synthetic tensor, no disk decode / resize / normalize).

---

## Shared input / head conventions

- Input tensor: `B×3×224×224` (ImageNet mean/std applied in the dataloader, **outside** the model).
- Both models reduce spatial size by **16×** with four stride-2 convolutions → feature map `B×C×14×14`.
- Tokens: flatten to sequence length **L = 196**, channel **C = 192**, add learnable positional embedding, then apply **depth = 4** residual SSM-inspired blocks.
- Head: LayerNorm → mean pool over tokens → `Linear(192 → 2)`.
- No official CUDA selective-scan kernel; scan is a **Python/PyTorch sequential loop** (see below). This can inflate measured latency relative to a fused kernel implementation.

---

## SelectiveSSM (shared core)

Used inside both LiteSSM-A blocks and LiteSSM-B blocks.

| Item | Setting |
|------|---------|
| Input | `B×L×D` |
| Expansion | LiteSSM-A MRFFI: `expand=1`, `d_state=8`; LiteSSM-B BiViM: `expand=1`, `d_state=16` |
| Depthwise conv | `Conv1d` k=3, groups=`d_inner` |
| Discretization | softplus Δ; `A = -exp(A_log)`; sequential recurrence over `L` |
| Gate | SiLU on z-branch (GLU-style) |
| Residual | applied by the outer block, not inside SelectiveSSM |

**Selective scan substitute:** for each time step `t=1..L`,

```text
h_t = exp(Δ_t ⊙ A) ⊙ h_{t-1} + (Δ_t ⊙ B_t) ⊙ x_t
y_t = ⟨h_t, C_t⟩ + D ⊙ x_t
```

implemented as an explicit Python `for t in range(L)` loop (no `mamba_ssm` / Triton / CUDA kernel).

---

## LiteSSM-A

### Stem (spatial CNN)

| Stage | Output size | Channels | Ops |
|-------|------------:|---------:|-----|
| Input | 224×224 | 3 | — |
| Stem-1 | 112×112 | 64 | Conv3×3/s2, BN, GELU |
| Stem-2 | 56×56 | 128 | Conv3×3/s2, BN, GELU |
| Stem-3 | 28×28 | 192 | Conv3×3/s2, BN, GELU |
| Stem-4 | 14×14 | 192 | Conv3×3/s2, BN, GELU |
| Tokenize | L=196 | 192 | flatten + pos emb |
| Blocks ×4 | L=196 | 192 | MRFFILite |
| Head | 1 | 2 | LN, mean pool, Linear |

### MRFFILite block

```text
x ─→ LayerNorm ─┬─→ DWConv1d{k=3,5,7} (sum) ─┐
                └─→ SelectiveSSM (d_state=8) ─┴─→ concat → Linear → +x
```

- Local branch: multi-receptive depthwise temporal convolutions on the token axis.
- Global branch: SelectiveSSM.
- Fusion: `Linear(2C → C)` then residual add.

**Locked size:** Params 1.74M; FLOPs 0.705G (`thop`).

---

## LiteSSM-B

### Stem (identical resolution schedule; named `patch` in code)

| Stage | Output size | Channels | Ops |
|-------|------------:|---------:|-----|
| Input | 224×224 | 3 | — |
| Stem-1 | 112×112 | 96 | Conv3×3/s2, BN, GELU (`embed_dim//2`) |
| Stem-2 | 56×56 | 96 | Conv3×3/s2, BN, GELU |
| Stem-3 | 28×28 | 192 | Conv3×3/s2, BN, GELU |
| Stem-4 | 14×14 | 192 | Conv3×3/s2, BN, GELU |
| Tokenize | L=196 | 192 | flatten + pos emb |
| Blocks ×4 | L=196 | 192 | BiViMBlock |
| Head | 1 | 2 | LN, mean pool, Linear |

### BiViMBlock (bidirectional SelectiveSSM)

```text
x ─→ LayerNorm ─┬─→ SelectiveSSM_fwd ──────────────┐
                └─→ flip → SelectiveSSM_bwd → flip ─┴─→ concat → Linear → +x
```

- Forward and backward SelectiveSSM (`d_state=16`, `expand=1`).
- Mix: `Linear(2C → C)` then residual.
- Distinctive operator is bidirectional SelectiveSSM; naming follows the real implementation.

**Locked size:** Params 2.48M; FLOPs 0.853G (`thop`).

---

## Latency / throughput (locked session only)

Canonical source: `freeze/frozen_numbers.json` (same RTX 4090D session as batch-1 lock).

| Metric | LiteSSM-A | LiteSSM-B | Includes preprocess? |
|--------|----------:|----------:|----------------------|
| Batch-1 p50 | **144.4 ms** | 237.6 ms | **No** (model-only) |
| Batch-1 p95 | 171.5 ms | 264.6 ms | **No** |
| Throughput @ bs32 | **226** img/s | 120 img/s | **No** |

A preliminary batch-32 figure of **367** img/s for LiteSSM-A is **superseded** and excluded from the manuscript (not produced under the final locked latency protocol).

| Metric | Notes |
|--------|-------|
| Batch-1 latency | Synthetic `1×3×224×224`, FP32, CUDA sync, warmup≥50, iters=500 |
| Throughput @ bs32 | Same locked session; whole-batch images/s, not 1000/latency |
| FFT modules | N/A for LiteSSM-A/B; freq ablations use DualBranch wrappers |

---

## DualBranch frequency wrappers (ablation only)

`mobilemamba_lite_freq` / `mambapsa_cls_freq` wrap the same SpatialStem + SSM stack with gated FFT magnitude fusion. They are **not** main-table models.
