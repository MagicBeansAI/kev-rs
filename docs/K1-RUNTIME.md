# K1: runtime spike and decision

Date: 2026-09-24. Host: MacBook Pro (M3 Pro, 18 GB). Upstream kev at
`557598f`; goldens and tolerances from K0.

## Decision

- **Metal (required)**: a hand-written Qwen3.5 hybrid backbone on
  **mlx-rs =0.32.0** with laya-rs's patched **mlx-sys 0.6.0** (git pin
  `23fff422`, the revision systemone already patches). Proven by the
  `spikes/mlx-backbone` parity run (numbers below).
- **CPU (portable)**: **llama.cpp via llama-cpp-2 =0.1.156** — the exact
  version systemone already links through openjev-llama, and it ships the
  `qwen35`/`qwen3next` architectures with a fused CPU `GATED_DELTA_NET` op
  and GGUF conversion for merged checkpoints. Kev's row form ("causal rows
  continuing a shared state prefix") maps directly onto llama.cpp sequence
  branching (`llama_memory_seq_cp`) with per-token positions and
  `embeddings=true` / `POOLING_TYPE_NONE` per-token hidden states. This
  works for **both** model generations, because rows are equivalent to the
  packed block-causal form on any architecture (upstream's own guarantee).
  Numerical parity on CPU is a K2 gate, not yet demonstrated; if GGUF
  precision (bf16/fp32 conversion of the merged model) cannot meet the K0
  goldens, the fallback is Candle with Qwen3 (kev-0.6b) as the honest
  CPU-generation limit.

## Rejected alternatives

- **Candle =0.11.0 for Qwen3.5**: no released candle-transformers
  implements the hybrid architecture (open PRs #3396/#3461 only, with a
  long correctness-bug tail). Candle remains viable for Qwen3 dense
  (kev-0.6b) and stays the documented fallback if llama.cpp parity fails.
- **mistral.rs**: supports the family but is a heavy server-oriented third
  inference stack; extracting pre-LM-head hidden states with custom rows
  would mean forking internals.
- **burn**: no GDN implementation; a from-scratch port with no oracle
  advantage over the mlx port we already validated.
- **PyTorch sidecar**: research baseline only, never shipped (hard rule).

## Spike evidence (`spikes/mlx-backbone`, results in `benchmarks/results/k1-spike/`)

Row-form forward (state + branch as one causal row), bf16 backbone, fp32
LoRA merge on the MLX CPU stream, fp32 pointer head from the converted
`head.safetensors`, temperature applied. Compared against the K0 goldens:

| Run | Fixtures | Questions | max \|dp\| | argmax flips |
|---|---|---|---|---|
| kev-0.8b vs upstream MLX | 22 | 30 | 0.0140 | 0 |
| kev-0.8b vs torch fp32 | 22 | 30 | 0.0052 | 1 (list-state, the same fp32 near-tie — gap 0.004 — where upstream's own MLX flips) |
| kev-4b vs upstream MLX | 22 | 30 | 0.0090 | 0 |
| kev-4b vs torch fp32 | 8 (K0 core subset) | 10 | 0.0033 | 0 |

Context: upstream's own Python-MLX-vs-fp32 envelope is 0.8B max 0.054 /
4 flips, 4B max 0.025 / 1 flip in 1264. The Rust backbone sits inside it on
every measured fixture; against fp32 it is currently *tighter* than
upstream's Python MLX on the same fixtures (0.0052 vs 0.0141 for 0.8B).

Implementation notes (ported from mlx-lm 0.31.3, math verified line by
line): sequential ops-form delta-rule scan (fp32 state; upstream's Metal
kernel is an optimization of the same recurrence, available for K2 if the
scan is too slow), depthwise causal conv k=4 + SiLU, l2-norm via unweighted
`fast.rms_norm` with the 1/√dk scaling folded exactly as mlx-lm does,
`g = exp(-exp(A_log_f32)·softplus(a+dt_bias))`, gated RMSNorm with fp32
swiglu, gated attention (q/gate fused in q_proj, QK RMSNorm, partial rotary
64 of 256 dims at theta 1e7, sigmoid output gate), raw-HF sanitize (+1 norm
shift, conv moveaxis, vision/MTP stripped), Hv>Hk head repeat for 4B.

## Linking

- `spikes/mlx-link`: mlx-rs 0.32.0 builds against the laya-rs git-pinned
  patched mlx-sys and runs GPU + CPU streams; the JIT build produces the
  same 1.4 MB residual metallib as laya.
- `spikes/link-all`: **passes** — laya-core (mlx feature, systemone's
  pinned revision), kev's mlx-rs usage and llama-cpp-2 0.1.156 build into
  one binary with one mlx-sys and initialize together in one process
  (MLX matmul + laya-core symbols + `llama_backend_init`, whose Metal
  library includes ggml's own `gated_delta_net` kernels).
- kev-rs carries **no vendored mlx-sys copy**; the root `[patch.crates-io]`
  points at laya-rs `23fff422`, so kev-metal and laya-metal cannot drift.

## Known limits and open items for K2

- Spike runs one row at a time with no state-prefix cache and per-layer
  eval; bench-m5-shape latency is ~0.4 s per question on 0.8B — the prefix
  cache and row batching are required to meet the frozen benchmark gate.
- llama.cpp CPU numerical parity (GGUF conversion precision, hidden-state
  extraction, row isolation via sequence branching) is unproven; it is the
  first K2 work item and the CPU gate blocks on it.
- The mlx scan is the ops reference; the Metal kernel port
  (`gated_delta_step`) is an optimization option only if benchmarks demand
  it, never at the cost of parity.
