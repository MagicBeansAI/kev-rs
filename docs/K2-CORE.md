# K2 — kev-core parity and performance gate report

Date: 2026-09-24. Host: Mac15,6 (M3 Pro, 18 GB), macOS 26.2.
Upstream kev pinned at `557598fced1dada75dfbf36ed144dce309ac6ceb`.
All gates below were frozen at K0 (`benchmarks/goldens/tolerances.json`,
`benchmarks/manifest.json`) before any Rust runtime existed; none were
relaxed.

## Scope

`crates/kev-core`: an independent Rust runtime for Kev decision models —
tokenizer + special-token escaping, request encoding, row isolation,
LoRA merge, pointer head + temperature calibration, and two backends:

- **mlx** (Apple Silicon): Qwen3.5 hybrid (Gated DeltaNet + gated
  attention) on mlx-rs =0.32.0, bf16 weights, fp32 GDN state, fp32 LoRA
  merge. The GDN recurrence runs as mlx-lm's fused Metal kernel
  (`gated_delta.py`'s scalar/unmasked variant, reproduced verbatim,
  called through raw mlx-sys FFI — `src/gdn_kernel.rs`), one launch per
  layer. The elementwise glue mlx-lm compiles (`swiglu`,
  `_precise_swiglu`, `compute_g`) is compiled here too (`mx.compile` via
  mlx-rs). Serving path mirrors upstream `mlx_model.py`: the state runs
  once into a prefix cache (KV + conv/recurrent state), branches run as
  one right-padded batch on a replicated copy, prefix kept for the next
  request.
- **candle** (portable CPU): Qwen3-generation attention-only bases
  (kev-0.6b) on candle =0.11.0 in fp32 — the exact-oracle precision
  (upstream loads the bf16 checkpoint with `dtype=fp32`). Same prefix
  cache contract, per-row branch passes. `candle-accelerate` enables
  Accelerate BLAS on macOS (identical parity, ~2× faster).

### K1 amendment (recorded before results): CPU backend

K1 chose llama.cpp/GGUF as the CPU candidate. That path was **not**
implemented at K2: GGUF conversion precision, hidden-state extraction and
row isolation were unverified risks, and the Qwen3.5-hybrid CPU parity
they gate was not needed for the K2 exit (the hybrid checkpoints serve on
Metal). K2's CPU backend is the named fallback — Candle with the Qwen3
attention-only generation (kev-0.6b). Qwen3.5 hybrid CPU inference is a
config error, not a silent fallback; llama.cpp remains the long-term
route to hybrid-on-CPU.

## Encoding parity (bit-exact, all checkpoints)

`cargo test -p kev-core` — `tests/goldens.rs::encoding_parity_*`,
against every K0 golden (kev-0.8b: 22, kev-0.6b: 22, kev-4b: 8-fixture
documented subset):

- state text, per-question instr/options: **exact**
- token ids, decide_idx, opt_idx, state_tokens: **exact**
- `rows_of` row decomposition (ids, decide, opts): **exact**
- wire answers rebuilt from golden probs (round_ties_even 4-decimal,
  confidence, legend, ordering): **exact**
- `usage.output_tokens` (Python-`json.dumps`-style serialization,
  unescaped tokenization): **exact**

## Backend parity vs the torch fp32 oracle (frozen gates)

`cargo test -p kev-core --release --features <backend> -- --ignored`

| checkpoint | backend | questions | max dp | gate | mean dp | gate | flips | gate |
|---|---|---|---|---|---|---|---|---|
| kev-0.8b | mlx | 30 | 0.012832 | 0.06 | 0.002242 | 0.005 | 0 | ≤2 near-tie |
| kev-4b | mlx | 10 (subset) | 0.006719 | 0.03 | 0.001073 | 0.004 | 0 | ≤1 near-tie |
| kev-0.6b | candle fp32 | 30 | 0.000008 | 0.005* | 0.000001 | 0.001* | 0 | 0 |

\* kev-0.6b has no frozen K0 Rust-CPU gate (tolerances.json froze
mlx-vs-fp32 only); 0.005/0.001 were written into the test before the
first run and passed at 8e-6. Parity is identical with and without
`candle-accelerate`.

Context: upstream's own MLX-vs-fp32 envelope at the pin is max 0.0542
(0.8b) / 0.0252 (4b); the K0 Python MLX goldens measured 0.0141 / 0.0071
on the same fixtures. The Rust MLX backend sits inside both.

## Benchmark acceptance (frozen bench-m5 protocol)

Timed scope = upstream `probs_and_prefix` / `probs_with_prefix`
(backbone + head + softmax on a precomputed encoding), median of 20
after 3 warmup, same host, results in `benchmarks/results/k2-bench/`.
Python reference medians from `benchmarks/manifest.json`.

Gate (k2_metal): Rust median ≤ 1.10× Python median; p95 ≤ 1.05× Python p95.

| config | phase | Rust ms | Python ms | ratio | Rust p95 | limit | pass |
|---|---|---|---|---|---|---|---|
| kev-0.8b mlx | new_state | 162.00 | 163.12 | **0.99×** | 162.78 | 171.86 | ✓ |
| kev-0.8b mlx | repeated | 67.45 | 66.30 | **1.02×** | 67.69 | 70.01 | ✓ |
| kev-4b mlx | new_state | 943.28 | 950.60 | **0.99×** | 945.42 | 999.14 | ✓ |
| kev-4b mlx | repeated | 378.16 | 372.71 | **1.01×** | 379.89 | 392.95 | ✓ |

CPU (k2_cpu — correctness-gated, speed recorded, not gated):

| config | phase | Rust ms | torch fp32 ms | ratio |
|---|---|---|---|---|
| kev-0.6b candle+accelerate | new_state | 907.56 | 491.63 | 1.85× |
| kev-0.6b candle+accelerate | repeated | 545.27 | 239.72 | 2.27× |
| kev-0.6b candle (portable) | new_state | 1911.50 | 491.63 | 3.89× |
| kev-0.6b candle (portable) | repeated | 1123.58 | 239.72 | 4.69× |

Honest caveats: the host was not idle (1-minute load 4.5–8.0 during
runs, a `ctx` indexer at ~2 cores; the Python baseline itself recorded
load 4.13). MLX numbers were stable to <1 ms across repeated runs. The
CPU numbers were taken at the highest ambient load and may read worse
than an idle host would give; they are records, not gates.

## What made the MLX gate (chronology)

First complete run was 6.1× over budget (naive per-token GDN scan:
1000 ms new / 422 ms repeated). No gate was touched; the path to 0.99×:

1. fused GDN Metal kernel via mlx-sys FFI (→ 224 / 121 ms),
2. batched branch rows on a replicated prefix, upstream's serving shape
   (→ 176 / 71 ms),
3. bench scope aligned to the frozen protocol (encoding out of the
   timed region, as in `kev_baseline.bench`) and picked-position gather
   (→ 175 / 70 ms),
4. compiled elementwise glue, matching mlx-lm's `mx.compile` usage
   (→ 162 / 67.5 ms).

Parity was re-verified after every step; the frozen tolerances never
moved.
