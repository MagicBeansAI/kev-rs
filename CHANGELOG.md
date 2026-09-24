# Changelog

All notable changes to kev-rs are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Release notes for
a tag `vX.Y.Z` are taken from the matching `## [X.Y.Z]` section by the
release workflow.

## [Unreleased]

### Added

- `LoadOptions::mlx` (`MlxOptions`), MLX-only memory options, both off by
  default so the reference path is unchanged:
  - `quantize: Some(Quantization::Q8)` — MLX affine 8-bit (group size 32)
    for every projection and the embedding, each weight quantized right
    after its fp32 LoRA merge so the bf16 model is never resident at once.
    Kev-4B: 4.41 GiB after load (bf16 6.65), inference peak 5.44 GiB
    (8.61), same latency. Gated by `mlx_q8_vs_fp32` in `tolerances.json`.
  - `state_chunk: Some(n)` — a new state runs through the backbone `n`
    tokens at a time, bounding activation memory on long states. Exact
    with fp32 weights (1.4e-6); bf16 boundary rounding is gated by
    `state_chunk_vs_single_pass`.
- `kev-bench`: `KEV_BENCH_Q8`, `KEV_BENCH_STATE_CHUNK`, and MLX memory
  reporting (active / cache after load, inference peak).
- Gated tests `mlx_q8_parity_kev_0_8b`, `mlx_q8_parity_kev_4b`,
  `mlx_state_chunks_match_single_pass_kev_0_8b`; `check_backend` prints each
  near-tie flip's top-2 gap.

## [0.1.1] - 2026-09-24

Documentation-only release so the pinned tag carries documentation that
describes its own publication state; the runtime code is identical to
0.1.0.

### Changed

- README and `docs/K3-ADAPTER.md` now describe the published repository,
  the CI added at publication, the release tag systemone pins, and the
  post-release tracking issue (#1), instead of the pre-publication
  "local-only path dependency" state.
- `kev-core` crate version bumped to 0.1.1 to match the tag.

## [0.1.0] - 2026-09-24

First tagged release: the `kev-core` library crate, an independent Rust
runtime for [Kev](https://github.com/jaredpalmer/kev) decision models
(upstream pinned at `557598fced1dada75dfbf36ed144dce309ac6ceb`), developed
under the frozen K0 gates and released only after every K2/K3 gate passed
(see `docs/`).

### Added

- `kev-core`: tokenizer and special-token escaping, request encoding and
  per-question row isolation (bit-exact against the frozen upstream
  goldens), fp32 LoRA merge, framework-free fp32 pointer head and
  temperature calibration, and upstream-exact answer math including the
  serialised-answer `output_tokens` count. `head.pt` is consumed only as a
  checksummed safetensors conversion; the runtime never executes pickle.
- MLX backend (`mlx` feature, Apple Silicon): the Qwen3.5 hybrid backbone
  (Gated DeltaNet + gated attention) with mlx-lm's fused gated-delta Metal
  kernel, batched branch rows on a replicated state prefix and a prefix
  cache. Parity vs the fp32 oracle: max |dp| 0.0128 (kev-0.8b) / 0.0067
  (kev-4b), zero argmax flips; bench-m5 at 0.99×/1.02× (0.8b) and
  0.99×/1.01× (4b) of the Python MLX medians on the same host.
- Candle CPU backend (`candle` feature, portable; `candle-accelerate` adds
  Apple BLAS): the Qwen3 attention-only generation (kev-0.6b) in fp32,
  parity max |dp| 8e-6. Qwen3.5 hybrid CPU inference is a load error, not
  a fallback (llama.cpp remains the long-term hybrid-CPU route).
- Baseline tooling (`benchmarks/baseline`, Python, oracle only — never
  shipped): pinned fetch with SHA-256 verification, `head.pt` →
  safetensors conversion, golden generation, benchmarks and gate tests.
- Frozen gate artifacts: goldens, `tolerances.json`, `manifest.json`,
  fixtures, and the K0–K3 gate reports under `docs/`.

### Known limits

- Weight-dependent parity/bench tests are `#[ignore]` and run on pinned
  Apple Silicon hardware with the model cache present; CI runs the
  weight-free suite (encoding tests skip gracefully without the cache).
- Not published to crates.io; consumers pin a Git revision (mirroring
  laya-core) and must repeat the `mlx-sys` `[patch.crates-io]` pin for
  Metal builds.
