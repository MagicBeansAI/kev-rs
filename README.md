# kev-rs

An independent Rust runtime for [Kev](https://github.com/jaredpalmer/kev):
small, self-trainable Jev-style decision models. Each checkpoint is a rank-16
LoRA adapter plus a pointer head over a Qwen base model.

kev-rs owns the tokenizer and special-token escaping, request encoding,
per-question row isolation, the LoRA merge, the pointer head and temperature
calibration. It has two backends:

- **CPU**: portable, tested in CI on Linux.
- **MLX** on Apple Silicon (via `mlx-rs`): required on macOS, mirroring
  upstream's own MLX path for the Qwen3.5 hybrid backbone.

Status: **K0 (baseline and goldens) in progress.** No Rust runtime exists yet.

## Upstream pin

Upstream kev is pinned at commit
`557598fced1dada75dfbf36ed144dce309ac6ceb` (2026-09-22). The Python code under
`benchmarks/baseline` is a research baseline and oracle only. It is never the
shipped implementation.

## Layout

- `benchmarks/baseline`: pinned uv environment for upstream kev, plus the
  golden-generation, head-conversion and benchmark CLIs.
- `benchmarks/fixtures/requests`: the frozen request set.
- `benchmarks/goldens`: oracle outputs (fp32 PyTorch and upstream MLX).
- `benchmarks/results`: immutable benchmark run records.
- `manifests/sources.json`: pinned model assets (Hugging Face revisions and
  SHA-256 checksums).
- `docs/plans`: the K0–K3 roadmap and per-gate reports.

## Rules

- Model weights live in `.cache/kev` or the Hugging Face cache, never in git.
- Golden files are generated, never hand-edited.
- Tolerances are frozen before Rust results are inspected. A tolerance is
  never relaxed to make a failure pass.
- `head.pt` is a PyTorch pickle. Rust never parses pickle; a converted,
  checksummed safetensors artifact is produced by `kev-convert-head`.

## License

Apache-2.0. Upstream kev, its adapter weights and the Qwen3 / Qwen3.5 base
models are also Apache-2.0.
