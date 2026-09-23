# kev-baseline

The pinned Python environment for upstream kev, and the tooling that produces
the kev-rs oracle evidence. This code is a research baseline only. It is never
the shipped implementation.

Upstream kev is pinned by commit
(`557598fced1dada75dfbf36ed144dce309ac6ceb`). The dependency pins match the
versions upstream's own `uv.lock` resolves (torch 2.8.0, transformers 5.17.0,
peft 0.21.0, mlx 0.32.2, mlx-lm 0.31.3, Python 3.13).

Every CLI prints JSON on stdout and diagnostics on stderr.

- `uv run kev-env`: report the resolved environment.
- `uv run kev-fetch`: download the pinned model files and write or verify
  `manifests/sources.json`. Weights go to `.cache/kev`, never into git.
- `uv run kev-convert-head`: convert a checkpoint's `head.pt` (a PyTorch
  pickle) into `head.safetensors` + `head.meta.json`, and record SHA-256
  checksums. Rust reads only the converted artifact.
- `uv run kev-goldens`: generate golden outputs for the frozen request
  fixtures, on the torch fp32 path and the upstream MLX path.
- `uv run kev-compare-goldens`: compare the MLX goldens against torch fp32.
- `uv run kev-bench`: run the upstream M5-table workload and record timings.

Golden files are generated, never hand-edited.
