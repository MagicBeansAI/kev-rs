# kev-rs roadmap (K0–K3)

Tracking issue: codesoda/systemone#17. The acceptance gate in that issue is
binding. Each phase ends with a user review before the next phase starts.

Upstream pin: `jaredpalmer/kev` @ `557598fced1dada75dfbf36ed144dce309ac6ceb`
(2026-09-22).

Template: `laya-rs` (layout, gate conventions, manifest format, the vendored
`mlx-sys` patch). kev-metal and laya-metal must share one mlx-sys
version/patch so both link into the same `s1` binary. kev-rs therefore patches
`mlx-sys` from the laya-rs git repository at the same revision systemone pins
(`23fff422666fd5039998fd8a55ca57f7e40d224b`), and does not carry a second
vendored copy.

## K0: baseline and goldens

- Pinned uv env for upstream kev in `benchmarks/baseline` (Python 3.12.*,
  torch 2.8.0, transformers 5.17.0, peft 0.21.0, mlx 0.32.2, mlx-lm 0.31.3 —
  the versions upstream's own `uv.lock` resolves).
- Asset manifest (`manifests/sources.json`) for `jaredpalmer/kev-0.8b` and
  `kev-4b` (Qwen3.5), plus `kev-0.6b` (Qwen3) as the CPU fallback. The Qwen
  base repositories are pinned too: the tokenizer loads from the base repo at
  `base_revision`, not from the adapter repo.
- `head.pt` is converted to `head.safetensors` + `head.meta.json` by
  `kev-convert-head`; SHA-256 checksums go into the manifest. Rust never
  executes pickle.
- Golden outputs on a fixed request set: noul/choice/score, packed vs
  separate questions, repeated state (prefix cache), delimiter escaping.
  Two oracle paths per fixture where supported:
  - `torch-fp32`: fp32 PyTorch (the reference).
  - `mlx`: upstream's MLX path (bf16 backbone, fp32 LoRA merge on the CPU
    stream). MLX refuses Qwen3 bases, so `kev-0.6b` has torch-fp32 goldens
    only.
- Record environment, precision, prefix-cache state and timings.
  `benchmarks/manifest.json` is frozen before any Rust timing is inspected.

## K1: runtime spike and decision

- Qwen3.5 (hybrid attention + Gated DeltaNet) backbone in Rust on MLX via
  mlx-rs, in `spikes/`. Check whether the laya-rs mlx-sys patch covers what
  kev needs.
- CPU options: Candle, and llama.cpp (already linked by systemone for
  openjev). Record the chosen runtime and the rejected alternatives with
  numbers. If Qwen3.5 on CPU is not feasible, say so and propose the Qwen3
  generation (kev-0.6b) as the honest first CPU step. Do not claim Qwen3.5
  CPU support without parity evidence.
- Verify early that kev-metal and laya-metal link into one binary with the
  shared mlx-sys patch.

## K2: kev-core parity

- Tokenizer, special-token escaping, encoding, row isolation, LoRA merge,
  pointer head and temperature against the K0 goldens.
- Tolerances are justified against upstream's own MLX-vs-fp32 evidence
  (Kev-4B max |dp| 0.02524, mean 0.0016, 1 argmax flip on 1,264 questions;
  Kev-0.8B max 0.054, mean 0.0023, 4 flips). Argmax flips are reported, not
  hidden. Packed and separate questions must match.
- Benchmark on the same workload as upstream's M5 table (five questions,
  three options each, ~270-token state; new vs repeated state). No speedup
  is assumed.

## K3: systemone adapter

- `systemone-kev` crate, `kev-cpu` / `kev-metal` features (`kev-metal`
  enables the CPU backend too), config validation, CHANGELOG / README /
  cross-repo.md updates.
- `device = "metal"` in a build without `kev-metal` is a configuration
  error, never a silent CPU fallback.
- JS SDK smoke passes with `SYSTEMONE_SMOKE_BACKEND=kev` on cpu and metal,
  with the shared assertions. A kev row is added to the per-backend
  expectation table; the `usage.output_tokens` semantics are decided and
  documented (upstream reports serialized-answer tokens; the other local
  backends report 0).
- `cargo build -p systemone-cli --features metal,laya-metal,gliner2,kev-metal`
  builds and serves with all backends in one binary.

## Decisions taken at K0 start

- `head.pt` handling: converted, checksummed safetensors artifact
  (user-approved 2026-09-24). A restricted Rust pickle reader was rejected:
  more code to audit for no capability gain.
- Base model downloads approved (~11 GB total): Qwen3.5-4B-Base,
  Qwen3.5-0.8B-Base, Qwen3-0.6B-Base.
