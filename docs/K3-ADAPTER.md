# K3 — systemone adapter gate report

Date: 2026-09-24. Host: Mac15,6 (M3 Pro, 18 GB), macOS 26.2.
The adapter lives in the systemone repository (branch `feat/kev-backend`):
`crates/systemone-kev`, `ProviderKind::Kev`, CLI features
`kev-cpu` / `kev-accelerate` / `kev-metal` (metal includes the CPU ladder,
mirroring laya). systemone consumes kev-core as a **path dependency**
(kev-rs is pre-release; the workspace notes it must become a pinned Git
revision before any release build).

## TypeSafe SDK smoke (`compat/sdk-js`, @typesafe-ai/sdk 0.6.0)

`SYSTEMONE_SMOKE_BACKEND=kev`, shared assertions, kev expectation row =
output tokens **counted** (upstream kev's serialised-answer semantics),
float state **accepted**:

| build | checkpoint / device | result | usage |
|---|---|---|---|
| `--features kev-accelerate` | kev-0.6b / cpu (candle) | **passed** | input 80, output 157 |
| `--features kev-metal` | kev-0.8b / metal (mlx) | **passed** | input 80, output 161 |
| `--features metal,laya-metal,kev-metal,gliner2` | kev-0.8b / metal | **passed** | input 80, output 161 |

The last row is the single-binary link check: OpenJev (llama.cpp) +
Laya (MLX/Candle) + Kev (MLX/Candle) + GLiNER2 (ONNX Runtime) in one
release `s1` (52.5 MB) over the one shared patched mlx-sys pin.

## Row isolation over the wire (frozen packed-vs-separate gate)

`tests/goldens.rs::*_packed_vs_separate_*` (mixed-packed-3 vs the three
separate-\* fixtures, same runtime):

- kev-0.8b MLX: max |dp| 0.00578 (frozen gate 0.01), no argmax flips.
- kev-0.6b Candle: max |dp| 0.0 (bitwise; frozen torch gate 1e-5), no flips.

## HTTP latency (bench-m5 workload through `s1 serve`, kev-0.8b metal)

Median of 20 after 3 warmup, loopback, release build; library numbers from
`docs/K2-CORE.md` for the same host:

| phase | HTTP median | HTTP p95 | library median | overhead |
|---|---|---|---|---|
| new state (unique state per request) | 163.04 ms | 163.37 | 162.00 | ~1.0 ms |
| repeated state (resident prefix cache) | 67.90 ms | 68.39 | 67.45 | ~0.5 ms |

The HTTP path includes systemone admission, conversion, kev-core encoding
and the serialised-answer token count. No speedup was assumed; none of the
frozen gates were touched.

## Known limits (disclosed, not gated away)

- CI (added at publication, `.github/workflows/ci.yml`) runs the
  weight-free suite on Linux and macOS; the weight-dependent parity and
  bench gates run on the pinned hardware and gate tags locally
  (`docs/RELEASE.md`, tracked in issue #1).
- Qwen3.5 hybrid CPU inference is not implemented (K1/K2 amendment):
  `device = "cpu"` serves the Qwen3 generation (kev-0.6b); llama.cpp
  remains the long-term hybrid-CPU route (issue #1).
- Third-party license regeneration for systemone release packaging is
  deferred until kev enters its binary release feature sets (issue #1).

## Publication (post-report addendum)

kev-rs was published to <https://github.com/codesoda/kev-rs> and tagged
`v0.1.0` (commit `799d552e`), followed by the documentation-only
`v0.1.1` so the pinned tag describes its own publication state.
systemone's `feat/kev-backend` (PR #18) depends on the `v0.1.1` tag
commit instead of the earlier path dependency, and the cpu smoke was
re-run against the pin (passed).
