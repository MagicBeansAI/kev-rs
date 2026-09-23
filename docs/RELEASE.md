# kev-rs releases

kev-rs releases a **library** (`kev-core`), not binaries or weights. A
release is a Git tag `vX.Y.Z` plus a GitHub Release whose notes come from
the matching `CHANGELOG.md` section. Consumers (systemone) depend on the
tag's commit as a pinned Git revision, the same way laya-core is consumed.
Releases never contain model weights, credentials, or the Python baseline
environment; the baseline under `benchmarks/baseline` is a research oracle
only and is never the shipped implementation.

## Gate before tagging

A tag may only be cut from a commit where, on the pinned hardware
(Apple Silicon with the model cache under `.cache/kev/`):

```sh
cargo fmt --all --check
cargo clippy -p kev-core --all-targets -- -D warnings
cargo clippy -p kev-core --all-targets --features candle,candle-accelerate -- -D warnings
cargo clippy -p kev-core --all-targets --features mlx -- -D warnings
cargo test -p kev-core                                                        # encoding parity
cargo test -p kev-core --release --features mlx --test goldens -- --ignored   # Metal parity gates
cargo test -p kev-core --release --features candle,candle-accelerate --test goldens -- --ignored
```

all pass, and the bench-m5 acceptance numbers in `docs/K2-CORE.md` still
hold for any change that touches a forward path. The frozen tolerances in
`benchmarks/goldens/tolerances.json` and `benchmarks/manifest.json` are
never relaxed to make a release; a failing configuration blocks the tag.

## Cutting a release

1. Move the `## [Unreleased]` items in `CHANGELOG.md` into a new dated
   `## [X.Y.Z]` section; commit.
2. Tag and push:

   ```sh
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

3. The `release` workflow re-runs the weight-free checks on CI and creates
   the GitHub Release with the changelog section as notes. Weight-dependent
   gates cannot run on CI; the tag itself asserts they passed locally (the
   gate reports under `docs/` carry the numbers).
4. Update consumers to the new commit, e.g. in systemone:

   ```toml
   kev-core = { git = "https://github.com/codesoda/kev-rs.git", rev = "<tag commit sha>", default-features = false }
   ```

   Metal consumers must repeat the `mlx-sys` `[patch.crates-io]` pin from
   this repository's `Cargo.toml` (Cargo does not propagate patch tables).
