//! kev-core: an independent Rust runtime for Kev decision models.
//!
//! Owns the tokenizer and special-token escaping, request encoding,
//! per-question row isolation, the LoRA merge, the pointer head and
//! temperature calibration. Backends: MLX on Apple Silicon (`mlx` feature)
//! for the Qwen3.5 hybrid generation, and a portable CPU path (`candle`
//! feature) for the Qwen3 attention-only generation (see the K1 decision
//! record in the kev-rs repository).

pub mod api;
pub mod encode;
pub mod error;
pub mod head;
pub mod tokenizer;

#[cfg(feature = "mlx")]
pub mod backend_mlx;
#[cfg(feature = "mlx")]
pub(crate) mod gdn_kernel;
#[cfg(feature = "candle")]
pub mod backend_candle;
pub mod runtime;

pub use api::{Record, SystemOneRequest};
pub use error::{KevError, Result};
pub use runtime::{Evaluation, LoadOptions, Runtime};

/// The upstream kev commit this runtime is verified against.
pub const UPSTREAM_KEV_SHA: &str = "557598fced1dada75dfbf36ed144dce309ac6ceb";
