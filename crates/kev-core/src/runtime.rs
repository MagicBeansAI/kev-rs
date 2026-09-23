//! Runtime: load a checkpoint directory, evaluate requests in the row form
//! (state + one branch per question as causal rows), fp32 pointer head and
//! calibration, upstream answer math.
//!
//! Device policy is explicit: `Metal` needs the `mlx` feature and a hybrid
//! (Qwen3.5) base; `Cpu` needs the `candle` feature and an attention-only
//! (Qwen3) base. Anything else is a load error, never a silent fallback.

use crate::api::{self, QuestionMeta, SystemOneRequest};
use crate::encode::{self, Encoding, SERVE_MAX_BRANCH, SERVE_MAX_STATE};
use crate::error::{KevError, Result};
use crate::head::PointerHead;
use crate::tokenizer::KevTokenizer;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
}

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Directory holding `base/` (model snapshot), `adapter/`,
    /// `head.safetensors` and `head.meta.json`.
    pub model_dir: PathBuf,
    pub device: Device,
    /// Override the checkpoint temperature (1.0 = raw logits). None = as stored.
    pub temperature: Option<f32>,
}

pub struct Evaluation {
    /// Wire answers object, upstream shape, 4-decimal rounding.
    pub answers: Value,
    /// Per-question calibrated probabilities (full precision).
    pub probs: Vec<Vec<f64>>,
    pub metas: Vec<QuestionMeta>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub latency_ms: f64,
    pub prefix_cache_hit: bool,
}

/// One branch row: its token ids and the branch-relative indices whose
/// hidden states the head needs (decide first, then the options).
pub struct RowQuery {
    pub ids: Vec<u32>,
    pub indices: Vec<usize>,
}

pub(crate) enum Backbone {
    #[cfg(feature = "mlx")]
    Mlx(crate::backend_mlx::MlxBackbone),
    #[cfg(feature = "candle")]
    Candle(crate::backend_candle::CandleBackbone),
}

impl Backbone {
    /// fp32 hidden states at each row's requested indices, for causal rows
    /// `state_ids ++ row.ids`, reusing a cached state prefix.
    fn hidden_rows(&mut self, state_ids: &[u32], rows: &[RowQuery]) -> Result<Vec<Vec<Vec<f32>>>> {
        // Consumed only when a backend feature is compiled in.
        let _ = (state_ids, rows);
        match self {
            #[cfg(feature = "mlx")]
            Backbone::Mlx(backend) => backend.hidden_rows(state_ids, rows),
            #[cfg(feature = "candle")]
            Backbone::Candle(backend) => backend.hidden_rows(state_ids, rows),
            #[allow(unreachable_patterns)]
            _ => unreachable!("backbone exists only with a backend feature"),
        }
    }

    fn clear_prefix(&mut self) {
        match self {
            #[cfg(feature = "mlx")]
            Backbone::Mlx(backend) => backend.clear_prefix(),
            #[cfg(feature = "candle")]
            Backbone::Candle(backend) => backend.clear_prefix(),
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    /// Whether the last `hidden_at` call replayed a cached state prefix.
    fn last_prefix_hit(&self) -> bool {
        match self {
            #[cfg(feature = "mlx")]
            Backbone::Mlx(backend) => backend.last_prefix_hit(),
            #[cfg(feature = "candle")]
            Backbone::Candle(backend) => backend.last_prefix_hit(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

pub struct Runtime {
    pub tokenizer: KevTokenizer,
    pub head: PointerHead,
    backbone: Backbone,
    pub hybrid: bool,
    pub backend_name: &'static str,
}

#[derive(serde::Deserialize)]
struct LayerTypesConfig {
    text_config: Option<TextLayerTypes>,
    layer_types: Option<Vec<String>>,
}

#[derive(serde::Deserialize)]
struct TextLayerTypes {
    layer_types: Option<Vec<String>>,
}

fn is_hybrid(base_dir: &std::path::Path) -> Result<bool> {
    let raw = std::fs::read_to_string(base_dir.join("config.json"))?;
    let config: LayerTypesConfig = serde_json::from_str(&raw)
        .map_err(|error| KevError::Load(format!("config.json: {error}")))?;
    let layer_types = config
        .text_config
        .and_then(|t| t.layer_types)
        .or(config.layer_types)
        .unwrap_or_default();
    Ok(layer_types.iter().any(|t| t == "linear_attention"))
}

impl Runtime {
    pub fn load(options: &LoadOptions) -> Result<Self> {
        let base_dir = options.model_dir.join("base");
        let adapter_dir = options.model_dir.join("adapter");
        let head_st = options.model_dir.join("head.safetensors");
        let head_meta = options.model_dir.join("head.meta.json");
        for path in [&base_dir, &adapter_dir] {
            if !path.is_dir() {
                return Err(KevError::Load(format!(
                    "missing directory {}",
                    path.display()
                )));
            }
        }
        let hybrid = is_hybrid(&base_dir)?;
        #[allow(unused_variables)]
        let tokenizer = KevTokenizer::load(&base_dir)?;
        #[allow(unused_mut, unused_variables)]
        let mut head = PointerHead::load(&head_st, &head_meta)?;
        if let Some(t) = options.temperature {
            head.temperature = t;
        }

        let (backbone, backend_name) = match options.device {
            Device::Metal => {
                if !hybrid {
                    return Err(KevError::Load(
                        "the MLX backend is for the hybrid (Qwen3.5) bases; this base is \
                         attention-only (Qwen3 generation) and runs on the cpu backend"
                            .into(),
                    ));
                }
                #[cfg(feature = "mlx")]
                {
                    (
                        Backbone::Mlx(crate::backend_mlx::MlxBackbone::load(
                            &base_dir,
                            &adapter_dir,
                            tokenizer.pad_id,
                        )?),
                        "mlx",
                    )
                }
                #[cfg(not(feature = "mlx"))]
                {
                    return Err(KevError::Load(
                        "device metal requires a build with the mlx feature".into(),
                    ));
                }
            }
            Device::Cpu => {
                if hybrid {
                    return Err(KevError::Load(
                        "the cpu backend supports the attention-only (Qwen3) generation only \
                         (e.g. kev-0.6b); Qwen3.5 hybrid CPU inference is not implemented — \
                         use device = \"metal\" or a Qwen3-generation checkpoint"
                            .into(),
                    ));
                }
                #[cfg(feature = "candle")]
                {
                    (
                        Backbone::Candle(crate::backend_candle::CandleBackbone::load(
                            &base_dir,
                            &adapter_dir,
                        )?),
                        "candle",
                    )
                }
                #[cfg(not(feature = "candle"))]
                {
                    return Err(KevError::Load(
                        "device cpu requires a build with the candle feature".into(),
                    ));
                }
            }
        };

        Ok(Self {
            tokenizer,
            head,
            backbone,
            hybrid,
            backend_name,
        })
    }

    /// Drop the cached state prefix (bench/testing control).
    pub fn clear_prefix(&mut self) {
        self.backbone.clear_prefix();
    }

    /// Encode a wire request with the serving limits.
    pub fn encode_request(
        &self,
        request: &SystemOneRequest,
    ) -> Result<(Encoding, Vec<QuestionMeta>)> {
        let (record, metas) = api::to_record(request)?;
        let enc = encode::encode(
            &self.tokenizer,
            &record,
            SERVE_MAX_STATE,
            SERVE_MAX_BRANCH,
            false,
        )?;
        Ok((enc, metas))
    }

    pub fn evaluate(&mut self, request: &SystemOneRequest) -> Result<Evaluation> {
        let (enc, metas) = self.encode_request(request)?;
        self.evaluate_encoded(&enc, metas)
    }

    /// Calibrated probabilities for an encoded request \u2014 the scope of
    /// upstream's `probs_and_prefix` / `probs_with_prefix` (backbone + head,
    /// no encoding or answer serialization). Returns per-question probs and
    /// whether the state prefix was replayed from cache.
    pub fn probs_encoded(&mut self, enc: &Encoding) -> Result<(Vec<Vec<f64>>, bool)> {
        let (state_ids, _state_pos, rows) = encode::rows_of(enc)?;
        let queries: Vec<RowQuery> = rows
            .iter()
            .map(|row| {
                let mut indices = vec![row.decide];
                indices.extend(&row.opts);
                RowQuery {
                    ids: row.ids.clone(),
                    indices,
                }
            })
            .collect();
        let picked_rows = self.backbone.hidden_rows(&state_ids, &queries)?;
        let prefix_cache_hit = self.backbone.last_prefix_hit();
        let mut probs = Vec::with_capacity(rows.len());
        for picked in &picked_rows {
            let raw = self.head.raw_logits(&picked[0], &picked[1..])?;
            probs.push(self.head.probs(&raw));
        }
        Ok((probs, prefix_cache_hit))
    }

    pub fn evaluate_encoded(
        &mut self,
        enc: &Encoding,
        metas: Vec<QuestionMeta>,
    ) -> Result<Evaluation> {
        let start = Instant::now();
        let (probs, prefix_cache_hit) = self.probs_encoded(enc)?;
        let answers = api::to_answers(&probs, &metas);
        let serialized = api::python_dumps(&answers);
        let output_tokens = self.tokenizer.raw_tokens(&serialized)?.len();
        Ok(Evaluation {
            answers,
            probs,
            metas,
            input_tokens: enc.ids.len(),
            output_tokens,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            prefix_cache_hit,
        })
    }
}
