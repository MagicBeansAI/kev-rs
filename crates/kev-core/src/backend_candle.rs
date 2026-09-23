//! Portable CPU backbone for the Qwen3 attention-only generation
//! (kev-0.6b), on Candle in fp32 — the same precision as the torch fp32
//! oracle (upstream loads the bf16 checkpoint with dtype=fp32 and merges
//! the LoRA in fp32). Row form with a single-slot state-prefix KV cache.
//!
//! The Qwen3.5 hybrid generation is *not* supported here; `Runtime::load`
//! refuses it with a clear error (see docs/K1-RUNTIME.md for the honest
//! CPU scope and the llama.cpp follow-up).

use crate::error::{KevError, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

type AnyResult<T> = anyhow::Result<T>;

#[derive(Debug, Deserialize)]
struct Config {
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
}

struct KvState {
    keys: Option<Tensor>,   // [kv_heads, S, head_dim]
    values: Option<Tensor>, // [kv_heads, S, head_dim]
}

struct Prefix {
    state_ids: Vec<u32>,
    layers: Vec<KvState>,
}

pub struct CandleBackbone {
    config: Config,
    weights: HashMap<String, Tensor>,
    device: Device,
    prefix: Option<Prefix>,
    last_hit: bool,
}

#[derive(Debug, Deserialize)]
struct AdapterConfig {
    r: f64,
    lora_alpha: f64,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    trainable_token_indices: Option<serde_json::Value>,
}

impl CandleBackbone {
    pub fn load(base_dir: &Path, adapter_dir: &Path) -> Result<Self> {
        Self::load_inner(base_dir, adapter_dir).map_err(|e| KevError::Load(e.to_string()))
    }

    fn load_inner(base_dir: &Path, adapter_dir: &Path) -> AnyResult<Self> {
        let raw = std::fs::read_to_string(base_dir.join("config.json"))?;
        let config: Config = serde_json::from_str(&raw)?;
        let device = Device::Cpu;

        let mut files: Vec<_> = std::fs::read_dir(base_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        anyhow::ensure!(
            !files.is_empty(),
            "no safetensors in {}",
            base_dir.display()
        );

        let mut weights = HashMap::new();
        for file in &files {
            for (key, tensor) in candle_core::safetensors::load(file, &device)? {
                if key == "lm_head.weight" {
                    continue;
                }
                let name = key
                    .strip_prefix("model.")
                    .ok_or_else(|| anyhow::anyhow!("unexpected key {key}"))?
                    .to_string();
                // fp32 everywhere: the exact-oracle precision.
                weights.insert(name, tensor.to_dtype(DType::F32)?);
            }
        }

        // fp32 LoRA merge, upstream's math (alpha / r, no rslora here).
        let adapter_config: AdapterConfig = serde_json::from_str(&std::fs::read_to_string(
            adapter_dir.join("adapter_config.json"),
        )?)?;
        anyhow::ensure!(
            adapter_config.trainable_token_indices.is_none(),
            "adapters with trainable token embeddings stay unmerged upstream; unsupported here"
        );
        let alpha = adapter_config.lora_alpha
            / if adapter_config.use_rslora {
                adapter_config.r.sqrt()
            } else {
                adapter_config.r
            };
        let adapter =
            candle_core::safetensors::load(adapter_dir.join("adapter_model.safetensors"), &device)?;
        let mut merged = 0usize;
        for (key, lora_a) in &adapter {
            let Some(path) = key
                .strip_prefix("base_model.model.")
                .and_then(|k| k.strip_suffix(".lora_A.weight"))
            else {
                continue;
            };
            let lora_b = adapter
                .get(&format!("base_model.model.{path}.lora_B.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing lora_B for {path}"))?;
            let target = format!("{path}.weight");
            let base = weights
                .get(&target)
                .ok_or_else(|| anyhow::anyhow!("adapter targets missing weight {target}"))?;
            let delta = lora_b
                .to_dtype(DType::F32)?
                .matmul(&lora_a.to_dtype(DType::F32)?)?
                .affine(alpha, 0.0)?;
            weights.insert(target, (base + delta)?);
            merged += 1;
        }
        anyhow::ensure!(merged > 0, "no LoRA pairs merged");

        Ok(Self {
            config,
            weights,
            device,
            prefix: None,
            last_hit: false,
        })
    }

    pub fn last_prefix_hit(&self) -> bool {
        self.last_hit
    }

    pub fn clear_prefix(&mut self) {
        self.prefix = None;
    }

    fn w(&self, name: &str) -> AnyResult<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing weight {name}"))
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f64) -> AnyResult<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
        Ok(normed.broadcast_mul(weight)?)
    }

    /// Non-interleaved (rotate-half) rope over the full head_dim.
    fn rope(&self, x: &Tensor, pos_offset: usize) -> AnyResult<Tensor> {
        // x: [heads, L, head_dim]
        let (_h, l, d) = x.dims3()?;
        let half = d / 2;
        let theta = self.config.rope_theta;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / d as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), &self.device)?;
        let positions: Vec<f32> = (0..l).map(|p| (p + pos_offset) as f32).collect();
        let positions = Tensor::from_vec(positions, (l, 1), &self.device)?;
        let angles = positions.matmul(&inv_freq)?; // [L, half]
        let cos = angles.cos()?.unsqueeze(0)?; // [1, L, half]
        let sin = angles.sin()?.unsqueeze(0)?;

        let x1 = x.i((.., .., ..half))?;
        let x2 = x.i((.., .., half..))?;
        let rotated_1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        Ok(Tensor::cat(&[&rotated_1, &rotated_2], D::Minus1)?)
    }

    /// Causal SDPA with cached prefix keys (bottom-right aligned mask).
    fn attention_scores(
        &self,
        queries: &Tensor, // [heads, Lq, d]
        keys: &Tensor,    // [heads, Lk, d]
        values: &Tensor,
    ) -> AnyResult<Tensor> {
        let (_h, lq, d) = queries.dims3()?;
        let lk = keys.dim(1)?;
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = (queries.matmul(&keys.transpose(1, 2)?)? * scale)?; // [h, Lq, Lk]
        if lq > 1 {
            // token i (global position lk-lq+i) may attend keys 0..=lk-lq+i
            let mut mask = vec![0f32; lq * lk];
            for i in 0..lq {
                let limit = lk - lq + i;
                for (j, slot) in mask[i * lk..(i + 1) * lk].iter_mut().enumerate() {
                    if j > limit {
                        *slot = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::from_vec(mask, (1, lq, lk), &self.device)?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        Ok(probs.matmul(values)?)
    }

    fn forward(&self, ids: &[u32], states: &mut [KvState], pos_offset: usize) -> AnyResult<Tensor> {
        let cfg = &self.config;
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let d = cfg.head_dim;
        let l = ids.len();

        let embed = self.w("embed_tokens.weight")?;
        let ids_t = Tensor::from_vec(ids.to_vec(), l, &self.device)?;
        let mut h = embed.index_select(&ids_t, 0)?; // [L, hidden]

        for (layer, state) in states.iter_mut().enumerate() {
            let p = format!("layers.{layer}");
            let normed = self.rms_norm(
                &h,
                self.w(&format!("{p}.input_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )?;

            let queries = normed
                .matmul(&self.w(&format!("{p}.self_attn.q_proj.weight"))?.t()?)?
                .reshape((l, heads, d))?;
            let keys = normed
                .matmul(&self.w(&format!("{p}.self_attn.k_proj.weight"))?.t()?)?
                .reshape((l, kv_heads, d))?;
            let values = normed
                .matmul(&self.w(&format!("{p}.self_attn.v_proj.weight"))?.t()?)?
                .reshape((l, kv_heads, d))?
                .transpose(0, 1)?; // [kv_heads, L, d]

            // Per-head QK RMSNorm (Qwen3), then rope with position offset.
            let queries = self
                .rms_norm(
                    &queries,
                    self.w(&format!("{p}.self_attn.q_norm.weight"))?,
                    cfg.rms_norm_eps,
                )?
                .transpose(0, 1)?; // [heads, L, d]
            let keys = self
                .rms_norm(
                    &keys,
                    self.w(&format!("{p}.self_attn.k_norm.weight"))?,
                    cfg.rms_norm_eps,
                )?
                .transpose(0, 1)?; // [kv_heads, L, d]
            let queries = self.rope(&queries, pos_offset)?;
            let keys = self.rope(&keys, pos_offset)?;

            let all_keys = match &state.keys {
                Some(cached) => Tensor::cat(&[cached, &keys], 1)?,
                None => keys,
            };
            let all_values = match &state.values {
                Some(cached) => Tensor::cat(&[cached, &values], 1)?,
                None => values,
            };
            state.keys = Some(all_keys.clone());
            state.values = Some(all_values.clone());

            // GQA: repeat kv heads.
            let group = heads / kv_heads;
            let lk = all_keys.dim(1)?;
            let rep_k = all_keys
                .unsqueeze(1)?
                .broadcast_as((kv_heads, group, lk, d))?
                .reshape((heads, lk, d))?
                .contiguous()?;
            let rep_v = all_values
                .unsqueeze(1)?
                .broadcast_as((kv_heads, group, lk, d))?
                .reshape((heads, lk, d))?
                .contiguous()?;

            let attn = self.attention_scores(&queries.contiguous()?, &rep_k, &rep_v)?; // [heads, L, d]
            let attn = attn.transpose(0, 1)?.reshape((l, heads * d))?;
            let attn_out = attn.matmul(&self.w(&format!("{p}.self_attn.o_proj.weight"))?.t()?)?;

            let mid = (&h + attn_out)?;
            let normed2 = self.rms_norm(
                &mid,
                self.w(&format!("{p}.post_attention_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )?;
            let gate = normed2.matmul(&self.w(&format!("{p}.mlp.gate_proj.weight"))?.t()?)?;
            let up = normed2.matmul(&self.w(&format!("{p}.mlp.up_proj.weight"))?.t()?)?;
            let silu = (&gate * candle_nn::ops::sigmoid(&gate)?)?;
            let mlp = (silu * up)?.matmul(&self.w(&format!("{p}.mlp.down_proj.weight"))?.t()?)?;
            h = (mid + mlp)?;
        }
        self.rms_norm(&h, self.w("norm.weight")?, cfg.rms_norm_eps)
    }

    fn fresh_states(&self) -> Vec<KvState> {
        (0..self.config.num_hidden_layers)
            .map(|_| KvState {
                keys: None,
                values: None,
            })
            .collect()
    }

    fn snapshot(states: &[KvState]) -> Vec<KvState> {
        states
            .iter()
            .map(|s| KvState {
                keys: s.keys.clone(),
                values: s.values.clone(),
            })
            .collect()
    }

    /// Per-row branch passes over the cached state prefix. The CPU path is
    /// correctness-gated only (benchmarked, not raced), so rows are not
    /// batched here.
    pub fn hidden_rows(
        &mut self,
        state_ids: &[u32],
        rows: &[crate::runtime::RowQuery],
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        self.hidden_rows_inner(state_ids, rows)
            .map_err(|e| KevError::Inference(e.to_string()))
    }

    fn hidden_rows_inner(
        &mut self,
        state_ids: &[u32],
        rows: &[crate::runtime::RowQuery],
    ) -> AnyResult<Vec<Vec<Vec<f32>>>> {
        let hit = self
            .prefix
            .as_ref()
            .is_some_and(|p| p.state_ids == state_ids);
        if !hit {
            let mut states = self.fresh_states();
            self.forward(state_ids, &mut states, 0)?;
            self.prefix = Some(Prefix {
                state_ids: state_ids.to_vec(),
                layers: states,
            });
        }
        self.last_hit = hit;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let mut states = Self::snapshot(&self.prefix.as_ref().unwrap().layers);
            let hidden = self.forward(&row.ids, &mut states, state_ids.len())?; // [L, d]
            let picked = row
                .indices
                .iter()
                .map(|i| {
                    anyhow::ensure!(*i < row.ids.len(), "index {i} out of branch");
                    Ok(hidden.i(*i)?.to_vec1::<f32>()?)
                })
                .collect::<AnyResult<Vec<_>>>()?;
            out.push(picked);
        }
        Ok(out)
    }
}
