//! Qwen3.5 hybrid backbone (Gated DeltaNet + gated attention) on MLX,
//! ported from mlx-lm 0.31.3 and verified against the K0 goldens at K1.
//! bf16 weights as stored, fp32 LoRA merge on the CPU stream, fp32 GDN
//! state. One causal row per call, with a single-slot state-prefix cache:
//! a repeated state pays only for its question branches (exact by
//! construction — the state's activations do not depend on the branches).

use crate::error::{KevError, Result};
use mlx_rs::{
    error::Exception,
    fast,
    ops::{self, indexing::TryIndexOp},
    transforms::{compile::compile, eval},
    Array, Device, Dtype, Stream,
};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

type AnyResult<T> = anyhow::Result<T>;

// ---------------------------------------------------------------- config --

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
    partial_rotary_factor: f32,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    full_attention_interval: usize,
    linear_conv_kernel_dim: i32,
    linear_key_head_dim: i32,
    linear_num_key_heads: i32,
    linear_num_value_heads: i32,
    linear_value_head_dim: i32,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
}

#[derive(Debug, Deserialize)]
struct TopConfig {
    text_config: TextConfig,
}

impl TextConfig {
    fn is_linear(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }
    fn rotary_dims(&self) -> i32 {
        (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as i32
    }
}

// ------------------------------------------------------------- layer state --

enum LayerState {
    /// Attention: cached keys/values [1, kv_heads, S, head_dim].
    Kv { keys: Array, values: Array },
    /// GDN: conv tail [1, K-1, conv_dim] and recurrent state [1, Hv, Dv, Dk] (fp32).
    Gdn { conv_tail: Array, state: Array },
}

struct Prefix {
    state_ids: Vec<u32>,
    layers: Vec<LayerState>,
}

/// Weights of one transformer block, resolved once at load so the forward
/// pass never touches a name map.
struct AttnWeights {
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    q_norm: Array,
    k_norm: Array,
}

struct GdnWeights {
    in_proj_qkv: Array,
    in_proj_z: Array,
    in_proj_b: Array,
    in_proj_a: Array,
    conv1d: Array,
    /// Precomputed `-exp(A_log.astype(f32))` — a weight-only constant that
    /// upstream recomputes inside `compute_g` every call.
    neg_exp_a_log: Array,
    dt_bias: Array,
    norm: Array,
    out_proj: Array,
}

enum Mixer {
    Attn(AttnWeights),
    Gdn(GdnWeights),
}

type Fn2 = Box<dyn for<'a> FnMut((&'a Array, &'a Array)) -> std::result::Result<Array, Exception>>;
type Fn3 = Box<
    dyn for<'a> FnMut(
        (&'a Array, &'a Array, &'a Array),
    ) -> std::result::Result<Array, Exception>,
>;

/// Compiled elementwise glue, mirroring the `mx.compile`d helpers the
/// Python baseline runs (`swiglu`, `_precise_swiglu`, `compute_g`): one
/// fused kernel launch instead of a chain of tiny ones.
struct CompiledOps {
    /// (gate, x) -> silu(gate) * x
    swiglu: Fn2,
    /// (y, z) -> (silu(z as f32) * (y as f32)) as y.dtype  \u2014 the gated
    /// RMSNorm's precise swiglu (norm already applied to y).
    gated_swiglu: Fn2,
    /// (neg_exp_a_log, a, dt_bias) -> exp(neg_exp_a_log * softplus(a + dt_bias))
    compute_g: Fn3,
}

impl CompiledOps {
    fn new() -> Self {
        Self {
            swiglu: Box::new(compile(
                |(gate, x): (&Array, &Array)| -> std::result::Result<Array, Exception> {
                    gate.multiply(&ops::sigmoid(gate)?)?.multiply(x)
                },
                true,
            )),
            gated_swiglu: Box::new(compile(
                |(normed, z): (&Array, &Array)| -> std::result::Result<Array, Exception> {
                    let dtype = z.dtype();
                    let z_f32 = z.as_dtype(Dtype::Float32)?;
                    let gate = z_f32.multiply(&ops::sigmoid(&z_f32)?)?;
                    gate.multiply(&normed.as_dtype(Dtype::Float32)?)?.as_dtype(dtype)
                },
                true,
            )),
            compute_g: Box::new(compile(
                |(neg_exp_a_log, a, dt_bias): (
                    &Array,
                    &Array,
                    &Array,
                )|
                 -> std::result::Result<Array, Exception> {
                    let zero = Array::from_f32(0.0).as_dtype(a.dtype())?;
                    let softplus = ops::logaddexp(&a.add(dt_bias)?, &zero)?;
                    ops::exp(&neg_exp_a_log.multiply(&softplus)?)
                },
                true,
            )),
        }
    }
}

struct Block {
    input_ln: Array,
    mixer: Mixer,
    post_ln: Array,
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,
}

pub struct MlxBackbone {
    config: TextConfig,
    embed: Array,
    blocks: Vec<Block>,
    final_norm: Array,
    /// Scalar constants in the compute dtype: q scale (dk^-1), k scale
    /// (dk^-0.5), and zero (softplus via logaddexp).
    q_scale: Array,
    k_scale: Array,
    zero: Array,
    kernel: crate::gdn_kernel::GdnKernel,
    compiled: RefCell<CompiledOps>,
    pad_id: u32,
    prefix: Option<Prefix>,
    last_hit: bool,
}

// ----------------------------------------------------------------- loading --

const SHIFTED_NORM_SUFFIXES: [&str; 5] = [
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
];

fn load_base(base_dir: &Path) -> AnyResult<HashMap<String, Array>> {
    let mut files: Vec<_> = std::fs::read_dir(base_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no safetensors in {}", base_dir.display());

    let mut raw = HashMap::new();
    for file in &files {
        raw.extend(Array::load_safetensors(file)?);
    }
    let has_mtp = raw.keys().any(|k| k.contains("mtp."));
    let mut weights = HashMap::new();
    for (key, value) in raw {
        if key.starts_with("model.visual") || key.contains("mtp.") || key == "lm_head.weight" {
            continue;
        }
        let name = key
            .strip_prefix("model.language_model.")
            .ok_or_else(|| anyhow::anyhow!("unexpected key {key}"))?
            .to_string();
        let mut tensor = value;
        if name.ends_with("conv1d.weight") && tensor.shape()[2] != 1 {
            tensor = ops::swap_axes(&tensor, 2, 1)?;
        }
        if has_mtp
            && tensor.ndim() == 1
            && SHIFTED_NORM_SUFFIXES
                .iter()
                .any(|s| name.ends_with(s) || format!("model.{name}").ends_with(s))
        {
            tensor = tensor.add(&Array::from_f32(1.0).as_dtype(tensor.dtype())?)?;
        }
        weights.insert(name, tensor);
    }
    Ok(weights)
}

#[derive(Debug, Deserialize)]
struct AdapterConfig {
    r: f32,
    lora_alpha: f32,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    trainable_token_indices: Option<serde_json::Value>,
}

fn merge_lora(weights: &mut HashMap<String, Array>, adapter_dir: &Path) -> AnyResult<usize> {
    let config: AdapterConfig =
        serde_json::from_str(&std::fs::read_to_string(adapter_dir.join("adapter_config.json"))?)?;
    anyhow::ensure!(
        config.trainable_token_indices.is_none(),
        "adapters with trainable_token_indices are not supported on the MLX path"
    );
    let alpha = config.lora_alpha
        / if config.use_rslora { config.r.sqrt() } else { config.r };
    let adapter = Array::load_safetensors(adapter_dir.join("adapter_model.safetensors"))?;

    let cpu = Stream::new_with_device(&Device::cpu());
    let mut merged = 0usize;
    mlx_rs::with_new_default_stream(cpu, || -> AnyResult<()> {
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
            let delta = ops::matmul(
                &lora_b.as_dtype(Dtype::Float32)?,
                &lora_a.as_dtype(Dtype::Float32)?,
            )?
            .multiply(&Array::from_f32(alpha))?;
            let out = base.as_dtype(Dtype::Float32)?.add(&delta)?.as_dtype(base.dtype())?;
            eval([&out])?;
            weights.insert(target, out);
            merged += 1;
        }
        Ok(())
    })?;
    anyhow::ensure!(merged > 0, "no LoRA pairs merged");
    Ok(merged)
}

// ----------------------------------------------------------------- forward --

fn linear(x: &Array, weight: &Array) -> AnyResult<Array> {
    Ok(ops::matmul(x, &ops::swap_axes(weight, -1, -2)?)?)
}

fn silu(x: &Array) -> AnyResult<Array> {
    Ok(x.multiply(&ops::sigmoid(x)?)?)
}

impl MlxBackbone {
    pub fn load(base_dir: &Path, adapter_dir: &Path, pad_id: u32) -> Result<Self> {
        let raw = std::fs::read_to_string(base_dir.join("config.json"))
            .map_err(|e| KevError::Load(e.to_string()))?;
        let top: TopConfig =
            serde_json::from_str(&raw).map_err(|e| KevError::Load(format!("config.json: {e}")))?;
        let mut weights =
            load_base(base_dir).map_err(|e| KevError::Load(format!("base weights: {e}")))?;
        merge_lora(&mut weights, adapter_dir)
            .map_err(|e| KevError::Load(format!("lora merge: {e}")))?;
        let kernel = crate::gdn_kernel::GdnKernel::new()?;

        let cfg = &top.text_config;
        let mut take = |name: String| -> Result<Array> {
            weights
                .remove(&name)
                .ok_or_else(|| KevError::Load(format!("missing weight {name}")))
        };
        let embed = take("embed_tokens.weight".into())?;
        let dtype = embed.dtype();
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in 0..cfg.num_hidden_layers {
            let p = format!("layers.{layer}");
            let mixer = if cfg.is_linear(layer) {
                let a_log = take(format!("{p}.linear_attn.A_log"))?;
                let neg_exp_a_log = (|| -> AnyResult<Array> {
                    let value =
                        ops::negative(&ops::exp(&a_log.as_dtype(Dtype::Float32)?)?)?;
                    eval([&value])?;
                    Ok(value)
                })()
                .map_err(|e| KevError::Load(format!("A_log: {e}")))?;
                Mixer::Gdn(GdnWeights {
                    in_proj_qkv: take(format!("{p}.linear_attn.in_proj_qkv.weight"))?,
                    in_proj_z: take(format!("{p}.linear_attn.in_proj_z.weight"))?,
                    in_proj_b: take(format!("{p}.linear_attn.in_proj_b.weight"))?,
                    in_proj_a: take(format!("{p}.linear_attn.in_proj_a.weight"))?,
                    conv1d: take(format!("{p}.linear_attn.conv1d.weight"))?,
                    neg_exp_a_log,
                    dt_bias: take(format!("{p}.linear_attn.dt_bias"))?,
                    norm: take(format!("{p}.linear_attn.norm.weight"))?,
                    out_proj: take(format!("{p}.linear_attn.out_proj.weight"))?,
                })
            } else {
                Mixer::Attn(AttnWeights {
                    q_proj: take(format!("{p}.self_attn.q_proj.weight"))?,
                    k_proj: take(format!("{p}.self_attn.k_proj.weight"))?,
                    v_proj: take(format!("{p}.self_attn.v_proj.weight"))?,
                    o_proj: take(format!("{p}.self_attn.o_proj.weight"))?,
                    q_norm: take(format!("{p}.self_attn.q_norm.weight"))?,
                    k_norm: take(format!("{p}.self_attn.k_norm.weight"))?,
                })
            };
            blocks.push(Block {
                input_ln: take(format!("{p}.input_layernorm.weight"))?,
                mixer,
                post_ln: take(format!("{p}.post_attention_layernorm.weight"))?,
                gate_proj: take(format!("{p}.mlp.gate_proj.weight"))?,
                up_proj: take(format!("{p}.mlp.up_proj.weight"))?,
                down_proj: take(format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let final_norm = take("norm.weight".into())?;

        let inv_scale = (cfg.linear_key_head_dim as f32).powf(-0.5);
        let make_const = |value: f32| -> Result<Array> {
            Array::from_f32(value)
                .as_dtype(dtype)
                .map_err(|e| KevError::Load(e.to_string()))
        };
        Ok(Self {
            config: top.text_config,
            embed,
            blocks,
            final_norm,
            q_scale: make_const(inv_scale * inv_scale)?,
            k_scale: make_const(inv_scale)?,
            zero: make_const(0.0)?,
            kernel,
            compiled: RefCell::new(CompiledOps::new()),
            pad_id,
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

    fn fresh_states(&self) -> AnyResult<Vec<LayerState>> {
        let cfg = &self.config;
        let conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let embed_dtype = self.embed.dtype();
        (0..cfg.num_hidden_layers)
            .map(|layer| {
                Ok(if cfg.is_linear(layer) {
                    LayerState::Gdn {
                        conv_tail: ops::zeros_dtype(
                            &[1, cfg.linear_conv_kernel_dim - 1, conv_dim],
                            embed_dtype,
                        )?,
                        state: ops::zeros_dtype(
                            &[
                                1,
                                cfg.linear_num_value_heads,
                                cfg.linear_value_head_dim,
                                cfg.linear_key_head_dim,
                            ],
                            Dtype::Float32,
                        )?,
                    }
                } else {
                    LayerState::Kv {
                        keys: ops::zeros_dtype(
                            &[1, cfg.num_key_value_heads, 0, cfg.head_dim],
                            embed_dtype,
                        )?,
                        values: ops::zeros_dtype(
                            &[1, cfg.num_key_value_heads, 0, cfg.head_dim],
                            embed_dtype,
                        )?,
                    }
                })
            })
            .collect()
    }

    /// Forward a right-padded batch of rows continuing `states` (mutated in
    /// place) with rope positions starting at `pos_offset`. Pads sit after
    /// every real token and both layer kinds are causal, so no real token
    /// sees a pad (upstream's batching argument). Returns hidden after the
    /// final norm, `[B, L, d]`.
    fn forward(
        &self,
        rows: &[Vec<u32>],
        states: &mut [LayerState],
        pos_offset: i32,
    ) -> AnyResult<Array> {
        let cfg = &self.config;
        let b = rows.len() as i32;
        let l = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut ids_i32 = Vec::with_capacity(rows.len() * l);
        for row in rows {
            ids_i32.extend(row.iter().map(|i| *i as i32));
            ids_i32.extend(std::iter::repeat_n(self.pad_id as i32, l - row.len()));
        }
        let ids_arr = Array::from_slice(&ids_i32, &[(rows.len() * l) as i32]);
        let mut h = self
            .embed
            .take_axis(&ids_arr, 0)?
            .reshape(&[b, l as i32, cfg.hidden_size])?;

        for (block, state) in self.blocks.iter().zip(states.iter_mut()) {
            let normed = fast::rms_norm(&h, Some(&block.input_ln), cfg.rms_norm_eps)?;
            let r = match &block.mixer {
                Mixer::Gdn(w) => self.gated_delta_net(&normed, w, state)?,
                Mixer::Attn(w) => self.attention(&normed, w, state, pos_offset)?,
            };
            let mid = h.add(&r)?;
            let normed2 = fast::rms_norm(&mid, Some(&block.post_ln), cfg.rms_norm_eps)?;
            let gate = linear(&normed2, &block.gate_proj)?;
            let up = linear(&normed2, &block.up_proj)?;
            let fused = (self.compiled.borrow_mut().swiglu)((&gate, &up))?;
            let mlp = linear(&fused, &block.down_proj)?;
            h = mid.add(&mlp)?;
        }
        Ok(fast::rms_norm(&h, Some(&self.final_norm), cfg.rms_norm_eps)?)
    }

    fn attention(
        &self,
        x: &Array,
        w: &AttnWeights,
        state: &mut LayerState,
        pos_offset: i32,
    ) -> AnyResult<Array> {
        let cfg = &self.config;
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let qg = linear(x, &w.q_proj)?.reshape(&[b, l, heads, 2 * head_dim])?;
        let parts = qg.split(2, -1)?;
        let (queries, gate) = (&parts[0], parts[1].reshape(&[b, l, -1])?);

        let keys = linear(x, &w.k_proj)?.reshape(&[b, l, kv_heads, head_dim])?;
        let values = linear(x, &w.v_proj)?
            .reshape(&[b, l, kv_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let queries = fast::rms_norm(queries, Some(&w.q_norm), cfg.rms_norm_eps)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = fast::rms_norm(&keys, Some(&w.k_norm), cfg.rms_norm_eps)?
            .transpose_axes(&[0, 2, 1, 3])?;

        let dims = cfg.rotary_dims();
        let base = cfg.rope_parameters.rope_theta;
        let queries = fast::rope(&queries, dims, false, base, 1.0, pos_offset, None)?;
        let keys = fast::rope(&keys, dims, false, base, 1.0, pos_offset, None)?;

        let LayerState::Kv { keys: cached_k, values: cached_v } = state else {
            anyhow::bail!("layer state mismatch: expected Kv");
        };
        let all_keys = if cached_k.shape()[2] == 0 {
            keys
        } else {
            ops::concatenate_axis(&[&*cached_k, &keys], 2)?
        };
        let all_values = if cached_v.shape()[2] == 0 {
            values
        } else {
            ops::concatenate_axis(&[&*cached_v, &values], 2)?
        };

        let scale = (head_dim as f32).powf(-0.5);
        let out = fast::scaled_dot_product_attention(
            &queries,
            &all_keys,
            &all_values,
            scale,
            fast::ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        *state = LayerState::Kv { keys: all_keys, values: all_values };

        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        let gated = out.multiply(&ops::sigmoid(&gate)?)?;
        linear(&gated, &w.o_proj)
    }

    fn gated_delta_net(
        &self,
        x: &Array,
        w: &GdnWeights,
        layer_state: &mut LayerState,
    ) -> AnyResult<Array> {
        let cfg = &self.config;
        let shape = x.shape();
        let (b, s) = (shape[0], shape[1]);
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let key_dim = hk * dk;
        let conv_dim = 2 * key_dim + hv * dv;
        let kernel = cfg.linear_conv_kernel_dim;

        let qkv = linear(x, &w.in_proj_qkv)?;
        let z = linear(x, &w.in_proj_z)?.reshape(&[b, s, hv, dv])?;
        let bb = linear(x, &w.in_proj_b)?;
        let a = linear(x, &w.in_proj_a)?;

        let LayerState::Gdn { conv_tail, state } = layer_state else {
            anyhow::bail!("layer state mismatch: expected Gdn");
        };

        let conv_input = ops::concatenate_axis(&[&*conv_tail, &qkv], 1)?;
        // Keep the last K-1 inputs for the next continuation.
        let total = conv_input.shape()[1];
        let new_tail = conv_input.try_index((.., (total - (kernel - 1))..total, ..))?;
        let conv_out = silu(&ops::conv1d(&conv_input, &w.conv1d, 1, 0, 1, conv_dim)?)?;

        let parts = conv_out.split_axis(&[key_dim, 2 * key_dim], -1)?;
        let q = parts[0].reshape(&[b, s, hk, dk])?;
        let k = parts[1].reshape(&[b, s, hk, dk])?;
        let v = parts[2].reshape(&[b, s, hv, dv])?;

        let q = fast::rms_norm(&q, None, 1e-6)?.multiply(&self.q_scale)?;
        let k = fast::rms_norm(&k, None, 1e-6)?.multiply(&self.k_scale)?;

        let beta = ops::sigmoid(&bb)?;
        let g = (self.compiled.borrow_mut().compute_g)((&w.neg_exp_a_log, &a, &w.dt_bias))?;

        // Fused Metal kernel: one launch for the whole [T]-step recurrence
        // (the same kernel mlx-lm uses; GQA head mapping happens inside).
        let (y, new_state) = self.kernel.apply(&q, &k, &v, &g, &beta, state)?;
        let _ = hv;

        *layer_state = LayerState::Gdn { conv_tail: new_tail, state: new_state };

        let normed = fast::rms_norm(&y, Some(&w.norm), cfg.rms_norm_eps)?;
        let out = (self.compiled.borrow_mut().gated_swiglu)((&normed, &z))?
            .reshape(&[b, s, -1])?;
        linear(&out, &w.out_proj)
    }

    /// Replicate the pristine state prefix for a batch of `b` branch rows
    /// (broadcast along the batch axis; concatenation and the GDN kernel
    /// materialize per-row copies, the cached arrays stay untouched).
    fn replicate(states: &[LayerState], b: i32) -> AnyResult<Vec<LayerState>> {
        states
            .iter()
            .map(|s| {
                Ok(match s {
                    LayerState::Kv { keys, values } => {
                        let mut k_shape = keys.shape().to_vec();
                        k_shape[0] = b;
                        LayerState::Kv {
                            keys: ops::broadcast_to(keys, &k_shape)?,
                            values: ops::broadcast_to(values, &k_shape)?,
                        }
                    }
                    LayerState::Gdn { conv_tail, state } => {
                        let mut c_shape = conv_tail.shape().to_vec();
                        c_shape[0] = b;
                        let mut s_shape = state.shape().to_vec();
                        s_shape[0] = b;
                        LayerState::Gdn {
                            conv_tail: ops::broadcast_to(conv_tail, &c_shape)?,
                            state: ops::broadcast_to(state, &s_shape)?,
                        }
                    }
                })
            })
            .collect()
    }

    /// fp32 hidden states at each row's branch-relative indices, all branch
    /// rows of one request batched on a replicated state prefix (upstream's
    /// serving path), the prefix cached for the next request.
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
            let mut states = self.fresh_states()?;
            self.forward(&[state_ids.to_vec()], &mut states, 0)?;
            self.prefix = Some(Prefix { state_ids: state_ids.to_vec(), layers: states });
        }
        self.last_hit = hit;

        // Rows per pass bounded by a token budget (like upstream's
        // rows_per_pass): every row costs state + padded-branch tokens.
        let max_branch = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let per_row = state_ids.len() + max_branch;
        let chunk_rows = (16384 / per_row.max(1)).clamp(1, 32);

        let mut out = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(chunk_rows) {
            let b = chunk.len();
            let padded_len = chunk.iter().map(|r| r.ids.len()).max().unwrap_or(0);
            let mut states = Self::replicate(&self.prefix.as_ref().unwrap().layers, b as i32)?;
            let row_ids: Vec<Vec<u32>> = chunk.iter().map(|r| r.ids.clone()).collect();
            let hidden = self.forward(&row_ids, &mut states, state_ids.len() as i32)?;
            // Gather just the picked positions on device (upstream's h[idx]),
            // then materialize them in fp32.
            let mut positions: Vec<i32> = Vec::new();
            for (i, row) in chunk.iter().enumerate() {
                for j in &row.indices {
                    anyhow::ensure!(*j < row.ids.len(), "index {j} out of branch");
                    positions.push((i * padded_len + j) as i32);
                }
            }
            let d = hidden.shape()[2];
            let pos_arr = Array::from_slice(&positions, &[positions.len() as i32]);
            let picked_f32 = hidden
                .reshape(&[-1, d])?
                .take_axis(&pos_arr, 0)?
                .as_dtype(Dtype::Float32)?;
            eval([&picked_f32])?;
            let flat: Vec<f32> = picked_f32.as_slice().to_vec();
            let d = d as usize;
            let mut cursor = 0usize;
            for row in chunk {
                let picked = row
                    .indices
                    .iter()
                    .map(|_| {
                        let v = flat[cursor * d..(cursor + 1) * d].to_vec();
                        cursor += 1;
                        v
                    })
                    .collect::<Vec<_>>();
                out.push(picked);
            }
        }
        Ok(out)
    }
}
