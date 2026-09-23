//! Qwen3.5 hybrid text backbone (Gated DeltaNet + gated attention), ported
//! from mlx-lm 0.31.3 `models/qwen3_5.py` + `gated_delta.py` + `qwen3_next.py`.
//! Batch = 1, one causal row per call (kev's row form), no KV cache: parity
//! first, speed later.

use crate::config::TextConfig;
use anyhow::{Context, Result};
use mlx_rs::{
    fast,
    ops::{self, indexing::TryIndexOp},
    Array, Dtype,
};
use std::collections::HashMap;

pub struct Model {
    pub config: TextConfig,
    weights: HashMap<String, Array>,
}

fn w<'a>(weights: &'a HashMap<String, Array>, name: &str) -> Result<&'a Array> {
    weights.get(name).with_context(|| format!("missing weight {name}"))
}

fn linear(x: &Array, weight: &Array) -> Result<Array> {
    // nn.Linear: x @ W.T
    Ok(ops::matmul(x, &ops::swap_axes(weight, -1, -2)?)?)
}

fn rms_norm_weighted(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    Ok(fast::rms_norm(x, Some(weight), eps)?)
}

fn silu(x: &Array) -> Result<Array> {
    Ok(x.multiply(&ops::sigmoid(x)?)?)
}

impl Model {
    pub fn new(config: TextConfig, weights: HashMap<String, Array>) -> Self {
        Self { config, weights }
    }

    /// Hidden states after the final norm for one causal row of token ids
    /// (positions are sequential from 0, kev's row-form guarantee).
    pub fn hidden_row(&self, ids: &[i32]) -> Result<Array> {
        let cfg = &self.config;
        let ids = Array::from_slice(ids, &[1, ids.len() as i32]);
        let embed = w(&self.weights, "embed_tokens.weight")?;
        let mut h = embed.take_axis(&ids.reshape(&[-1])?, 0)?;
        h = h.reshape(&[1, -1, cfg.hidden_size])?;

        for layer in 0..cfg.num_hidden_layers {
            let p = format!("layers.{layer}");
            let normed = rms_norm_weighted(
                &h,
                w(&self.weights, &format!("{p}.input_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )?;
            let r = if cfg.is_linear(layer) {
                self.gated_delta_net(&normed, &p)?
            } else {
                self.attention(&normed, &p)?
            };
            let mid = h.add(&r)?;
            let normed2 = rms_norm_weighted(
                &mid,
                w(&self.weights, &format!("{p}.post_attention_layernorm.weight"))?,
                cfg.rms_norm_eps,
            )?;
            let mlp = self.mlp(&normed2, &p)?;
            h = mid.add(&mlp)?;
            // Keep the lazy graph bounded: force one layer at a time.
            mlx_rs::transforms::eval([&h])?;
        }
        rms_norm_weighted(&h, w(&self.weights, "norm.weight")?, cfg.rms_norm_eps)
    }

    fn mlp(&self, x: &Array, p: &str) -> Result<Array> {
        let gate = linear(x, w(&self.weights, &format!("{p}.mlp.gate_proj.weight"))?)?;
        let up = linear(x, w(&self.weights, &format!("{p}.mlp.up_proj.weight"))?)?;
        let inner = silu(&gate)?.multiply(&up)?;
        linear(&inner, w(&self.weights, &format!("{p}.mlp.down_proj.weight"))?)
    }

    fn attention(&self, x: &Array, p: &str) -> Result<Array> {
        let cfg = &self.config;
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let qg = linear(x, w(&self.weights, &format!("{p}.self_attn.q_proj.weight"))?)?
            .reshape(&[b, l, heads, 2 * head_dim])?;
        let parts = qg.split(2, -1)?;
        let (queries, gate) = (&parts[0], parts[1].reshape(&[b, l, -1])?);

        let keys = linear(x, w(&self.weights, &format!("{p}.self_attn.k_proj.weight"))?)?
            .reshape(&[b, l, kv_heads, head_dim])?;
        let values = linear(x, w(&self.weights, &format!("{p}.self_attn.v_proj.weight"))?)?
            .reshape(&[b, l, kv_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let queries = rms_norm_weighted(
            queries,
            w(&self.weights, &format!("{p}.self_attn.q_norm.weight"))?,
            cfg.rms_norm_eps,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let keys = rms_norm_weighted(
            &keys,
            w(&self.weights, &format!("{p}.self_attn.k_norm.weight"))?,
            cfg.rms_norm_eps,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;

        let dims = cfg.rotary_dims();
        let base = cfg.rope_parameters.rope_theta;
        let queries = fast::rope(&queries, dims, false, base, 1.0, 0, None)?;
        let keys = fast::rope(&keys, dims, false, base, 1.0, 0, None)?;

        let scale = (head_dim as f32).powf(-0.5);
        let out = fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            fast::ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        let gated = out.multiply(&ops::sigmoid(&gate)?)?;
        linear(&gated, w(&self.weights, &format!("{p}.self_attn.o_proj.weight"))?)
    }

    fn gated_delta_net(&self, x: &Array, p: &str) -> Result<Array> {
        let cfg = &self.config;
        let shape = x.shape();
        let (b, s) = (shape[0], shape[1]);
        let hk = cfg.linear_num_key_heads;
        let hv = cfg.linear_num_value_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = cfg.linear_conv_kernel_dim;

        let qkv = linear(x, w(&self.weights, &format!("{p}.linear_attn.in_proj_qkv.weight"))?)?;
        let z = linear(x, w(&self.weights, &format!("{p}.linear_attn.in_proj_z.weight"))?)?
            .reshape(&[b, s, hv, dv])?;
        let bb = linear(x, w(&self.weights, &format!("{p}.linear_attn.in_proj_b.weight"))?)?;
        let a = linear(x, w(&self.weights, &format!("{p}.linear_attn.in_proj_a.weight"))?)?;

        // Causal depthwise conv, kernel K, zero initial state, SiLU.
        let conv_state = ops::zeros_dtype(&[b, kernel - 1, conv_dim], qkv.dtype())?;
        let conv_input = ops::concatenate_axis(&[&conv_state, &qkv], 1)?;
        let conv_weight = w(&self.weights, &format!("{p}.linear_attn.conv1d.weight"))?;
        let conv_out = ops::conv1d(&conv_input, conv_weight, 1, 0, 1, conv_dim)?;
        let conv_out = silu(&conv_out)?;

        let parts = conv_out.split_axis(&[key_dim, 2 * key_dim], -1)?;
        let q = parts[0].reshape(&[b, s, hk, dk])?;
        let k = parts[1].reshape(&[b, s, hk, dk])?;
        let v = parts[2].reshape(&[b, s, hv, dv])?;

        // l2-ish norm with the attention scaling folded in (mlx-lm exact form).
        let inv_scale = (dk as f32).powf(-0.5);
        let q = fast::rms_norm(&q, None, 1e-6)?
            .multiply(&Array::from_f32(inv_scale * inv_scale).as_dtype(q.dtype())?)?;
        let k = fast::rms_norm(&k, None, 1e-6)?
            .multiply(&Array::from_f32(inv_scale).as_dtype(k.dtype())?)?;

        // beta = sigmoid(b); g = exp(-exp(A_log_f32) * softplus(a + dt_bias)).
        let beta = ops::sigmoid(&bb)?; // [B,S,Hv]
        let a_log = w(&self.weights, &format!("{p}.linear_attn.A_log"))?;
        let dt_bias = w(&self.weights, &format!("{p}.linear_attn.dt_bias"))?;
        let softplus = ops::logaddexp(&a.add(dt_bias)?, &Array::from_f32(0.0).as_dtype(a.dtype())?)?;
        let g = ops::exp(
            &ops::negative(&ops::exp(&a_log.as_dtype(Dtype::Float32)?)?)?.multiply(&softplus)?,
        )?; // [B,S,Hv] f32

        // Hv > Hk: repeat q/k across value heads (mlx-lm gated_delta_ops:
        // mx.repeat(x, Hv//Hk, -2), i.e. each head duplicated in place).
        let repeat_heads = |x: &Array, factor: i32| -> Result<Array> {
            let shape = x.shape().to_vec(); // [B,S,H,D]
            let expanded = ops::expand_dims_axes(x, &[-2])?; // [B,S,H,1,D]
            let target = [shape[0], shape[1], shape[2], factor, shape[3]];
            let broadcast = ops::broadcast_to(&expanded, &target)?;
            Ok(broadcast.reshape(&[shape[0], shape[1], shape[2] * factor, shape[3]])?)
        };
        let (q, k) = if hv > hk {
            (repeat_heads(&q, hv / hk)?, repeat_heads(&k, hv / hk)?)
        } else {
            (q, k)
        };

        // Sequential delta-rule scan (mlx-lm reference ops), state fp32.
        let mut state = ops::zeros_dtype(&[b, hv, dv, dk], Dtype::Float32)?;
        let mut ys: Vec<Array> = Vec::with_capacity(s as usize);
        for t in 0..s {
            let qt = q.try_index((.., t))?; // [B,Hk,Dk]
            let kt = k.try_index((.., t))?;
            let vt = v.try_index((.., t))?; // [B,Hv,Dv]
            let gt = g.try_index((.., t))?; // [B,Hv] f32
            let bt = beta.try_index((.., t))?; // [B,Hv]

            let decay = ops::expand_dims_axes(&gt, &[-1, -2])?; // [B,Hv,1,1]
            state = state.multiply(&decay)?;
            let k_row = ops::expand_dims_axes(&kt, &[-2])?; // [B,H,1,Dk]
            let kv_mem = state.multiply(&k_row)?.sum_axis(-1, false)?; // [B,H,Dv]
            let delta = vt
                .subtract(&kv_mem)?
                .multiply(&ops::expand_dims_axes(&bt, &[-1])?)?; // [B,H,Dv]
            state = state.add(&k_row.multiply(&ops::expand_dims_axes(&delta, &[-1])?)?)?;
            let q_row = ops::expand_dims_axes(&qt, &[-2])?;
            let y = state.multiply(&q_row)?.sum_axis(-1, false)?; // [B,H,Dv] f32
            ys.push(y.as_dtype(q.dtype())?);
        }
        let y = ops::stack_axis(&ys.iter().collect::<Vec<_>>(), 1)?; // [B,S,Hv,Dv]

        // Gated RMSNorm: silu(z_f32) * rms_norm(y, w, eps)_f32, cast back.
        let norm_weight = w(&self.weights, &format!("{p}.linear_attn.norm.weight"))?;
        let normed = fast::rms_norm(&y, Some(norm_weight), cfg.rms_norm_eps)?;
        let gate_f32 = silu(&z.as_dtype(Dtype::Float32)?)?;
        let out = gate_f32
            .multiply(&normed.as_dtype(Dtype::Float32)?)?
            .as_dtype(y.dtype())?;

        let out = out.reshape(&[b, s, -1])?;
        linear(&out, w(&self.weights, &format!("{p}.linear_attn.out_proj.weight"))?)
    }
}
