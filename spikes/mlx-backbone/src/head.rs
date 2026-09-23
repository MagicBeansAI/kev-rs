//! Pointer head: fp32 q/k Linear (with bias), z = (K(h_opts) @ Q(h_decide)) / sqrt(dp),
//! calibrated by the checkpoint temperature. Loaded from the converted,
//! checksummed head.safetensors (never the pickle).

use anyhow::{Context, Result};
use mlx_rs::{ops, Array, Dtype};
use std::path::Path;

pub struct PointerHead {
    q_weight: Array,
    q_bias: Array,
    k_weight: Array,
    k_bias: Array,
    scale: f32,
    pub temperature: f32,
}

impl PointerHead {
    pub fn load(head_st: &Path, head_meta: &Path) -> Result<Self> {
        let tensors = Array::load_safetensors(head_st)
            .with_context(|| format!("load {}", head_st.display()))?;
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(head_meta)?)?;
        let temperature = meta["meta"]["temperature"].as_f64().unwrap_or(1.0) as f32;
        let dp = tensors["q.weight"].shape()[0] as f32;
        Ok(Self {
            q_weight: tensors["q.weight"].clone(),
            q_bias: tensors["q.bias"].clone(),
            k_weight: tensors["k.weight"].clone(),
            k_bias: tensors["k.bias"].clone(),
            scale: dp.powf(-0.5),
            temperature,
        })
    }

    /// h_decide [d], h_opts [K, d] (any float dtype; promoted to fp32) -> raw logits [K].
    pub fn raw_logits(&self, h_decide: &Array, h_opts: &Array) -> Result<Array> {
        let hd = h_decide.as_dtype(Dtype::Float32)?;
        let ho = h_opts.as_dtype(Dtype::Float32)?;
        let q = ops::matmul(&self.q_weight, &hd)?.add(&self.q_bias)?; // [dp]
        let k = ops::matmul(&ho, &ops::swap_axes(&self.k_weight, -1, -2)?)?.add(&self.k_bias)?; // [K, dp]
        Ok(ops::matmul(&k, &q)?.multiply(&Array::from_f32(self.scale))?)
    }

    pub fn probs(&self, raw: &Array) -> Result<Array> {
        let calibrated = raw.divide(&Array::from_f32(self.temperature))?;
        Ok(ops::softmax_axis(&calibrated, -1, true)?)
    }
}
