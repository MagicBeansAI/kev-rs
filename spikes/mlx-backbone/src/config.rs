//! Qwen3.5 text_config, read from the base snapshot's config.json.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

#[derive(Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub full_attention_interval: usize,
    pub linear_conv_kernel_dim: i32,
    pub linear_key_head_dim: i32,
    pub linear_num_key_heads: i32,
    pub linear_num_value_heads: i32,
    pub linear_value_head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub tie_word_embeddings: bool,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Deserialize)]
struct TopConfig {
    text_config: TextConfig,
}

impl TextConfig {
    pub fn load(base_dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(base_dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", base_dir.display()))?;
        let top: TopConfig = serde_json::from_str(&raw).context("parse text_config")?;
        Ok(top.text_config)
    }

    pub fn is_linear(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }

    pub fn rotary_dims(&self) -> i32 {
        (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as i32
    }
}
