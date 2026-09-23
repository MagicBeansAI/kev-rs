//! Weight loading with mlx-lm's sanitize semantics, plus the fp32 LoRA merge
//! on the CPU stream (upstream kev's exact recipe).

use anyhow::{bail, Context, Result};
use mlx_rs::{ops, transforms::eval, Array, Device, Dtype, Stream};
use std::collections::HashMap;
use std::path::Path;

/// Norm-weight suffixes that get the +1.0 shift when loading a raw HF
/// checkpoint (mlx-lm qwen3_5 sanitize). `linear_attn.norm.weight` (the
/// gated norm) is deliberately not in this list.
const SHIFTED_NORM_SUFFIXES: [&str; 5] = [
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
];

/// Load the text-model weights from the base snapshot, sanitized to plain
/// `layers.N.*` / `embed_tokens.weight` / `norm.weight` names.
pub fn load_base(base_dir: &Path) -> Result<HashMap<String, Array>> {
    let mut files: Vec<_> = std::fs::read_dir(base_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no safetensors in {}", base_dir.display());
    }

    let mut raw = HashMap::new();
    for file in &files {
        let part = Array::load_safetensors(file)
            .with_context(|| format!("load {}", file.display()))?;
        raw.extend(part);
    }

    let has_mtp = raw.keys().any(|k| k.contains("mtp."));
    let mut weights = HashMap::new();
    for (key, value) in raw {
        if key.starts_with("model.visual") || key.contains("mtp.") || key == "lm_head.weight" {
            continue;
        }
        let name = key
            .strip_prefix("model.language_model.")
            .with_context(|| format!("unexpected key {key}"))?
            .to_string();

        let mut tensor = value;
        // Torch conv1d weight (C, 1, K) -> mlx (C, K, 1).
        if name.ends_with("conv1d.weight") && tensor.shape()[2] != 1 {
            tensor = ops::swap_axes(&tensor, 2, 1)?;
        }
        // Raw HF checkpoints store zero-centered norm weights.
        if has_mtp
            && tensor.ndim() == 1
            && SHIFTED_NORM_SUFFIXES
                .iter()
                .any(|suffix| name.ends_with(suffix) || format!("model.{name}").ends_with(suffix))
        {
            tensor = tensor.add(&Array::from_f32(1.0).as_dtype(tensor.dtype())?)?;
        }
        weights.insert(name, tensor);
    }
    Ok(weights)
}

#[derive(Debug, serde::Deserialize)]
struct AdapterConfig {
    r: f32,
    lora_alpha: f32,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    trainable_token_indices: Option<serde_json::Value>,
}

/// Merge the LoRA adapter into the base weights in fp32 on the CPU stream,
/// exactly as upstream `kev/mlx_model.py::merge_lora` does, then cast back
/// to the base dtype.
pub fn merge_lora(
    weights: &mut HashMap<String, Array>,
    adapter_dir: &Path,
    lora_scale: f32,
) -> Result<usize> {
    let config: AdapterConfig = serde_json::from_str(
        &std::fs::read_to_string(adapter_dir.join("adapter_config.json"))
            .context("read adapter_config.json")?,
    )?;
    if config.trainable_token_indices.is_some() {
        bail!("adapters with trainable_token_indices are not supported on the MLX path");
    }
    let alpha = config.lora_alpha
        / if config.use_rslora {
            config.r.sqrt()
        } else {
            config.r
        };

    let adapter = Array::load_safetensors(adapter_dir.join("adapter_model.safetensors"))
        .context("load adapter_model.safetensors")?;

    let cpu = Stream::new_with_device(&Device::cpu());
    let mut merged = 0usize;
    mlx_rs::with_new_default_stream(cpu, || merge_all(weights, &adapter, alpha * lora_scale, &mut merged))?;
    if merged == 0 {
        bail!("no LoRA pairs merged");
    }
    Ok(merged)
}

fn merge_all(
    weights: &mut HashMap<String, Array>,
    adapter: &HashMap<String, Array>,
    scale: f32,
    merged: &mut usize,
) -> Result<()> {
    for (key, lora_a) in adapter {
        let Some(path) = key
            .strip_prefix("base_model.model.")
            .and_then(|k| k.strip_suffix(".lora_A.weight"))
        else {
            continue;
        };
        let b_key = format!("base_model.model.{path}.lora_B.weight");
        let lora_b = adapter
            .get(&b_key)
            .with_context(|| format!("missing {b_key}"))?;
        let target = format!("{path}.weight");
        let base = weights
            .get(&target)
            .with_context(|| format!("adapter targets missing weight {target}"))?;

        // delta = (B @ A) * (alpha * scale), fp32, on the CPU stream.
        let delta = ops::matmul(
            &lora_b.as_dtype(Dtype::Float32)?,
            &lora_a.as_dtype(Dtype::Float32)?,
        )?
        .multiply(&Array::from_f32(scale))?;
        let out = base
            .as_dtype(Dtype::Float32)?
            .add(&delta)?
            .as_dtype(base.dtype())?;
        eval([&out])?;
        weights.insert(target, out);
        *merged += 1;
    }
    Ok(())
}
