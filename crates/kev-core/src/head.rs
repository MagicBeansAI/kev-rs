//! The pointer head: two fp32 Linear layers (q, k with bias);
//! `z = (K(h_opts) @ Q(h_decide)) / sqrt(dp)`, softmax(z / T) with the
//! per-checkpoint temperature. Framework-free fp32 math so every backend
//! shares one head. Loaded from the converted `head.safetensors` +
//! `head.meta.json` (kev-rs never executes pickle).

use crate::error::{KevError, Result};
use std::path::Path;

pub struct PointerHead {
    /// [dp, d] row-major.
    q_weight: Vec<f32>,
    q_bias: Vec<f32>,
    k_weight: Vec<f32>,
    k_bias: Vec<f32>,
    pub dp: usize,
    pub d: usize,
    pub temperature: f32,
}

impl PointerHead {
    pub fn load(head_st: &Path, head_meta: &Path) -> Result<Self> {
        let file = SafeTensors::read(head_st)?;
        let (q_weight, q_shape) = file.tensor_f32("q.weight")?;
        let (q_bias, _) = file.tensor_f32("q.bias")?;
        let (k_weight, k_shape) = file.tensor_f32("k.weight")?;
        let (k_bias, _) = file.tensor_f32("k.bias")?;
        if q_shape.len() != 2 || q_shape != k_shape {
            return Err(KevError::Load(format!("unexpected head shapes {q_shape:?} {k_shape:?}")));
        }
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(head_meta)?)
            .map_err(|error| KevError::Load(format!("head.meta.json: {error}")))?;
        // Older checkpoints (Qwen3 generation) carry no fitted temperature;
        // upstream serves them at 1.0.
        let temperature = meta["meta"]["temperature"].as_f64().unwrap_or(1.0) as f32;
        Ok(Self {
            dp: q_shape[0],
            d: q_shape[1],
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            temperature,
        })
    }

    /// h_decide `[d]`, h_opts `[K][d]` (fp32) -> raw logits `[K]` (T = 1).
    pub fn raw_logits(&self, h_decide: &[f32], h_opts: &[Vec<f32>]) -> Result<Vec<f32>> {
        if h_decide.len() != self.d || h_opts.iter().any(|h| h.len() != self.d) {
            return Err(KevError::Inference(format!(
                "head expects hidden size {}, got {}",
                self.d,
                h_decide.len()
            )));
        }
        let scale = (self.dp as f32).powf(-0.5);
        // q = Q h_decide + bq
        let mut q = self.q_bias.clone();
        for (row, slot) in q.iter_mut().enumerate() {
            let weights = &self.q_weight[row * self.d..(row + 1) * self.d];
            *slot += dot(weights, h_decide);
        }
        // z_k = (K h_opt + bk) . q * scale
        let mut logits = Vec::with_capacity(h_opts.len());
        for h_opt in h_opts {
            let mut z = 0f32;
            for (row, bias) in self.k_bias.iter().enumerate() {
                let weights = &self.k_weight[row * self.d..(row + 1) * self.d];
                z += (dot(weights, h_opt) + bias) * q[row];
            }
            logits.push(z * scale);
        }
        Ok(logits)
    }

    /// Calibrated probabilities: softmax(z / T) in fp32, upstream's math.
    pub fn probs(&self, raw: &[f32]) -> Vec<f64> {
        let calibrated: Vec<f32> = raw.iter().map(|z| z / self.temperature).collect();
        let max = calibrated.iter().cloned().fold(f32::MIN, f32::max);
        let exp: Vec<f32> = calibrated.iter().map(|z| (z - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|e| (*e / sum) as f64).collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimal read-only safetensors parser for little-endian fp32 tensors.
struct SafeTensors {
    header: serde_json::Value,
    data: Vec<u8>,
}

impl SafeTensors {
    fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 8 {
            return Err(KevError::Load("safetensors too short".into()));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
            .map_err(|error| KevError::Load(format!("safetensors header: {error}")))?;
        Ok(Self { header, data: bytes[8 + header_len..].to_vec() })
    }

    fn tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let entry = &self.header[name];
        if entry.is_null() {
            return Err(KevError::Load(format!("safetensors lacks {name}")));
        }
        if entry["dtype"] != "F32" {
            return Err(KevError::Load(format!("{name} is {}, expected F32", entry["dtype"])));
        }
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
            .unwrap_or_default();
        let offsets = entry["data_offsets"].as_array().ok_or_else(|| {
            KevError::Load(format!("{name}: missing data_offsets"))
        })?;
        let (start, end) = (
            offsets[0].as_u64().unwrap_or(0) as usize,
            offsets[1].as_u64().unwrap_or(0) as usize,
        );
        let bytes = &self.data[start..end];
        let values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok((values, shape))
    }
}
