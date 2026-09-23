//! K0 golden files: the encoding rows and reference outputs.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct GoldenRow {
    pub ids: Vec<i32>,
    pub decide: usize,
    pub opts: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct GoldenEncoding {
    pub ids: Vec<i32>,
    pub state_tokens: usize,
    pub rows: Vec<GoldenRow>,
}

#[derive(Debug, Deserialize)]
pub struct GoldenNative {
    pub probs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct GoldenMeta {
    pub temperature: f64,
    pub oracle_path: String,
}

#[derive(Debug, Deserialize)]
pub struct Golden {
    pub fixture: String,
    pub encoding: GoldenEncoding,
    pub native: GoldenNative,
    pub logits_raw: Vec<Vec<f64>>,
    pub meta: GoldenMeta,
}

impl Golden {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}
