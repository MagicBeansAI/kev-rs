//! The Qwen tokenizer with upstream kev's special-token handling. Kev reuses
//! five rarely-used Qwen control tokens as delimiters (state, q, opt, /opt,
//! decide); caller text is escaped so it can never produce them.

use crate::error::{KevError, Result};
use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

/// Delimiter token strings, in order: state, q, opt, /opt, decide.
pub const SPECIAL: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];

fn special_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\|([A-Za-z0-9_]+)\|>").expect("static regex"))
}

pub struct KevTokenizer {
    inner: Tokenizer,
    /// state, q, opt, /opt, decide token ids.
    pub special_ids: [u32; 5],
    pub pad_id: u32,
}

impl KevTokenizer {
    /// Load `tokenizer.json` (+ `tokenizer_config.json` for the pad token)
    /// from a base-model snapshot directory. The tokenizer lives in the base
    /// repository, not the adapter repository, exactly as upstream loads it.
    pub fn load(base_dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(base_dir.join("tokenizer.json"))
            .map_err(|error| KevError::Load(format!("tokenizer.json: {error}")))?;
        let mut special_ids = [0u32; 5];
        for (slot, token) in special_ids.iter_mut().zip(SPECIAL) {
            *slot = inner
                .token_to_id(token)
                .ok_or_else(|| KevError::Load(format!("tokenizer lacks {token}")))?;
        }
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(base_dir.join("tokenizer_config.json"))?,
        )
        .map_err(|error| KevError::Load(format!("tokenizer_config.json: {error}")))?;
        // Upstream pad_id(): the tokenizer's pad token, falling back to 0.
        let pad_id = config["pad_token"]
            .as_str()
            .and_then(|token| inner.token_to_id(token))
            .unwrap_or(0);
        Ok(Self { inner, special_ids, pad_id })
    }

    /// Plain tokenization without escaping (upstream uses this for the
    /// `usage.output_tokens` count of the serialized answers).
    pub fn raw_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| KevError::Inference(format!("tokenize: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Upstream `user_tokens`: `<|name|>` is rewritten to `<¦name¦>` before
    /// tokenizing, so option boundaries are unforgeable.
    pub fn user_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let escaped = special_re().replace_all(text, "<\u{00a6}$1\u{00a6}>");
        let encoding = self
            .inner
            .encode(escaped.as_ref(), false)
            .map_err(|error| KevError::Inference(format!("tokenize: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }
}
