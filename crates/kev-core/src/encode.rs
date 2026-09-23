//! `kev/model.py::encode` and `rows_of`, ported exactly. Packs one record as
//! `[<state> ...]` then per-question `[<q> instr (<opt> o </opt>)* <decide>]`.
//! Branch positions restart right after the state, so state + one branch is
//! a plain causal row with sequential positions (the hybrid serving form).

use crate::api::Record;
use crate::error::{KevError, Result};
use crate::tokenizer::KevTokenizer;

/// Serving limits (`kev.serve`); training limits are 384/1024/2048.
pub const SERVE_MAX_STATE: usize = 8192;
pub const SERVE_MAX_BRANCH: usize = 8192;
pub const SERVE_MAX_PACKED: usize = SERVE_MAX_STATE + SERVE_MAX_BRANCH;

#[derive(Debug, Clone)]
pub struct Encoding {
    pub ids: Vec<u32>,
    /// 0 = state, k = question k (1-based).
    pub seg: Vec<usize>,
    pub pos: Vec<usize>,
    pub decide_idx: Vec<usize>,
    pub opt_idx: Vec<Vec<usize>>,
    pub state_truncated: bool,
}

impl Encoding {
    pub fn state_tokens(&self) -> usize {
        self.seg.iter().take_while(|s| **s == 0).count()
    }
}

/// One question's branch, with readout offsets *within the branch*.
#[derive(Debug, Clone)]
pub struct Row {
    pub ids: Vec<u32>,
    pub pos: Vec<usize>,
    pub decide: usize,
    pub opts: Vec<usize>,
}

pub fn encode(
    tok: &KevTokenizer,
    rec: &Record,
    max_state: usize,
    max_branch: usize,
    strict: bool,
) -> Result<Encoding> {
    let state_tokens = tok.user_tokens(&rec.state)?;
    if strict && state_tokens.len() + 1 > max_state {
        return Err(KevError::ContextOverflow(format!(
            "state exceeds {max_state} tokens: {}",
            state_tokens.len() + 1
        )));
    }
    let [state_id, q_id, o_id, c_id, d_id] = tok.special_ids;
    let mut state: Vec<u32> = vec![state_id];
    state.extend(state_tokens.iter().take(max_state - 1));
    let state_len = state.len();

    let mut ids = state;
    let mut seg = vec![0usize; state_len];
    let mut pos: Vec<usize> = (0..state_len).collect();
    let mut decide_idx = Vec::new();
    let mut opt_idx = Vec::new();

    for (k, question) in rec.questions.iter().enumerate() {
        let mut branch: Vec<u32> = vec![q_id];
        branch.extend(tok.user_tokens(&question.instr)?);
        let instr_len = branch.len();
        let mut span_lens = Vec::with_capacity(question.options.len());
        for option in &question.options {
            let body = tok.user_tokens(option)?;
            branch.push(o_id);
            branch.extend(&body);
            branch.push(c_id);
            span_lens.push(body.len() + 2);
        }
        branch.push(d_id);

        if branch.len() > max_branch - state_len {
            return Err(KevError::ContextOverflow(format!(
                "branch too long: {} tokens with a {state_len}-token state (row limit {max_branch})",
                branch.len()
            )));
        }

        let base = ids.len();
        let p0 = state_len;
        let mut ends = Vec::with_capacity(span_lens.len());
        let mut cursor = instr_len;
        for span in &span_lens {
            cursor += span;
            ends.push(cursor - 1);
        }
        pos.extend(p0..p0 + branch.len());
        seg.extend(std::iter::repeat_n(k + 1, branch.len()));
        decide_idx.push(base + branch.len() - 1);
        opt_idx.push(ends.iter().map(|e| base + e).collect());
        ids.extend(branch);
    }

    Ok(Encoding {
        ids,
        seg,
        pos,
        decide_idx,
        opt_idx,
        state_truncated: state_tokens.len() + 1 > max_state,
    })
}

/// `rows_of`: split a packed encoding into (state_ids, state_pos, rows).
/// Feeding state + rows[k] as one causal row is equivalent to the packed
/// block-causal form for that question, on any architecture.
pub fn rows_of(enc: &Encoding) -> Result<(Vec<u32>, Vec<usize>, Vec<Row>)> {
    let ls = enc.state_tokens();
    let mut rows = Vec::with_capacity(enc.decide_idx.len());
    let mut start = ls;
    for (k, (decide, opts)) in enc.decide_idx.iter().zip(&enc.opt_idx).enumerate() {
        let end = decide + 1;
        if enc.seg[start] != k + 1 || enc.seg[end - 1] != k + 1 {
            return Err(KevError::Inference("branch layout mismatch".into()));
        }
        rows.push(Row {
            ids: enc.ids[start..end].to_vec(),
            pos: enc.pos[start..end].to_vec(),
            decide: decide - start,
            opts: opts.iter().map(|o| o - start).collect(),
        });
        start = end;
    }
    Ok((enc.ids[..ls].to_vec(), enc.pos[..ls].to_vec(), rows))
}
