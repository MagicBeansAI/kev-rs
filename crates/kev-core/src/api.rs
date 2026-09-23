//! Wire shapes and answer math, ported exactly from upstream `kev/api.py`
//! at the pinned commit. Noul -> 2 options [false, true]; Choice -> named
//! options; Score -> ordered level descriptions. Values round to 4 decimals
//! (upstream `round_prob`); nothing is renormalized.

use crate::error::{KevError, Result};
use serde::Deserialize;
use serde_json::{Map, Value};

pub const MAX_OPTIONS: usize = 255;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Noul {
        #[serde(default)]
        instructions: Value,
        #[serde(default)]
        criteria: Option<Map<String, Value>>,
    },
    Choice {
        #[serde(default)]
        instructions: Value,
        criteria: Map<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneRequest {
    pub state: Value,
    #[serde(default = "default_model")]
    pub model: String,
    pub questions: Map<String, Value>,
}

fn default_model() -> String {
    "kev-latest".to_string()
}

/// Python `str()` for the scalar JSON values `render` can meet.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                // Python str(float) is the shortest round-trip repr, which
                // Rust's {:?} for f64 also produces.
                format!("{:?}", n.as_f64().unwrap_or(f64::NAN))
            }
        }
        _ => String::new(),
    }
}

/// `api.render`: flatten str | object | array into the text the model sees.
pub fn render(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        Value::String(_) | Value::Number(_) | Value::Bool(_) => py_str(value),
        Value::Array(items) => items
            .iter()
            .map(|item| format!("{pad}- {}", render(item, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => map
            .iter()
            .map(|(key, item)| match item {
                Value::Object(_) | Value::Array(_) => {
                    format!("{pad}{key}:\n{}", render(item, indent + 1))
                }
                _ => format!("{pad}{key}: {}", render(item, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn option_text(name: &str, desc: Option<&Value>) -> String {
    match desc {
        None | Some(Value::Null) => name.to_string(),
        Some(Value::String(s)) if s.is_empty() => name.to_string(),
        Some(v) => format!("{name}: {}", render(v, 0)),
    }
}

/// One encoded question: rendered instruction and option texts.
#[derive(Debug, Clone)]
pub struct RecordQuestion {
    pub instr: String,
    pub options: Vec<String>,
}

/// The internal record `encode()` consumes.
#[derive(Debug, Clone)]
pub struct Record {
    pub state: String,
    pub questions: Vec<RecordQuestion>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuestionKind {
    Noul,
    Choice,
    Score,
}

/// Per-question metadata to map probabilities back to the wire.
#[derive(Debug, Clone)]
pub struct QuestionMeta {
    pub id: String,
    pub kind: QuestionKind,
    pub keys: Vec<String>,
    /// Score only: level index -> rendered description.
    pub legend: Option<Vec<(String, String)>>,
}

/// `api.to_record`: request -> internal record + per-question metadata.
pub fn to_record(request: &SystemOneRequest) -> Result<(Record, Vec<QuestionMeta>)> {
    if request.questions.is_empty() {
        return Err(KevError::InvalidRequest("questions must not be empty".into()));
    }
    let mut questions = Vec::new();
    let mut metas = Vec::new();
    for (id, raw) in &request.questions {
        let question: Question = serde_json::from_value(raw.clone())
            .map_err(|error| KevError::InvalidRequest(format!("question {id:?}: {error}")))?;
        let (instr, options, kind, keys, legend) = match &question {
            Question::Noul { instructions, criteria } => {
                let empty = Map::new();
                let c = criteria.as_ref().unwrap_or(&empty);
                let options = vec![
                    option_text("no", c.get("false")),
                    option_text("yes", c.get("true")),
                ];
                (
                    render(instructions, 0),
                    options,
                    QuestionKind::Noul,
                    vec!["false".to_string(), "true".to_string()],
                    None,
                )
            }
            Question::Choice { instructions, criteria } => {
                if criteria.is_empty() || criteria.len() > MAX_OPTIONS {
                    return Err(KevError::InvalidRequest(format!(
                        "question {id:?}: criteria must have 1..{MAX_OPTIONS} options"
                    )));
                }
                let options = criteria
                    .iter()
                    .map(|(name, desc)| option_text(name, Some(desc)))
                    .collect();
                let keys = criteria.keys().cloned().collect();
                (render(instructions, 0), options, QuestionKind::Choice, keys, None)
            }
            Question::Score { instructions, criteria } => {
                if criteria.is_empty() || criteria.len() > MAX_OPTIONS {
                    return Err(KevError::InvalidRequest(format!(
                        "question {id:?}: criteria must have 1..{MAX_OPTIONS} levels"
                    )));
                }
                let options: Vec<String> = criteria.iter().map(|c| render(c, 0)).collect();
                let keys: Vec<String> = (0..criteria.len()).map(|i| i.to_string()).collect();
                let legend = keys.iter().cloned().zip(options.iter().cloned()).collect();
                (render(instructions, 0), options, QuestionKind::Score, keys, Some(legend))
            }
        };
        questions.push(RecordQuestion { instr, options });
        metas.push(QuestionMeta { id: id.clone(), kind, keys, legend });
    }
    Ok((
        Record { state: render(&request.state, 0), questions },
        metas,
    ))
}

/// Upstream `round_prob`: Python round(x, 4) (round-half-to-even at the
/// fourth decimal of the binary double).
pub fn round_prob(x: f64) -> f64 {
    (x * 10_000.0).round_ties_even() / 10_000.0
}

pub fn choice_confidence(p: &[f64]) -> f64 {
    let k = p.len() as f64;
    if p.len() == 1 {
        return 1.0;
    }
    let max = p.iter().cloned().fold(f64::MIN, f64::max);
    (max - 1.0 / k) / (1.0 - 1.0 / k)
}

pub fn score_confidence(p: &[f64]) -> f64 {
    let l = p.len();
    if l == 1 {
        return 1.0;
    }
    let mode = p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let expected: f64 = p
        .iter()
        .enumerate()
        .map(|(i, pi)| pi * (i as f64 - mode as f64).abs())
        .sum();
    1.0 - expected / (l as f64 - 1.0)
}

/// `api.to_answers`: probabilities + metadata -> the wire answers object
/// (insertion order preserved, 4-decimal rounding, no renormalization).
pub fn to_answers(probs: &[Vec<f64>], metas: &[QuestionMeta]) -> Value {
    let mut out = Map::new();
    for (p, meta) in probs.iter().zip(metas) {
        let answer = match meta.kind {
            QuestionKind::Noul => {
                let mut m = Map::new();
                m.insert("type".into(), "noul".into());
                m.insert("noul".into(), json_num(round_prob(p[1])));
                m
            }
            QuestionKind::Choice => {
                let top = p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut dist = Map::new();
                for (key, value) in meta.keys.iter().zip(p) {
                    dist.insert(key.clone(), json_num(round_prob(*value)));
                }
                let mut m = Map::new();
                m.insert("type".into(), "choice".into());
                m.insert("choice".into(), meta.keys[top].clone().into());
                m.insert("confidence".into(), json_num(round_prob(choice_confidence(p))));
                m.insert("probabilities".into(), Value::Object(dist));
                m
            }
            QuestionKind::Score => {
                let score: f64 = p.iter().enumerate().map(|(i, pi)| i as f64 * pi).sum();
                let mut legend = Map::new();
                for (key, text) in meta.legend.as_deref().unwrap_or(&[]) {
                    legend.insert(key.clone(), text.clone().into());
                }
                let mut dist = Map::new();
                for (i, value) in p.iter().enumerate() {
                    dist.insert(i.to_string(), json_num(round_prob(*value)));
                }
                let mut m = Map::new();
                m.insert("type".into(), "score".into());
                m.insert("score".into(), json_num(round_prob(score)));
                m.insert("legend".into(), Value::Object(legend));
                m.insert("probabilities".into(), Value::Object(dist));
                m.insert("confidence".into(), json_num(round_prob(score_confidence(p))));
                m
            }
        };
        out.insert(meta.id.clone(), Value::Object(answer));
    }
    Value::Object(out)
}

fn json_num(x: f64) -> Value {
    serde_json::Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}

/// Python `json.dumps(answers)` with the default separators, for the
/// `usage.output_tokens` count ("billing-style figure: tokens of the
/// serialised answers; not a measure of generation").
pub fn python_dumps(value: &Value) -> String {
    let mut out = String::new();
    dumps_into(value, &mut out);
    out
}

fn dumps_into(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&format!("{:?}", n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into()))
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dumps_into(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_else(|_| "\"\"".into()));
                out.push_str(": ");
                dumps_into(item, out);
            }
            out.push('}');
        }
    }
}
