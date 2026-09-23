//! K1 spike: the Qwen3.5 hybrid backbone + kev pointer head in Rust on MLX,
//! checked against the K0 goldens. Row form: each question runs as one
//! causal row (state ids + branch ids), positions sequential — exactly
//! upstream's hybrid path.

mod config;
mod golden;
mod head;
mod model;
mod weights;

use anyhow::{Context, Result};
use clap::Parser;
use mlx_rs::{ops::indexing::TryIndexOp, transforms::eval, Array};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Base model snapshot directory (config.json + safetensors).
    #[arg(long)]
    base_dir: PathBuf,
    /// Adapter snapshot directory (adapter_config.json + adapter_model.safetensors).
    #[arg(long)]
    adapter_dir: PathBuf,
    /// Converted head.safetensors.
    #[arg(long)]
    head: PathBuf,
    /// head.meta.json next to the converted head.
    #[arg(long)]
    head_meta: PathBuf,
    /// Golden JSON files to check against.
    #[arg(long, required = true)]
    golden: Vec<PathBuf>,
    /// Skip the LoRA merge (base model only; for debugging).
    #[arg(long)]
    no_merge: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cfg = config::TextConfig::load(&args.base_dir)?;
    eprintln!(
        "config: {} layers, hidden {}, interval {}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.full_attention_interval
    );

    let start = Instant::now();
    let mut wts = weights::load_base(&args.base_dir)?;
    eprintln!("loaded {} base tensors in {:.1?}", wts.len(), start.elapsed());
    if !args.no_merge {
        let start = Instant::now();
        let merged = weights::merge_lora(&mut wts, &args.adapter_dir, 1.0)?;
        eprintln!("merged {merged} LoRA pairs (fp32, cpu stream) in {:.1?}", start.elapsed());
    }
    let model = model::Model::new(cfg, wts);
    let head = head::PointerHead::load(&args.head, &args.head_meta)?;
    eprintln!("head temperature: {}", head.temperature);

    let mut report = Vec::new();
    for path in &args.golden {
        let golden = golden::Golden::load(path).with_context(|| path.display().to_string())?;
        let state_ids = &golden.encoding.ids[..golden.encoding.state_tokens];

        let mut fixture_max_dp = 0f64;
        let mut fixture_max_dlogit = 0f64;
        let mut flips = 0usize;
        let started = Instant::now();
        for (q_idx, row) in golden.encoding.rows.iter().enumerate() {
            let mut ids = state_ids.to_vec();
            ids.extend_from_slice(&row.ids);
            let hidden = model.hidden_row(&ids)?; // [1, L, d]

            let base = golden.encoding.state_tokens;
            let h_decide = hidden.try_index((0, (base + row.decide) as i32))?;
            let opt_idx: Vec<i32> = row.opts.iter().map(|o| (base + o) as i32).collect();
            let h_opts = hidden.try_index((0,))?.take_axis(
                &Array::from_slice(&opt_idx, &[opt_idx.len() as i32]),
                0,
            )?;

            let raw = head.raw_logits(&h_decide, &h_opts)?;
            let probs = head.probs(&raw)?;
            eval([&raw, &probs])?;

            let raw_v: Vec<f32> = raw.as_slice().to_vec();
            let probs_v: Vec<f32> = probs.as_slice().to_vec();
            let gold_probs = &golden.native.probs[q_idx];
            let gold_raw = &golden.logits_raw[q_idx];

            let max_dp = probs_v
                .iter()
                .zip(gold_probs)
                .map(|(p, g)| (*p as f64 - g).abs())
                .fold(0f64, f64::max);
            let max_dl = raw_v
                .iter()
                .zip(gold_raw)
                .map(|(p, g)| (*p as f64 - g).abs())
                .fold(0f64, f64::max);
            let ours_top = probs_v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i);
            let gold_top = gold_probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i);
            if ours_top != gold_top {
                flips += 1;
            }
            fixture_max_dp = fixture_max_dp.max(max_dp);
            fixture_max_dlogit = fixture_max_dlogit.max(max_dl);
        }
        eprintln!(
            "{}: max|dp| {:.6} max|dlogit_raw| {:.6} flips {} ({} questions, {:.1?})",
            golden.fixture,
            fixture_max_dp,
            fixture_max_dlogit,
            flips,
            golden.encoding.rows.len(),
            started.elapsed()
        );
        report.push(serde_json::json!({
            "fixture": golden.fixture,
            "reference": golden.meta.oracle_path,
            "max_abs_dp": fixture_max_dp,
            "max_abs_dlogit_raw": fixture_max_dlogit,
            "argmax_flips": flips,
            "questions": golden.encoding.rows.len(),
        }));
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "results": report }))?);
    Ok(())
}
