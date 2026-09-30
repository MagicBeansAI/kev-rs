//! Acceptance benchmark: the frozen bench-m5 protocol from
//! benchmarks/manifest.json. new_state = full pass with the prefix
//! rebuilt (upstream `probs_and_prefix`); repeated_state = warm-prefix
//! branches only (`probs_with_prefix`). The timed scope matches the Python
//! baseline: backbone + head + softmax on a precomputed encoding —
//! encoding and answer serialization sit outside the loop, exactly as in
//! `kev_baseline.bench`. Median of 20 iterations after 3 warmup, wall
//! clock (device sync is implicit — fp32 hidden states are materialized
//! every pass).
//!
//! Usage: kev-bench <checkpoint> <cpu|metal> [out.json]
//!
//! `KEV_BENCH_Q8=1` loads with `Quantization::Q8`; `KEV_BENCH_STATE_CHUNK=N`
//! runs states in N-token chunks (MLX memory options, off by default).
//! Q8 is off when unset, empty, or `0`; any other non-empty value enables it.
//! The state chunk must be a positive integer; invalid values are errors.

#[cfg(any(test, feature = "mlx", feature = "candle"))]
fn parse_mlx_options(
    q8: Option<&str>,
    state_chunk: Option<&str>,
) -> anyhow::Result<kev_core::runtime::MlxOptions> {
    Ok(kev_core::runtime::MlxOptions {
        quantize: q8
            .filter(|v| !v.is_empty() && *v != "0")
            .map(|_| kev_core::runtime::Quantization::Q8),
        state_chunk: state_chunk
            .map(|n| {
                n.parse().map_err(|_| {
                    anyhow::anyhow!("KEV_BENCH_STATE_CHUNK must be a positive integer, got {n:?}")
                })
            })
            .transpose()?,
    })
}

#[cfg(any(feature = "mlx", feature = "candle"))]
fn main() -> anyhow::Result<()> {
    use kev_core::runtime::{Device, LoadOptions, Runtime};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let checkpoint = args
        .next()
        .expect("usage: kev-bench <checkpoint> <cpu|metal> [out.json]");
    let device_name = args
        .next()
        .expect("usage: kev-bench <checkpoint> <cpu|metal> [out.json]");
    let out_path = args.next();
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        other => anyhow::bail!("unknown device {other}"),
    };
    let q8 = std::env::var("KEV_BENCH_Q8").ok();
    let state_chunk = std::env::var("KEV_BENCH_STATE_CHUNK").ok();
    let mlx = parse_mlx_options(q8.as_deref(), state_chunk.as_deref())?;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("manifests/sources.json"),
    )?)?;
    let facts = &manifest["checkpoints"][&checkpoint];
    let snapshot = |repo: &str, rev: &str| -> PathBuf {
        root.join(".cache/kev/hf")
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(rev)
    };
    let base_dir = snapshot(
        facts["base"].as_str().unwrap(),
        facts["base_revision"].as_str().unwrap(),
    );
    let adapter_repo = manifest["sources"][&checkpoint]["url"]
        .as_str()
        .unwrap()
        .strip_prefix("https://huggingface.co/")
        .unwrap();
    let adapter_dir = snapshot(
        adapter_repo,
        manifest["sources"][&checkpoint]["revision"]
            .as_str()
            .unwrap(),
    );
    let converted = root.join(".cache/kev/converted").join(&checkpoint);
    let model_dir = std::env::temp_dir().join(format!("kev-bench-{checkpoint}"));
    let _ = std::fs::remove_dir_all(&model_dir);
    std::fs::create_dir_all(&model_dir)?;
    std::os::unix::fs::symlink(&base_dir, model_dir.join("base"))?;
    std::os::unix::fs::symlink(&adapter_dir, model_dir.join("adapter"))?;
    std::os::unix::fs::symlink(
        converted.join("head.safetensors"),
        model_dir.join("head.safetensors"),
    )?;
    std::os::unix::fs::symlink(
        converted.join("head.meta.json"),
        model_dir.join("head.meta.json"),
    )?;

    let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("benchmarks/fixtures/requests/bench-m5.json"),
    )?)?;
    let request: kev_core::SystemOneRequest = serde_json::from_value(fixture["request"].clone())?;

    let mut runtime = Runtime::load(&LoadOptions {
        model_dir,
        device,
        temperature: None,
        mlx,
    })?;
    let (enc, _metas) = runtime.encode_request(&request)?;
    #[cfg(feature = "mlx")]
    let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
    #[cfg(feature = "mlx")]
    {
        let rss_now = || -> f64 {
            std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.trim().parse::<f64>().ok())
                .map_or(0.0, |kb| kb / (1u64 << 20) as f64)
        };
        eprintln!(
            "memory after load: mlx active {:.2} GiB, mlx cache {:.2} GiB, process RSS {:.2} GiB",
            gib(mlx_rs::memory::active_memory()?),
            gib(mlx_rs::memory::cache_memory()?),
            rss_now()
        );
        mlx_rs::memory::reset_peak_memory()?;
    }

    let warmup = 3usize;
    let iterations = 20usize;

    // Warmup + new_state: prefix cleared before every evaluation
    // (probs_and_prefix: full pass, prefix kept).
    let mut new_state = Vec::with_capacity(iterations);
    for i in 0..warmup + iterations {
        runtime.clear_prefix();
        let start = Instant::now();
        let (probs, hit) = runtime.probs_encoded(&enc)?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(!hit && !probs.is_empty(), "expected a cold prefix");
        if i >= warmup {
            new_state.push(ms);
        }
    }
    // repeated_state: prefix warm from the last pass (probs_with_prefix).
    let mut repeated_state = Vec::with_capacity(iterations);
    for i in 0..warmup + iterations {
        let start = Instant::now();
        let (probs, hit) = runtime.probs_encoded(&enc)?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        anyhow::ensure!(hit && !probs.is_empty(), "expected a warm prefix");
        if i >= warmup {
            repeated_state.push(ms);
        }
    }

    let stats = |samples: &[f64]| -> serde_json::Value {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0;
        let p95 = sorted[((sorted.len() as f64 * 0.95).ceil() as usize).min(sorted.len()) - 1];
        serde_json::json!({
            "median_ms": (median * 100.0).round() / 100.0,
            "p95_ms": (p95 * 100.0).round() / 100.0,
            "min_ms": (sorted[0] * 100.0).round() / 100.0,
            "max_ms": (sorted[sorted.len() - 1] * 100.0).round() / 100.0,
            "samples_ms": samples.iter().map(|s| (s * 100.0).round() / 100.0).collect::<Vec<_>>(),
        })
    };

    let loadavg = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let result = serde_json::json!({
        "runtime": "kev-rs (kev-core)",
        "backend": runtime.backend_name,
        "checkpoint": checkpoint,
        "device": device_name,
        "fixture": "bench-m5",
        "iterations": iterations,
        "warmup": warmup,
        "loadavg_raw": loadavg.trim(),
        "new_state": stats(&new_state),
        "repeated_state": stats(&repeated_state),
    });
    #[cfg(feature = "mlx")]
    eprintln!(
        "memory during inference: mlx peak {:.2} GiB",
        gib(mlx_rs::memory::peak_memory()?)
    );
    let rendered = serde_json::to_string_pretty(&result)?;
    println!("{rendered}");
    if let Some(path) = out_path {
        std::fs::create_dir_all(Path::new(&path).parent().unwrap())?;
        std::fs::write(&path, rendered + "\n")?;
    }
    Ok(())
}

#[cfg(not(any(feature = "mlx", feature = "candle")))]
fn main() {
    eprintln!("kev-bench needs the mlx or candle feature");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::parse_mlx_options;
    use kev_core::runtime::{MlxOptions, Quantization};

    #[test]
    fn unset_options_use_the_reference_path() {
        assert_eq!(
            parse_mlx_options(None, None).unwrap(),
            MlxOptions::default()
        );
    }

    #[test]
    fn q8_empty_and_zero_are_off() {
        for value in ["", "0"] {
            assert_eq!(
                parse_mlx_options(Some(value), None).unwrap().quantize,
                None,
                "KEV_BENCH_Q8={value:?}"
            );
        }
    }

    #[test]
    fn q8_nonempty_nonzero_values_enable_quantization() {
        for value in ["1", "true"] {
            assert_eq!(
                parse_mlx_options(Some(value), None).unwrap().quantize,
                Some(Quantization::Q8),
                "KEV_BENCH_Q8={value:?}"
            );
        }
    }

    #[test]
    fn chunk_size_can_be_combined_with_q8() {
        assert_eq!(
            parse_mlx_options(Some("1"), Some("16")).unwrap(),
            MlxOptions {
                quantize: Some(Quantization::Q8),
                state_chunk: Some(16),
            }
        );
        assert_eq!(
            parse_mlx_options(Some("0"), Some("1")).unwrap().state_chunk,
            Some(1)
        );
    }

    #[test]
    fn invalid_chunk_sizes_fail_with_the_setting_and_value() {
        let overflow = format!("{}0", usize::MAX);
        for value in ["", "abc", "-1", "1.5", overflow.as_str()] {
            let error = parse_mlx_options(None, Some(value)).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("KEV_BENCH_STATE_CHUNK must be a positive integer, got {value:?}")
            );
        }
    }
}
