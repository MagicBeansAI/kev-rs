//! Integration tests against the K0 goldens. The encoding tests are
//! bit-exact and run everywhere; the backend parity tests need the pinned
//! model cache and a backend feature, and are `#[ignore]` by default:
//!
//! ```sh
//! cargo test -p kev-core                                   # encoding + wire math
//! cargo test -p kev-core --features mlx -- --ignored       # Metal parity (Apple Silicon)
//! cargo test -p kev-core --features candle -- --ignored    # CPU parity (kev-0.6b)
//! ```

use kev_core::api::{self, SystemOneRequest};
use kev_core::encode::{encode, rows_of, SERVE_MAX_BRANCH, SERVE_MAX_STATE};
use kev_core::tokenizer::KevTokenizer;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().to_path_buf()
}

fn manifest() -> Value {
    let raw = std::fs::read_to_string(repo_root().join("manifests/sources.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn snapshot_dir(repo: &str, revision: &str) -> PathBuf {
    repo_root()
        .join(".cache/kev/hf")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision)
}

/// (base_dir, adapter_dir) for a checkpoint, from the pinned manifest.
fn checkpoint_dirs(manifest: &Value, name: &str) -> (PathBuf, PathBuf) {
    let facts = &manifest["checkpoints"][name];
    let base = facts["base"].as_str().unwrap();
    let base_rev = facts["base_revision"].as_str().unwrap();
    let adapter_rev = manifest["sources"][name]["revision"].as_str().unwrap();
    let adapter_repo = manifest["sources"][name]["url"]
        .as_str()
        .unwrap()
        .strip_prefix("https://huggingface.co/")
        .unwrap()
        .to_string();
    (snapshot_dir(base, base_rev), snapshot_dir(&adapter_repo, adapter_rev))
}

fn golden_files(checkpoint: &str, path: &str) -> Vec<PathBuf> {
    let dir = repo_root().join("benchmarks/goldens").join(checkpoint).join(path);
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|_| panic!("missing goldens {}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "json")
                && !matches!(
                    p.file_stem().and_then(|s| s.to_str()),
                    Some("run") | Some("packed-vs-separate")
                )
        })
        .collect();
    files.sort();
    files
}

fn fixture_request(id: &str) -> SystemOneRequest {
    let raw = std::fs::read_to_string(
        repo_root().join("benchmarks/fixtures/requests").join(format!("{id}.json")),
    )
    .unwrap();
    let doc: Value = serde_json::from_str(&raw).unwrap();
    serde_json::from_value(doc["request"].clone()).unwrap()
}

/// Tokenizer + encoding parity: bit-exact ids against every golden of the
/// given checkpoint's fp32 path.
fn check_encoding(checkpoint: &str, min_goldens: usize) {
    let manifest = manifest();
    let (base_dir, _) = checkpoint_dirs(&manifest, checkpoint);
    if !base_dir.is_dir() {
        eprintln!("skipping {checkpoint}: pinned cache missing at {}", base_dir.display());
        return;
    }
    let tok = KevTokenizer::load(&base_dir).unwrap();
    let mut checked = 0;
    for golden_path in golden_files(checkpoint, "torch-fp32") {
        let golden: Value =
            serde_json::from_str(&std::fs::read_to_string(&golden_path).unwrap()).unwrap();
        let fixture = golden["fixture"].as_str().unwrap();
        let request = fixture_request(fixture);
        let (record, metas) = api::to_record(&request).unwrap();

        assert_eq!(record.state, golden["state_text"].as_str().unwrap(), "{fixture}: state text");
        for (question, gq) in record.questions.iter().zip(golden["questions"].as_array().unwrap())
        {
            assert_eq!(question.instr, gq["instr"].as_str().unwrap(), "{fixture}: instr");
            let opts: Vec<&str> =
                gq["options"].as_array().unwrap().iter().map(|o| o.as_str().unwrap()).collect();
            assert_eq!(question.options, opts, "{fixture}: options");
        }

        let enc = encode(&tok, &record, SERVE_MAX_STATE, SERVE_MAX_BRANCH, false).unwrap();
        let golden_ids: Vec<u32> = golden["encoding"]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(enc.ids, golden_ids, "{fixture}: token ids");
        let golden_decide: Vec<usize> = golden["encoding"]["decide_idx"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(enc.decide_idx, golden_decide, "{fixture}: decide_idx");
        let golden_opts: Vec<Vec<usize>> = golden["encoding"]["opt_idx"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect())
            .collect();
        assert_eq!(enc.opt_idx, golden_opts, "{fixture}: opt_idx");
        assert_eq!(
            enc.state_tokens(),
            golden["encoding"]["state_tokens"].as_u64().unwrap() as usize,
            "{fixture}: state tokens"
        );

        // rows_of round-trips the golden row decomposition.
        let (state_ids, _, rows) = rows_of(&enc).unwrap();
        let golden_rows = golden["encoding"]["rows"].as_array().unwrap();
        assert_eq!(rows.len(), golden_rows.len());
        assert_eq!(state_ids.len(), enc.state_tokens());
        for (row, grow) in rows.iter().zip(golden_rows) {
            let gids: Vec<u32> =
                grow["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(row.ids, gids, "{fixture}: row ids");
            assert_eq!(row.decide, grow["decide"].as_u64().unwrap() as usize);
            let gopts: Vec<usize> =
                grow["opts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(row.opts, gopts, "{fixture}: row opts");
        }

        // Wire answer math: rebuilding answers from the golden probabilities
        // must reproduce the golden wire answers (rounding, confidence,
        // legend, ordering), and output_tokens must match upstream's count.
        let probs: Vec<Vec<f64>> = golden["native"]["probs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
            .collect();
        let answers = api::to_answers(&probs, &metas);
        assert_eq!(&answers, &golden["wire"]["answers"], "{fixture}: wire answers");
        let serialized = api::python_dumps(&answers);
        let output_tokens = tok.raw_tokens(&serialized).unwrap().len();
        assert_eq!(
            output_tokens,
            golden["wire"]["usage"]["output_tokens"].as_u64().unwrap() as usize,
            "{fixture}: output_tokens"
        );
        checked += 1;
    }
    assert!(
        checked >= min_goldens,
        "expected >= {min_goldens} goldens for {checkpoint}, checked {checked}"
    );
}

#[test]
fn encoding_parity_kev_0_8b() {
    check_encoding("kev-0.8b", 20);
}

#[test]
fn encoding_parity_kev_0_6b() {
    check_encoding("kev-0.6b", 20);
}

#[test]
fn encoding_parity_kev_4b() {
    // kev-4b torch-fp32 is a documented 8-fixture core subset (host RAM
    // limit, see docs/K0-BASELINE.md).
    check_encoding("kev-4b", 8);
}

// ------------------------------------------------------------------
// Backend parity (needs weights; run with --ignored and a feature).

#[cfg(any(feature = "mlx", feature = "candle"))]
fn assemble_model_dir(checkpoint: &str) -> PathBuf {
    let manifest = manifest();
    let (base_dir, adapter_dir) = checkpoint_dirs(&manifest, checkpoint);
    let converted = repo_root().join(".cache/kev/converted").join(checkpoint);
    let out = std::env::temp_dir().join(format!("kev-core-test-{checkpoint}"));
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    std::os::unix::fs::symlink(&base_dir, out.join("base")).unwrap();
    std::os::unix::fs::symlink(&adapter_dir, out.join("adapter")).unwrap();
    std::os::unix::fs::symlink(converted.join("head.safetensors"), out.join("head.safetensors"))
        .unwrap();
    std::os::unix::fs::symlink(converted.join("head.meta.json"), out.join("head.meta.json"))
        .unwrap();
    out
}

#[cfg(any(feature = "mlx", feature = "candle"))]
fn check_backend(
    checkpoint: &str,
    device: kev_core::runtime::Device,
    gate_max_dp: f64,
    gate_mean_dp: f64,
) {
    use kev_core::runtime::{LoadOptions, Runtime};

    let mut runtime = Runtime::load(&LoadOptions {
        model_dir: assemble_model_dir(checkpoint),
        device,
        temperature: None,
    })
    .unwrap();

    let mut worst: (f64, String) = (0.0, String::new());
    let mut flips: Vec<String> = Vec::new();
    let mut questions = 0usize;
    let mut dp_sum = 0f64;
    let mut dp_count = 0usize;
    for golden_path in golden_files(checkpoint, "torch-fp32") {
        let golden: Value =
            serde_json::from_str(&std::fs::read_to_string(&golden_path).unwrap()).unwrap();
        let fixture = golden["fixture"].as_str().unwrap().to_string();
        let request = fixture_request(&fixture);
        let evaluation = runtime.evaluate(&request).unwrap();
        let golden_probs: Vec<Vec<f64>> = golden["native"]["probs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
            .collect();
        for (ours, gold) in evaluation.probs.iter().zip(&golden_probs) {
            questions += 1;
            for (a, b) in ours.iter().zip(gold) {
                dp_sum += (a - b).abs();
                dp_count += 1;
            }
            let max_dp = ours
                .iter()
                .zip(gold)
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            if max_dp > worst.0 {
                worst = (max_dp, fixture.clone());
            }
            let ours_top = ours
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i);
            let gold_top = gold
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i);
            if ours_top != gold_top {
                // A flip is tolerable only on an fp32 near-tie (upstream rule).
                let mut sorted = gold.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let gap = if sorted.len() > 1 { sorted[0] - sorted[1] } else { 1.0 };
                assert!(gap < 0.02, "{fixture}: argmax flip on gap {gap}");
                flips.push(fixture.clone());
            }
        }
        assert_eq!(
            evaluation.input_tokens,
            golden["wire"]["usage"]["input_tokens"].as_u64().unwrap() as usize
        );
    }
    let mean_dp = dp_sum / dp_count.max(1) as f64;
    eprintln!(
        "{checkpoint} [{}]: {questions} questions, max|dp| {:.6} ({}), mean|dp| {:.6}, near-tie flips: {:?}",
        runtime.backend_name, worst.0, worst.1, mean_dp, flips
    );
    assert!(worst.0 <= gate_max_dp, "max|dp| {} over gate {gate_max_dp}", worst.0);
    assert!(mean_dp <= gate_mean_dp, "mean|dp| {mean_dp} over gate {gate_mean_dp}");
}

/// The frozen packed-vs-separate gate (tolerances.json): a packed request
/// and the same questions asked separately must match \u2014 row isolation.
#[cfg(any(feature = "mlx", feature = "candle"))]
fn check_packed_vs_separate(
    checkpoint: &str,
    device: kev_core::runtime::Device,
    gate: f64,
) {
    use kev_core::runtime::{LoadOptions, Runtime};

    let mut runtime = Runtime::load(&LoadOptions {
        model_dir: assemble_model_dir(checkpoint),
        device,
        temperature: None,
    })
    .unwrap();

    let packed = runtime.evaluate(&fixture_request("mixed-packed-3")).unwrap();
    let mut max_dp = 0f64;
    for (index, separate_fixture) in ["separate-route", "separate-review", "separate-urgency"]
        .iter()
        .enumerate()
    {
        let separate = runtime.evaluate(&fixture_request(separate_fixture)).unwrap();
        assert_eq!(separate.probs.len(), 1);
        for (a, b) in packed.probs[index].iter().zip(&separate.probs[0]) {
            max_dp = max_dp.max((a - b).abs());
        }
        let packed_top = packed.probs[index]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);
        let separate_top = separate.probs[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);
        assert_eq!(packed_top, separate_top, "{separate_fixture}: argmax flip");
    }
    eprintln!("{checkpoint} packed-vs-separate max|dp| {max_dp:.8}");
    assert!(max_dp <= gate, "max|dp| {max_dp} over gate {gate}");
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "needs the pinned model cache and Apple Silicon"]
fn mlx_packed_vs_separate_kev_0_8b() {
    // Frozen K0 gate: packed_vs_separate mlx = 0.01, no flips.
    check_packed_vs_separate("kev-0.8b", kev_core::runtime::Device::Metal, 0.01);
}

#[cfg(feature = "candle")]
#[test]
#[ignore = "needs the pinned model cache"]
fn candle_packed_vs_separate_kev_0_6b() {
    // Frozen K0 gate: packed_vs_separate torch-fp32 = 1e-5, no flips; the
    // candle path runs the same fp32 precision.
    check_packed_vs_separate("kev-0.6b", kev_core::runtime::Device::Cpu, 1e-5);
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "needs the pinned model cache and Apple Silicon"]
fn mlx_parity_kev_0_8b() {
    // Gate: the frozen K0 mlx_vs_fp32 gate for kev-0.8b (tolerances.json).
    check_backend("kev-0.8b", kev_core::runtime::Device::Metal, 0.06, 0.005);
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "needs the pinned model cache and Apple Silicon"]
fn mlx_parity_kev_4b() {
    check_backend("kev-4b", kev_core::runtime::Device::Metal, 0.03, 0.004);
}

#[cfg(feature = "candle")]
#[test]
#[ignore = "needs the pinned model cache"]
fn candle_parity_kev_0_6b() {
    // fp32 vs fp32: far tighter than the bf16 Metal gates.
    check_backend("kev-0.6b", kev_core::runtime::Device::Cpu, 0.005, 0.001);
}
