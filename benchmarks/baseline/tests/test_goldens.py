"""The generated goldens are internally consistent and honor the frozen gates.

These tests read golden files only; they load no models. The tolerance gate
lives in ``benchmarks/goldens/tolerances.json`` and is frozen: relaxing a
tolerance after a failure is prohibited (a failing configuration gets its own
gate version instead).
"""

import json
import math
from pathlib import Path

import pytest

GOLDENS = Path(__file__).resolve().parents[2] / "goldens"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "requests"

pytestmark = pytest.mark.skipif(not GOLDENS.exists(), reason="goldens not generated yet")


def golden_files():
    return sorted(
        p for p in GOLDENS.glob("*/*/*.json") if p.name not in ("run.json", "packed-vs-separate.json")
    )


def tolerances():
    return json.loads((GOLDENS / "tolerances.json").read_text())


@pytest.mark.parametrize("path", golden_files(), ids=lambda p: f"{p.parts[-3]}/{p.parts[-2]}/{p.stem}")
def test_golden_consistency(path):
    doc = json.loads(path.read_text())
    tol = tolerances()

    # The instrumented raw-logit pass reproduces the native probabilities.
    assert doc["instrumented_matches_native"]["max_abs_diff"] <= tol["instrumented_matches_native"]

    for probs, logits_raw, logits_cal, question in zip(
        doc["native"]["probs"], doc["logits_raw"], doc["logits_calibrated"], doc["questions"]
    ):
        assert len(probs) == len(question["keys"]) == len(logits_raw) == len(logits_cal)
        assert abs(sum(probs) - 1.0) < 1e-5
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert all(math.isfinite(z) for z in logits_raw)
        # Calibration is z / T, computed in fp32 (the stored values are fp32;
        # this recheck runs in float64, so allow fp32 rounding).
        temperature = doc["meta"]["temperature"]
        for raw, cal in zip(logits_raw, logits_cal):
            assert cal == pytest.approx(raw / temperature, rel=1e-6, abs=1e-7)

    # Readout indices point where the encoding says they point.
    enc = doc["encoding"]
    assert len(enc["decide_idx"]) == len(doc["questions"])
    for decide, opts, question in zip(enc["decide_idx"], enc["opt_idx"], doc["questions"]):
        assert len(opts) == len(question["options"])
        assert all(o < decide for o in opts)

    # The prefix-cache passes match the native pass within the frozen gate.
    if "prefix" in doc:
        assert doc["prefix"]["max_abs_diff_vs_native"] <= tol["prefix_vs_native"][doc["meta"]["oracle_path"]]

    # usage.output_tokens counts serialized-answer tokens: always positive.
    assert doc["wire"]["usage"]["input_tokens"] == len(enc["ids"])
    assert doc["wire"]["usage"]["output_tokens"] > 0


@pytest.mark.parametrize("report_path", sorted(GOLDENS.glob("*/*/packed-vs-separate.json")), ids=str)
def test_packed_vs_separate(report_path):
    report = json.loads(report_path.read_text())
    tol = tolerances()
    assert report["max_abs_dp"] <= tol["packed_vs_separate"][report["oracle_path"]]
    assert report["argmax_flips"] == 0


@pytest.mark.parametrize("report_path", sorted(GOLDENS.glob("*/mlx-vs-fp32.json")), ids=str)
def test_mlx_vs_fp32(report_path):
    report = json.loads(report_path.read_text())
    gate = tolerances()["mlx_vs_fp32"][report["checkpoint"]]
    assert report["max_abs_dp"] <= gate["max_abs_dp"]
    assert report["mean_abs_dp"] <= gate["mean_abs_dp"]
    assert report["argmax_flips"] <= gate["argmax_flips"]
    # A flip is tolerable only on a near-tie, upstream's own rule.
    for flip in report["argmax_flip_details"]:
        assert flip["ref_top2_gap"] < gate["flip_top2_gap"]


def test_escape_fixtures_leak_no_special_tokens():
    """User text must never produce the five structural special-token ids."""
    from kev.model import SPECIAL
    from transformers import AutoTokenizer

    for checkpoint_dir in sorted(GOLDENS.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        for path_dir in sorted(p for p in checkpoint_dir.iterdir() if p.is_dir()):
            golden = path_dir / "escape-state.json"
            if not golden.exists():
                continue
            doc = json.loads(golden.read_text())
            tok = AutoTokenizer.from_pretrained(
                doc["meta"]["base"], revision=doc["meta"]["base_revision"]
            )
            special_ids = {tok.convert_tokens_to_ids(t) for t in SPECIAL}
            enc = doc["encoding"]
            structural = 1 + sum(  # <state> + per question: <q> ... <decide>
                2 + 2 * len(q["options"]) for q in doc["questions"]
            )
            found = sum(1 for i in enc["ids"] if i in special_ids)
            assert found == structural, (
                f"{golden}: {found} special ids, expected {structural}; user text leaked structure"
            )


def test_every_fixture_has_goldens():
    fixture_ids = {p.stem for p in FIXTURES.glob("*.json")}
    for checkpoint_dir in sorted(GOLDENS.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        for path_dir in sorted(p for p in checkpoint_dir.iterdir() if p.is_dir()):
            run = path_dir / "run.json"
            if not run.exists():
                continue
            run_doc = json.loads(run.read_text())
            expected = set(run_doc["fixtures"])
            if expected != fixture_ids:
                # A subset run must say why; the gap is documented, not silent.
                assert run_doc.get("subset_reason"), f"{path_dir}: subset without a reason"
            produced = {p.stem for p in path_dir.glob("*.json")} - {"run", "packed-vs-separate"}
            assert produced == expected, f"{path_dir}: {sorted(expected ^ produced)}"
            npz = {p.stem for p in path_dir.glob("*.npz")}
            assert npz == expected, f"{path_dir}: missing npz sidecars"
