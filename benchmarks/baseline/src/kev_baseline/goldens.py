"""Generate golden oracle outputs for the frozen request fixtures.

Two oracle paths, both upstream kev code, never a reimplementation:

- ``torch-fp32``: the exact reference (``LoadOptions(dtype=fp32, backend="torch")``
  on CPU). Every upstream reported number uses this path.
- ``mlx``: upstream's Apple Silicon path (``backend="mlx"``: bf16 mlx-lm
  backbone, fp32 LoRA merge on the CPU stream, torch fp32 pointer head).
  Refused for attention-only (Qwen3) bases, so ``kev-0.6b`` is torch-fp32 only.

Per fixture the golden records the encoding (ids, positions, segments, readout
indices, row decomposition), the native per-question probabilities with wall
time, an instrumented raw-logit pass (temperature forced to 1.0, then the
calibrated logits and probabilities recomputed with the same torch ops and
checked against the native pass), the wire answers and usage from upstream
``api.to_answers`` / ``api.output_tokens``, and, for ``repeat`` fixtures, the
state-prefix-cache passes exactly as ``kev.serve`` runs them. Full-precision
arrays go to an ``.npz`` sidecar.

Golden files are generated, never hand-edited. Every model file loads from
the shared pinned cache at an immutable commit hash (huggingface_hub 1.32
cannot list a repo tree offline, so hard offline mode is not enforced here;
``kev-fetch --verify-only`` re-hashes the cache against the manifest
instead).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time

from . import paths

# Set before huggingface_hub is imported anywhere (it reads this at import
# time): all model files come from the pinned cache.
os.environ.setdefault("HF_HUB_CACHE", str(paths.hf_cache_dir()))

from .fetch import CHECKPOINTS  # noqa: E402 - needs the env set first

FORBIDDEN_ENV = ("KEV_DTYPE", "KEV_MERGE", "KEV_ATTN", "KEV_LORA_SCALE", "KEV_TEMPERATURE", "KEV_BACKEND")

PATHS = ("torch-fp32", "mlx")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_model(checkpoint: str, oracle_path: str):
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions

    pin = CHECKPOINTS[checkpoint]
    run = f"{pin['repo']}@{pin['revision']}"
    ck = Checkpoint(run)
    if oracle_path == "torch-fp32":
        device = "cpu"
        opts = LoadOptions(dtype=torch.float32, backend="torch")
    elif oracle_path == "mlx":
        if not ck.hybrid_base():
            raise SystemExit(f"{checkpoint}: base {ck.meta.base} is attention-only; the MLX path refuses it")
        device = "mps"
        opts = LoadOptions(backend="mlx")
    else:
        raise ValueError(oracle_path)
    tok, model = ck.load(device, opts)
    return ck, tok, model, device


def encode_capture(enc) -> dict:
    from kev.model import rows_of

    state_ids, state_pos, rows = rows_of(enc)
    return {
        "ids": list(enc["ids"]),
        "pos": list(enc["pos"]),
        "seg": list(enc["seg"]),
        "decide_idx": list(enc["decide_idx"]),
        "opt_idx": [list(o) for o in enc["opt_idx"]],
        "state_tokens": len(state_ids),
        "state_truncated": bool(enc.get("state_truncated")),
        "option_isolation": bool(enc.get("option_isolation")),
        "rows": [
            {"ids": list(r["ids"]), "pos": list(r["pos"]), "decide": r["decide"], "opts": list(r["opts"])}
            for r in rows
        ],
    }


def timed_probs(model, enc, device):
    from kev.device import sync

    sync(device)
    start = time.time()
    ps = model.probs(enc)
    sync(device)
    return ps, round((time.time() - start) * 1000, 2)


def instrumented(model, enc, device):
    """Raw logits at T=1, then upstream's own calibration math re-applied."""
    import torch
    import torch.nn.functional as F
    from kev.device import sync

    temperature = model.head.temperature
    model.head.temperature = 1.0
    try:
        with torch.no_grad():
            sync(device)
            raw = [z.cpu() for z in model.forward(enc)]
            sync(device)
    finally:
        model.head.temperature = temperature
    calibrated = [z if temperature == 1.0 else z / temperature for z in raw]
    probs = [F.softmax(z, -1) for z in calibrated]
    return raw, calibrated, probs


def prefix_passes(model, enc, device, repeats: int):
    """The kev.serve prefix-cache path: miss on pass 1, hits after."""
    from kev.device import sync

    passes = []
    sync(device)
    start = time.time()
    ps, prefix = model.probs_and_prefix(enc)
    sync(device)
    passes.append({"probs": [p.tolist() for p in ps], "latency_ms": round((time.time() - start) * 1000, 2), "prefix_cache_hit": False})
    for _ in range(repeats - 1):
        sync(device)
        start = time.time()
        ps = model.probs_with_prefix(enc, prefix)
        sync(device)
        passes.append({"probs": [p.tolist() for p in ps], "latency_ms": round((time.time() - start) * 1000, 2), "prefix_cache_hit": True})
    return passes


def max_abs_diff(a, b) -> float:
    return max(
        (abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb)),
        default=0.0,
    )


def run_fixture(fixture: dict, tok, model, device) -> tuple[dict, dict]:
    import numpy as np
    from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

    req = SystemOneRequest.model_validate(fixture["request"])
    rec, meta = to_record(req)
    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)

    native, latency_ms = timed_probs(model, enc, device)
    native_lists = [p.tolist() for p in native]
    raw, calibrated, probs_check = instrumented(model, enc, device)
    instrumented_diff = max_abs_diff([p.tolist() for p in probs_check], native_lists)

    answers = to_answers(native_lists, meta)
    doc = {
        "fixture": fixture["id"],
        "encoding": encode_capture(enc),
        "questions": [
            {"id": m["id"], "type": m["type"], "keys": m["keys"], "instr": q["instr"], "options": q["options"]}
            for m, q in zip(meta, rec["questions"])
        ],
        "state_text": rec["state"],
        "native": {"probs": native_lists, "latency_ms": latency_ms},
        "logits_raw": [z.tolist() for z in raw],
        "logits_calibrated": [z.tolist() for z in calibrated],
        "instrumented_matches_native": {"max_abs_diff": instrumented_diff},
        "wire": {
            "answers": answers,
            "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(tok, answers)},
        },
    }
    if fixture.get("repeat"):
        passes = prefix_passes(model, enc, device, int(fixture["repeat"]))
        doc["prefix"] = {
            "passes": passes,
            "max_abs_diff_vs_native": max(max_abs_diff(p["probs"], native_lists) for p in passes),
        }
    arrays = {}
    for i, (z_raw, z_cal, p) in enumerate(zip(raw, calibrated, native)):
        arrays[f"q{i}_logits_raw"] = np.asarray(z_raw, dtype=np.float32)
        arrays[f"q{i}_logits_calibrated"] = np.asarray(z_cal, dtype=np.float32)
        arrays[f"q{i}_probs"] = np.asarray(p, dtype=np.float32)
    return doc, arrays


def meta_block(checkpoint: str, oracle_path: str, ck, model, device) -> dict:
    from kev.model import SPECIAL

    from . import env as env_mod

    info = env_mod.gather()
    pin = CHECKPOINTS[checkpoint]
    return {
        "checkpoint": checkpoint,
        "run": f"{pin['repo']}@{pin['revision']}",
        "oracle_path": oracle_path,
        "backend": model.backend,
        "dtype": model.dtype,
        "device": device,
        "hybrid": bool(model.hybrid),
        "temperature": model.head.temperature,
        "prefix_min_tokens": model.prefix_min_tokens,
        "base": ck.meta.base,
        "base_revision": ck.meta.base_revision,
        "special_tokens": list(SPECIAL),
        "packages": info["packages"],
        "python": info["python"],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "upstream_kev_sha": env_mod.UPSTREAM_KEV_SHA,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def determinism_check(model, enc, device) -> dict:
    a, _ = timed_probs(model, enc, device)
    b, _ = timed_probs(model, enc, device)
    return {"max_abs_diff": max_abs_diff([p.tolist() for p in a], [p.tolist() for p in b])}


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=[*sorted(CHECKPOINTS), "all"], default="all")
    parser.add_argument("--path", choices=[*PATHS, "all"], default="all", dest="oracle_path")
    parser.add_argument("--fixture", action="append", help="restrict to these fixture ids")
    parser.add_argument(
        "--subset-reason",
        help="why this run covers a fixture subset; recorded in run.json (required with --fixture)",
    )
    args = parser.parse_args()
    if args.fixture and not args.subset_reason:
        raise SystemExit("--fixture needs --subset-reason so the gap is documented, not silent")

    present = [name for name in FORBIDDEN_ENV if os.environ.get(name)]
    if present:
        raise SystemExit(f"unset {present} first; goldens must not inherit KEV_* overrides")

    fixtures = []
    for path in sorted(paths.fixtures_dir().glob("*.json")):
        fixture = json.loads(path.read_text())
        if args.fixture and fixture["id"] not in args.fixture:
            continue
        fixtures.append(fixture)
    if not fixtures:
        raise SystemExit("no fixtures selected")

    checkpoints = sorted(CHECKPOINTS) if args.checkpoint == "all" else [args.checkpoint]
    summary = {}
    for checkpoint in checkpoints:
        for oracle_path in PATHS if args.oracle_path == "all" else [args.oracle_path]:
            if oracle_path == "mlx" and checkpoint == "kev-0.6b":
                log(f"{checkpoint}/mlx: skipped (attention-only Qwen3 base; upstream MLX refuses it)")
                summary[f"{checkpoint}/mlx"] = "skipped: attention-only base"
                continue
            log(f"loading {checkpoint} [{oracle_path}]")
            ck, tok, model, device = load_model(checkpoint, oracle_path)
            out_dir = paths.goldens_dir() / checkpoint / oracle_path
            out_dir.mkdir(parents=True, exist_ok=True)
            run_meta = meta_block(checkpoint, oracle_path, ck, model, device)
            total = time.time()
            determinism = None
            for index, fixture in enumerate(fixtures):
                log(f"  {fixture['id']}")
                doc, arrays = run_fixture(fixture, tok, model, device)
                if index == 0:
                    from kev.api import SystemOneRequest, to_record
                    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

                    rec, _ = to_record(SystemOneRequest.model_validate(fixture["request"]))
                    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
                    determinism = determinism_check(model, enc, device)
                doc["meta"] = run_meta
                doc["hidden_states_npz"] = f"{fixture['id']}.npz"
                (out_dir / f"{fixture['id']}.json").write_text(
                    json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
                )
                np.savez_compressed(out_dir / f"{fixture['id']}.npz", **arrays)
            run_doc = {
                "meta": run_meta,
                "fixtures": [f["id"] for f in fixtures],
                "determinism_first_fixture": determinism,
                "total_seconds": round(time.time() - total, 1),
            }
            if args.subset_reason:
                run_doc["subset_reason"] = args.subset_reason
            (out_dir / "run.json").write_text(json.dumps(run_doc, indent=2, sort_keys=True) + "\n")
            summary[f"{checkpoint}/{oracle_path}"] = f"{len(fixtures)} fixtures in {run_doc['total_seconds']}s"
            del model
    json.dump({"ok": True, "runs": summary}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
