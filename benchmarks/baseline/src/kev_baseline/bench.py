"""Benchmark the upstream M5-table workload on this host.

Workload: the frozen ``bench-m5`` fixture (five 3-option choice questions on
a ~296-token state), the shape behind upstream's README table (Kev-0.8B:
149 ms new / 28 ms repeated; Kev-4B: 721 / 136 ms, both MLX on an M5).

Protocol, mirroring upstream serving:

- ``new_state``: ``probs_and_prefix`` — one full pass, the state prefix is
  computed and kept.
- ``repeated_state``: ``probs_with_prefix`` — only the question branches
  run against the cached state.

Results are immutable: a run id refuses to overwrite existing files. Rerun
with a new run id instead. Record load averages; benchmark on an idle host.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

from . import paths

os.environ.setdefault("HF_HUB_CACHE", str(paths.hf_cache_dir()))

from .fetch import CHECKPOINTS  # noqa: E402 - needs the env set first
from .goldens import FORBIDDEN_ENV, load_model  # noqa: E402

BENCH_FIXTURE = "bench-m5"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def host_info() -> dict:
    brand = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": brand or platform.processor(),
        "loadavg_1m": os.getloadavg()[0],
    }


def run_bench(checkpoint: str, oracle_path: str, iterations: int, warmup: int) -> dict:
    from kev.api import SystemOneRequest, to_record
    from kev.device import sync
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

    fixture = json.loads((paths.fixtures_dir() / f"{BENCH_FIXTURE}.json").read_text())
    ck, tok, model, device = load_model(checkpoint, oracle_path)
    rec, _ = to_record(SystemOneRequest.model_validate(fixture["request"]))
    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)

    def timed(fn):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        return result, (time.perf_counter() - start) * 1000

    for _ in range(warmup):
        (_, prefix), _ = timed(lambda: model.probs_and_prefix(enc))
        timed(lambda: model.probs_with_prefix(enc, prefix))

    new_state, repeated = [], []
    for _ in range(iterations):
        (_, prefix), ms = timed(lambda: model.probs_and_prefix(enc))
        new_state.append(ms)
        _, ms = timed(lambda: model.probs_with_prefix(enc, prefix))
        repeated.append(ms)

    def stats(samples: list[float]) -> dict:
        ordered = sorted(samples)
        return {
            "median_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
            "min_ms": round(ordered[0], 2),
            "max_ms": round(ordered[-1], 2),
            "samples_ms": [round(s, 2) for s in samples],
        }

    pin = CHECKPOINTS[checkpoint]
    return {
        "checkpoint": checkpoint,
        "run": f"{pin['repo']}@{pin['revision']}",
        "oracle_path": oracle_path,
        "backend": model.backend,
        "dtype": model.dtype,
        "device": device,
        "temperature": model.head.temperature,
        "fixture": BENCH_FIXTURE,
        "state_tokens": enc["seg"].count(0),
        "total_tokens": len(enc["ids"]),
        "questions": len(enc["decide_idx"]),
        "iterations": iterations,
        "warmup": warmup,
        "new_state": stats(new_state),
        "repeated_state": stats(repeated),
        "host": host_info(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--path", choices=("torch-fp32", "mlx"), required=True, dest="oracle_path")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--run-id", required=True, help="results/<run-id>/; existing files are never overwritten")
    args = parser.parse_args()

    present = [name for name in FORBIDDEN_ENV if os.environ.get(name)]
    if present:
        raise SystemExit(f"unset {present} first; benchmarks must not inherit KEV_* overrides")

    out_dir = paths.results_dir() / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench-{args.checkpoint}-{args.oracle_path}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; results are immutable, use a new --run-id")

    log(f"benchmarking {args.checkpoint} [{args.oracle_path}] x{args.iterations}")
    report = run_bench(args.checkpoint, args.oracle_path, args.iterations, args.warmup)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    json.dump({"ok": True, "report": str(out_path), "new_state": report["new_state"]["median_ms"], "repeated_state": report["repeated_state"]["median_ms"]}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
