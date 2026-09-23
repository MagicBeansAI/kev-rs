"""Compare golden oracle paths and record the evidence.

Two comparisons, written next to the goldens:

- ``<checkpoint>/mlx-vs-fp32.json``: per-fixture, per-question probability
  deltas between the MLX path and the torch fp32 reference, with argmax
  flips and the fp32 top-2 gap for each flip (upstream allows a flip only
  on a near-tie). These figures justify the K2 tolerances against
  upstream's own parity evidence.
- ``<checkpoint>/<path>/packed-vs-separate.json``: the questions of
  ``mixed-packed-3`` against the same questions asked alone
  (``separate-*``). Upstream guarantees row isolation; packed and separate
  must match to fp32 noise.

Reads goldens only; loads no models.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import paths
from .fetch import CHECKPOINTS

MATCH_GROUP = {
    "packed": "mixed-packed-3",
    "separate": {"route": "separate-route", "review": "separate-review", "urgency": "separate-urgency"},
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_golden(checkpoint: str, oracle_path: str, fixture: str) -> dict | None:
    path = paths.goldens_dir() / checkpoint / oracle_path / f"{fixture}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def question_stats(p_ref: list[float], p_alt: list[float]) -> dict:
    deltas = [abs(a - b) for a, b in zip(p_ref, p_alt)]
    ref_top = max(range(len(p_ref)), key=p_ref.__getitem__)
    alt_top = max(range(len(p_alt)), key=p_alt.__getitem__)
    ordered = sorted(p_ref, reverse=True)
    top2_gap = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
    return {
        "max_abs_dp": max(deltas),
        "mean_abs_dp": sum(deltas) / len(deltas),
        "argmax_flip": ref_top != alt_top,
        "ref_top2_gap": top2_gap,
    }


def compare_mlx(checkpoint: str) -> dict | None:
    fixtures = sorted(p.stem for p in (paths.goldens_dir() / checkpoint / "torch-fp32").glob("*.json") if p.stem != "run")
    per_fixture = {}
    all_stats = []
    for fixture in fixtures:
        ref = load_golden(checkpoint, "torch-fp32", fixture)
        alt = load_golden(checkpoint, "mlx", fixture)
        if ref is None or alt is None:
            continue
        rows = []
        for meta, p_ref, p_alt in zip(ref["questions"], ref["native"]["probs"], alt["native"]["probs"]):
            stats = question_stats(p_ref, p_alt)
            stats["question"] = meta["id"]
            rows.append(stats)
            all_stats.append(stats)
        answers_changed = {
            qid: {"fp32": ref["wire"]["answers"][qid], "mlx": alt["wire"]["answers"][qid]}
            for qid in ref["wire"]["answers"]
            if ref["wire"]["answers"][qid] != alt["wire"]["answers"][qid]
        }
        per_fixture[fixture] = {"questions": rows, "wire_answers_changed": answers_changed}
    if not all_stats:
        return None
    flips = [s for s in all_stats if s["argmax_flip"]]
    return {
        "checkpoint": checkpoint,
        "reference": "torch-fp32",
        "alternative": "mlx",
        "questions": len(all_stats),
        "max_abs_dp": max(s["max_abs_dp"] for s in all_stats),
        "mean_abs_dp": sum(s["mean_abs_dp"] for s in all_stats) / len(all_stats),
        "argmax_flips": len(flips),
        "argmax_flip_details": [
            {"question": s["question"], "ref_top2_gap": s["ref_top2_gap"]} for s in flips
        ],
        "upstream_reference_figures": {
            "kev-4b": {"max_abs_dp": 0.025242, "mean_abs_dp": 0.0016, "argmax_flips": "1 of 1264"},
            "kev-0.8b": {"max_abs_dp": 0.05423, "mean_abs_dp": 0.0023, "argmax_flips": "4 (0.3%)"},
            "source": "upstream runs/r4-mlx-parity-*/report.json at the pinned commit",
        },
        "fixtures": per_fixture,
    }


def compare_packed_separate(checkpoint: str, oracle_path: str) -> dict | None:
    packed = load_golden(checkpoint, oracle_path, MATCH_GROUP["packed"])
    if packed is None:
        return None
    rows = []
    for meta, p_packed in zip(packed["questions"], packed["native"]["probs"]):
        separate = load_golden(checkpoint, oracle_path, MATCH_GROUP["separate"][meta["id"]])
        if separate is None:
            return None
        stats = question_stats(p_packed, separate["native"]["probs"][0])
        stats["question"] = meta["id"]
        stats["wire_answers_equal"] = (
            packed["wire"]["answers"][meta["id"]] == separate["wire"]["answers"][meta["id"]]
        )
        rows.append(stats)
    return {
        "checkpoint": checkpoint,
        "oracle_path": oracle_path,
        "max_abs_dp": max(r["max_abs_dp"] for r in rows),
        "argmax_flips": sum(r["argmax_flip"] for r in rows),
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    summary = {}
    for checkpoint in sorted(CHECKPOINTS):
        report = compare_mlx(checkpoint)
        if report is not None:
            out = paths.goldens_dir() / checkpoint / "mlx-vs-fp32.json"
            out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            summary[f"{checkpoint}/mlx-vs-fp32"] = {
                "max_abs_dp": report["max_abs_dp"],
                "mean_abs_dp": report["mean_abs_dp"],
                "argmax_flips": report["argmax_flips"],
                "questions": report["questions"],
            }
            log(f"wrote {out}")
        for oracle_path in ("torch-fp32", "mlx"):
            report = compare_packed_separate(checkpoint, oracle_path)
            if report is not None:
                out = paths.goldens_dir() / checkpoint / oracle_path / "packed-vs-separate.json"
                out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                summary[f"{checkpoint}/{oracle_path}/packed-vs-separate"] = {
                    "max_abs_dp": report["max_abs_dp"],
                    "argmax_flips": report["argmax_flips"],
                }
                log(f"wrote {out}")
    json.dump({"ok": True, "summary": summary}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
