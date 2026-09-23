"""Convert each checkpoint's ``head.pt`` into a pickle-free artifact.

``head.pt`` is a PyTorch pickle (a plain dict). Rust must never execute
pickle, so this tool writes, per checkpoint:

- ``head.safetensors``: the four fp32 pointer-head tensors
  (``q.weight``, ``q.bias``, ``k.weight``, ``k.bias``).
- ``head.meta.json``: every non-tensor field, plus provenance (the SHA-256
  of the source ``head.pt``) and the tensor shapes.

Both artifacts are checksummed into ``manifests/sources.json`` under
``converted``. The artifacts live in the cache, not in git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import paths
from .fetch import CHECKPOINTS, sha256_of

TENSOR_KEYS = ("q.weight", "q.bias", "k.weight", "k.bias")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def json_safe(value):
    """Return value if JSON-serializable, else its repr (provenance only)."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return {"unserializable_repr": repr(value)}


def convert_one(name: str, repo: str, revision: str) -> dict:
    import torch
    from safetensors.torch import save_file

    source = Path(
        hf_hub_download(
            repo_id=repo,
            filename="head.pt",
            revision=revision,
            cache_dir=str(paths.hf_cache_dir()),
            local_files_only=True,
        )
    )
    meta = torch.load(source, map_location="cpu", weights_only=True)
    head = meta["head"]
    missing = [k for k in TENSOR_KEYS if k not in head]
    unexpected = [k for k in head if k not in TENSOR_KEYS]
    if missing or unexpected:
        raise ValueError(f"{name}: head tensors missing={missing} unexpected={unexpected}")
    for key in TENSOR_KEYS:
        if head[key].dtype != torch.float32:
            raise ValueError(f"{name}: {key} is {head[key].dtype}, expected fp32")

    out_dir = paths.converted_dir() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    st_path = out_dir / "head.safetensors"
    save_file({k: head[k].contiguous() for k in TENSOR_KEYS}, str(st_path))

    scalar_meta = {k: json_safe(v) for k, v in meta.items() if k != "head"}
    meta_doc = {
        "checkpoint": name,
        "source": {
            "repo": repo,
            "revision": revision,
            "file": "head.pt",
            "sha256": sha256_of(source),
        },
        "tensors": {k: {"shape": list(head[k].shape), "dtype": "float32"} for k in TENSOR_KEYS},
        "meta": scalar_meta,
    }
    meta_path = out_dir / "head.meta.json"
    meta_path.write_text(json.dumps(meta_doc, indent=2, sort_keys=True) + "\n")

    entry = {
        "directory": str(out_dir.relative_to(paths.repo_root())),
        "files": {
            "head.safetensors": {"bytes": st_path.stat().st_size, "sha256": sha256_of(st_path)},
            "head.meta.json": {"bytes": meta_path.stat().st_size, "sha256": sha256_of(meta_path)},
        },
        "source_head_pt_sha256": meta_doc["source"]["sha256"],
    }
    log(f"{name}: wrote {st_path} ({entry['files']['head.safetensors']['bytes']} bytes)")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), action="append")
    args = parser.parse_args()
    selected = args.checkpoint or sorted(CHECKPOINTS)

    manifest_path = paths.manifests_dir() / "sources.json"
    manifest = json.loads(manifest_path.read_text())
    converted = manifest.setdefault("converted", {})

    for name in selected:
        pin = CHECKPOINTS[name]
        converted[name] = convert_one(name, pin["repo"], pin["revision"])

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    json.dump({"ok": True, "converted": {n: converted[n]["files"] for n in selected}}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
