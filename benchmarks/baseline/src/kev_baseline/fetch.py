"""Download the pinned model assets and write ``manifests/sources.json``.

Adapter repositories are pinned by revision below. The base model repository
and revision for each checkpoint come from ``head.pt`` (``meta.base`` and
``meta.base_revision``), which is the same source upstream kev uses. The
tokenizer loads from the base repository, not from the adapter repository.

Every downloaded file is verified by size and SHA-256. Weights live under
the cache root only, never in git. JSON goes to stdout, diagnostics to
stderr.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from . import env as env_mod
from . import paths

SCHEMA_VERSION = 1

# Adapter repositories, pinned 2026-09-24 (main at time of pinning).
CHECKPOINTS = {
    "kev-0.8b": {"repo": "jaredpalmer/kev-0.8b", "revision": "54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8"},
    "kev-4b": {"repo": "jaredpalmer/kev-4b", "revision": "485ace8703592fcf405488b262449990824cfed1"},
    "kev-0.6b": {"repo": "jaredpalmer/kev-0.6b", "revision": "dece6dba8d43f0f7ded45e9f5b9df12474d90843"},
}

# Everything upstream's snapshot_download would take (checkpoint.resolve_run
# uses allow_patterns *.json/*.safetensors/*.pt/*.txt/*.jinja), plus the model
# card for provenance. Pinning the same set keeps every file the runtime can
# touch under the manifest.
ADAPTER_SUFFIXES = (".json", ".safetensors", ".pt", ".txt", ".jinja")

BASE_PATTERNS_SUFFIXES = (".json", ".safetensors", ".txt", ".jinja")
BASE_EXCLUDE = (".gitattributes",)

LICENSE = "Apache-2.0"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_files(api: HfApi, repo: str, revision: str) -> dict[str, int]:
    info = api.repo_info(repo, revision=revision, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings}


def fetch_files(repo: str, revision: str, names: list[str]) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for name in sorted(names):
        local = hf_hub_download(
            repo_id=repo,
            filename=name,
            revision=revision,
            cache_dir=str(paths.hf_cache_dir()),
        )
        local_path = Path(local)
        entries[name] = {
            "bytes": local_path.stat().st_size,
            "sha256": sha256_of(local_path),
        }
        log(f"  {name}: {entries[name]['bytes']} bytes")
    return entries


def read_head_meta(repo: str, revision: str) -> dict:
    import torch

    local = hf_hub_download(
        repo_id=repo,
        filename="head.pt",
        revision=revision,
        cache_dir=str(paths.hf_cache_dir()),
    )
    meta = torch.load(local, map_location="cpu", weights_only=True)
    if not isinstance(meta, dict):
        raise TypeError(f"{repo}/head.pt did not load as a dict")
    return meta


def base_file_names(files: dict[str, int]) -> list[str]:
    names = []
    for name in files:
        if name in BASE_EXCLUDE:
            continue
        if name.endswith(BASE_PATTERNS_SUFFIXES) or name == "README.md":
            names.append(name)
    return names


def python_environment() -> dict:
    info = env_mod.gather()
    return {
        "python": info["python"],
        "packages": info["packages"],
        "reason": (
            "Pins match the versions upstream kev's own uv.lock resolves at "
            f"commit {env_mod.UPSTREAM_KEV_SHA}."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-bases", action="store_true", help="fetch adapters only")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing manifest against the cache; download nothing",
    )
    args = parser.parse_args()

    manifest_path = paths.manifests_dir() / "sources.json"

    if args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        failures = verify(manifest)
        json.dump({"ok": not failures, "failures": failures}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        if failures:
            sys.exit(1)
        return

    api = HfApi()
    sources: dict[str, dict] = {}
    checkpoints: dict[str, dict] = {}

    for name, pin in CHECKPOINTS.items():
        repo, revision = pin["repo"], pin["revision"]
        log(f"adapter {repo} @ {revision[:12]}")
        available = repo_files(api, repo, revision)
        wanted = [
            f
            for f in available
            if f == "README.md" or (f.endswith(ADAPTER_SUFFIXES) and f not in BASE_EXCLUDE)
        ]
        entries = fetch_files(repo, revision, wanted)
        sources[name] = {
            "url": f"https://huggingface.co/{repo}",
            "revision": revision,
            "license": LICENSE,
            "files": entries,
        }
        meta = read_head_meta(repo, revision)
        facts = {
            "base": meta.get("base"),
            "base_revision": meta.get("base_revision"),
            "head_dim": meta.get("head_dim"),
            "temperature": meta.get("temperature"),
            "option_isolation": meta.get("option_isolation"),
            "special_embeddings": meta.get("special_embeddings"),
            "weights_dtype": meta.get("weights_dtype"),
            "lora": meta.get("lora"),
        }
        checkpoints[name] = facts
        log(f"  base: {facts['base']} @ {facts['base_revision']} T={facts['temperature']}")

    if not args.skip_bases:
        for name, facts in checkpoints.items():
            base, base_rev = facts["base"], facts["base_revision"]
            if base is None or base_rev is None:
                raise ValueError(f"{name}: head.pt lacks base/base_revision")
            key = f"base:{base}"
            if key in sources:
                if sources[key]["revision"] != base_rev:
                    raise ValueError(f"{base}: conflicting base revisions")
                continue
            log(f"base {base} @ {base_rev[:12]}")
            available = repo_files(api, base, base_rev)
            wanted = base_file_names(available)
            total = sum(available[f] for f in wanted)
            log(f"  total download: {total} bytes ({total / 1e9:.2f} GB)")
            entries = fetch_files(base, base_rev, wanted)
            sources[key] = {
                "url": f"https://huggingface.co/{base}",
                "revision": base_rev,
                "license": LICENSE,
                "files": entries,
            }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "upstream": {
            "url": "https://github.com/jaredpalmer/kev",
            "revision": env_mod.UPSTREAM_KEV_SHA,
            "license": LICENSE,
        },
        "checkpoints": checkpoints,
        "sources": sources,
        "python_environment": python_environment(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(f"wrote {manifest_path}")
    json.dump({"ok": True, "manifest": str(manifest_path)}, sys.stdout, indent=2)
    sys.stdout.write("\n")


def verify(manifest: dict) -> list[str]:
    """Re-hash every cached file against the manifest."""
    failures: list[str] = []
    for source_name, source in manifest["sources"].items():
        repo = source["url"].removeprefix("https://huggingface.co/")
        for file_name, entry in source["files"].items():
            try:
                local = hf_hub_download(
                    repo_id=repo,
                    filename=file_name,
                    revision=source["revision"],
                    cache_dir=str(paths.hf_cache_dir()),
                    local_files_only=True,
                )
            except Exception as error:  # noqa: BLE001 - report, do not mask
                failures.append(f"{source_name}/{file_name}: missing ({error})")
                continue
            local_path = Path(local)
            if local_path.stat().st_size != entry["bytes"]:
                failures.append(f"{source_name}/{file_name}: size mismatch")
            elif sha256_of(local_path) != entry["sha256"]:
                failures.append(f"{source_name}/{file_name}: sha256 mismatch")
    return failures


if __name__ == "__main__":
    main()
