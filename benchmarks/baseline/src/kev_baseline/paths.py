"""Repository paths and the local cache root.

The cache root is ``$KEV_RS_HOME``, falling back to ``<repo>/.cache/kev``.
Model weights live only under the cache root, never in git.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the kev-rs repository root (three levels above this file)."""
    return Path(__file__).resolve().parents[4]


def cache_root() -> Path:
    override = os.environ.get("KEV_RS_HOME")
    if override:
        return Path(override).expanduser()
    return repo_root() / ".cache" / "kev"


def hf_cache_dir() -> Path:
    return cache_root() / "hf"


def converted_dir() -> Path:
    """Converted artifacts (head.safetensors) live here, outside git."""
    return cache_root() / "converted"


def manifests_dir() -> Path:
    return repo_root() / "manifests"


def benchmarks_dir() -> Path:
    return repo_root() / "benchmarks"


def fixtures_dir() -> Path:
    return benchmarks_dir() / "fixtures" / "requests"


def goldens_dir() -> Path:
    return benchmarks_dir() / "goldens"


def results_dir() -> Path:
    return benchmarks_dir() / "results"
