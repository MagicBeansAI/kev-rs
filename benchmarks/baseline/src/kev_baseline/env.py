"""Report the resolved baseline environment as JSON on stdout."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

PINNED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "numpy",
    "safetensors",
    "tokenizers",
    "huggingface-hub",
    "mlx",
    "mlx-lm",
    "pydantic",
    "kev",
]

UPSTREAM_KEV_SHA = "557598fced1dada75dfbf36ed144dce309ac6ceb"


def gather() -> dict:
    import torch

    packages = {}
    for name in PINNED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "upstream_kev_sha": UPSTREAM_KEV_SHA,
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_num_threads": torch.get_num_threads(),
    }


def main() -> None:
    json.dump(gather(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
