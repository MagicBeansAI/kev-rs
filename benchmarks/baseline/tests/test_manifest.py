"""The asset manifest is complete, pinned and internally consistent."""

import json
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[3] / "manifests" / "sources.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not generated yet")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_schema(manifest):
    for key in ("schema_version", "generated_at", "upstream", "checkpoints", "sources", "python_environment"):
        assert key in manifest


def test_upstream_pin(manifest):
    assert manifest["upstream"]["revision"] == "557598fced1dada75dfbf36ed144dce309ac6ceb"
    assert manifest["upstream"]["license"] == "Apache-2.0"


def test_every_file_is_pinned(manifest):
    for name, source in manifest["sources"].items():
        assert len(source["revision"]) == 40, name
        assert source["files"], name
        for file_name, entry in source["files"].items():
            assert entry["bytes"] > 0, f"{name}/{file_name}"
            assert len(entry["sha256"]) == 64, f"{name}/{file_name}"


def test_checkpoint_facts(manifest):
    checkpoints = manifest["checkpoints"]
    assert set(checkpoints) == {"kev-0.8b", "kev-4b", "kev-0.6b"}
    for name, facts in checkpoints.items():
        assert facts["base"].startswith("Qwen/"), name
        assert len(facts["base_revision"]) == 40, name
        assert facts["head_dim"] == 256, name
        assert facts["lora"] == 16, name
        # The base repository each checkpoint points at is itself pinned.
        assert f"base:{facts['base']}" in manifest["sources"], name
        assert manifest["sources"][f"base:{facts['base']}"]["revision"] == facts["base_revision"], name
    # Qwen3.5 checkpoints carry a fitted temperature; the Qwen3 one does not.
    assert 2.1 < checkpoints["kev-0.8b"]["temperature"] < 2.5
    assert 2.1 < checkpoints["kev-4b"]["temperature"] < 2.5
    assert checkpoints["kev-0.6b"]["temperature"] is None


def test_converted_heads(manifest):
    converted = manifest.get("converted", {})
    assert set(converted) == {"kev-0.8b", "kev-4b", "kev-0.6b"}
    for name, entry in converted.items():
        assert len(entry["source_head_pt_sha256"]) == 64, name
        assert entry["source_head_pt_sha256"] == manifest["sources"][name]["files"]["head.pt"]["sha256"], name
        for file_name in ("head.safetensors", "head.meta.json"):
            assert len(entry["files"][file_name]["sha256"]) == 64, f"{name}/{file_name}"
