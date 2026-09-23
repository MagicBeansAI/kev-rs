"""Point every test at the pinned model cache before huggingface_hub loads."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HUB_CACHE", str(REPO_ROOT / ".cache" / "kev" / "hf"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
