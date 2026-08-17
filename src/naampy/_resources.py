"""Resolve the immutable gender checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO = "gojiberries/naampy"
HF_REVISION = "f7f2b7ac62a17f9bbfd102bf88388f1e3da5322a"
MODEL_DIR_ENV = "NAAMPY_MODEL_DIR"


def resolve_model(filename: str) -> str:
    """Return a local path for a pinned model artifact."""
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, filename, revision=HF_REVISION)
