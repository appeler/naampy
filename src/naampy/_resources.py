"""Resolve immutable Naampy runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_REPOSITORY = "gojiberries/naampy"
MODEL_REVISION = "72d5ec16d3ede38e3627504a108b69ffd114c813"
MODEL_MANIFEST_FILENAME = "first_name_pattern_manifest.json"
MODEL_DIRECTORY_ENVIRONMENT_VARIABLE = "NAAMPY_MODEL_DIR"

LOOKUP_TABLE_REPOSITORY = "gojiberries/naampy"
LOOKUP_TABLE_REVISION = "72d5ec16d3ede38e3627504a108b69ffd114c813"
LOOKUP_TABLE_MANIFEST_FILENAME = "first_name_composition_manifest.json"
LOOKUP_TABLE_DIRECTORY_ENVIRONMENT_VARIABLE = "NAAMPY_LOOKUP_TABLE_DIR"


def resolve_artifact(
    filename: str,
    *,
    repository: str,
    revision: str | None,
    local_directory_environment_variable: str,
) -> Path:
    """Resolve one artifact from a local override or immutable HF revision.

    Args:
        filename: Artifact filename within the repository.
        repository: Hugging Face repository identifier.
        revision: Immutable Hugging Face commit hash.
        local_directory_environment_variable: Environment variable naming a local
            artifact directory.

    Returns:
        The local artifact path.

    Raises:
        RuntimeError: If no local artifact exists and no immutable revision is set.
        FileNotFoundError: If a configured local directory lacks the artifact.
        ValueError: If the configured revision is not an immutable commit hash.
    """
    local_directory = os.environ.get(local_directory_environment_variable)
    if local_directory is not None:
        local_path = Path(local_directory) / filename
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Configured artifact directory lacks {filename!r}: {local_path}"
            )
        return local_path

    if revision is None:
        raise RuntimeError(
            f"No immutable revision is configured for {repository!r}; "
            f"set {local_directory_environment_variable} for a local bundle"
        )
    if len(revision) != 40 or set(revision) - set("0123456789abcdef"):
        raise ValueError(
            "Artifact revision must be a 40-character lowercase commit hash"
        )

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repository, filename, revision=revision))


def resolve_model_artifact(filename: str) -> Path:
    """Resolve one learned-model artifact."""
    return resolve_artifact(
        filename,
        repository=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        local_directory_environment_variable=MODEL_DIRECTORY_ENVIRONMENT_VARIABLE,
    )


def resolve_lookup_table_artifact(filename: str) -> Path:
    """Resolve one exact-lookup artifact."""
    return resolve_artifact(
        filename,
        repository=LOOKUP_TABLE_REPOSITORY,
        revision=LOOKUP_TABLE_REVISION,
        local_directory_environment_variable=LOOKUP_TABLE_DIRECTORY_ENVIRONMENT_VARIABLE,
    )
