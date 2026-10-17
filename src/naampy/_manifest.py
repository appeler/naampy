"""Validation shared by the lookup-table and model manifests."""

import hashlib
from pathlib import Path
from typing import Any


def verify_sha256(path: Path, expected_digest: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            f"Artifact hash mismatch for {path.name!r}: "
            f"expected {expected_digest}, found {actual_digest}"
        )


def require_exact_keys(
    payload: dict[str, Any], expected: set[str], description: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{description} keys must be {sorted(expected)!r}; found {sorted(actual)!r}"
        )


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} must be a JSON object")
    return value


def require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be non-empty text")
    return value


def require_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer")
    return value


def require_positive_integer(payload: dict[str, Any], key: str) -> int:
    value = require_integer(payload, key)
    if value <= 0:
        raise ValueError(f"{key!r} must be positive")
    return value
