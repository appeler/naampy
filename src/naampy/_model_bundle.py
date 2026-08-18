"""Typed, validated learned-model bundles for first-name pattern estimates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors.torch import load_file

from . import _resources
from .nnets import CharacterBiLSTM

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL_MANIFEST_SCHEMA_VERSION = 1
ENSEMBLE_METHOD = "equal-probability-mean"
CALIBRATION_METHOD = "positive-slope-logit-affine"


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """Architecture parameters required to reconstruct a checkpoint."""

    vocabulary: str
    embedding_dimension: int
    hidden_dimension: int
    layer_count: int
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    """One hashed SafeTensors ensemble member."""

    filename: str
    sha256: str
    training_seed: int


@dataclass(frozen=True, slots=True)
class ScoreCalibration:
    """Affine calibration on the logit of the ensemble mean probability."""

    method: str
    slope: float
    intercept: float
    population: str


@dataclass(frozen=True, slots=True)
class NamePatternModelManifest:
    """Validated metadata controlling learned inference."""

    schema_version: int
    model_version: str
    score_target: str
    reference_population: str
    label_source: str
    architecture: ModelArchitecture
    ensemble_method: str
    ensemble_members: tuple[EnsembleMember, EnsembleMember]
    calibration: ScoreCalibration


@dataclass(frozen=True, slots=True)
class NamePatternModelBundle:
    """Two validated inference models and their immutable provenance."""

    manifest: NamePatternModelManifest
    models: tuple[CharacterBiLSTM, CharacterBiLSTM]
    repository: str
    revision: str


def load_default_model_bundle() -> NamePatternModelBundle:
    """Load the package's configured immutable model bundle."""
    repository, revision = _resources.artifact_provenance(
        repository=_resources.MODEL_REPOSITORY,
        revision=_resources.MODEL_REVISION,
        local_directory_environment_variable=(
            _resources.MODEL_DIRECTORY_ENVIRONMENT_VARIABLE
        ),
    )
    manifest_path = _resources.resolve_model_artifact(
        _resources.MODEL_MANIFEST_FILENAME
    )
    return load_model_bundle(
        manifest_path,
        artifact_resolver=_resources.resolve_model_artifact,
        repository=repository,
        revision=revision,
    )


def load_model_bundle(
    manifest_path: str | Path,
    *,
    artifact_resolver: Callable[[str], str | Path],
    repository: str,
    revision: str,
) -> NamePatternModelBundle:
    """Load and validate a manifest and both SafeTensors members.

    Args:
        manifest_path: JSON model-manifest path.
        artifact_resolver: Callable resolving a manifest filename to a local path.
        repository: Artifact repository identifier recorded in outputs.
        revision: Immutable artifact revision recorded in outputs.

    Returns:
        A fully validated, evaluation-mode model bundle.

    Raises:
        ValueError: If metadata, hashes, or checkpoint state violate the contract.
    """
    manifest = parse_model_manifest(manifest_path)
    loaded_models: list[CharacterBiLSTM] = []
    for member in manifest.ensemble_members:
        member_path = Path(artifact_resolver(member.filename))
        _verify_sha256(member_path, member.sha256)
        model = _build_model(manifest.architecture)
        try:
            state_dictionary = load_file(member_path, device="cpu")
            model.load_state_dict(state_dictionary, strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"SafeTensors member {member.filename!r} does not match its architecture"
            ) from error
        model.eval()
        loaded_models.append(model)

    return NamePatternModelBundle(
        manifest=manifest,
        models=cast("tuple[CharacterBiLSTM, CharacterBiLSTM]", tuple(loaded_models)),
        repository=repository,
        revision=revision,
    )


def parse_model_manifest(path: str | Path) -> NamePatternModelManifest:
    """Parse a strict model manifest from JSON."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read model manifest {path!s}") from error
    if not isinstance(payload, dict):
        raise ValueError("Model manifest must contain a JSON object")

    _require_exact_keys(
        payload,
        {
            "schema_version",
            "model_version",
            "score_target",
            "reference_population",
            "label_source",
            "architecture",
            "ensemble",
            "calibration",
        },
        "model manifest",
    )
    schema_version = _require_integer(payload, "schema_version")
    if schema_version != MODEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model manifest schema_version {schema_version!r}"
        )

    architecture_payload = _require_mapping(payload, "architecture")
    _require_exact_keys(
        architecture_payload,
        {
            "vocabulary",
            "embedding_dimension",
            "hidden_dimension",
            "layer_count",
            "dropout_probability",
        },
        "architecture",
    )
    architecture = ModelArchitecture(
        vocabulary=_require_text(architecture_payload, "vocabulary"),
        embedding_dimension=_require_positive_integer(
            architecture_payload, "embedding_dimension"
        ),
        hidden_dimension=_require_positive_integer(
            architecture_payload, "hidden_dimension"
        ),
        layer_count=_require_positive_integer(architecture_payload, "layer_count"),
        dropout_probability=_require_probability(
            architecture_payload, "dropout_probability", upper_inclusive=False
        ),
    )
    if architecture.vocabulary != "abcdefghijklmnopqrstuvwxyz":
        raise ValueError("Model vocabulary must be exactly lowercase ASCII a-z")

    ensemble_payload = _require_mapping(payload, "ensemble")
    _require_exact_keys(ensemble_payload, {"method", "members"}, "ensemble")
    ensemble_method = _require_text(ensemble_payload, "method")
    if ensemble_method != ENSEMBLE_METHOD:
        raise ValueError(f"Unsupported ensemble method {ensemble_method!r}")
    members_payload = ensemble_payload["members"]
    if not isinstance(members_payload, list) or len(members_payload) != 2:
        raise ValueError("Ensemble must contain exactly two members")
    members = tuple(_parse_member(member) for member in members_payload)
    if members[0].filename == members[1].filename:
        raise ValueError("Ensemble member filenames must be distinct")
    if members[0].training_seed == members[1].training_seed:
        raise ValueError("Ensemble member training seeds must be distinct")

    calibration_payload = _require_mapping(payload, "calibration")
    _require_exact_keys(
        calibration_payload,
        {"method", "slope", "intercept", "population"},
        "calibration",
    )
    calibration = ScoreCalibration(
        method=_require_text(calibration_payload, "method"),
        slope=_require_finite_number(calibration_payload, "slope"),
        intercept=_require_finite_number(calibration_payload, "intercept"),
        population=_require_text(calibration_payload, "population"),
    )
    if calibration.method != CALIBRATION_METHOD:
        raise ValueError(f"Unsupported calibration method {calibration.method!r}")
    if calibration.slope <= 0:
        raise ValueError("Calibration slope must be positive")

    return NamePatternModelManifest(
        schema_version=schema_version,
        model_version=_require_text(payload, "model_version"),
        score_target=_require_text(payload, "score_target"),
        reference_population=_require_text(payload, "reference_population"),
        label_source=_require_text(payload, "label_source"),
        architecture=architecture,
        ensemble_method=ensemble_method,
        ensemble_members=cast("tuple[EnsembleMember, EnsembleMember]", members),
        calibration=calibration,
    )


def calibrated_ensemble_score(
    bundle: NamePatternModelBundle,
    encoded_names: torch.Tensor,
    name_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return calibrated scores after equal-probability member averaging."""
    with torch.inference_mode():
        member_probabilities = [
            torch.sigmoid(model(encoded_names, name_lengths)).squeeze(1)
            for model in bundle.models
        ]
        mean_probability = torch.stack(member_probabilities).mean(dim=0)
        stable_probability = mean_probability.clamp(1e-7, 1 - 1e-7)
        uncalibrated_logit = torch.logit(stable_probability)
        calibration = bundle.manifest.calibration
        return torch.sigmoid(
            calibration.slope * uncalibrated_logit + calibration.intercept
        )


def _parse_member(payload: object) -> EnsembleMember:
    if not isinstance(payload, dict):
        raise ValueError("Each ensemble member must be a JSON object")
    _require_exact_keys(payload, {"filename", "sha256", "training_seed"}, "member")
    filename = _require_text(payload, "filename")
    if Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError("Ensemble member filename must be a bare .safetensors name")
    digest = _require_text(payload, "sha256")
    if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ValueError("Ensemble member sha256 must be 64 lowercase hex characters")
    return EnsembleMember(
        filename=filename,
        sha256=digest,
        training_seed=_require_integer(payload, "training_seed"),
    )


def _build_model(architecture: ModelArchitecture) -> CharacterBiLSTM:
    return CharacterBiLSTM(
        vocabulary_size=len(architecture.vocabulary) + 1,
        output_dimension=1,
        embedding_dimension=architecture.embedding_dimension,
        hidden_dimension=architecture.hidden_dimension,
        layer_count=architecture.layer_count,
        dropout_probability=architecture.dropout_probability,
    )


def _verify_sha256(path: Path, expected_digest: str) -> None:
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


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], description: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{description} keys must be {sorted(expected)!r}; found {sorted(actual)!r}"
        )


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} must be a JSON object")
    return value


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be non-empty text")
    return value


def _require_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer")
    return value


def _require_positive_integer(payload: dict[str, Any], key: str) -> int:
    value = _require_integer(payload, key)
    if value <= 0:
        raise ValueError(f"{key!r} must be positive")
    return value


def _require_finite_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key!r} must be a number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{key!r} must be finite")
    return numeric_value


def _require_probability(
    payload: dict[str, Any], key: str, *, upper_inclusive: bool
) -> float:
    value = _require_finite_number(payload, key)
    valid_upper = value <= 1 if upper_inclusive else value < 1
    if value < 0 or not valid_upper:
        upper_bound = "less than or equal to 1" if upper_inclusive else "less than 1"
        raise ValueError(f"{key!r} must be nonnegative and {upper_bound}")
    return value
