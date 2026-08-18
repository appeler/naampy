"""Train, calibrate, and evaluate the Naampy character BiLSTM in gated stages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as torch_functional
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save as serialize_safetensors

from model_training.evaluation import (
    bootstrap_metric_intervals,
    fit_logistic_calibration,
    probability_metrics,
    report_metrics,
)
from naampy._model_bundle import (
    CALIBRATION_METHOD,
    ENSEMBLE_METHOD,
    MODEL_MANIFEST_SCHEMA_VERSION,
    parse_model_manifest,
)
from naampy.nnets import (
    CHARACTER_VOCABULARY_SIZE,
    LSTM_DROPOUT_PROBABILITY,
    LSTM_EMBEDDING_DIMENSION,
    LSTM_HIDDEN_DIMENSION,
    LSTM_LAYER_COUNT,
    CharacterBiLSTM,
    encode_normalized_name,
    pad_encoded_names,
)

LOGGER = logging.getLogger(__name__)
STAGE_SCHEMA_VERSION: Final = 1
DATA_MANIFEST_SCHEMA_VERSION: Final = 1
PARTITIONS: Final = ("training", "validation", "calibration", "test")
DEVELOPMENT_PARTITIONS: Final = ("training", "validation")
FINAL_FIT_PARTITIONS: Final = ("training", "validation")
STANDARD_TRAINING_SEEDS: Final = (0, 1)
FINAL_MODEL_VERSION: Final = "0.11.0"
MODEL_SCORE_TARGET: Final = (
    "female source-label share among represented female and male electoral-roll labels"
)
MODEL_LABEL_SOURCE: Final = (
    "female and male source-label counts from Indian electoral-roll registration "
    "records in Dataverse DOI 10.7910/DVN/WZGJBM"
)
CALIBRATION_POPULATION: Final = (
    "calibration partition of the frozen Naampy v3 aggregate first-name dataset"
)
EXPECTED_TRAINING_COLUMNS: Final = (
    "normalized_name",
    "female_label_record_count",
    "male_label_record_count",
    "represented_binary_label_record_count",
    "partition",
)
EXPECTED_TRAINING_SCHEMA: Final = pa.schema(
    [
        pa.field("normalized_name", pa.string(), nullable=False),
        pa.field("female_label_record_count", pa.int64(), nullable=False),
        pa.field("male_label_record_count", pa.int64(), nullable=False),
        pa.field("represented_binary_label_record_count", pa.int64(), nullable=False),
        pa.field("partition", pa.string(), nullable=False),
    ]
)
_REPEATED_CHARACTER = re.compile(r"(.)\1\1")


@dataclass(frozen=True)
class PartitionData:
    """Names, model features, targets, and weights for permitted partitions."""

    names: list[str]
    encoded_names: list[list[int]]
    female_proportions: np.ndarray
    represented_record_counts: np.ndarray
    partitions: tuple[str, ...]


@dataclass(frozen=True)
class TrainingConfiguration:
    """Frozen optimization settings shared by development and final fitting."""

    training_seeds: tuple[int, ...]
    epochs: int
    samples_per_epoch: int
    batch_size: int
    learning_rate: float

    def validate(self) -> None:
        """Reject configurations that cannot define a training run."""
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("training seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.training_seeds):
            raise ValueError("training seeds must be nonnegative")
        if self.epochs < 1 or self.samples_per_epoch < 1 or self.batch_size < 1:
            raise ValueError(
                "epochs, samples per epoch, and batch size must be positive"
            )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")

    def as_manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "training_seeds": list(self.training_seeds),
            "epochs_requested": self.epochs,
            "samples_per_epoch": self.samples_per_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "loss": "binary_cross_entropy_with_logits",
            "training_name_sampling": (
                "represented_binary_label_record_count_weighted_with_replacement"
            ),
            "canonical_input_order": "normalized_name ascending",
        }

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> TrainingConfiguration:
        """Load and validate settings from a development manifest."""
        configuration = cls(
            training_seeds=tuple(int(seed) for seed in manifest["training_seeds"]),
            epochs=int(manifest["epochs_requested"]),
            samples_per_epoch=int(manifest["samples_per_epoch"]),
            batch_size=int(manifest["batch_size"]),
            learning_rate=float(manifest["learning_rate"]),
        )
        configuration.validate()
        expected = configuration.as_manifest()
        if manifest != expected:
            raise ValueError("development training configuration is not recognized")
        return configuration


def load_names(
    path: str | Path, max_rows: int | None = None
) -> tuple[list[str], list[list[int]], list[float], list[float]]:
    """Load the legacy gzip source for the deployed-checkpoint audit only."""
    with gzip.open(path, "rt", encoding="utf-8") as compressed_file:
        source_table = pd.read_csv(
            compressed_file, nrows=max_rows, dtype={"first_name": str}
        )
    grouped_names = cast(
        "pd.DataFrame",
        source_table.groupby("first_name", as_index=False)[
            ["n_female", "n_male"]
        ].sum(),
    )
    grouped_names = grouped_names[
        (grouped_names["n_female"] + grouped_names["n_male"]) > 0
    ]
    names: list[str] = []
    encoded_names: list[list[int]] = []
    female_proportions: list[float] = []
    represented_record_counts: list[float] = []
    for name_value, female_value, male_value in grouped_names.itertuples(
        index=False, name=None
    ):
        name = str(name_value)
        female_count = float(female_value)
        male_count = float(male_value)
        if (
            not (2 < len(name) < 20)
            or not name.isascii()
            or not name.isalpha()
            or _REPEATED_CHARACTER.search(name)
        ):
            continue
        encoded_name = encode_normalized_name(name)
        if not encoded_name:
            continue
        names.append(name)
        encoded_names.append(encoded_name)
        female_proportions.append(female_count / (female_count + male_count))
        represented_record_counts.append(female_count + male_count)
    return names, encoded_names, female_proportions, represented_record_counts


@torch.no_grad()
def predict_probabilities(
    model: CharacterBiLSTM,
    encoded_names: list[list[int]],
    device: str,
) -> np.ndarray:
    """Return raw female-label name-pattern probabilities."""
    model.eval()
    probabilities: list[float] = []
    for start in range(0, len(encoded_names), 512):
        inputs, lengths = pad_encoded_names(encoded_names[start : start + 512])
        batch_probabilities = (
            torch.sigmoid(model(inputs.to(device), lengths)).squeeze(1).cpu().numpy()
        )
        probabilities.extend(batch_probabilities.tolist())
    return np.asarray(probabilities, dtype=np.float64)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_expected_sha256(value: str, argument_name: str) -> str:
    """Return a normalized expected digest or reject malformed input."""
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{argument_name} must be a 64-character SHA-256 digest")
    return normalized


def _require_exact_keys(
    document: dict[str, Any], expected_keys: set[str], description: str
) -> None:
    """Reject missing and unrecognized manifest fields."""
    actual_keys = set(document)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{description} keys must be {sorted(expected_keys)!r}; "
            f"found {sorted(actual_keys)!r}"
        )


def _require_artifact_filename_and_hash(
    document: dict[str, Any], description: str
) -> tuple[str, str]:
    """Validate one bare SafeTensors filename and digest."""
    filename = document.get("filename")
    digest = document.get("sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".safetensors")
    ):
        raise ValueError(f"{description} filename must be a bare .safetensors name")
    if not isinstance(digest, str):
        raise ValueError(f"{description} sha256 must be text")
    return filename, _validated_expected_sha256(digest, f"{description} sha256")


def _verify_file_hash(path: Path, expected_sha256: str, artifact_name: str) -> None:
    """Verify a frozen artifact before parsing or loading it."""
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{artifact_name} hash mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )


def _write_new_bytes(path: Path, content: bytes) -> None:
    """Atomically create a file without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_path = Path(temporary_file.name)
    try:
        os.link(temporary_path, path)
        path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic(report: dict[str, Any], output_path: Path) -> None:
    """Create a frozen JSON artifact without overwriting an existing file."""
    content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new_bytes(output_path, content)


def _load_frozen_json(
    path: Path, expected_sha256: str, artifact_name: str
) -> dict[str, Any]:
    """Hash-verify and parse a frozen JSON artifact."""
    _verify_file_hash(path, expected_sha256, artifact_name)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{artifact_name} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{artifact_name} must contain a JSON object")
    return cast("dict[str, Any]", document)


def _verify_training_artifact(
    parquet_path: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify training metadata and Parquet bytes without reading any row."""
    manifest = _load_frozen_json(
        manifest_path, expected_manifest_sha256, "training data manifest"
    )
    if manifest.get("schema_version") != DATA_MANIFEST_SCHEMA_VERSION:
        raise ValueError("training data manifest schema version is not recognized")
    if manifest.get("artifact_role") != "name_pattern_model_training_data":
        raise ValueError("training data artifact role is not recognized")
    if manifest.get("output", {}).get("format") != "parquet":
        raise ValueError("training data output must be Parquet")
    expected_parquet_sha256 = cast("str", manifest["output"]["sha256"])
    _verify_file_hash(parquet_path, expected_parquet_sha256, "training Parquet")
    if pq.read_schema(parquet_path) != EXPECTED_TRAINING_SCHEMA:
        raise ValueError("training Parquet does not use the required typed schema")
    split = manifest.get("split", {})
    if split.get("seed") != 0:
        raise ValueError("training data must use the frozen split seed 0")
    if split.get("fractions") != {
        "training": 0.70,
        "validation": 0.10,
        "calibration": 0.10,
        "test": 0.10,
    }:
        raise ValueError("training data split fractions are not recognized")
    membership = split.get("exported_membership_sha256", {})
    if set(membership) != set(PARTITIONS):
        raise ValueError("training data manifest must hash every partition")
    return manifest


def _canonical_name_set_sha256(names: list[str]) -> str:
    """Hash a name set in the exporter-compatible canonical representation."""
    encoded = json.dumps(
        sorted(names), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_partition_data(
    parquet_path: Path,
    data_manifest: dict[str, Any],
    permitted_partitions: tuple[str, ...],
) -> PartitionData:
    """Load only explicitly permitted partitions in canonical name order."""
    if (
        not permitted_partitions
        or len(set(permitted_partitions)) != len(permitted_partitions)
        or any(partition not in PARTITIONS for partition in permitted_partitions)
    ):
        raise ValueError("permitted partitions must be unique recognized partitions")
    tables = [
        pq.read_table(
            parquet_path,
            columns=list(EXPECTED_TRAINING_COLUMNS),
            filters=[("partition", "=", partition)],
        )
        for partition in permitted_partitions
    ]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if table.schema != EXPECTED_TRAINING_SCHEMA:
        raise ValueError("filtered training data has an unexpected schema")
    names = cast("list[str]", table["normalized_name"].to_pylist())
    assigned_partitions = cast("list[str]", table["partition"].to_pylist())
    if not names or len(names) != len(set(names)):
        raise ValueError("loaded partitions must contain unique names")
    if any(partition not in permitted_partitions for partition in assigned_partitions):
        raise ValueError("Parquet filter returned a non-permitted partition")
    expected_membership = data_manifest["split"]["exported_membership_sha256"]
    for partition in permitted_partitions:
        partition_names = [
            name
            for name, assigned_partition in zip(names, assigned_partitions, strict=True)
            if assigned_partition == partition
        ]
        if not partition_names:
            raise ValueError(f"partition {partition!r} is empty")
        if (
            _canonical_name_set_sha256(partition_names)
            != expected_membership[partition]
        ):
            raise ValueError(f"partition {partition!r} membership hash mismatch")

    female_counts = np.asarray(
        table["female_label_record_count"].to_pylist(), dtype=np.float64
    )
    male_counts = np.asarray(
        table["male_label_record_count"].to_pylist(), dtype=np.float64
    )
    represented_counts = np.asarray(
        table["represented_binary_label_record_count"].to_pylist(),
        dtype=np.float64,
    )
    if not np.array_equal(female_counts + male_counts, represented_counts):
        raise ValueError(
            "represented binary-label record count must equal female plus male counts"
        )
    if np.any(represented_counts <= 0):
        raise ValueError("represented binary-label record counts must be positive")
    order = np.argsort(np.asarray(names, dtype=object), kind="stable")
    ordered_names = [names[int(index)] for index in order]
    encoded_names = [encode_normalized_name(name) for name in ordered_names]
    if any(not encoded_name for encoded_name in encoded_names):
        raise ValueError(
            "training Parquet contains a name outside the model vocabulary"
        )
    return PartitionData(
        names=ordered_names,
        encoded_names=encoded_names,
        female_proportions=(female_counts / represented_counts)[order],
        represented_record_counts=represented_counts[order],
        partitions=permitted_partitions,
    )


def _resolve_device(requested_device: str) -> str:
    """Resolve an explicit or automatic torch device."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(device: str) -> CharacterBiLSTM:
    """Build the frozen Naampy BiLSTM architecture."""
    return CharacterBiLSTM(
        CHARACTER_VOCABULARY_SIZE,
        1,
        LSTM_EMBEDDING_DIMENSION,
        LSTM_HIDDEN_DIMENSION,
        LSTM_LAYER_COUNT,
        LSTM_DROPOUT_PROBABILITY,
    ).to(device)


def _architecture_manifest() -> dict[str, Any]:
    """Return the exact architecture contract."""
    return {
        "type": "character_bidirectional_lstm",
        "vocabulary_size": CHARACTER_VOCABULARY_SIZE,
        "embedding_dimension": LSTM_EMBEDDING_DIMENSION,
        "hidden_dimension_per_direction": LSTM_HIDDEN_DIMENSION,
        "layers": LSTM_LAYER_COUNT,
        "dropout_between_lstm_layers": LSTM_DROPOUT_PROBABILITY,
        "output_logits": 1,
    }


def _optimizer_manifest(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Return the frozen Adam configuration."""
    return {
        "type": "torch.optim.Adam",
        "learning_rate": optimizer.defaults["lr"],
        "betas": list(optimizer.defaults["betas"]),
        "epsilon": optimizer.defaults["eps"],
        "weight_decay": optimizer.defaults["weight_decay"],
        "amsgrad": optimizer.defaults["amsgrad"],
    }


def _train_epoch(
    model: CharacterBiLSTM,
    optimizer: torch.optim.Optimizer,
    data: PartitionData,
    random_generator: random.Random,
    *,
    samples_per_epoch: int,
    batch_size: int,
    device: str,
) -> float:
    """Train one record-count-weighted sampled epoch."""
    model.train()
    sampled_positions = random_generator.choices(
        range(len(data.names)),
        weights=data.represented_record_counts.tolist(),
        k=samples_per_epoch,
    )
    running_loss = 0.0
    for start in range(0, len(sampled_positions), batch_size):
        positions = sampled_positions[start : start + batch_size]
        inputs, lengths = pad_encoded_names(
            [data.encoded_names[position] for position in positions]
        )
        targets = torch.as_tensor(
            data.female_proportions[positions, None],
            dtype=torch.float32,
            device=device,
        )
        logits = model(inputs.to(device), lengths)
        loss = torch_functional.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach().cpu()) * len(positions)
    return running_loss / len(sampled_positions)


def _cpu_state_dict(model: CharacterBiLSTM) -> dict[str, torch.Tensor]:
    """Copy a model state to contiguous CPU tensors."""
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _train_development_model(
    training_data: PartitionData,
    validation_data: PartitionData,
    configuration: TrainingConfiguration,
    training_seed: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], np.ndarray]:
    """Select one checkpoint using validation record-weighted log loss."""
    random_generator = random.Random(training_seed)  # noqa: S311
    torch.manual_seed(training_seed)
    model = _build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, configuration.epochs + 1):
        training_loss = _train_epoch(
            model,
            optimizer,
            training_data,
            random_generator,
            samples_per_epoch=configuration.samples_per_epoch,
            batch_size=configuration.batch_size,
            device=device,
        )
        validation_probabilities = predict_probabilities(
            model, validation_data.encoded_names, device
        )
        validation_metrics = probability_metrics(
            validation_probabilities,
            validation_data.female_proportions,
            validation_data.represented_record_counts,
        )
        validation_loss = validation_metrics.expected_binary_log_loss
        history.append(
            {
                "epoch": epoch,
                "sampled_training_expected_binary_log_loss": training_loss,
                "validation_record_weighted_expected_binary_log_loss": (
                    validation_loss
                ),
            }
        )
        selected = validation_loss < best_loss
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
        LOGGER.info(
            "Development seed %s epoch %s/%s: sampled training log loss %.6f; "
            "validation record-weighted log loss %.6f%s",
            training_seed,
            epoch,
            configuration.epochs,
            training_loss,
            validation_loss,
            "; new best" if selected else "",
        )
    if best_state is None:
        raise RuntimeError("development training did not select a checkpoint")
    model.load_state_dict(best_state)
    selected_probabilities = predict_probabilities(
        model, validation_data.encoded_names, device
    )
    return (
        best_state,
        {
            "training_seed": training_seed,
            "selected_epoch": best_epoch,
            "selection_metric": ("validation_record_weighted_expected_binary_log_loss"),
            "selection_value": best_loss,
            "selection_history": history,
            "selected_validation_metrics": report_metrics(
                selected_probabilities,
                validation_data.female_proportions,
                validation_data.represented_record_counts,
            ),
            "optimizer": _optimizer_manifest(optimizer),
        },
        selected_probabilities,
    )


def _train_fixed_epoch_model(
    data: PartitionData,
    configuration: TrainingConfiguration,
    training_seed: int,
    epochs: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]], dict[str, Any]]:
    """Fit one final constituent for a precommitted epoch count."""
    random_generator = random.Random(training_seed)  # noqa: S311
    torch.manual_seed(training_seed)
    model = _build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        training_loss = _train_epoch(
            model,
            optimizer,
            data,
            random_generator,
            samples_per_epoch=configuration.samples_per_epoch,
            batch_size=configuration.batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "sampled_training_expected_binary_log_loss": training_loss,
            }
        )
        LOGGER.info(
            "Final fit seed %s epoch %s/%s: sampled training log loss %.6f",
            training_seed,
            epoch,
            epochs,
            training_loss,
        )
    return _cpu_state_dict(model), history, _optimizer_manifest(optimizer)


def _save_state_dict(
    state_dict: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]
) -> str:
    """Create a SafeTensors checkpoint and return its digest."""
    content = serialize_safetensors(state_dict, metadata=metadata)
    _write_new_bytes(path, content)
    loaded_state = load_safetensors(path, device="cpu")
    verification_model = _build_model("cpu")
    verification_model.load_state_dict(loaded_state, strict=True)
    return file_sha256(path)


def _load_verified_model(path: Path, expected_sha256: str) -> CharacterBiLSTM:
    """Hash-verify and load one SafeTensors checkpoint."""
    _verify_file_hash(path, expected_sha256, "SafeTensors checkpoint")
    model = _build_model("cpu")
    model.load_state_dict(load_safetensors(path, device="cpu"), strict=True)
    model.eval()
    return model


def _data_reference(
    parquet_path: Path,
    data_manifest_path: Path,
    data_manifest_sha256: str,
    data_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return immutable training-data references for a stage manifest."""
    return {
        "parquet_filename": parquet_path.name,
        "parquet_sha256": data_manifest["output"]["sha256"],
        "manifest_filename": data_manifest_path.name,
        "manifest_sha256": data_manifest_sha256,
        "split_seed": data_manifest["split"]["seed"],
        "exported_membership_sha256": data_manifest["split"][
            "exported_membership_sha256"
        ],
    }


def _final_fit_source_code_hashes() -> dict[str, str]:
    """Return hashes of code that defines final fitting and runtime inference."""
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = (
        Path(__file__).resolve(),
        repository_root / "model_training" / "evaluation.py",
        repository_root / "src" / "naampy" / "nnets.py",
        repository_root / "src" / "naampy" / "_model_bundle.py",
    )
    return {
        str(source_path.relative_to(repository_root)): file_sha256(source_path)
        for source_path in source_paths
    }


def run_development_stage(
    *,
    parquet_path: Path,
    data_manifest_path: Path,
    data_manifest_sha256: str,
    output_directory: Path,
    configuration: TrainingConfiguration,
    requested_device: str,
) -> Path:
    """Train candidates on training and select them on validation only."""
    configuration.validate()
    expected_manifest_hash = _validated_expected_sha256(
        data_manifest_sha256, "data manifest SHA-256"
    )
    data_manifest = _verify_training_artifact(
        parquet_path, data_manifest_path, expected_manifest_hash
    )
    training_data = load_partition_data(parquet_path, data_manifest, ("training",))
    validation_data = load_partition_data(parquet_path, data_manifest, ("validation",))
    device = _resolve_device(requested_device)
    manifest_path = output_directory / "development_manifest.json"
    model_paths = [
        output_directory / f"development_seed_{seed}.safetensors"
        for seed in configuration.training_seeds
    ]
    for path in [manifest_path, *model_paths]:
        if path.exists():
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")

    model_reports: list[dict[str, Any]] = []
    selected_probabilities: list[np.ndarray] = []
    for training_seed, model_path in zip(
        configuration.training_seeds, model_paths, strict=True
    ):
        state_dict, model_report, probabilities = _train_development_model(
            training_data,
            validation_data,
            configuration,
            training_seed,
            device,
        )
        model_sha256 = _save_state_dict(
            state_dict,
            model_path,
            {
                "artifact_role": "naampy_development_checkpoint",
                "training_seed": str(training_seed),
            },
        )
        model_report.update({"filename": model_path.name, "sha256": model_sha256})
        model_reports.append(model_report)
        selected_probabilities.append(probabilities)
    ensemble_probabilities = np.mean(np.stack(selected_probabilities), axis=0)
    manifest = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage": "development",
        "data": _data_reference(
            parquet_path,
            data_manifest_path,
            expected_manifest_hash,
            data_manifest,
        ),
        "partition_access": {
            "loaded": list(DEVELOPMENT_PARTITIONS),
            "reserved": ["calibration", "test"],
        },
        "training_configuration": configuration.as_manifest(),
        "architecture": _architecture_manifest(),
        "models": model_reports,
        "candidate": {
            "type": "equal_probability_average",
            "constituent_training_seeds": list(configuration.training_seeds),
            "validation_metrics": report_metrics(
                ensemble_probabilities,
                validation_data.female_proportions,
                validation_data.represented_record_counts,
            ),
        },
        "device": {"requested": requested_device, "resolved": device},
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "Development selection uses validation only.",
            "Calibration and test were not loaded, transformed, scored, or summarized.",
            "Canonical name ordering differs from the historical development benchmark sampling path.",
        ],
    }
    write_json_atomic(manifest, manifest_path)
    return manifest_path


def _verify_development_manifest(
    manifest_path: Path,
    expected_sha256: str,
    data_reference: dict[str, Any],
) -> dict[str, Any]:
    """Verify the frozen development decision before final fitting."""
    manifest = _load_frozen_json(manifest_path, expected_sha256, "development manifest")
    if manifest.get("schema_version") != STAGE_SCHEMA_VERSION:
        raise ValueError("development manifest schema version is not recognized")
    if manifest.get("stage") != "development":
        raise ValueError("expected a development-stage manifest")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "stage",
            "data",
            "partition_access",
            "training_configuration",
            "architecture",
            "models",
            "candidate",
            "device",
            "software_versions",
            "limitations",
        },
        "development manifest",
    )
    if manifest.get("data") != data_reference:
        raise ValueError("development manifest training data does not match")
    if manifest.get("architecture") != _architecture_manifest():
        raise ValueError("development architecture is not recognized")
    if manifest.get("partition_access") != {
        "loaded": ["training", "validation"],
        "reserved": ["calibration", "test"],
    }:
        raise ValueError("development partition-access declaration is invalid")
    configuration = TrainingConfiguration.from_manifest(
        cast("dict[str, Any]", manifest["training_configuration"])
    )
    if configuration.training_seeds != STANDARD_TRAINING_SEEDS:
        raise ValueError("development decision must use exactly training seeds 0 and 1")
    model_reports = manifest.get("models")
    if not isinstance(model_reports, list) or len(model_reports) != 2:
        raise ValueError("development manifest must contain exactly two model reports")
    model_filenames: set[str] = set()
    model_hashes: set[str] = set()
    model_seeds: set[int] = set()
    for model_report_value in model_reports:
        if not isinstance(model_report_value, dict):
            raise ValueError("each development model report must be an object")
        model_report = cast("dict[str, Any]", model_report_value)
        _require_exact_keys(
            model_report,
            {
                "training_seed",
                "selected_epoch",
                "selection_metric",
                "selection_value",
                "selection_history",
                "selected_validation_metrics",
                "optimizer",
                "filename",
                "sha256",
            },
            "development model report",
        )
        filename, digest = _require_artifact_filename_and_hash(
            model_report, "development checkpoint"
        )
        training_seed = model_report.get("training_seed")
        selected_epoch = model_report.get("selected_epoch")
        if isinstance(training_seed, bool) or not isinstance(training_seed, int):
            raise ValueError("development training_seed must be an integer")
        if (
            isinstance(selected_epoch, bool)
            or not isinstance(selected_epoch, int)
            or not 1 <= selected_epoch <= configuration.epochs
        ):
            raise ValueError(
                "development selected_epoch must be within the requested epoch budget"
            )
        if model_report.get("selection_metric") != (
            "validation_record_weighted_expected_binary_log_loss"
        ):
            raise ValueError("development selection metric is not recognized")
        model_filenames.add(filename)
        model_hashes.add(digest)
        model_seeds.add(training_seed)
    if len(model_filenames) != 2 or len(model_hashes) != 2:
        raise ValueError("development checkpoint filenames and hashes must be unique")
    if model_seeds != set(STANDARD_TRAINING_SEEDS):
        raise ValueError("development model reports must use exactly seeds 0 and 1")
    candidate = manifest.get("candidate", {})
    if not isinstance(candidate, dict):
        raise ValueError("development candidate must be an object")
    _require_exact_keys(
        cast("dict[str, Any]", candidate),
        {"type", "constituent_training_seeds", "validation_metrics"},
        "development candidate",
    )
    if candidate.get("type") != "equal_probability_average":
        raise ValueError("development candidate must be a probability-average ensemble")
    if candidate.get("constituent_training_seeds") != list(STANDARD_TRAINING_SEEDS):
        raise ValueError("development candidate must combine seeds 0 and 1")
    return manifest


def run_final_fit_stage(
    *,
    parquet_path: Path,
    data_manifest_path: Path,
    data_manifest_sha256: str,
    development_manifest_path: Path,
    development_manifest_sha256: str,
    output_directory: Path,
    requested_device: str,
) -> Path:
    """Fit the frozen ensemble constituents on training plus validation."""
    if requested_device != "cpu":
        raise ValueError("final ensemble artifacts must be trained on CPU")
    expected_data_manifest_hash = _validated_expected_sha256(
        data_manifest_sha256, "data manifest SHA-256"
    )
    expected_development_hash = _validated_expected_sha256(
        development_manifest_sha256, "development manifest SHA-256"
    )
    data_manifest = _verify_training_artifact(
        parquet_path, data_manifest_path, expected_data_manifest_hash
    )
    data_reference = _data_reference(
        parquet_path,
        data_manifest_path,
        expected_data_manifest_hash,
        data_manifest,
    )
    development_manifest = _verify_development_manifest(
        development_manifest_path, expected_development_hash, data_reference
    )
    configuration = TrainingConfiguration.from_manifest(
        development_manifest["training_configuration"]
    )
    model_reports = cast("list[dict[str, Any]]", development_manifest["models"])
    selected_epochs = {
        int(model_report["training_seed"]): int(model_report["selected_epoch"])
        for model_report in model_reports
    }
    if set(selected_epochs) != set(configuration.training_seeds):
        raise ValueError("development model seeds do not match training configuration")
    final_data = load_partition_data(parquet_path, data_manifest, FINAL_FIT_PARTITIONS)
    device = _resolve_device(requested_device)
    ensemble_manifest_path = output_directory / "ensemble_manifest.json"
    checkpoint_paths = [
        output_directory / f"final_seed_{seed}.safetensors"
        for seed in configuration.training_seeds
    ]
    for path in [ensemble_manifest_path, *checkpoint_paths]:
        if path.exists():
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    final_models = []
    for training_seed, checkpoint_path in zip(
        configuration.training_seeds, checkpoint_paths, strict=True
    ):
        state_dict, history, optimizer = _train_fixed_epoch_model(
            final_data,
            configuration,
            training_seed,
            selected_epochs[training_seed],
            device,
        )
        checkpoint_hash = _save_state_dict(
            state_dict,
            checkpoint_path,
            {
                "artifact_role": "naampy_final_ensemble_constituent",
                "training_seed": str(training_seed),
                "epochs": str(selected_epochs[training_seed]),
            },
        )
        final_models.append(
            {
                "filename": checkpoint_path.name,
                "sha256": checkpoint_hash,
                "training_seed": training_seed,
                "epochs": selected_epochs[training_seed],
                "training_history": history,
                "optimizer": optimizer,
            }
        )
    manifest = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage": "final_ensemble",
        "data": data_reference,
        "partition_access": {
            "loaded": list(FINAL_FIT_PARTITIONS),
            "reserved": ["calibration", "test"],
        },
        "development_manifest": {
            "filename": development_manifest_path.name,
            "sha256": expected_development_hash,
        },
        "training_configuration": configuration.as_manifest(),
        "final_fit": {
            "partitions": list(FINAL_FIT_PARTITIONS),
            "canonical_input_order": "normalized_name ascending",
            "development_sampling_path_equivalence": False,
        },
        "architecture": _architecture_manifest(),
        "source_code_sha256": _final_fit_source_code_hashes(),
        "ensemble": {
            "type": "equal_probability_average",
            "models": final_models,
        },
        "device": {"requested": requested_device, "resolved": device},
        "limitations": [
            "Calibration and test were not loaded, transformed, scored, or summarized.",
            "The final fit is a precommitted refit, not a bitwise reproduction of development sampling.",
        ],
    }
    write_json_atomic(manifest, ensemble_manifest_path)
    return ensemble_manifest_path


def _verified_ensemble_models(
    manifest_path: Path,
    expected_sha256: str,
    data_reference: dict[str, Any],
) -> tuple[dict[str, Any], list[CharacterBiLSTM]]:
    """Verify the frozen ensemble manifest and every constituent checkpoint."""
    manifest = _load_frozen_json(manifest_path, expected_sha256, "ensemble manifest")
    if manifest.get("schema_version") != STAGE_SCHEMA_VERSION:
        raise ValueError("ensemble manifest schema version is not recognized")
    if manifest.get("stage") != "final_ensemble":
        raise ValueError("expected a final-ensemble manifest")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "stage",
            "data",
            "partition_access",
            "development_manifest",
            "training_configuration",
            "final_fit",
            "architecture",
            "source_code_sha256",
            "ensemble",
            "device",
            "limitations",
        },
        "ensemble manifest",
    )
    if manifest.get("data") != data_reference:
        raise ValueError("ensemble manifest training data does not match")
    if manifest.get("architecture") != _architecture_manifest():
        raise ValueError("ensemble architecture is not recognized")
    if manifest.get("source_code_sha256") != _final_fit_source_code_hashes():
        raise ValueError("final-fit source-code hashes do not match the current code")
    if manifest.get("partition_access") != {
        "loaded": ["training", "validation"],
        "reserved": ["calibration", "test"],
    }:
        raise ValueError("final-fit partition-access declaration is invalid")
    if manifest.get("device") != {"requested": "cpu", "resolved": "cpu"}:
        raise ValueError("final ensemble must be trained on CPU")
    configuration = TrainingConfiguration.from_manifest(
        cast("dict[str, Any]", manifest["training_configuration"])
    )
    if configuration.training_seeds != STANDARD_TRAINING_SEEDS:
        raise ValueError("final ensemble must use exactly training seeds 0 and 1")
    ensemble = manifest.get("ensemble", {})
    if not isinstance(ensemble, dict):
        raise ValueError("ensemble must be an object")
    _require_exact_keys(
        cast("dict[str, Any]", ensemble), {"type", "models"}, "ensemble"
    )
    if ensemble.get("type") != "equal_probability_average":
        raise ValueError("ensemble combination rule is not recognized")
    model_documents_value = ensemble.get("models")
    if not isinstance(model_documents_value, list) or len(model_documents_value) != 2:
        raise ValueError("final ensemble must contain exactly two models")
    model_documents = cast("list[dict[str, Any]]", model_documents_value)
    models = []
    filenames: set[str] = set()
    digests: set[str] = set()
    seeds: list[int] = []
    for model_document in model_documents:
        if not isinstance(model_document, dict):
            raise ValueError("each final ensemble model must be an object")
        _require_exact_keys(
            model_document,
            {
                "filename",
                "sha256",
                "training_seed",
                "epochs",
                "training_history",
                "optimizer",
            },
            "final ensemble model",
        )
        filename, digest = _require_artifact_filename_and_hash(
            model_document, "final ensemble checkpoint"
        )
        training_seed = model_document.get("training_seed")
        epochs = model_document.get("epochs")
        if isinstance(training_seed, bool) or not isinstance(training_seed, int):
            raise ValueError("final ensemble training_seed must be an integer")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("final ensemble epochs must be a positive integer")
        filenames.add(filename)
        digests.add(digest)
        seeds.append(training_seed)
        models.append(
            _load_verified_model(
                manifest_path.parent / filename,
                digest,
            )
        )
    if len(filenames) != 2 or len(digests) != 2:
        raise ValueError("final ensemble filenames and hashes must be distinct")
    if tuple(seeds) != STANDARD_TRAINING_SEEDS:
        raise ValueError("final ensemble models must use exactly seeds 0 and 1")
    return manifest, models


def _ensemble_probabilities(
    models: list[CharacterBiLSTM], encoded_names: list[list[int]], device: str
) -> np.ndarray:
    """Average raw probabilities across frozen ensemble constituents."""
    return np.mean(
        np.stack(
            [
                predict_probabilities(model.to(device), encoded_names, device)
                for model in models
            ]
        ),
        axis=0,
    )


def _required_human_readable_text(
    document: dict[str, Any], key: str, description: str
) -> str:
    """Return explicit nonempty human-readable metadata."""
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must provide nonempty {key}")
    return value


def _runtime_architecture_manifest() -> dict[str, Any]:
    """Return the exact runtime parser architecture section."""
    return {
        "vocabulary": "abcdefghijklmnopqrstuvwxyz",
        "embedding_dimension": LSTM_EMBEDDING_DIMENSION,
        "hidden_dimension": LSTM_HIDDEN_DIMENSION,
        "layer_count": LSTM_LAYER_COUNT,
        "dropout_probability": LSTM_DROPOUT_PROBABILITY,
    }


def _runtime_manifest_document(
    ensemble_manifest: dict[str, Any],
    data_manifest: dict[str, Any],
    *,
    slope: float,
    intercept: float,
) -> dict[str, Any]:
    """Generate the strict public runtime manifest from verified internal state."""
    reference_population = _required_human_readable_text(
        data_manifest, "reference_population", "training data manifest"
    )
    model_documents = cast(
        "list[dict[str, Any]]", ensemble_manifest["ensemble"]["models"]
    )
    return {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "model_version": FINAL_MODEL_VERSION,
        "score_target": MODEL_SCORE_TARGET,
        "reference_population": reference_population,
        "label_source": MODEL_LABEL_SOURCE,
        "architecture": _runtime_architecture_manifest(),
        "ensemble": {
            "method": ENSEMBLE_METHOD,
            "members": [
                {
                    "filename": model_document["filename"],
                    "sha256": model_document["sha256"],
                    "training_seed": model_document["training_seed"],
                }
                for model_document in model_documents
            ],
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "slope": slope,
            "intercept": intercept,
            "population": CALIBRATION_POPULATION,
        },
    }


def run_calibration_stage(
    *,
    parquet_path: Path,
    data_manifest_path: Path,
    data_manifest_sha256: str,
    ensemble_manifest_path: Path,
    ensemble_manifest_sha256: str,
    calibration_manifest_output_path: Path,
    runtime_manifest_output_path: Path,
    requested_device: str,
) -> Path:
    """Fit one calibrator after verifying the complete frozen ensemble."""
    if requested_device != "cpu":
        raise ValueError("calibration artifacts must be fitted on CPU")
    for output_path in (
        calibration_manifest_output_path,
        runtime_manifest_output_path,
    ):
        if output_path.exists():
            raise FileExistsError(f"refusing to replace frozen artifact: {output_path}")
    expected_data_manifest_hash = _validated_expected_sha256(
        data_manifest_sha256, "data manifest SHA-256"
    )
    expected_ensemble_hash = _validated_expected_sha256(
        ensemble_manifest_sha256, "ensemble manifest SHA-256"
    )
    data_manifest = _verify_training_artifact(
        parquet_path, data_manifest_path, expected_data_manifest_hash
    )
    data_reference = _data_reference(
        parquet_path,
        data_manifest_path,
        expected_data_manifest_hash,
        data_manifest,
    )
    ensemble_manifest, models = _verified_ensemble_models(
        ensemble_manifest_path, expected_ensemble_hash, data_reference
    )
    calibration_data = load_partition_data(
        parquet_path, data_manifest, ("calibration",)
    )
    device = _resolve_device(requested_device)
    raw_probabilities = _ensemble_probabilities(
        models, calibration_data.encoded_names, device
    )
    calibration = fit_logistic_calibration(
        raw_probabilities,
        calibration_data.female_proportions,
        calibration_data.represented_record_counts,
    )
    calibrated_probabilities = calibration.apply(raw_probabilities)
    runtime_manifest = _runtime_manifest_document(
        ensemble_manifest,
        data_manifest,
        slope=calibration.scale,
        intercept=calibration.intercept,
    )
    write_json_atomic(runtime_manifest, runtime_manifest_output_path)
    parse_model_manifest(runtime_manifest_output_path)
    runtime_manifest_hash = file_sha256(runtime_manifest_output_path)
    calibration_manifest = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage": "calibration",
        "data": data_reference,
        "ensemble_manifest": {
            "filename": ensemble_manifest_path.name,
            "sha256": expected_ensemble_hash,
        },
        "partition_access": {"loaded": ["calibration"], "reserved": ["test"]},
        "calibrator": {
            "method": CALIBRATION_METHOD,
            "scale": calibration.scale,
            "intercept": calibration.intercept,
            "input": "equal_probability_average_raw_probability",
            "population": CALIBRATION_POPULATION,
        },
        "calibration_partition_metrics": {
            "raw": report_metrics(
                raw_probabilities,
                calibration_data.female_proportions,
                calibration_data.represented_record_counts,
            ),
            "calibrated": report_metrics(
                calibrated_probabilities,
                calibration_data.female_proportions,
                calibration_data.represented_record_counts,
            ),
        },
        "device": {"requested": requested_device, "resolved": device},
        "runtime_manifest": {
            "filename": runtime_manifest_output_path.name,
            "sha256": runtime_manifest_hash,
        },
        "limitations": ["Test was not loaded, transformed, scored, or summarized."],
    }
    write_json_atomic(calibration_manifest, calibration_manifest_output_path)
    return calibration_manifest_output_path


def _metric_report_with_intervals(
    probabilities: np.ndarray,
    data: PartitionData,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Return point estimates and name-cluster bootstrap intervals."""
    return {
        "point_estimates": report_metrics(
            probabilities,
            data.female_proportions,
            data.represented_record_counts,
        ),
        "intervals": {
            "method": "name-cluster percentile bootstrap",
            "confidence_level": 0.95,
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "name_weighted": bootstrap_metric_intervals(
                probabilities,
                data.female_proportions,
                np.ones_like(data.represented_record_counts, dtype=np.float64),
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
            "record_weighted": bootstrap_metric_intervals(
                probabilities,
                data.female_proportions,
                data.represented_record_counts,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
        },
    }


def run_test_stage(
    *,
    parquet_path: Path,
    data_manifest_path: Path,
    data_manifest_sha256: str,
    ensemble_manifest_path: Path,
    ensemble_manifest_sha256: str,
    calibration_manifest_path: Path,
    calibration_manifest_sha256: str,
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
    output_path: Path,
    requested_device: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Path:
    """Score test once after every prior artifact passes hash verification."""
    if requested_device != "cpu":
        raise ValueError("final test evaluation must run on CPU")
    if output_path.exists():
        raise FileExistsError(f"refusing to replace frozen artifact: {output_path}")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    expected_data_manifest_hash = _validated_expected_sha256(
        data_manifest_sha256, "data manifest SHA-256"
    )
    expected_ensemble_hash = _validated_expected_sha256(
        ensemble_manifest_sha256, "ensemble manifest SHA-256"
    )
    expected_calibration_hash = _validated_expected_sha256(
        calibration_manifest_sha256, "calibration manifest SHA-256"
    )
    expected_runtime_hash = _validated_expected_sha256(
        runtime_manifest_sha256, "runtime manifest SHA-256"
    )
    data_manifest = _verify_training_artifact(
        parquet_path, data_manifest_path, expected_data_manifest_hash
    )
    data_reference = _data_reference(
        parquet_path,
        data_manifest_path,
        expected_data_manifest_hash,
        data_manifest,
    )
    ensemble_manifest, models = _verified_ensemble_models(
        ensemble_manifest_path, expected_ensemble_hash, data_reference
    )
    calibration_manifest = _load_frozen_json(
        calibration_manifest_path,
        expected_calibration_hash,
        "calibration manifest",
    )
    if calibration_manifest.get("schema_version") != STAGE_SCHEMA_VERSION:
        raise ValueError("calibration manifest schema version is not recognized")
    if calibration_manifest.get("stage") != "calibration":
        raise ValueError("expected a calibration-stage manifest")
    _require_exact_keys(
        calibration_manifest,
        {
            "schema_version",
            "stage",
            "data",
            "ensemble_manifest",
            "partition_access",
            "calibrator",
            "calibration_partition_metrics",
            "device",
            "runtime_manifest",
            "limitations",
        },
        "calibration manifest",
    )
    if calibration_manifest.get("data") != data_reference:
        raise ValueError("calibration manifest training data does not match")
    if calibration_manifest.get("ensemble_manifest") != {
        "filename": ensemble_manifest_path.name,
        "sha256": expected_ensemble_hash,
    }:
        raise ValueError("calibration manifest ensemble reference does not match")
    if calibration_manifest.get("partition_access") != {
        "loaded": ["calibration"],
        "reserved": ["test"],
    }:
        raise ValueError("calibration partition-access declaration is invalid")
    if calibration_manifest.get("device") != {
        "requested": "cpu",
        "resolved": "cpu",
    }:
        raise ValueError("calibration artifact must be fitted on CPU")
    if calibration_manifest.get("runtime_manifest") != {
        "filename": runtime_manifest_path.name,
        "sha256": expected_runtime_hash,
    }:
        raise ValueError("calibration runtime-manifest reference does not match")
    calibrator_value = calibration_manifest.get("calibrator")
    if not isinstance(calibrator_value, dict):
        raise ValueError("calibrator must be an object")
    calibrator = cast("dict[str, Any]", calibrator_value)
    _require_exact_keys(
        calibrator,
        {"method", "scale", "intercept", "input", "population"},
        "calibrator",
    )
    if calibrator.get("method") != CALIBRATION_METHOD:
        raise ValueError("calibration method is not recognized")
    if calibrator.get("input") != "equal_probability_average_raw_probability":
        raise ValueError("calibration input is not recognized")
    if calibrator.get("population") != CALIBRATION_POPULATION:
        raise ValueError("calibration population is not recognized")
    scale = float(calibrator["scale"])
    intercept = float(calibrator["intercept"])
    if not np.isfinite([scale, intercept]).all() or scale <= 0:
        raise ValueError("calibration parameters are invalid")

    _verify_file_hash(runtime_manifest_path, expected_runtime_hash, "runtime manifest")
    runtime_manifest = parse_model_manifest(runtime_manifest_path)
    if runtime_manifest.model_version != FINAL_MODEL_VERSION:
        raise ValueError("runtime model version is not recognized")
    if runtime_manifest.score_target != MODEL_SCORE_TARGET:
        raise ValueError("runtime score target does not match the frozen target")
    if runtime_manifest.reference_population != _required_human_readable_text(
        data_manifest, "reference_population", "training data manifest"
    ):
        raise ValueError("runtime reference population does not match training data")
    if runtime_manifest.label_source != MODEL_LABEL_SOURCE:
        raise ValueError("runtime label source does not match the frozen source")
    if (
        runtime_manifest.calibration.method != CALIBRATION_METHOD
        or not np.isclose(
            [
                runtime_manifest.calibration.slope,
                runtime_manifest.calibration.intercept,
            ],
            [scale, intercept],
            rtol=0,
            atol=0,
        ).all()
    ):
        raise ValueError("runtime calibration does not match calibration manifest")
    if runtime_manifest.calibration.population != CALIBRATION_POPULATION:
        raise ValueError("runtime calibration population is not recognized")
    internal_model_documents = cast(
        "list[dict[str, Any]]", ensemble_manifest["ensemble"]["models"]
    )
    expected_runtime_members = [
        (
            model_document["filename"],
            model_document["sha256"],
            model_document["training_seed"],
        )
        for model_document in internal_model_documents
    ]
    actual_runtime_members = [
        (member.filename, member.sha256, member.training_seed)
        for member in runtime_manifest.ensemble_members
    ]
    if actual_runtime_members != expected_runtime_members:
        raise ValueError("runtime ensemble members do not match frozen final ensemble")

    test_data = load_partition_data(parquet_path, data_manifest, ("test",))
    device = _resolve_device(requested_device)
    raw_probabilities = _ensemble_probabilities(models, test_data.encoded_names, device)
    clipped = np.clip(raw_probabilities, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    calibrated_probabilities = 1 / (1 + np.exp(-(scale * raw_logits + intercept)))
    report = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage": "test_evaluation",
        "data": data_reference,
        "ensemble_manifest": {
            "filename": ensemble_manifest_path.name,
            "sha256": expected_ensemble_hash,
        },
        "calibration_manifest": {
            "filename": calibration_manifest_path.name,
            "sha256": expected_calibration_hash,
        },
        "runtime_manifest": {
            "filename": runtime_manifest_path.name,
            "sha256": expected_runtime_hash,
        },
        "partition_access": {"loaded": ["test"]},
        "metrics": {
            "raw": _metric_report_with_intervals(
                raw_probabilities,
                test_data,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            ),
            "calibrated": _metric_report_with_intervals(
                calibrated_probabilities,
                test_data,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            ),
        },
        "device": {"requested": requested_device, "resolved": device},
        "limitations": [
            "Outputs are aggregate electoral-roll label-pattern estimates, not individual identity labels.",
            "The test report is valid only for the exact frozen artifact hashes recorded here.",
        ],
    }
    write_json_atomic(report, output_path)
    return output_path


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add immutable typed-data arguments shared by every stage."""
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--data-manifest-sha256", required=True)


def _parse_arguments() -> argparse.Namespace:
    """Parse one explicit workflow stage."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    development = subparsers.add_parser("development")
    _add_data_arguments(development)
    development.add_argument("--output-directory", required=True, type=Path)
    development.add_argument(
        "--training-seeds", nargs="+", type=int, default=list(STANDARD_TRAINING_SEEDS)
    )
    development.add_argument("--epochs", type=int, default=12)
    development.add_argument("--samples-per-epoch", type=int, default=300_000)
    development.add_argument("--batch-size", type=int, default=256)
    development.add_argument("--learning-rate", type=float, default=1e-3)
    development.add_argument("--device", default="auto")

    final_fit = subparsers.add_parser("fit-final")
    _add_data_arguments(final_fit)
    final_fit.add_argument("--development-manifest", required=True, type=Path)
    final_fit.add_argument("--development-manifest-sha256", required=True)
    final_fit.add_argument("--output-directory", required=True, type=Path)
    final_fit.add_argument("--device", default="cpu")

    calibration = subparsers.add_parser("calibrate")
    _add_data_arguments(calibration)
    calibration.add_argument("--ensemble-manifest", required=True, type=Path)
    calibration.add_argument("--ensemble-manifest-sha256", required=True)
    calibration.add_argument("--calibration-manifest-output", required=True, type=Path)
    calibration.add_argument("--runtime-manifest-output", required=True, type=Path)
    calibration.add_argument("--device", default="cpu")

    test = subparsers.add_parser("score-test")
    _add_data_arguments(test)
    test.add_argument("--ensemble-manifest", required=True, type=Path)
    test.add_argument("--ensemble-manifest-sha256", required=True)
    test.add_argument("--calibration-manifest", required=True, type=Path)
    test.add_argument("--calibration-manifest-sha256", required=True)
    test.add_argument("--runtime-manifest", required=True, type=Path)
    test.add_argument("--runtime-manifest-sha256", required=True)
    test.add_argument("--output", required=True, type=Path)
    test.add_argument("--device", default="cpu")
    test.add_argument("--bootstrap-iterations", type=int, default=1_000)
    test.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run exactly one gated training, calibration, or evaluation stage."""
    arguments = _parse_arguments()
    common = {
        "parquet_path": arguments.data,
        "data_manifest_path": arguments.data_manifest,
        "data_manifest_sha256": arguments.data_manifest_sha256,
    }
    if arguments.stage == "development":
        configuration = TrainingConfiguration(
            training_seeds=tuple(arguments.training_seeds),
            epochs=arguments.epochs,
            samples_per_epoch=arguments.samples_per_epoch,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
        )
        run_development_stage(
            **common,
            output_directory=arguments.output_directory,
            configuration=configuration,
            requested_device=arguments.device,
        )
    elif arguments.stage == "fit-final":
        run_final_fit_stage(
            **common,
            development_manifest_path=arguments.development_manifest,
            development_manifest_sha256=arguments.development_manifest_sha256,
            output_directory=arguments.output_directory,
            requested_device=arguments.device,
        )
    elif arguments.stage == "calibrate":
        run_calibration_stage(
            **common,
            ensemble_manifest_path=arguments.ensemble_manifest,
            ensemble_manifest_sha256=arguments.ensemble_manifest_sha256,
            calibration_manifest_output_path=(arguments.calibration_manifest_output),
            runtime_manifest_output_path=arguments.runtime_manifest_output,
            requested_device=arguments.device,
        )
    else:
        run_test_stage(
            **common,
            ensemble_manifest_path=arguments.ensemble_manifest,
            ensemble_manifest_sha256=arguments.ensemble_manifest_sha256,
            calibration_manifest_path=arguments.calibration_manifest,
            calibration_manifest_sha256=arguments.calibration_manifest_sha256,
            runtime_manifest_path=arguments.runtime_manifest,
            runtime_manifest_sha256=arguments.runtime_manifest_sha256,
            output_path=arguments.output,
            requested_device=arguments.device,
            bootstrap_iterations=arguments.bootstrap_iterations,
            bootstrap_seed=arguments.bootstrap_seed,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
