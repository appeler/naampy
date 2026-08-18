"""Evaluate the shipped gender-pattern checkpoint on its held-out names.

Run from the repository root:

    python -m model_training.evaluate_gender_model \
        --data model_training/data/naampy_v3.csv.gz \
        --output model_training/reports/gender_lstm_f7f2b7ac.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from model_training.evaluation import (
    NameDatasetSplit,
    balanced_calibration_test_split,
    bootstrap_metric_intervals,
    fit_logistic_calibration,
    legacy_development_split,
    name_membership_sha256,
    partition_summary,
    report_metrics,
    split_summary,
    stratified_name_split,
)
from model_training.train_gender_lstm import load_names
from naampy._resources import HF_REPO, HF_REVISION, MODEL_DIR_ENV, resolve_model
from naampy.nnets import (
    LSTM_DROPOUT,
    LSTM_EMB,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    VOCAB_SIZE,
    CharBiLSTM,
    pad_encoded,
)

LOGGER = logging.getLogger(__name__)
SHIPPED_CHECKPOINT_SPLIT_SEED = 0
SHIPPED_CHECKPOINT_DATA_SHA256 = (
    "a548226d9fe4c1dd7193d79487f51e55d77ad58e327b1b58e966b910e484ccf4"
)
SHIPPED_GENDER_MODEL_SHA256 = (
    "98fdcfe9016b48e3a2639c6f2c55eb9d2a56946a51dea6ac0a0c64fafc33b9f7"
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shipped_evaluation_artifacts(data_path: Path, model_path: Path) -> None:
    """Verify the exact data and checkpoint used by the shipped-model audit."""
    if file_sha256(data_path) != SHIPPED_CHECKPOINT_DATA_SHA256:
        raise ValueError(
            "--data does not match the immutable input used to train the shipped "
            "checkpoint; evaluate a custom data/model pair with --model and "
            "--training-manifest"
        )
    if file_sha256(model_path) != SHIPPED_GENDER_MODEL_SHA256:
        raise ValueError(
            "the resolved gender_lstm.pt does not match the checkpoint pinned by "
            "Naampy; evaluate a custom checkpoint with --model and "
            "--training-manifest"
        )


@torch.no_grad()
def predict_probabilities(
    model: CharBiLSTM, encoded_names: list[list[int]]
) -> np.ndarray:
    """Return female probabilities for encoded names."""
    model.eval()
    probabilities: list[float] = []
    for start in range(0, len(encoded_names), 512):
        inputs, lengths = pad_encoded(encoded_names[start : start + 512])
        batch_probabilities = (
            torch.sigmoid(model(inputs, lengths)).squeeze(1).cpu().numpy()
        )
        probabilities.extend(batch_probabilities.tolist())
    return np.asarray(probabilities)


def metric_report_with_intervals(
    probabilities: np.ndarray,
    female_proportions: np.ndarray,
    person_counts: np.ndarray,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Return point estimates and name-cluster bootstrap intervals."""
    return {
        "point_estimates": report_metrics(
            probabilities, female_proportions, person_counts
        ),
        "intervals": {
            "method": "name-cluster percentile bootstrap",
            "confidence_level": 0.95,
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "name_weighted": bootstrap_metric_intervals(
                probabilities,
                female_proportions,
                np.ones_like(person_counts, dtype=np.float64),
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
            "person_weighted": bootstrap_metric_intervals(
                probabilities,
                female_proportions,
                person_counts,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
        },
    }


def write_json_atomic(report: dict[str, Any], output_path: Path) -> None:
    """Write a JSON report without exposing a partial file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(report, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(output_path)
    output_path.chmod(0o644)


def load_verified_training_split(
    manifest_path: Path,
    data_path: Path,
    model_path: Path,
    names: list[str],
    female_proportions: np.ndarray,
    person_counts: np.ndarray,
) -> tuple[NameDatasetSplit, dict[str, Any]]:
    """Load and verify the split provenance for a custom checkpoint."""
    manifest = cast(
        "dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if manifest.get("schema_version") != 2:
        raise ValueError("training manifest must use schema version 2")
    for metadata_field in ("target", "reference_population"):
        metadata_value = manifest.get(metadata_field)
        if not isinstance(metadata_value, str) or not metadata_value.strip():
            raise ValueError(
                f"training manifest {metadata_field} must be a nonempty string"
            )
    if manifest["data"]["sha256"] != file_sha256(data_path):
        raise ValueError("training manifest data hash does not match --data")
    if manifest["model"]["sha256"] != file_sha256(model_path):
        raise ValueError("training manifest model hash does not match --model")
    split_manifest = manifest["split"]
    if split_manifest["method"] != "stratified disjoint unique-name split":
        raise ValueError("unsupported training manifest split method")
    fractions = split_manifest["fractions"]
    declared_fractions = {
        partition: float(fractions[partition])
        for partition in ("training", "validation", "calibration", "test")
    }
    if not math.isclose(
        sum(declared_fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("training manifest split fractions must sum to 1")
    strata = split_manifest["strata"]
    split = stratified_name_split(
        female_proportions,
        person_counts,
        seed=int(split_manifest["seed"]),
        training_fraction=declared_fractions["training"],
        validation_fraction=declared_fractions["validation"],
        calibration_fraction=declared_fractions["calibration"],
        count_strata=int(strata["person_count_rank_bins"]),
        proportion_strata=int(strata["female_proportion_bins"]),
    )
    expected_hashes = split_manifest["membership_sha256"]
    actual_hashes = {
        "training": name_membership_sha256(names, split.training),
        "validation": name_membership_sha256(names, split.validation),
        "calibration": name_membership_sha256(names, split.calibration),
        "test": name_membership_sha256(names, split.test),
    }
    if expected_hashes != actual_hashes:
        raise ValueError("training manifest membership hashes do not match --data")
    return split, manifest


def main() -> None:
    """Evaluate the pinned checkpoint and write its evidence report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    arguments = parser.parse_args()

    max_source_rows: int | None = None
    if arguments.model is not None:
        if arguments.training_manifest is None:
            parser.error("--model requires --training-manifest")
        try:
            manifest_preview = cast(
                "dict[str, Any]",
                json.loads(arguments.training_manifest.read_text(encoding="utf-8")),
            )
            recorded_row_cap = manifest_preview["training"]["hyperparameters"][
                "max_source_rows"
            ]
            if recorded_row_cap is not None:
                if (
                    isinstance(recorded_row_cap, bool)
                    or not isinstance(recorded_row_cap, int)
                    or recorded_row_cap < 1
                ):
                    raise ValueError(
                        "training manifest max_source_rows must be null or a positive integer"
                    )
                max_source_rows = recorded_row_cap
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))

    shipped_model_path: Path | None = None
    if arguments.model is None:
        if arguments.training_manifest is not None:
            parser.error("--training-manifest requires --model")
        shipped_model_path = Path(resolve_model("gender_lstm.pt"))
        try:
            verify_shipped_evaluation_artifacts(arguments.data, shipped_model_path)
        except ValueError as error:
            parser.error(str(error))

    names, encoded_names, female_proportions, person_counts = load_names(
        arguments.data, max_source_rows
    )
    proportions = np.asarray(female_proportions)
    counts = np.asarray(person_counts)
    if arguments.model is None:
        if shipped_model_path is None:
            raise RuntimeError("shipped checkpoint path was not resolved")
        model_path = shipped_model_path
        training_indices, held_out_indices = legacy_development_split(
            len(names), seed=SHIPPED_CHECKPOINT_SPLIT_SEED
        )
        calibration_indices, test_indices = balanced_calibration_test_split(
            held_out_indices, proportions, counts
        )
        local_model_directory = os.environ.get(MODEL_DIR_ENV)
        expected_local_path = (
            Path(local_model_directory) / model_path.name
            if local_model_directory
            else None
        )
        model_provenance = {
            "source": (
                "verified_local_mirror"
                if expected_local_path == model_path
                else "hugging_face_hub"
            ),
            "repository": HF_REPO,
            "revision": HF_REVISION,
        }
        data_artifact_encoding = (
            "legacy pandas gzip output predating the canonical deterministic writer"
        )
        target_description = "female share among female and male electoral-roll labels"
        reference_population = (
            "aggregated Indian electoral-roll registration records represented "
            "in the local naampy v3 construction"
        )
        split_method = (
            "original seeded 80/20 unique-name split; held-out names balanced "
            "into calibration and test halves by support and label composition"
        )
        split_provenance: dict[str, Any] = {
            "membership_provenance": {
                "recipe": "Python random.Random shuffle followed by an 80/20 cutoff",
                "fixed_seed": SHIPPED_CHECKPOINT_SPLIT_SEED,
                "source": "shipped checkpoint training recipe",
            },
            "membership_sha256": {
                "original_training": name_membership_sha256(names, training_indices),
                "original_held_out": name_membership_sha256(names, held_out_indices),
                "calibration": name_membership_sha256(names, calibration_indices),
                "test": name_membership_sha256(names, test_indices),
            },
            "summary": {
                "original_training": partition_summary(
                    training_indices, proportions, counts
                ),
                "calibration": partition_summary(
                    calibration_indices, proportions, counts
                ),
                "test": partition_summary(test_indices, proportions, counts),
            },
        }
        original_held_out_probabilities_required = True
    else:
        if arguments.training_manifest is None:
            parser.error("--model requires --training-manifest")
        training_manifest_path = arguments.training_manifest
        model_path = arguments.model
        try:
            custom_split, training_manifest = load_verified_training_split(
                training_manifest_path,
                arguments.data,
                model_path,
                names,
                proportions,
                counts,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        training_indices = custom_split.training
        held_out_indices = np.asarray([], dtype=np.int64)
        calibration_indices = custom_split.calibration
        test_indices = custom_split.test
        model_provenance = {
            "source": "user_provided_local_path",
            "path": str(model_path),
            "training_manifest": {
                "filename": training_manifest_path.name,
                "sha256": file_sha256(training_manifest_path),
            },
        }
        data_artifact_encoding = training_manifest["data"].get(
            "artifact_encoding", "unspecified"
        )
        target_description = training_manifest["target"]
        reference_population = training_manifest["reference_population"]
        split_method = training_manifest["split"]["method"]
        split_provenance = {
            "seed": training_manifest["split"]["seed"],
            "fractions": training_manifest["split"]["fractions"],
            "strata": training_manifest["split"]["strata"],
            "membership_sha256": training_manifest["split"]["membership_sha256"],
            "summary": split_summary(custom_split, proportions, counts),
        }
        original_held_out_probabilities_required = False
    model = CharBiLSTM(VOCAB_SIZE, 1, LSTM_EMB, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    held_out_probabilities = (
        predict_probabilities(
            model, [encoded_names[index] for index in held_out_indices]
        )
        if original_held_out_probabilities_required
        else None
    )
    calibration_probabilities = predict_probabilities(
        model, [encoded_names[index] for index in calibration_indices]
    )
    test_probabilities = predict_probabilities(
        model, [encoded_names[index] for index in test_indices]
    )
    calibration = fit_logistic_calibration(
        calibration_probabilities,
        proportions[calibration_indices],
        counts[calibration_indices],
    )
    calibrated_test_probabilities = calibration.apply(test_probabilities)
    limitations = [
        (
            "The binary model excludes the extremely sparse third-gender label "
            "and does not estimate gender identity."
        ),
        (
            "The calibration in this report is an offline evaluation transform; "
            "the current Naampy runtime serves the raw checkpoint probabilities."
        ),
    ]
    if original_held_out_probabilities_required:
        limitations.insert(
            0,
            (
                "The checkpoint training loop monitored the complete original "
                "held-out partition after every epoch; this is not confirmatory evidence."
            ),
        )
    else:
        limitations.insert(
            0,
            (
                "The training manifest verifies data, checkpoint, and split membership, "
                "but this evaluator does not independently verify the training process."
            ),
        )

    report = {
        "schema_version": 2,
        "evidence_role": "developmental",
        "target": target_description,
        "reference_population": reference_population,
        "limitations": limitations,
        "data": {
            "filename": arguments.data.name,
            "sha256": file_sha256(arguments.data),
            "usable_unique_names": len(names),
            "artifact_encoding": data_artifact_encoding,
            "source_row_limit": max_source_rows,
        },
        "model": {
            "filename": model_path.name,
            "sha256": file_sha256(model_path),
            "provenance": model_provenance,
            "architecture": {
                "type": "character_bidirectional_lstm",
                "vocabulary_size": VOCAB_SIZE,
                "embedding_dimension": LSTM_EMB,
                "hidden_dimension": LSTM_HIDDEN,
                "layers": LSTM_LAYERS,
                "dropout": LSTM_DROPOUT,
            },
        },
        "split": {"method": split_method, **split_provenance},
        "calibration": {
            "method": "positive-scale logistic calibration",
            "scale": calibration.scale,
            "intercept": calibration.intercept,
            "fit_partition": "calibration",
            "required_for_calibrated_serving": True,
            "current_runtime_status": "not_applied",
        },
        "metrics": {
            "balanced_test_raw": metric_report_with_intervals(
                test_probabilities,
                proportions[test_indices],
                counts[test_indices],
                bootstrap_iterations=arguments.bootstrap_iterations,
                bootstrap_seed=arguments.bootstrap_seed,
            ),
            "balanced_test_calibrated": metric_report_with_intervals(
                calibrated_test_probabilities,
                proportions[test_indices],
                counts[test_indices],
                bootstrap_iterations=arguments.bootstrap_iterations,
                bootstrap_seed=arguments.bootstrap_seed,
            ),
        },
    }
    if held_out_probabilities is not None:
        report["metrics"]["original_held_out_raw"] = report_metrics(
            held_out_probabilities,
            proportions[held_out_indices],
            counts[held_out_indices],
        )
    write_json_atomic(report, arguments.output)
    LOGGER.info("Wrote evaluation report to %s", arguments.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
