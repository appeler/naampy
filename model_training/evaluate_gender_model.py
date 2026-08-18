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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_training.evaluation import (
    balanced_calibration_test_split,
    bootstrap_metric_intervals,
    fit_logistic_calibration,
    legacy_development_split,
    partition_summary,
    report_metrics,
)
from model_training.train_gender_lstm import load_names
from naampy._resources import HF_REPO, HF_REVISION, resolve_model
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


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    seed: int,
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
            "name_weighted": bootstrap_metric_intervals(
                probabilities,
                female_proportions,
                np.ones_like(person_counts, dtype=np.float64),
                iterations=bootstrap_iterations,
                seed=seed,
            ),
            "person_weighted": bootstrap_metric_intervals(
                probabilities,
                female_proportions,
                person_counts,
                iterations=bootstrap_iterations,
                seed=seed,
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


def main() -> None:
    """Evaluate the pinned checkpoint and write its evidence report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    names, encoded_names, female_proportions, person_counts = load_names(arguments.data)
    proportions = np.asarray(female_proportions)
    counts = np.asarray(person_counts)
    training_indices, held_out_indices = legacy_development_split(
        len(names), seed=arguments.seed
    )
    calibration_indices, test_indices = balanced_calibration_test_split(
        held_out_indices, proportions, counts
    )
    model_path = (
        arguments.model
        if arguments.model is not None
        else Path(resolve_model("gender_lstm.pt"))
    )
    model = CharBiLSTM(VOCAB_SIZE, 1, LSTM_EMB, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    held_out_probabilities = predict_probabilities(
        model, [encoded_names[index] for index in held_out_indices]
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

    report = {
        "schema_version": 1,
        "evidence_role": "developmental",
        "target": "female share among female and male electoral-roll labels",
        "reference_population": (
            "aggregated Indian electoral-roll registration records represented "
            "in the local naampy v3 construction"
        ),
        "limitations": [
            (
                "The checkpoint training loop monitored the complete original "
                "held-out partition after every epoch; this is not confirmatory evidence."
            ),
            (
                "The binary model excludes the extremely sparse third-gender label "
                "and does not estimate gender identity."
            ),
        ],
        "data": {
            "filename": arguments.data.name,
            "sha256": file_sha256(arguments.data),
            "usable_unique_names": len(names),
        },
        "model": {
            "filename": model_path.name,
            "sha256": file_sha256(model_path),
            "hugging_face_repository": HF_REPO,
            "hugging_face_revision": HF_REVISION,
            "architecture": {
                "type": "character_bidirectional_lstm",
                "vocabulary_size": VOCAB_SIZE,
                "embedding_dimension": LSTM_EMB,
                "hidden_dimension": LSTM_HIDDEN,
                "layers": LSTM_LAYERS,
                "dropout": LSTM_DROPOUT,
            },
        },
        "split": {
            "method": (
                "original seeded 80/20 unique-name split; held-out names balanced "
                "into calibration and test halves by support and label composition"
            ),
            "seed": arguments.seed,
            "summary": {
                "original_training": partition_summary(
                    training_indices,
                    proportions,
                    counts,
                ),
                "calibration": partition_summary(
                    calibration_indices,
                    proportions,
                    counts,
                ),
                "test": partition_summary(
                    test_indices,
                    proportions,
                    counts,
                ),
            },
        },
        "calibration": {
            "method": "positive-scale logistic calibration",
            "scale": calibration.scale,
            "intercept": calibration.intercept,
            "fit_partition": "calibration",
        },
        "metrics": {
            "original_held_out_raw": report_metrics(
                held_out_probabilities,
                proportions[held_out_indices],
                counts[held_out_indices],
            ),
            "balanced_test_raw": metric_report_with_intervals(
                test_probabilities,
                proportions[test_indices],
                counts[test_indices],
                bootstrap_iterations=arguments.bootstrap_iterations,
                seed=arguments.seed,
            ),
            "balanced_test_calibrated": metric_report_with_intervals(
                calibrated_test_probabilities,
                proportions[test_indices],
                counts[test_indices],
                bootstrap_iterations=arguments.bootstrap_iterations,
                seed=arguments.seed,
            ),
        },
    }
    write_json_atomic(report, arguments.output)
    LOGGER.info("Wrote evaluation report to %s", arguments.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
