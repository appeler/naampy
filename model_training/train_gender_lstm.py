"""Train the Naampy character BiLSTM on the v3 data.

Replaces the legacy TensorFlow char-CNN. Aggregates the (state,year,first_name) table to a
global ``first_name -> female_prop`` target (weighted by count), trains a torch ``CharBiLSTM``
with a single sigmoid output. CPU-trainable.

    uv run python -m model_training.train_gender_lstm \
        --data model_training/data/naampy_v3.csv.gz \
        --out gender_lstm.pt --epochs 12
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import platform
import random
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional

from model_training.evaluation import (
    fit_logistic_calibration,
    name_membership_sha256,
    probability_metrics,
    report_metrics,
    split_summary,
    stratified_name_split,
)
from naampy.nnets import (
    LSTM_DROPOUT,
    LSTM_EMB,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    VOCAB_SIZE,
    CharBiLSTM,
    encode_name,
    pad_encoded,
)

_REPEAT3 = re.compile(r"(.)\1\1")
LOGGER = logging.getLogger(__name__)


def load_names(
    path: str | Path, max_rows: int | None = None
) -> tuple[list[str], list[list[int]], list[float], list[float]]:
    """Aggregate to global first_name -> (female_prop, count); apply naampy's name filters."""
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
    person_counts: list[float] = []
    for name_value, female_value, male_value in grouped_names.itertuples(
        index=False, name=None
    ):
        name = str(name_value)
        female_count = float(female_value)
        male_count = float(male_value)
        if not (2 < len(name) < 20) or not name.isalpha() or _REPEAT3.search(name):
            continue
        encoded_name = encode_name(name)
        if not encoded_name:
            continue
        names.append(name)
        encoded_names.append(encoded_name)
        female_proportions.append(female_count / (female_count + male_count))
        person_counts.append(female_count + male_count)
    return names, encoded_names, female_proportions, person_counts


@torch.no_grad()
def predict_probabilities(
    model: CharBiLSTM,
    encoded_names: list[list[int]],
    device: str,
) -> np.ndarray:
    """Return female-name-pattern probabilities for encoded names."""
    model.eval()
    probabilities: list[float] = []
    for start in range(0, len(encoded_names), 512):
        inputs, lengths = pad_encoded(encoded_names[start : start + 512])
        batch_probabilities = (
            torch.sigmoid(model(inputs.to(device), lengths)).squeeze(1).cpu().numpy()
        )
        probabilities.extend(batch_probabilities.tolist())
    return np.asarray(probabilities)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(report: dict[str, Any], output_path: Path) -> None:
    """Write a JSON document without exposing a partial file."""
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
    """Train and save a character BiLSTM checkpoint."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True)
    argument_parser.add_argument("--out", required=True)
    argument_parser.add_argument("--metadata-out", type=Path)
    argument_parser.add_argument("--epochs", type=int, default=12)
    argument_parser.add_argument("--samples-per-epoch", type=int, default=300_000)
    argument_parser.add_argument("--batch-size", type=int, default=256)
    argument_parser.add_argument("--learning-rate", type=float, default=1e-3)
    argument_parser.add_argument(
        "--max-rows", type=int, default=None, help="cap for a smoke test"
    )
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--seed", type=int, default=0)
    arguments = argument_parser.parse_args()

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if arguments.device == "auto"
        else arguments.device
    )

    names, encoded_names, female_proportions, person_counts = load_names(
        arguments.data, arguments.max_rows
    )
    proportions = np.asarray(female_proportions)
    counts = np.asarray(person_counts)
    split = stratified_name_split(proportions, counts, seed=arguments.seed)
    training_weights = counts[split.training].tolist()
    LOGGER.info(
        "Names: %s; training: %s; validation: %s; calibration: %s; "
        "test: %s; device: %s",
        f"{len(names):,}",
        f"{len(split.training):,}",
        f"{len(split.validation):,}",
        f"{len(split.calibration):,}",
        f"{len(split.test):,}",
        device,
    )

    model = CharBiLSTM(
        VOCAB_SIZE, 1, LSTM_EMB, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    batch_size = arguments.batch_size
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    selection_history: list[dict[str, float | int]] = []

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        sampled_indices = random.choices(  # noqa: S311
            split.training.tolist(),
            weights=training_weights,
            k=arguments.samples_per_epoch,
        )
        running = 0.0
        for start in range(0, len(sampled_indices), batch_size):
            batch_indices = sampled_indices[start : start + batch_size]
            inputs, lengths = pad_encoded(
                [encoded_names[index] for index in batch_indices]
            )
            target = torch.tensor(
                [[proportions[index]] for index in batch_indices], dtype=torch.float32
            )
            logits = model(inputs.to(device), lengths)
            loss = torch_functional.binary_cross_entropy_with_logits(
                logits, target.to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch_indices)
        validation_probabilities = predict_probabilities(
            model,
            [encoded_names[index] for index in split.validation],
            device,
        )
        validation_metrics = probability_metrics(
            validation_probabilities,
            proportions[split.validation],
            counts[split.validation],
        )
        training_loss = running / len(sampled_indices)
        validation_loss = validation_metrics.expected_binary_log_loss
        selection_history.append(
            {
                "epoch": epoch,
                "training_expected_binary_log_loss": training_loss,
                "validation_person_weighted_expected_binary_log_loss": (
                    validation_loss
                ),
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        LOGGER.info(
            "Epoch %s: training loss %.4f; validation log loss %.4f; "
            "validation root Brier score %.4f; validation accuracy %.4f",
            epoch,
            training_loss,
            validation_loss,
            validation_metrics.expected_binary_root_brier_score,
            validation_metrics.majority_name_label_accuracy,
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)

    calibration_probabilities = predict_probabilities(
        model,
        [encoded_names[index] for index in split.calibration],
        device,
    )
    calibration = fit_logistic_calibration(
        calibration_probabilities,
        proportions[split.calibration],
        counts[split.calibration],
    )
    test_probabilities = predict_probabilities(
        model,
        [encoded_names[index] for index in split.test],
        device,
    )
    calibrated_test_probabilities = calibration.apply(test_probabilities)

    checkpoint_path = Path(arguments.out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    metadata_path = arguments.metadata_out or checkpoint_path.with_suffix(".json")
    training_report = {
        "schema_version": 2,
        "target": "female share among female and male electoral-roll labels",
        "reference_population": (
            "aggregated Indian electoral-roll registration records represented "
            "in the Naampy v3 construction"
        ),
        "data": {
            "filename": Path(arguments.data).name,
            "sha256": file_sha256(Path(arguments.data)),
            "usable_unique_names": len(names),
        },
        "model": {
            "filename": checkpoint_path.name,
            "sha256": file_sha256(checkpoint_path),
            "architecture": {
                "type": "character_bidirectional_lstm",
                "vocabulary_size": VOCAB_SIZE,
                "embedding_dimension": LSTM_EMB,
                "hidden_dimension": LSTM_HIDDEN,
                "layers": LSTM_LAYERS,
                "dropout": LSTM_DROPOUT,
            },
            "selected_epoch": best_epoch,
            "selection_metric": ("validation_person_weighted_expected_binary_log_loss"),
            "selection_value": best_validation_loss,
            "selection_history": selection_history,
        },
        "training": {
            "random_seed": arguments.seed,
            "hyperparameters": {
                "epochs_requested": arguments.epochs,
                "samples_per_epoch": arguments.samples_per_epoch,
                "batch_size": arguments.batch_size,
                "max_source_rows": arguments.max_rows,
                "loss": "binary_cross_entropy_with_logits",
                "training_name_sampling": "person_count_weighted_with_replacement",
            },
            "optimizer": {
                "type": "torch.optim.Adam",
                "learning_rate": arguments.learning_rate,
                "betas": list(optimizer.defaults["betas"]),
                "epsilon": optimizer.defaults["eps"],
                "weight_decay": optimizer.defaults["weight_decay"],
                "amsgrad": optimizer.defaults["amsgrad"],
            },
            "device": {
                "requested": arguments.device,
                "resolved": device,
            },
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
            },
        },
        "split": {
            "method": "stratified disjoint unique-name split",
            "seed": arguments.seed,
            "fractions": {
                "training": 0.70,
                "validation": 0.10,
                "calibration": 0.10,
                "test": 0.10,
            },
            "strata": {
                "person_count_rank_bins": 100,
                "female_proportion_bins": 10,
            },
            "summary": split_summary(split, proportions, counts),
            "membership_sha256": {
                "training": name_membership_sha256(names, split.training),
                "validation": name_membership_sha256(names, split.validation),
                "calibration": name_membership_sha256(names, split.calibration),
                "test": name_membership_sha256(names, split.test),
            },
        },
        "calibration": {
            "method": "positive-scale logistic calibration",
            "scale": calibration.scale,
            "intercept": calibration.intercept,
            "fit_partition": "calibration",
            "required_for_calibrated_serving": True,
            "current_runtime_status": "not_applied",
        },
        "serving": {
            "checkpoint_probabilities": "uncalibrated",
            "calibrated_serving_requires": metadata_path.name,
            "current_naampy_runtime_applies_manifest_calibration": False,
        },
        "test_metrics": {
            "raw": report_metrics(
                test_probabilities,
                proportions[split.test],
                counts[split.test],
            ),
            "calibrated": report_metrics(
                calibrated_test_probabilities,
                proportions[split.test],
                counts[split.test],
            ),
        },
    }
    write_json_atomic(training_report, metadata_path)
    LOGGER.info(
        "Saved epoch %s checkpoint to %s and metadata to %s",
        best_epoch,
        checkpoint_path,
        metadata_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
