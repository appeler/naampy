import json
import subprocess
import sys

import pandas as pd
import pytest
import torch

from model_training.evaluate_gender_model import (
    SHIPPED_CHECKPOINT_DATA_SHA256,
    file_sha256,
    load_verified_training_split,
    verify_shipped_evaluation_artifacts,
)
from model_training.evaluation import name_membership_sha256, stratified_name_split
from model_training.train_gender_lstm import load_names
from naampy.nnets import (
    LSTM_DROPOUT,
    LSTM_EMB,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    VOCAB_SIZE,
    CharBiLSTM,
)


def _write_smoke_data(path):
    names = [
        "aarav",
        "aditi",
        "akash",
        "ananya",
        "arjun",
        "diya",
        "esha",
        "isha",
        "kabir",
        "kavya",
        "kiara",
        "meera",
        "neha",
        "priya",
        "rahul",
        "rohan",
        "saanvi",
        "tara",
        "varun",
        "vivaan",
    ]
    table = pd.DataFrame(
        {
            "state": "smoke",
            "birth_year": 1990,
            "first_name": names,
            "n_female": [(index * 3) % 11 + 1 for index in range(len(names))],
            "n_male": [(index * 5) % 13 + 1 for index in range(len(names))],
        }
    )
    table.to_csv(path, index=False, compression="gzip")


def test_shipped_evaluation_rejects_unverified_data(tmp_path):
    data_path = tmp_path / "data.csv.gz"
    model_path = tmp_path / "gender_lstm.pt"
    data_path.write_bytes(b"unverified data")
    model_path.write_bytes(b"unverified checkpoint")

    with pytest.raises(ValueError, match="--data does not match"):
        verify_shipped_evaluation_artifacts(data_path, model_path)


def test_shipped_evaluation_rejects_unverified_checkpoint(tmp_path, monkeypatch):
    data_path = tmp_path / "data.csv.gz"
    model_path = tmp_path / "gender_lstm.pt"
    data_path.write_bytes(b"data hash is replaced below")
    model_path.write_bytes(b"unverified checkpoint")
    monkeypatch.setattr(
        "model_training.evaluate_gender_model.file_sha256",
        lambda path: (
            SHIPPED_CHECKPOINT_DATA_SHA256
            if path == data_path
            else "unverified-checkpoint"
        ),
    )

    with pytest.raises(ValueError, match="does not match the checkpoint"):
        verify_shipped_evaluation_artifacts(data_path, model_path)


def test_custom_checkpoint_evaluation_reports_local_provenance(tmp_path):
    data_path = tmp_path / "smoke.csv.gz"
    model_path = tmp_path / "custom.pt"
    manifest_path = tmp_path / "custom.json"
    report_path = tmp_path / "evaluation.json"
    _write_smoke_data(data_path)
    model = CharBiLSTM(VOCAB_SIZE, 1, LSTM_EMB, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT)
    torch.save(model.state_dict(), model_path)
    names, _, female_proportions, person_counts = load_names(data_path)
    split = stratified_name_split(
        female_proportions, person_counts, seed=37, count_strata=5
    )
    manifest = {
        "schema_version": 2,
        "data": {"sha256": file_sha256(data_path)},
        "model": {"sha256": file_sha256(model_path)},
        "training": {"hyperparameters": {"max_source_rows": None}},
        "split": {
            "method": "stratified disjoint unique-name split",
            "seed": 37,
            "fractions": {
                "training": 0.70,
                "validation": 0.10,
                "calibration": 0.10,
                "test": 0.10,
            },
            "strata": {
                "person_count_rank_bins": 5,
                "female_proportion_bins": 10,
            },
            "membership_sha256": {
                "training": name_membership_sha256(names, split.training),
                "validation": name_membership_sha256(names, split.validation),
                "calibration": name_membership_sha256(names, split.calibration),
                "test": name_membership_sha256(names, split.test),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest))

    completed_process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.evaluate_gender_model",
            "--data",
            str(data_path),
            "--model",
            str(model_path),
            "--training-manifest",
            str(manifest_path),
            "--output",
            str(report_path),
            "--bootstrap-iterations",
            "5",
            "--bootstrap-seed",
            "91",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    report = json.loads(report_path.read_text())
    assert report["model"]["provenance"]["path"] == str(model_path)
    assert report["model"]["provenance"]["source"] == "user_provided_local_path"
    assert report["model"]["provenance"]["training_manifest"]["sha256"] == (
        file_sha256(manifest_path)
    )
    assert report["split"]["seed"] == 37
    assert report["metrics"]["balanced_test_raw"]["intervals"]["seed"] == 91


def test_training_manifest_records_reproducibility_contract(tmp_path):
    data_path = tmp_path / "smoke.csv.gz"
    model_path = tmp_path / "trained.pt"
    manifest_path = tmp_path / "trained.json"
    _write_smoke_data(data_path)

    completed_process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.train_gender_lstm",
            "--data",
            str(data_path),
            "--out",
            str(model_path),
            "--metadata-out",
            str(manifest_path),
            "--epochs",
            "2",
            "--samples-per-epoch",
            "8",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--max-rows",
            "10",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"]["selection_value"] >= 0
    assert len(manifest["model"]["selection_history"]) == 2
    assert manifest["training"]["hyperparameters"] == {
        "batch_size": 4,
        "epochs_requested": 2,
        "loss": "binary_cross_entropy_with_logits",
        "max_source_rows": 10,
        "samples_per_epoch": 8,
        "training_name_sampling": "person_count_weighted_with_replacement",
    }
    assert manifest["training"]["optimizer"]["type"] == "torch.optim.Adam"
    assert manifest["training"]["device"] == {
        "requested": "cpu",
        "resolved": "cpu",
    }
    assert set(manifest["training"]["software_versions"]) == {
        "numpy",
        "pandas",
        "python",
        "torch",
    }
    assert set(manifest["split"]["membership_sha256"]) == {
        "training",
        "validation",
        "calibration",
        "test",
    }
    assert manifest["serving"] == {
        "calibrated_serving_requires": manifest_path.name,
        "checkpoint_probabilities": "uncalibrated",
        "current_naampy_runtime_applies_manifest_calibration": False,
    }

    evaluation_path = tmp_path / "trained-evaluation.json"
    evaluation_process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.evaluate_gender_model",
            "--data",
            str(data_path),
            "--model",
            str(model_path),
            "--training-manifest",
            str(manifest_path),
            "--output",
            str(evaluation_path),
            "--bootstrap-iterations",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert evaluation_process.returncode == 0, evaluation_process.stderr
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation["data"]["source_row_limit"] == 10


def test_custom_manifest_rejects_inconsistent_test_fraction(tmp_path):
    data_path = tmp_path / "smoke.csv.gz"
    model_path = tmp_path / "custom.pt"
    manifest_path = tmp_path / "custom.json"
    _write_smoke_data(data_path)
    model = CharBiLSTM(VOCAB_SIZE, 1, LSTM_EMB, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT)
    torch.save(model.state_dict(), model_path)
    names, _, female_proportions, person_counts = load_names(data_path)
    split = stratified_name_split(
        female_proportions, person_counts, seed=37, count_strata=5
    )
    manifest = {
        "schema_version": 2,
        "data": {"sha256": file_sha256(data_path)},
        "model": {"sha256": file_sha256(model_path)},
        "training": {"hyperparameters": {"max_source_rows": None}},
        "split": {
            "method": "stratified disjoint unique-name split",
            "seed": 37,
            "fractions": {
                "training": 0.70,
                "validation": 0.10,
                "calibration": 0.10,
                "test": 0.20,
            },
            "strata": {
                "person_count_rank_bins": 5,
                "female_proportion_bins": 10,
            },
            "membership_sha256": {
                partition: name_membership_sha256(names, indices)
                for partition, indices in (
                    ("training", split.training),
                    ("validation", split.validation),
                    ("calibration", split.calibration),
                    ("test", split.test),
                )
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="split fractions must sum to 1"):
        load_verified_training_split(
            manifest_path,
            data_path,
            model_path,
            names,
            female_proportions,
            person_counts,
        )
