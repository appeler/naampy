import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from model_training.train_gender_ngram import (
    build_soft_label_training_data,
    fit_logistic_challengers,
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


def test_soft_labels_remain_sparse_and_preserve_record_weight_ratios():
    features = sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.float32)
    proportions = np.array([0.25, 1.0])
    counts = np.array([8.0, 4.0])

    duplicated, targets, weights = build_soft_label_training_data(
        features, proportions, counts
    )

    assert sparse.isspmatrix_csr(duplicated)
    assert duplicated.shape == (3, 2)
    assert targets.tolist() == [1.0, 1.0, 0.0]
    assert weights[0] / weights[2] == pytest.approx(2 / 6)
    assert weights[1] / weights[2] == pytest.approx(4 / 6)
    assert weights.mean() == pytest.approx(1.0)


def test_validation_loss_selects_one_fitted_candidate():
    training_features = sparse.csr_matrix(
        [[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=np.float32
    )
    validation_features = sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.float32)
    fitted = fit_logistic_challengers(
        training_features,
        np.array([1.0, 0.9, 0.0, 0.1]),
        np.ones(4),
        validation_features,
        np.array([1.0, 0.0]),
        np.ones(2),
        inverse_regularization_strengths=[0.1, 1.0],
        maximum_iterations=100,
        model_seed=7,
    )

    losses = [
        row["validation_record_weighted_expected_binary_log_loss"]
        for row in fitted.selection_history
    ]
    selected_row = fitted.selection_history[int(np.argmin(losses))]
    assert (
        fitted.selected_inverse_regularization_strength
        == selected_row["inverse_regularization_strength"]
    )
    assert len(fitted.selection_history) == 2


def test_challenger_command_writes_reproducible_evidence_report(tmp_path):
    data_path = tmp_path / "smoke.csv.gz"
    report_path = tmp_path / "challenger.json"
    _write_smoke_data(data_path)

    completed_process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.train_gender_ngram",
            "--data",
            str(data_path),
            "--output",
            str(report_path),
            "--inverse-regularization-strengths",
            "0.1",
            "1.0",
            "--minimum-document-frequency",
            "1",
            "--maximum-iterations",
            "100",
            "--minimum-training-name-supports",
            "1",
            "5",
            "--bootstrap-iterations",
            "2",
            "--bootstrap-seed",
            "0",
            "--split-seed",
            "17",
            "--model-seed",
            "17",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 3
    assert report["evidence_role"] == "developmental_challenger"
    assert report["data"]["usable_unique_names"] == 20
    assert report["model"]["architecture"]["type"] == (
        "character_tfidf_logistic_regression"
    )
    assert report["model"]["selected_inverse_regularization_strength"] in {
        0.1,
        1.0,
    }
    assert report["model"]["selected_minimum_training_name_support"] in {1, 5}
    assert len(report["model"]["selection_history"]) == 4
    assert {
        row["minimum_training_name_support"]
        for row in report["model"]["selection_history"]
    } == {1, 5}
    assert set(report["split"]["membership_sha256"]) == {
        "training",
        "validation",
        "calibration",
        "test",
    }
    assert set(report["metrics"]) == {"selected_validation_raw"}
    assert "intervals" in report["metrics"]["selected_validation_raw"]
    assert report["calibration"]["status"] == "reserved_untouched"
    assert report["model"]["candidate_artifact_status"] == ("not_persisted_not_frozen")
    assert report["reserved_partitions"] == {
        "calibration_status": "untouched",
        "policy": (
            "Do not transform, inspect targets for, fit on, or score calibration "
            "or test until the candidate artifact and selection are frozen."
        ),
        "test_status": "untouched",
    }
    assert set(report["split"]["development_summary"]) == {
        "training",
        "validation",
    }


def test_challenger_command_skips_unusable_support_floor(tmp_path):
    data_path = tmp_path / "smoke.csv.gz"
    report_path = tmp_path / "challenger.json"
    _write_smoke_data(data_path)

    completed_process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.train_gender_ngram",
            "--data",
            str(data_path),
            "--output",
            str(report_path),
            "--inverse-regularization-strengths",
            "1.0",
            "--minimum-document-frequency",
            "1",
            "--minimum-training-name-supports",
            "1",
            "1000000",
            "--bootstrap-iterations",
            "2",
            "--bootstrap-seed",
            "0",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    report = json.loads(report_path.read_text())
    assert report["training"]["skipped_minimum_training_name_supports"] == [
        {
            "minimum_training_name_support": 1000000,
            "reason": "fewer_than_two_training_names",
            "retained_training_name_count": 0,
        }
    ]


def test_challenger_command_errors_when_no_support_floor_is_usable(tmp_path):
    data_path = tmp_path / "smoke.csv.gz"
    report_path = tmp_path / "challenger.json"
    _write_smoke_data(data_path)

    completed_process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.train_gender_ngram",
            "--data",
            str(data_path),
            "--output",
            str(report_path),
            "--minimum-training-name-supports",
            "1000000",
            "--bootstrap-iterations",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode != 0
    assert "No usable minimum training-name support floor" in completed_process.stderr
    assert not report_path.exists()
