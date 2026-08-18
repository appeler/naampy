import subprocess
import sys

import numpy as np
import pytest

from model_training.evaluation import (
    LogisticCalibration,
    balanced_calibration_test_split,
    bootstrap_metric_intervals,
    expected_calibration_error,
    fit_logistic_calibration,
    legacy_development_split,
    probability_metrics,
    split_summary,
    stratified_name_split,
)


def test_legacy_development_split_matches_original_recipe():
    training, held_out = legacy_development_split(10, seed=0)

    assert training.tolist() == [7, 8, 1, 5, 3, 4, 2, 0]
    assert held_out.tolist() == [9, 6]


@pytest.mark.parametrize(
    "module_name",
    [
        "model_training.evaluate_gender_model",
        "model_training.train_gender_lstm",
    ],
)
def test_model_command_help_runs(module_name):
    completed_process = subprocess.run(  # noqa: S603
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert "usage:" in completed_process.stdout


def test_balanced_calibration_test_split_preserves_held_out_names():
    proportions = np.linspace(0, 1, 100)
    counts = np.arange(1, 101)
    held_out = np.arange(20, 100)

    calibration, test = balanced_calibration_test_split(held_out, proportions, counts)

    assert set(calibration).isdisjoint(test)
    assert set(np.concatenate((calibration, test))) == set(held_out)
    assert abs(counts[calibration].sum() - counts[test].sum()) <= counts.max()


def test_stratified_split_is_deterministic_disjoint_and_exhaustive():
    proportions = np.linspace(0, 1, 1_000)
    counts = np.arange(1, 1_001)

    first = stratified_name_split(proportions, counts, seed=17)
    second = stratified_name_split(proportions, counts, seed=17)

    assert np.array_equal(first.training, second.training)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.calibration, second.calibration)
    assert np.array_equal(first.test, second.test)
    first.validate(len(proportions))
    assert set(first.training).isdisjoint(first.validation)
    assert set(first.training).isdisjoint(first.calibration)
    assert set(first.training).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.calibration)
    assert set(first.validation).isdisjoint(first.test)
    assert set(first.calibration).isdisjoint(first.test)


def test_stratified_split_preserves_support_and_target_composition():
    proportions = np.tile(np.linspace(0, 1, 100), 20)
    counts = np.tile(np.arange(1, 101), 20)
    split = stratified_name_split(proportions, counts, seed=5)

    summary = split_summary(split, proportions, counts)

    person_totals = np.array([values["people"] for values in summary.values()])
    person_shares = person_totals / person_totals.sum()
    assert person_shares == pytest.approx([0.70, 0.10, 0.10, 0.10], abs=0.02)
    female_shares = [values["female_share"] for values in summary.values()]
    assert max(female_shares) - min(female_shares) < 0.02


def test_probability_metrics_match_hand_calculation():
    probabilities = np.array([0.9, 0.8, 0.4, 0.1])
    proportions = np.array([1.0, 0.25, 0.75, 0.0])
    weights = np.ones(4)

    metrics = probability_metrics(probabilities, proportions, weights)

    assert metrics.majority_label_accuracy == pytest.approx(0.5)
    assert metrics.expected_person_accuracy == pytest.approx(0.625)
    assert metrics.female_precision == pytest.approx(0.5)
    assert metrics.female_recall == pytest.approx(0.5)
    assert metrics.male_recall == pytest.approx(0.5)
    assert metrics.female_f1 == pytest.approx(0.5)
    assert metrics.brier_score == pytest.approx(
        np.mean(np.square(probabilities - proportions))
    )


def test_expected_calibration_error_is_zero_for_matching_bin_means():
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    proportions = np.array([0.0, 0.3, 0.7, 1.0])

    error = expected_calibration_error(
        probabilities, proportions, np.ones(4), number_of_bins=2
    )

    assert error == pytest.approx(0.0)


def test_logistic_calibration_improves_overconfident_probabilities():
    probabilities = np.array([0.001, 0.01, 0.99, 0.999])
    proportions = np.array([0.2, 0.3, 0.7, 0.8])
    weights = np.ones(4)

    calibration = fit_logistic_calibration(probabilities, proportions, weights)
    calibrated = calibration.apply(probabilities)

    assert calibration.scale < 1
    assert (
        probability_metrics(calibrated, proportions, weights).soft_log_loss
        < probability_metrics(probabilities, proportions, weights).soft_log_loss
    )


def test_identity_calibration_preserves_probabilities():
    probabilities = np.array([0.2, 0.5, 0.8])

    calibrated = LogisticCalibration(scale=1, intercept=0).apply(probabilities)

    assert calibrated == pytest.approx(probabilities)


def test_bootstrap_intervals_are_reproducible_and_contain_estimate():
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    proportions = np.array([0.0, 0.2, 0.8, 1.0])
    weights = np.ones(4)

    first = bootstrap_metric_intervals(
        probabilities, proportions, weights, iterations=50, seed=11
    )
    second = bootstrap_metric_intervals(
        probabilities, proportions, weights, iterations=50, seed=11
    )

    assert first == second
    for interval in first.values():
        assert interval["lower"] <= interval["estimate"] <= interval["upper"]


@pytest.mark.parametrize(
    ("probabilities", "proportions", "weights"),
    [
        (np.array([]), np.array([]), np.array([])),
        (np.array([1.1]), np.array([1.0]), np.array([1.0])),
        (np.array([0.5]), np.array([np.nan]), np.array([1.0])),
        (np.array([0.5]), np.array([0.5]), np.array([-1.0])),
    ],
)
def test_probability_metrics_reject_invalid_inputs(probabilities, proportions, weights):
    with pytest.raises(ValueError):
        probability_metrics(probabilities, proportions, weights)
