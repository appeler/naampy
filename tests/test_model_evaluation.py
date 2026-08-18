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


def test_shipped_checkpoint_split_seed_is_not_configurable():
    completed_process = subprocess.run(
        [sys.executable, "-m", "model_training.evaluate_gender_model", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert "--bootstrap-seed" in completed_process.stdout
    assert "--seed " not in completed_process.stdout


def test_balanced_calibration_test_split_preserves_held_out_names():
    proportions = np.linspace(0, 1, 100)
    counts = np.arange(1, 101)
    held_out = np.arange(20, 100)

    calibration, test = balanced_calibration_test_split(held_out, proportions, counts)

    assert set(calibration).isdisjoint(test)
    assert set(np.concatenate((calibration, test))) == set(held_out)
    assert abs(counts[calibration].sum() - counts[test].sum()) <= counts.max()


@pytest.mark.parametrize("proportion", [0.0, 1.0])
def test_balanced_calibration_test_split_supports_single_class_targets(proportion):
    proportions = np.full(8, proportion)
    counts = np.arange(1, 9)
    held_out = np.arange(8)

    calibration, test = balanced_calibration_test_split(held_out, proportions, counts)

    assert len(calibration) > 0
    assert len(test) > 0
    assert set(calibration).isdisjoint(test)
    assert set(np.concatenate((calibration, test))) == set(held_out)


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


@pytest.mark.parametrize("number_of_names", [4, 5, 10])
def test_stratified_split_keeps_small_partitions_nonempty(number_of_names):
    proportions = np.linspace(0, 1, number_of_names)
    counts = np.arange(1, number_of_names + 1)

    split = stratified_name_split(proportions, counts, seed=5)

    split.validate(number_of_names)
    assert all(
        len(partition) > 0
        for partition in (
            split.training,
            split.validation,
            split.calibration,
            split.test,
        )
    )


def test_nonempty_quota_adjustment_balances_against_realized_sizes():
    proportions = np.full(5, 0.5)
    counts = np.arange(1, 6)

    split = stratified_name_split(proportions, counts, seed=5)
    support_shares = (
        np.array(
            [
                counts[partition].sum()
                for partition in (
                    split.training,
                    split.validation,
                    split.calibration,
                    split.test,
                )
            ]
        )
        / counts.sum()
    )

    assert support_shares[0] == pytest.approx(0.40, abs=0.07)


@pytest.mark.parametrize("number_of_names", [20, 100, 500])
def test_stratified_split_respects_global_partition_sizes(number_of_names):
    proportions = np.linspace(0, 1, number_of_names)
    counts = np.arange(1, number_of_names + 1)

    split = stratified_name_split(proportions, counts, seed=5)

    assert [
        len(split.training),
        len(split.validation),
        len(split.calibration),
        len(split.test),
    ] == pytest.approx(
        np.array([0.70, 0.10, 0.10, 0.10]) * number_of_names,
        abs=1,
    )


def test_sparse_strata_do_not_systematically_shift_support():
    proportions = np.linspace(0, 1, 20)
    counts = np.arange(1, 21)

    split = stratified_name_split(proportions, counts, seed=5)
    support_shares = (
        np.array(
            [
                counts[partition].sum()
                for partition in (
                    split.training,
                    split.validation,
                    split.calibration,
                    split.test,
                )
            ]
        )
        / counts.sum()
    )

    assert support_shares == pytest.approx([0.70, 0.10, 0.10, 0.10], abs=0.02)
    female_compositions = np.array(
        [
            np.average(proportions[partition], weights=counts[partition])
            for partition in (
                split.training,
                split.validation,
                split.calibration,
                split.test,
            )
        ]
    )
    overall_composition = np.average(proportions, weights=counts)
    assert female_compositions == pytest.approx(overall_composition, abs=0.025)


def test_stratified_split_requires_one_name_per_partition():
    with pytest.raises(ValueError, match="At least four usable unique names"):
        stratified_name_split(np.array([0.0, 0.5, 1.0]), np.ones(3))


def test_probability_metrics_match_hand_calculation():
    probabilities = np.array([0.9, 0.8, 0.4, 0.1])
    proportions = np.array([1.0, 0.25, 0.75, 0.0])
    weights = np.ones(4)

    metrics = probability_metrics(probabilities, proportions, weights)

    assert metrics.majority_name_label_accuracy == pytest.approx(0.5)
    assert metrics.expected_person_accuracy == pytest.approx(0.625)
    assert metrics.expected_female_precision == pytest.approx(0.625)
    assert metrics.expected_female_recall == pytest.approx(0.625)
    assert metrics.expected_male_recall == pytest.approx(0.625)
    assert metrics.expected_female_f1 == pytest.approx(0.625)
    assert metrics.aggregate_composition_mean_squared_error == pytest.approx(
        np.mean(np.square(probabilities - proportions))
    )
    expected_binary_brier = np.mean(
        np.square(probabilities - proportions) + proportions * (1 - proportions)
    )
    assert metrics.expected_binary_brier_score == pytest.approx(expected_binary_brier)
    assert metrics.expected_binary_brier_score > (
        metrics.aggregate_composition_mean_squared_error
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
        probability_metrics(calibrated, proportions, weights).expected_binary_log_loss
        < probability_metrics(
            probabilities, proportions, weights
        ).expected_binary_log_loss
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
