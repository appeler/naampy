"""Evaluation contracts for first-name gender-pattern models."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional


@dataclass(frozen=True)
class NameDatasetSplit:
    """Indices for disjoint model-development partitions."""

    training: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def validate(self, number_of_names: int) -> None:
        """Validate that the partitions are disjoint and exhaustive."""
        partitions = (self.training, self.validation, self.calibration, self.test)
        if any(len(partition) == 0 for partition in partitions):
            raise ValueError(
                "Training, validation, calibration, and test must be nonempty"
            )
        combined = np.concatenate(partitions)
        if len(combined) != number_of_names:
            raise ValueError("Split does not contain every name exactly once")
        if len(np.unique(combined)) != number_of_names:
            raise ValueError("Split partitions overlap or contain duplicate indices")
        if combined.min(initial=0) < 0 or combined.max(initial=-1) >= number_of_names:
            raise ValueError("Split contains an out-of-range name index")


@dataclass(frozen=True)
class LogisticCalibration:
    """Logistic calibration applied to a model logit."""

    scale: float
    intercept: float

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        checked = _probability_array("probabilities", probabilities)
        clipped = np.clip(checked, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped))
        calibrated_logits = self.scale * logits + self.intercept
        return 1 / (1 + np.exp(-calibrated_logits))


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Aggregate-composition and expected individual-label metrics."""

    majority_name_label_accuracy: float
    expected_person_accuracy: float
    expected_female_precision: float
    expected_female_recall: float
    expected_male_recall: float
    expected_female_f1: float
    aggregate_composition_mean_squared_error: float
    aggregate_composition_root_mean_squared_error: float
    expected_binary_brier_score: float
    expected_binary_root_brier_score: float
    expected_binary_log_loss: float
    calibration_error_10_bins: float

    def as_dict(self) -> dict[str, float]:
        """Return JSON-serializable metric values."""
        return asdict(self)


def legacy_development_split(
    number_of_names: int, *, seed: int = 0, training_fraction: float = 0.80
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the split used to train the shipped v0.10.0 checkpoint."""
    if number_of_names < 2:
        raise ValueError("At least two names are required")
    if not 0 < training_fraction < 1:
        raise ValueError("training_fraction must be between 0 and 1")
    indices = list(range(number_of_names))
    legacy_random = random.Random(seed)  # noqa: S311
    legacy_random.shuffle(indices)
    cutoff = int(training_fraction * number_of_names)
    return (
        np.asarray(indices[:cutoff], dtype=np.int64),
        np.asarray(indices[cutoff:], dtype=np.int64),
    )


def balanced_calibration_test_split(
    held_out_indices: np.ndarray,
    female_proportions: np.ndarray,
    person_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance an existing held-out set into calibration and test halves.

    Names are considered from highest to lowest support. Each name is assigned
    to the half that minimizes imbalance in name count, represented records,
    female count, and male count. The algorithm is deterministic.
    """
    proportions, counts = _validated_targets_and_weights(
        female_proportions, person_counts
    )
    held_out = np.asarray(held_out_indices, dtype=np.int64)
    if held_out.ndim != 1 or len(held_out) < 2:
        raise ValueError("held_out_indices must contain at least two indices")
    if len(np.unique(held_out)) != len(held_out):
        raise ValueError("held_out_indices must not contain duplicates")
    if held_out.min() < 0 or held_out.max() >= len(counts):
        raise ValueError("held_out_indices contain an out-of-range index")

    target = np.array(
        [
            len(held_out) / 2,
            counts[held_out].sum() / 2,
            (counts[held_out] * proportions[held_out]).sum() / 2,
            (counts[held_out] * (1 - proportions[held_out])).sum() / 2,
        ]
    )
    balance_scale = np.where(target > 0, target, 1.0)
    totals = [np.zeros(4), np.zeros(4)]
    partitions: tuple[list[int], list[int]] = ([], [])
    ordered = sorted(held_out.tolist(), key=lambda index: (-counts[index], index))
    for index in ordered:
        contribution = np.array(
            [
                1,
                counts[index],
                counts[index] * proportions[index],
                counts[index] * (1 - proportions[index]),
            ]
        )
        costs = []
        for side in (0, 1):
            candidate = [totals[0].copy(), totals[1].copy()]
            candidate[side] += contribution
            costs.append(np.square((candidate[0] - candidate[1]) / balance_scale).sum())
        selected_side = 0 if costs[0] <= costs[1] else 1
        totals[selected_side] += contribution
        partitions[selected_side].append(index)
    calibration = np.asarray(partitions[0], dtype=np.int64)
    test = np.asarray(partitions[1], dtype=np.int64)
    if len(calibration) == 0 or len(test) == 0:
        raise RuntimeError("balanced split unexpectedly produced an empty partition")
    return calibration, test


def _partition_balance_cost(
    partition_contributions: np.ndarray, allocation_fractions: np.ndarray
) -> float:
    """Return the largest support-share or composition imbalance."""
    overall_contributions = partition_contributions.sum(axis=0)
    support_shares = partition_contributions[:, 0] / overall_contributions[0]
    female_compositions = partition_contributions[:, 1] / partition_contributions[:, 0]
    overall_female_composition = overall_contributions[1] / overall_contributions[0]
    return float(
        max(
            np.abs(support_shares - allocation_fractions).max(),
            np.abs(female_compositions - overall_female_composition).max(),
        )
    )


def _refine_partition_balance(
    partitions: list[list[int]],
    contributions: np.ndarray,
    allocation_fractions: np.ndarray,
    random_generator: np.random.Generator,
) -> None:
    """Improve final balance through exact-size-preserving pair swaps."""
    partition_contributions = np.asarray(
        [contributions[partition].sum(axis=0) for partition in partitions]
    )
    current_cost = _partition_balance_cost(
        partition_contributions, allocation_fractions
    )
    candidate_limit = 64
    for _ in range(25):
        candidate_positions: list[np.ndarray] = []
        for partition in partitions:
            if len(partition) <= candidate_limit:
                candidate_positions.append(np.arange(len(partition)))
                continue
            extreme_positions = {
                int(np.argmin(contributions[partition, feature_index]))
                for feature_index in range(contributions.shape[1])
            } | {
                int(np.argmax(contributions[partition, feature_index]))
                for feature_index in range(contributions.shape[1])
            }
            remaining_positions = np.setdiff1d(
                np.arange(len(partition)),
                np.fromiter(extreme_positions, dtype=np.int64),
            )
            sampled_positions = random_generator.choice(
                remaining_positions,
                size=candidate_limit - len(extreme_positions),
                replace=False,
            )
            candidate_positions.append(
                np.concatenate(
                    (np.fromiter(extreme_positions, dtype=np.int64), sampled_positions)
                )
            )

        best_swap: tuple[int, int, int, int, np.ndarray] | None = None
        best_cost = current_cost
        for first_partition in range(4):
            for second_partition in range(first_partition + 1, 4):
                for first_position in candidate_positions[first_partition]:
                    first_index = partitions[first_partition][first_position]
                    for second_position in candidate_positions[second_partition]:
                        second_index = partitions[second_partition][second_position]
                        candidate_contributions = partition_contributions.copy()
                        difference = (
                            contributions[second_index] - contributions[first_index]
                        )
                        candidate_contributions[first_partition] += difference
                        candidate_contributions[second_partition] -= difference
                        candidate_cost = _partition_balance_cost(
                            candidate_contributions, allocation_fractions
                        )
                        if candidate_cost < best_cost - 1e-12:
                            best_cost = candidate_cost
                            best_swap = (
                                first_partition,
                                second_partition,
                                int(first_position),
                                int(second_position),
                                candidate_contributions,
                            )
        if best_swap is None:
            break
        (
            first_partition,
            second_partition,
            first_position,
            second_position,
            partition_contributions,
        ) = best_swap
        (
            partitions[first_partition][first_position],
            partitions[second_partition][second_position],
        ) = (
            partitions[second_partition][second_position],
            partitions[first_partition][first_position],
        )
        current_cost = best_cost


def stratified_name_split(
    female_proportions: np.ndarray,
    person_counts: np.ndarray,
    *,
    seed: int = 0,
    training_fraction: float = 0.70,
    validation_fraction: float = 0.10,
    calibration_fraction: float = 0.10,
    count_strata: int = 100,
    proportion_strata: int = 10,
) -> NameDatasetSplit:
    """Split unique names while balancing support and target proportion.

    Names are ranked into support strata and binned by their observed female
    proportion. Each stratum is shuffled independently before being allocated
    to training, validation, calibration, and test partitions. Exact names
    therefore never cross partitions, while all four retain similar support and
    target distributions.
    """
    proportions, counts = _validated_targets_and_weights(
        female_proportions, person_counts
    )
    if len(counts) < 4:
        raise ValueError(
            "At least four usable unique names are required for nonempty training, "
            "validation, calibration, and test partitions"
        )
    if count_strata < 1 or proportion_strata < 1:
        raise ValueError("Stratum counts must be positive")
    if not 0 < training_fraction < 1:
        raise ValueError("training_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 - training_fraction:
        raise ValueError(
            "validation_fraction must be positive and leave room for later partitions"
        )
    if not 0 < calibration_fraction < 1 - training_fraction - validation_fraction:
        raise ValueError(
            "calibration_fraction must be positive and leave a nonempty test fraction"
        )

    number_of_names = len(counts)
    count_order = np.argsort(counts, kind="stable")
    count_rank = np.empty(number_of_names, dtype=np.int64)
    count_rank[count_order] = np.arange(number_of_names)
    count_bins = np.minimum(
        count_rank * count_strata // number_of_names, count_strata - 1
    )
    proportion_bins = np.minimum(
        (proportions * proportion_strata).astype(np.int64),
        proportion_strata - 1,
    )

    random_generator = np.random.default_rng(seed)
    partition_fractions = np.array(
        [
            training_fraction,
            validation_fraction,
            calibration_fraction,
            1 - training_fraction - validation_fraction - calibration_fraction,
        ]
    )
    raw_partition_sizes = partition_fractions * number_of_names
    partition_sizes = np.floor(raw_partition_sizes).astype(np.int64)
    unassigned_names = number_of_names - int(partition_sizes.sum())
    remainder_order = np.argsort(
        -(raw_partition_sizes - partition_sizes), kind="stable"
    )
    partition_sizes[remainder_order[:unassigned_names]] += 1
    for empty_partition in np.flatnonzero(partition_sizes == 0):
        donor_partition = int(np.argmax(partition_sizes))
        partition_sizes[donor_partition] -= 1
        partition_sizes[empty_partition] = 1
    allocation_fractions = partition_sizes / number_of_names

    strata: list[tuple[int, np.ndarray]] = []
    for count_bin in range(count_strata):
        for proportion_bin in range(proportion_strata):
            stratum = np.flatnonzero(
                (count_bins == count_bin) & (proportion_bins == proportion_bin)
            )
            if len(stratum) == 0:
                continue
            random_generator.shuffle(stratum)
            strata.append((count_bin, stratum))
    random_generator.shuffle(strata)
    strata.sort(key=lambda item: item[0], reverse=True)
    allocation_order = np.concatenate([stratum for _, stratum in strata])

    contributions = np.column_stack(
        (counts, counts * proportions, counts * (1 - proportions))
    )
    final_targets = allocation_fractions[:, None] * contributions.sum(axis=0)
    cost_scale = np.where(final_targets > 0, final_targets, 1)
    processed_contributions = np.zeros(3, dtype=np.float64)
    partition_contributions = np.zeros((4, 3), dtype=np.float64)
    partitions: list[list[int]] = [[], [], [], []]
    for step, name_index in enumerate(allocation_order, start=1):
        contribution = contributions[name_index]
        processed_contributions += contribution
        running_targets = allocation_fractions[:, None] * processed_contributions
        running_name_targets = allocation_fractions * step
        candidate_costs = np.full(4, np.inf)
        for partition_index in range(4):
            if len(partitions[partition_index]) >= partition_sizes[partition_index]:
                continue
            candidate_contributions = partition_contributions.copy()
            candidate_contributions[partition_index] += contribution
            candidate_name_counts = np.asarray(
                [len(partition) for partition in partitions], dtype=np.float64
            )
            candidate_name_counts[partition_index] += 1
            candidate_costs[partition_index] = (
                np.square(
                    (candidate_contributions - running_targets) / cost_scale
                ).sum()
                + np.square(
                    (candidate_name_counts - running_name_targets)
                    / np.maximum(partition_sizes, 1)
                ).sum()
            )
        selected_partition = int(np.argmin(candidate_costs))
        partitions[selected_partition].append(int(name_index))
        partition_contributions[selected_partition] += contribution
    _refine_partition_balance(
        partitions, contributions, allocation_fractions, random_generator
    )

    split = NameDatasetSplit(
        training=np.asarray(partitions[0], dtype=np.int64),
        validation=np.asarray(partitions[1], dtype=np.int64),
        calibration=np.asarray(partitions[2], dtype=np.int64),
        test=np.asarray(partitions[3], dtype=np.int64),
    )
    split.validate(number_of_names)
    return split


def probability_metrics(
    probabilities: np.ndarray,
    female_proportions: np.ndarray,
    sample_weights: np.ndarray,
) -> ProbabilityMetrics:
    """Compute probability and threshold metrics for soft binary targets."""
    predicted_probabilities = _probability_array("probabilities", probabilities)
    targets, weights = _validated_targets_and_weights(
        female_proportions, sample_weights
    )
    if len(predicted_probabilities) != len(targets):
        raise ValueError("probabilities, targets, and weights must have equal lengths")

    majority_name_labels = targets > 0.5
    predicted_labels = predicted_probabilities > 0.5
    true_positive_weight = (weights[predicted_labels] * targets[predicted_labels]).sum()
    false_positive_weight = (
        weights[predicted_labels] * (1 - targets[predicted_labels])
    ).sum()
    false_negative_weight = (
        weights[~predicted_labels] * targets[~predicted_labels]
    ).sum()
    true_negative_weight = (
        weights[~predicted_labels] * (1 - targets[~predicted_labels])
    ).sum()

    expected_female_precision = _safe_ratio(
        true_positive_weight, true_positive_weight + false_positive_weight
    )
    expected_female_recall = _safe_ratio(
        true_positive_weight, true_positive_weight + false_negative_weight
    )
    expected_male_recall = _safe_ratio(
        true_negative_weight, true_negative_weight + false_positive_weight
    )
    expected_female_f1 = _safe_ratio(
        2 * expected_female_precision * expected_female_recall,
        expected_female_precision + expected_female_recall,
    )
    aggregate_composition_mean_squared_error = float(
        np.average(np.square(predicted_probabilities - targets), weights=weights)
    )
    expected_binary_brier_score = float(
        np.average(
            np.square(predicted_probabilities - targets) + targets * (1 - targets),
            weights=weights,
        )
    )
    clipped = np.clip(predicted_probabilities, 1e-7, 1 - 1e-7)
    expected_binary_log_loss = float(
        np.average(
            -targets * np.log(clipped) - (1 - targets) * np.log(1 - clipped),
            weights=weights,
        )
    )
    return ProbabilityMetrics(
        majority_name_label_accuracy=float(
            np.average(predicted_labels == majority_name_labels, weights=weights)
        ),
        expected_person_accuracy=float(
            np.average(
                np.where(predicted_labels, targets, 1 - targets), weights=weights
            )
        ),
        expected_female_precision=expected_female_precision,
        expected_female_recall=expected_female_recall,
        expected_male_recall=expected_male_recall,
        expected_female_f1=expected_female_f1,
        aggregate_composition_mean_squared_error=(
            aggregate_composition_mean_squared_error
        ),
        aggregate_composition_root_mean_squared_error=float(
            np.sqrt(aggregate_composition_mean_squared_error)
        ),
        expected_binary_brier_score=expected_binary_brier_score,
        expected_binary_root_brier_score=float(np.sqrt(expected_binary_brier_score)),
        expected_binary_log_loss=expected_binary_log_loss,
        calibration_error_10_bins=expected_calibration_error(
            predicted_probabilities, targets, weights, number_of_bins=10
        ),
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    female_proportions: np.ndarray,
    sample_weights: np.ndarray,
    *,
    number_of_bins: int,
) -> float:
    """Return weighted absolute calibration error across equal-width bins."""
    predicted_probabilities = _probability_array("probabilities", probabilities)
    targets, weights = _validated_targets_and_weights(
        female_proportions, sample_weights
    )
    if len(predicted_probabilities) != len(targets):
        raise ValueError("probabilities, targets, and weights must have equal lengths")
    if number_of_bins < 1:
        raise ValueError("number_of_bins must be positive")

    boundaries = np.linspace(0, 1, number_of_bins + 1)
    bin_indices = np.minimum(
        np.digitize(predicted_probabilities, boundaries[1:-1]), number_of_bins - 1
    )
    total_weight = weights.sum()
    calibration_error = 0.0
    for bin_index in range(number_of_bins):
        selected = bin_indices == bin_index
        if not selected.any():
            continue
        bin_weight = weights[selected].sum()
        mean_prediction = np.average(
            predicted_probabilities[selected], weights=weights[selected]
        )
        mean_target = np.average(targets[selected], weights=weights[selected])
        calibration_error += (
            bin_weight / total_weight * abs(mean_prediction - mean_target)
        )
    return float(calibration_error)


def fit_logistic_calibration(
    probabilities: np.ndarray,
    female_proportions: np.ndarray,
    sample_weights: np.ndarray,
) -> LogisticCalibration:
    """Fit positive-scale logistic calibration on a held-out partition."""
    predicted_probabilities = _probability_array("probabilities", probabilities)
    targets, weights = _validated_targets_and_weights(
        female_proportions, sample_weights
    )
    if len(predicted_probabilities) != len(targets):
        raise ValueError("probabilities, targets, and weights must have equal lengths")

    clipped = np.clip(predicted_probabilities, 1e-7, 1 - 1e-7)
    logits = torch.tensor(np.log(clipped / (1 - clipped)), dtype=torch.float64)
    target_tensor = torch.tensor(targets, dtype=torch.float64)
    weight_tensor = torch.tensor(weights / weights.sum(), dtype=torch.float64)
    log_scale = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    intercept = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_scale, intercept],
        max_iter=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated_logits = torch.exp(log_scale) * logits + intercept
        loss = (
            torch_functional.binary_cross_entropy_with_logits(
                calibrated_logits, target_tensor, reduction="none"
            )
            * weight_tensor
        ).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return LogisticCalibration(
        scale=float(torch.exp(log_scale).detach()),
        intercept=float(intercept.detach()),
    )


def split_summary(
    split: NameDatasetSplit,
    female_proportions: np.ndarray,
    person_counts: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize support and composition for each split partition."""
    proportions, counts = _validated_targets_and_weights(
        female_proportions, person_counts
    )
    summary: dict[str, dict[str, float | int]] = {}
    for partition_name, indices in (
        ("training", split.training),
        ("validation", split.validation),
        ("calibration", split.calibration),
        ("test", split.test),
    ):
        summary[partition_name] = partition_summary(indices, proportions, counts)
    return summary


def partition_summary(
    indices: np.ndarray,
    female_proportions: np.ndarray,
    person_counts: np.ndarray,
) -> dict[str, float | int]:
    """Summarize support and composition for one nonempty partition."""
    proportions, counts = _validated_targets_and_weights(
        female_proportions, person_counts
    )
    selected_indices = np.asarray(indices, dtype=np.int64)
    if selected_indices.ndim != 1 or len(selected_indices) == 0:
        raise ValueError("indices must be a nonempty one-dimensional array")
    if selected_indices.min() < 0 or selected_indices.max() >= len(counts):
        raise ValueError("indices contain an out-of-range value")
    selected_counts = counts[selected_indices]
    total_people = selected_counts.sum()
    return {
        "names": len(selected_indices),
        "people": int(total_people),
        "female_share": float(
            np.average(proportions[selected_indices], weights=selected_counts)
        ),
        "effective_names": float(total_people**2 / np.square(selected_counts).sum()),
        "largest_name_share": float(selected_counts.max() / total_people),
    }


def report_metrics(
    probabilities: np.ndarray,
    female_proportions: np.ndarray,
    person_counts: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return name-weighted and person-weighted metrics."""
    return {
        "name_weighted": probability_metrics(
            probabilities,
            female_proportions,
            np.ones_like(person_counts, dtype=np.float64),
        ).as_dict(),
        "person_weighted": probability_metrics(
            probabilities, female_proportions, person_counts
        ).as_dict(),
    }


def bootstrap_metric_intervals(
    probabilities: np.ndarray,
    female_proportions: np.ndarray,
    sample_weights: np.ndarray,
    *,
    iterations: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Return name-cluster bootstrap intervals for every reported metric."""
    predicted_probabilities = _probability_array("probabilities", probabilities)
    targets, weights = _validated_targets_and_weights(
        female_proportions, sample_weights
    )
    if len(predicted_probabilities) != len(targets):
        raise ValueError("probabilities, targets, and weights must have equal lengths")
    if iterations < 2:
        raise ValueError("iterations must be at least two")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    estimate = probability_metrics(predicted_probabilities, targets, weights).as_dict()
    draws = {metric_name: np.empty(iterations) for metric_name in estimate}
    random_generator = np.random.default_rng(seed)
    for iteration in range(iterations):
        sampled_indices = random_generator.integers(0, len(targets), size=len(targets))
        sampled_metrics = probability_metrics(
            predicted_probabilities[sampled_indices],
            targets[sampled_indices],
            weights[sampled_indices],
        ).as_dict()
        for metric_name, metric_value in sampled_metrics.items():
            draws[metric_name][iteration] = metric_value

    tail_probability = (1 - confidence_level) / 2
    return {
        metric_name: {
            "estimate": metric_value,
            "lower": float(np.quantile(draws[metric_name], tail_probability)),
            "upper": float(np.quantile(draws[metric_name], 1 - tail_probability)),
        }
        for metric_name, metric_value in estimate.items()
    }


def name_membership_sha256(names: list[str], indices: np.ndarray) -> str:
    """Return a canonical SHA-256 digest of a name partition's membership."""
    selected_indices = np.asarray(indices, dtype=np.int64)
    if selected_indices.ndim != 1 or len(selected_indices) == 0:
        raise ValueError("indices must be a nonempty one-dimensional array")
    if selected_indices.min() < 0 or selected_indices.max() >= len(names):
        raise ValueError("indices contain an out-of-range value")
    members = sorted(names[index] for index in selected_indices)
    encoded = json.dumps(members, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _probability_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must contain finite values between 0 and 1")
    return array


def _validated_targets_and_weights(
    female_proportions: np.ndarray, sample_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = _probability_array("female_proportions", female_proportions)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != len(targets):
        raise ValueError("sample_weights must be one-dimensional and match the targets")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(
            "sample_weights must be finite, nonnegative, and sum above zero"
        )
    return targets, weights


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
