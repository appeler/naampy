"""Evaluate a sparse character n-gram challenger on the Naampy v3 data.

The command fits TF-IDF features and candidate logistic regressions on the
training partition and selects inverse regularization strength on validation.
It freezes partition hashes but does not transform or score the reserved
calibration and test partitions.

Run from the repository root::

    python -m model_training.train_gender_ngram \
        --data model_training/data/naampy_v3.csv.gz \
        --output model_training/reports/gender_ngram_v3_validation.json
"""

from __future__ import annotations

import argparse
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from model_training.evaluate_gender_model import metric_report_with_intervals
from model_training.evaluation import (
    name_membership_sha256,
    partition_summary,
    probability_metrics,
    stratified_name_split,
)
from model_training.train_gender_lstm import file_sha256, load_names, write_json_atomic

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FittedChallenger:
    """Selected sparse model and its validation evidence."""

    classifier: LogisticRegression
    selection_history: list[dict[str, float | int]]
    selected_inverse_regularization_strength: float


def build_soft_label_training_data(
    name_features: sparse.csr_matrix,
    female_proportions: np.ndarray,
    record_counts: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Represent aggregate female shares as weighted binary observations."""
    proportions = np.asarray(female_proportions, dtype=np.float64)
    counts = np.asarray(record_counts, dtype=np.float64)
    feature_shape = name_features.shape
    if feature_shape is None:
        raise ValueError("name features must be a two-dimensional sparse matrix")
    if feature_shape[0] != len(proportions) or len(proportions) != len(counts):
        raise ValueError("features, female proportions, and counts must align")
    if (
        len(proportions) == 0
        or not np.isfinite(proportions).all()
        or np.any((proportions < 0) | (proportions > 1))
    ):
        raise ValueError("female proportions must be finite values from zero to one")
    if not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("record counts must be finite and positive")

    binary_targets = np.concatenate(
        (np.ones(len(proportions)), np.zeros(len(proportions)))
    )
    sample_weights = np.concatenate((counts * proportions, counts * (1 - proportions)))
    nonzero_weight = sample_weights > 0
    stacked_features = cast(
        "sparse.csr_matrix",
        sparse.vstack((name_features, name_features), format="csr"),
    )
    duplicated_features = stacked_features[nonzero_weight]
    binary_targets = binary_targets[nonzero_weight]
    sample_weights = sample_weights[nonzero_weight]

    # Preserve relative record support while keeping the mean optimization
    # weight at one, so regularization strengths are stable across data sizes.
    sample_weights *= len(sample_weights) / sample_weights.sum()
    return duplicated_features, binary_targets, sample_weights


def fit_logistic_challengers(
    training_features: sparse.csr_matrix,
    training_proportions: np.ndarray,
    training_counts: np.ndarray,
    validation_features: sparse.csr_matrix,
    validation_proportions: np.ndarray,
    validation_counts: np.ndarray,
    *,
    inverse_regularization_strengths: list[float],
    maximum_iterations: int,
    model_seed: int,
) -> FittedChallenger:
    """Select a sparse logistic model using validation log loss only."""
    if not inverse_regularization_strengths or any(
        value <= 0 for value in inverse_regularization_strengths
    ):
        raise ValueError("inverse regularization strengths must be positive")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    features, targets, sample_weights = build_soft_label_training_data(
        training_features, training_proportions, training_counts
    )
    selected_model: LogisticRegression | None = None
    selected_strength = 0.0
    selected_loss = float("inf")
    selection_history: list[dict[str, float | int]] = []
    for inverse_regularization_strength in inverse_regularization_strengths:
        fit_started = perf_counter()
        candidate = LogisticRegression(
            C=inverse_regularization_strength,
            solver="liblinear",
            max_iter=maximum_iterations,
            tol=1e-5,
            random_state=model_seed,
        )
        candidate.fit(features, targets, sample_weight=sample_weights)
        validation_probabilities = candidate.predict_proba(validation_features)[:, 1]
        validation_metrics = probability_metrics(
            validation_probabilities, validation_proportions, validation_counts
        )
        validation_loss = validation_metrics.expected_binary_log_loss
        selection_history.append(
            {
                "inverse_regularization_strength": inverse_regularization_strength,
                "iterations": int(candidate.n_iter_[0]),
                "fit_and_validation_seconds": perf_counter() - fit_started,
                "validation_record_weighted_expected_binary_log_loss": (
                    validation_loss
                ),
                "validation_record_weighted_expected_binary_brier_score": (
                    validation_metrics.expected_binary_brier_score
                ),
                "validation_record_weighted_expected_record_accuracy": (
                    validation_metrics.expected_record_accuracy
                ),
                "validation_record_weighted_expected_female_f1": (
                    validation_metrics.expected_female_f1
                ),
            }
        )
        if validation_loss < selected_loss:
            selected_model = candidate
            selected_strength = inverse_regularization_strength
            selected_loss = validation_loss

    if selected_model is None:
        raise RuntimeError("No logistic challenger was fitted")
    return FittedChallenger(
        classifier=selected_model,
        selection_history=selection_history,
        selected_inverse_regularization_strength=selected_strength,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--minimum-ngram-length", type=int, default=2)
    parser.add_argument("--maximum-ngram-length", type=int, default=5)
    parser.add_argument("--minimum-document-frequency", type=int, default=2)
    parser.add_argument(
        "--inverse-regularization-strengths",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 30.0, 100.0],
    )
    parser.add_argument(
        "--minimum-training-name-supports",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50],
    )
    parser.add_argument("--maximum-iterations", type=int, default=500)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.minimum_ngram_length < 1:
        parser.error("--minimum-ngram-length must be positive")
    if arguments.maximum_ngram_length < arguments.minimum_ngram_length:
        parser.error("--maximum-ngram-length must be at least --minimum-ngram-length")
    if arguments.minimum_document_frequency < 1:
        parser.error("--minimum-document-frequency must be positive")
    if not arguments.minimum_training_name_supports or any(
        support < 1 for support in arguments.minimum_training_name_supports
    ):
        parser.error("--minimum-training-name-supports must be positive")
    if arguments.bootstrap_iterations < 2:
        parser.error("--bootstrap-iterations must be at least two")
    return arguments


def main() -> None:
    """Fit and select the sparse n-gram challenger on validation only."""
    arguments = _parse_arguments()
    command_started = perf_counter()
    names, _, female_proportions, record_counts = load_names(
        arguments.data, arguments.max_rows
    )
    proportions = np.asarray(female_proportions, dtype=np.float64)
    counts = np.asarray(record_counts, dtype=np.float64)
    split = stratified_name_split(proportions, counts, seed=arguments.split_seed)

    selection_started = perf_counter()
    selection_history: list[dict[str, float | int]] = []
    selected_classifier: LogisticRegression | None = None
    selected_vectorizer: TfidfVectorizer | None = None
    selected_validation_features: sparse.csr_matrix | None = None
    selected_support_floor = 0
    selected_training_name_count = 0
    selected_represented_record_fraction = 0.0
    selected_feature_fit_seconds = 0.0
    selected_validation_transform_seconds = 0.0
    selected_validation_loss = float("inf")
    full_training_record_count = counts[split.training].sum()
    skipped_support_floors: list[dict[str, int | str]] = []
    validation_names = [names[index] for index in split.validation]
    for support_floor in sorted(set(arguments.minimum_training_name_supports)):
        retained_training_indices = split.training[
            counts[split.training] >= support_floor
        ]
        if len(retained_training_indices) < 2:
            skipped_support_floors.append(
                {
                    "minimum_training_name_support": support_floor,
                    "retained_training_name_count": len(retained_training_indices),
                    "reason": "fewer_than_two_training_names",
                }
            )
            continue
        if arguments.minimum_document_frequency > len(retained_training_indices):
            skipped_support_floors.append(
                {
                    "minimum_training_name_support": support_floor,
                    "retained_training_name_count": len(retained_training_indices),
                    "reason": "minimum_document_frequency_exceeds_training_names",
                }
            )
            continue
        retained_female_records = float(
            np.sum(
                counts[retained_training_indices]
                * proportions[retained_training_indices]
            )
        )
        retained_male_records = float(
            np.sum(
                counts[retained_training_indices]
                * (1 - proportions[retained_training_indices])
            )
        )
        if retained_female_records == 0 or retained_male_records == 0:
            skipped_support_floors.append(
                {
                    "minimum_training_name_support": support_floor,
                    "retained_training_name_count": len(retained_training_indices),
                    "reason": "retained_records_contain_only_one_source_label",
                }
            )
            continue
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(
                arguments.minimum_ngram_length,
                arguments.maximum_ngram_length,
            ),
            lowercase=False,
            dtype=np.float64,
            sublinear_tf=True,
            min_df=arguments.minimum_document_frequency,
        )
        feature_fit_started = perf_counter()
        try:
            training_features = cast(
                "sparse.csr_matrix",
                vectorizer.fit_transform(
                    [names[index] for index in retained_training_indices]
                ),
            )
        except ValueError as error:
            if "empty vocabulary" not in str(error) and "After pruning" not in str(
                error
            ):
                raise
            skipped_support_floors.append(
                {
                    "minimum_training_name_support": support_floor,
                    "retained_training_name_count": len(retained_training_indices),
                    "reason": "no_character_ngrams_meet_document_frequency",
                }
            )
            continue
        feature_fit_seconds = perf_counter() - feature_fit_started
        validation_transform_started = perf_counter()
        validation_features = cast(
            "sparse.csr_matrix", vectorizer.transform(validation_names)
        )
        validation_transform_seconds = perf_counter() - validation_transform_started
        fitted = fit_logistic_challengers(
            training_features,
            proportions[retained_training_indices],
            counts[retained_training_indices],
            validation_features,
            proportions[split.validation],
            counts[split.validation],
            inverse_regularization_strengths=(
                arguments.inverse_regularization_strengths
            ),
            maximum_iterations=arguments.maximum_iterations,
            model_seed=arguments.model_seed,
        )
        represented_record_fraction = float(
            counts[retained_training_indices].sum() / full_training_record_count
        )
        selection_history.extend(
            [
                {
                    "minimum_training_name_support": support_floor,
                    "training_name_count": len(retained_training_indices),
                    "excluded_training_name_count": (
                        len(split.training) - len(retained_training_indices)
                    ),
                    "represented_record_fraction": represented_record_fraction,
                    "feature_count": len(vectorizer.vocabulary_),
                    "feature_fit_seconds": feature_fit_seconds,
                    "validation_transform_seconds": validation_transform_seconds,
                    **candidate_result,
                }
                for candidate_result in fitted.selection_history
            ]
        )
        floor_validation_loss = min(
            float(
                candidate_result["validation_record_weighted_expected_binary_log_loss"]
            )
            for candidate_result in fitted.selection_history
        )
        if floor_validation_loss < selected_validation_loss:
            selected_classifier = fitted.classifier
            selected_vectorizer = vectorizer
            selected_validation_features = validation_features
            selected_support_floor = support_floor
            selected_training_name_count = len(retained_training_indices)
            selected_represented_record_fraction = represented_record_fraction
            selected_feature_fit_seconds = feature_fit_seconds
            selected_validation_transform_seconds = validation_transform_seconds
            selected_validation_loss = floor_validation_loss

    if (
        selected_classifier is None
        or selected_vectorizer is None
        or selected_validation_features is None
    ):
        raise ValueError(
            "No usable minimum training-name support floor remained after validation"
        )
    selection_seconds = perf_counter() - selection_started
    validation_prediction_started = perf_counter()
    selected_validation_probabilities = selected_classifier.predict_proba(
        selected_validation_features
    )[:, 1]
    validation_prediction_seconds = perf_counter() - validation_prediction_started
    validation_metric_report = metric_report_with_intervals(
        selected_validation_probabilities,
        proportions[split.validation],
        counts[split.validation],
        bootstrap_iterations=arguments.bootstrap_iterations,
        bootstrap_seed=arguments.bootstrap_seed,
    )

    coefficient_bytes = (
        np.asarray(selected_classifier.coef_).nbytes
        + np.asarray(selected_classifier.intercept_).nbytes
        + np.asarray(selected_vectorizer.idf_).nbytes
    )
    report: dict[str, Any] = {
        "schema_version": 3,
        "evidence_role": "developmental_challenger",
        "target": "female share among female and male electoral-roll labels",
        "reference_population": (
            "aggregated Indian electoral-roll registration records represented "
            "in the Naampy v3 construction"
        ),
        "limitations": [
            (
                "The binary target excludes the extremely sparse third-gender "
                "label and does not estimate gender identity."
            ),
            (
                "This report selects a development challenger on validation; "
                "the current Naampy runtime does not serve this model."
            ),
            (
                "The selected candidate artifact is not persisted or frozen. Until "
                "it is frozen, the calibration and test partitions must not be "
                "transformed, inspected by target, fitted, or scored."
            ),
        ],
        "data": {
            "filename": arguments.data.name,
            "sha256": file_sha256(arguments.data),
            "usable_unique_names": len(names),
            "artifact_encoding": "gzip-compressed CSV",
            "source_row_limit": arguments.max_rows,
        },
        "model": {
            "architecture": {
                "type": "character_tfidf_logistic_regression",
                "analyzer": "character",
                "ngram_range": [
                    arguments.minimum_ngram_length,
                    arguments.maximum_ngram_length,
                ],
                "minimum_document_frequency": (arguments.minimum_document_frequency),
                "sublinear_term_frequency": True,
                "tfidf_normalization": "l2",
                "classifier": "logistic_regression",
                "penalty": "l2",
                "solver": "liblinear",
                "feature_count": len(selected_vectorizer.vocabulary_),
                "numeric_array_bytes": coefficient_bytes,
            },
            "selected_minimum_training_name_support": selected_support_floor,
            "selected_inverse_regularization_strength": (selected_classifier.C),
            "selection_metric": ("validation_record_weighted_expected_binary_log_loss"),
            "selection_history": selection_history,
            "candidate_artifact_status": "not_persisted_not_frozen",
        },
        "training": {
            "model_random_seed": arguments.model_seed,
            "maximum_iterations": arguments.maximum_iterations,
            "minimum_training_name_support_grid": sorted(
                set(arguments.minimum_training_name_supports)
            ),
            "skipped_minimum_training_name_supports": skipped_support_floors,
            "selected_training_name_count": selected_training_name_count,
            "excluded_training_name_count": (
                len(split.training) - selected_training_name_count
            ),
            "selected_represented_record_fraction": (
                selected_represented_record_fraction
            ),
            "training_weighting": (
                "female and male registration-record counts, rescaled to mean optimization "
                "weight one"
            ),
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        "split": {
            "method": "stratified disjoint unique-name split",
            "seed": arguments.split_seed,
            "fractions": {
                "training": 0.70,
                "validation": 0.10,
                "calibration": 0.10,
                "test": 0.10,
            },
            "strata": {
                "record_count_rank_bins": 100,
                "female_proportion_bins": 10,
            },
            "development_summary": {
                "training": partition_summary(split.training, proportions, counts),
                "validation": partition_summary(split.validation, proportions, counts),
            },
            "reserved_name_counts": {
                "calibration": len(split.calibration),
                "test": len(split.test),
            },
            "membership_sha256": {
                "training": name_membership_sha256(names, split.training),
                "validation": name_membership_sha256(names, split.validation),
                "calibration": name_membership_sha256(names, split.calibration),
                "test": name_membership_sha256(names, split.test),
            },
        },
        "reserved_partitions": {
            "policy": (
                "Do not transform, inspect targets for, fit on, or score calibration "
                "or test until the candidate artifact and selection are frozen."
            ),
            "calibration_status": "untouched",
            "test_status": "untouched",
        },
        "calibration": {"status": "reserved_untouched"},
        "metrics": {
            "selected_validation_raw": {
                "point_estimates": validation_metric_report["point_estimates"],
                "intervals": validation_metric_report["intervals"],
            },
        },
        "runtime": {
            "selection_seconds": selection_seconds,
            "selected_training_feature_fit_seconds": (selected_feature_fit_seconds),
            "validation_names": len(split.validation),
            "validation_feature_nonzero_values": selected_validation_features.nnz,
            "selected_validation_transform_seconds": (
                selected_validation_transform_seconds
            ),
            "validation_prediction_seconds": validation_prediction_seconds,
            "validation_transform_and_prediction_seconds": (
                selected_validation_transform_seconds + validation_prediction_seconds
            ),
            "validation_names_per_transform_and_prediction_second": (
                len(split.validation)
                / (
                    selected_validation_transform_seconds
                    + validation_prediction_seconds
                )
            ),
            "total_command_seconds": perf_counter() - command_started,
        },
    }
    write_json_atomic(report, arguments.output)
    LOGGER.info(
        "Selected C=%s and wrote challenger report to %s",
        selected_classifier.C,
        arguments.output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
