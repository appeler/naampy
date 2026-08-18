"""Public first-name pattern estimation and exact-composition lookup APIs."""

from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from typing import Any, cast

import pandas as pd

from . import _resources
from ._lookup_bundle import (
    FirstNameCompositionBundle,
    load_default_lookup_bundle,
)
from ._model_bundle import (
    NamePatternModelBundle,
    calibrated_ensemble_score,
    load_default_model_bundle,
)
from ._normalization import (
    NameCollection,
    NormalizedFirstName,
    coerce_name_collection,
    normalize_for_lookup,
    normalize_for_model,
)
from .nnets import encode_normalized_name, pad_encoded_names

_INFERENCE_BATCH_SIZE = 1024

ESTIMATE_COLUMNS: tuple[str, ...] = (
    "input_name",
    "normalized_name",
    "female_label_score",
    "abstained",
    "abstention_reason",
    "detected_script",
    "script_supported",
    "score_target",
    "reference_population",
    "label_source",
    "calibration_population",
    "model_manifest_schema_version",
    "model_version",
    "model_repository",
    "model_revision",
    "model_bundle_sha256",
)

LOOKUP_COLUMNS: tuple[str, ...] = (
    "input_name",
    "normalized_name",
    "female_label_record_count",
    "male_label_record_count",
    "represented_binary_label_record_count",
    "female_label_share_among_binary_labels",
    "male_label_share_among_binary_labels",
    "lookup_status",
    "lookup_reason",
    "detected_script",
    "script_supported",
    "reference_population",
    "label_source",
    "third_gender_exclusion",
    "lookup_manifest_schema_version",
    "lookup_artifact_version",
    "lookup_artifact_repository",
    "lookup_artifact_revision",
    "lookup_artifact_sha256",
)


def estimate_first_name_pattern(names: NameCollection) -> pd.DataFrame:
    """Estimate the female source-label share associated with name patterns.

    This is a population-level name-pattern estimate. It is not an observation of
    any person's gender and must not be used for individual or consequential
    decisions. Inputs outside the frozen training domain are returned as explicit
    abstentions rather than transformed, truncated, or guessed.

    Args:
        names: One first name or a one-dimensional collection of first names.

    Returns:
        A new DataFrame with a calibrated nullable score, eligibility status, and
        immutable model provenance for every input row.
    """
    normalized_names = [
        normalize_for_model(name) for name in coerce_name_collection(names)
    ]
    if not normalized_names:
        return _empty_estimate_frame()
    model_bundle = _cached_default_model_bundle()
    scores = _score_eligible_names(normalized_names, model_bundle)
    manifest = model_bundle.manifest
    bundle_sha256 = _model_bundle_sha256(model_bundle)

    return pd.DataFrame(
        {
            "input_name": pd.array(
                [name.name for name in normalized_names], dtype="string"
            ),
            "normalized_name": pd.array(
                [name.normalized_name for name in normalized_names], dtype="string"
            ),
            "female_label_score": pd.array(scores, dtype="Float64"),
            "abstained": pd.array(
                [name.abstained for name in normalized_names], dtype="boolean"
            ),
            "abstention_reason": pd.array(
                [name.abstention_reason for name in normalized_names], dtype="string"
            ),
            "detected_script": pd.array(
                [name.detected_script for name in normalized_names], dtype="string"
            ),
            "script_supported": pd.array(
                [name.script_supported for name in normalized_names], dtype="boolean"
            ),
            "score_target": pd.array(
                [manifest.score_target] * len(normalized_names), dtype="string"
            ),
            "reference_population": pd.array(
                [manifest.reference_population] * len(normalized_names), dtype="string"
            ),
            "label_source": pd.array(
                [manifest.label_source] * len(normalized_names), dtype="string"
            ),
            "calibration_population": pd.array(
                [manifest.calibration.population] * len(normalized_names),
                dtype="string",
            ),
            "model_manifest_schema_version": pd.array(
                [manifest.schema_version] * len(normalized_names), dtype="Int64"
            ),
            "model_version": pd.array(
                [manifest.model_version] * len(normalized_names), dtype="string"
            ),
            "model_repository": pd.array(
                [model_bundle.repository] * len(normalized_names), dtype="string"
            ),
            "model_revision": pd.array(
                [model_bundle.revision] * len(normalized_names), dtype="string"
            ),
            "model_bundle_sha256": pd.array(
                [bundle_sha256] * len(normalized_names), dtype="string"
            ),
        },
        columns=ESTIMATE_COLUMNS,
    )


def lookup_first_name_composition(
    names: NameCollection,
) -> pd.DataFrame:
    """Return exact electoral-roll source-label composition for first names.

    This function performs only an exact table lookup after lossless normalization.
    It never falls back to a learned estimate.

    Args:
        names: One first name or a one-dimensional collection of first names.

    Returns:
        A new DataFrame with nullable counts, shares, lookup status, and immutable
        table provenance for every input row.
    """
    input_names = coerce_name_collection(names)
    if not input_names:
        return _empty_lookup_frame()
    lookup_bundle = _cached_default_lookup_bundle()
    manifest = lookup_bundle.manifest
    normalized_names = [
        normalize_for_lookup(name, manifest.supported_scripts) for name in input_names
    ]
    composition_by_name = _load_composition_by_name(lookup_bundle)

    rows = [_lookup_row(name, composition_by_name) for name in normalized_names]
    row_count = len(rows)
    return pd.DataFrame(
        {
            "input_name": pd.array([row["input_name"] for row in rows], dtype="string"),
            "normalized_name": pd.array(
                [row["normalized_name"] for row in rows], dtype="string"
            ),
            "female_label_record_count": pd.array(
                [row["female_label_record_count"] for row in rows], dtype="Int64"
            ),
            "male_label_record_count": pd.array(
                [row["male_label_record_count"] for row in rows], dtype="Int64"
            ),
            "represented_binary_label_record_count": pd.array(
                [row["represented_binary_label_record_count"] for row in rows],
                dtype="Int64",
            ),
            "female_label_share_among_binary_labels": pd.array(
                [row["female_label_share_among_binary_labels"] for row in rows],
                dtype="Float64",
            ),
            "male_label_share_among_binary_labels": pd.array(
                [row["male_label_share_among_binary_labels"] for row in rows],
                dtype="Float64",
            ),
            "lookup_status": pd.array(
                [row["lookup_status"] for row in rows], dtype="string"
            ),
            "lookup_reason": pd.array(
                [row["lookup_reason"] for row in rows], dtype="string"
            ),
            "detected_script": pd.array(
                [row["detected_script"] for row in rows], dtype="string"
            ),
            "script_supported": pd.array(
                [row["script_supported"] for row in rows], dtype="boolean"
            ),
            "reference_population": pd.array(
                [manifest.reference_population] * row_count, dtype="string"
            ),
            "label_source": pd.array(
                [manifest.label_source] * row_count, dtype="string"
            ),
            "third_gender_exclusion": pd.array(
                [manifest.label_contract.third_gender_exclusion] * row_count,
                dtype="string",
            ),
            "lookup_manifest_schema_version": pd.array(
                [manifest.schema_version] * row_count, dtype="Int64"
            ),
            "lookup_artifact_version": pd.array(
                [manifest.artifact_version] * row_count, dtype="string"
            ),
            "lookup_artifact_repository": pd.array(
                [lookup_bundle.repository] * row_count, dtype="string"
            ),
            "lookup_artifact_revision": pd.array(
                [lookup_bundle.revision] * row_count, dtype="string"
            ),
            "lookup_artifact_sha256": pd.array(
                [manifest.output.sha256] * row_count, dtype="string"
            ),
        },
        columns=LOOKUP_COLUMNS,
    )


@cache
def _cached_default_model_bundle() -> NamePatternModelBundle:
    bundle = load_default_model_bundle()
    if os.environ.get(_resources.MODEL_DIRECTORY_ENVIRONMENT_VARIABLE) is None:
        return bundle
    return NamePatternModelBundle(
        manifest=bundle.manifest,
        models=bundle.models,
        repository="local-artifact-directory",
        revision="local-artifact-directory",
    )


@cache
def _cached_default_lookup_bundle() -> FirstNameCompositionBundle:
    bundle = load_default_lookup_bundle()
    if os.environ.get(_resources.LOOKUP_TABLE_DIRECTORY_ENVIRONMENT_VARIABLE) is None:
        return bundle
    return FirstNameCompositionBundle(
        manifest=bundle.manifest,
        table_path=bundle.table_path,
        repository="local-artifact-directory",
        revision="local-artifact-directory",
    )


def _model_bundle_sha256(bundle: NamePatternModelBundle) -> str:
    """Return a stable digest of the validated model bundle semantics."""
    manifest = bundle.manifest
    architecture = manifest.architecture
    calibration = manifest.calibration
    fingerprint = {
        "schema_version": manifest.schema_version,
        "model_version": manifest.model_version,
        "score_target": manifest.score_target,
        "reference_population": manifest.reference_population,
        "label_source": manifest.label_source,
        "architecture": {
            "vocabulary": architecture.vocabulary,
            "embedding_dimension": architecture.embedding_dimension,
            "hidden_dimension": architecture.hidden_dimension,
            "layer_count": architecture.layer_count,
            "dropout_probability": architecture.dropout_probability,
        },
        "ensemble": {
            "method": manifest.ensemble_method,
            "members": [
                {
                    "filename": member.filename,
                    "sha256": member.sha256,
                    "training_seed": member.training_seed,
                }
                for member in manifest.ensemble_members
            ],
        },
        "calibration": {
            "method": calibration.method,
            "slope": calibration.slope,
            "intercept": calibration.intercept,
            "population": calibration.population,
        },
    }
    canonical_json = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _empty_estimate_frame() -> pd.DataFrame:
    integer_columns = {"model_manifest_schema_version"}
    float_columns = {"female_label_score"}
    boolean_columns = {"abstained", "script_supported"}
    return pd.DataFrame(
        {
            column: pd.Series(
                dtype=_nullable_dtype(
                    column,
                    integer_columns=integer_columns,
                    float_columns=float_columns,
                    boolean_columns=boolean_columns,
                )
            )
            for column in ESTIMATE_COLUMNS
        },
        columns=ESTIMATE_COLUMNS,
    )


def _empty_lookup_frame() -> pd.DataFrame:
    integer_columns = {
        "female_label_record_count",
        "male_label_record_count",
        "represented_binary_label_record_count",
        "lookup_manifest_schema_version",
    }
    float_columns = {
        "female_label_share_among_binary_labels",
        "male_label_share_among_binary_labels",
    }
    boolean_columns = {"script_supported"}
    return pd.DataFrame(
        {
            column: pd.Series(
                dtype=_nullable_dtype(
                    column,
                    integer_columns=integer_columns,
                    float_columns=float_columns,
                    boolean_columns=boolean_columns,
                )
            )
            for column in LOOKUP_COLUMNS
        },
        columns=LOOKUP_COLUMNS,
    )


def _nullable_dtype(
    column: str,
    *,
    integer_columns: set[str],
    float_columns: set[str],
    boolean_columns: set[str],
) -> str:
    if column in integer_columns:
        return "Int64"
    if column in float_columns:
        return "Float64"
    if column in boolean_columns:
        return "boolean"
    return "string"


def _score_eligible_names(
    normalized_names: list[NormalizedFirstName], bundle: NamePatternModelBundle
) -> list[float | None]:
    scores: list[float | None] = [None] * len(normalized_names)
    eligible_positions = [
        position for position, name in enumerate(normalized_names) if not name.abstained
    ]
    for batch_start in range(0, len(eligible_positions), _INFERENCE_BATCH_SIZE):
        batch_positions = eligible_positions[
            batch_start : batch_start + _INFERENCE_BATCH_SIZE
        ]
        encoded_batch = [
            encode_normalized_name(
                cast("str", normalized_names[position].normalized_name)
            )
            for position in batch_positions
        ]
        encoded_names, name_lengths = pad_encoded_names(encoded_batch)
        batch_scores = calibrated_ensemble_score(
            bundle, encoded_names, name_lengths
        ).tolist()
        for position, score in zip(batch_positions, batch_scores, strict=True):
            scores[position] = float(score)
    return scores


@cache
def _load_composition_by_name(
    bundle: FirstNameCompositionBundle,
) -> pd.DataFrame:
    return cast(
        "pd.DataFrame",
        pd.read_parquet(
            bundle.table_path,
            columns=[
                "normalized_name",
                "female_label_record_count",
                "male_label_record_count",
            ],
        ).set_index("normalized_name"),
    )


def _lookup_row(
    normalized_name: NormalizedFirstName, composition_by_name: pd.DataFrame
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "input_name": normalized_name.name,
        "normalized_name": normalized_name.normalized_name,
        "female_label_record_count": None,
        "male_label_record_count": None,
        "represented_binary_label_record_count": None,
        "female_label_share_among_binary_labels": None,
        "male_label_share_among_binary_labels": None,
        "lookup_status": "abstained" if normalized_name.abstained else "not-found",
        "lookup_reason": normalized_name.abstention_reason
        if normalized_name.abstained
        else "not-released",
        "detected_script": normalized_name.detected_script,
        "script_supported": normalized_name.script_supported,
    }
    if normalized_name.abstained:
        return row

    if normalized_name.normalized_name is None:
        raise RuntimeError(
            "eligible normalization unexpectedly produced a missing name"
        )
    if normalized_name.normalized_name not in composition_by_name.index:
        return row
    composition = cast(
        "pd.Series", composition_by_name.loc[normalized_name.normalized_name]
    )
    female_count = int(cast("int", composition.at["female_label_record_count"]))
    male_count = int(cast("int", composition.at["male_label_record_count"]))
    represented_count = female_count + male_count
    row.update(
        {
            "female_label_record_count": female_count,
            "male_label_record_count": male_count,
            "represented_binary_label_record_count": represented_count,
            "female_label_share_among_binary_labels": female_count / represented_count,
            "male_label_share_among_binary_labels": male_count / represented_count,
            "lookup_status": "matched",
            "lookup_reason": None,
        }
    )
    return row
