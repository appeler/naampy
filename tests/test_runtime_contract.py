from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

import naampy.inference as inference
from naampy import estimate_first_name_pattern, lookup_first_name_composition
from naampy._lookup_bundle import (
    LOOKUP_ARTIFACT_VERSION,
    LOOKUP_LABEL_SOURCE,
    LOOKUP_MINIMUM_BINARY_LABEL_RECORD_COUNT,
    LOOKUP_REFERENCE_POPULATION,
    LOOKUP_TABLE_SCHEMA,
    THIRD_GENDER_EXCLUSION,
    load_default_lookup_bundle,
    load_lookup_bundle,
)
from naampy._model_bundle import (
    load_default_model_bundle,
    load_model_bundle,
    parse_model_manifest,
)
from naampy.nnets import CharacterBiLSTM, encode_normalized_name

if TYPE_CHECKING:
    from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_model_bundle(directory: Path) -> tuple[Path, dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for seed, output_bias in [(0, 0.0), (1, math.log(4.0))]:
        model = CharacterBiLSTM(
            vocabulary_size=27,
            output_dimension=1,
            embedding_dimension=3,
            hidden_dimension=4,
            layer_count=1,
            dropout_probability=0.0,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.fc.bias.fill_(output_bias)
        filename = f"member_seed_{seed}.safetensors"
        member_path = directory / filename
        save_file(model.state_dict(), member_path)
        members.append(
            {
                "filename": filename,
                "sha256": _sha256(member_path),
                "training_seed": seed,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_version": "fixture-v1",
        "score_target": "female share among female and male source labels",
        "reference_population": "fixture electoral-roll records",
        "label_source": "fixture source-reported electoral-roll labels",
        "architecture": {
            "vocabulary": "abcdefghijklmnopqrstuvwxyz",
            "embedding_dimension": 3,
            "hidden_dimension": 4,
            "layer_count": 1,
            "dropout_probability": 0.0,
        },
        "ensemble": {"method": "equal-probability-mean", "members": members},
        "calibration": {
            "method": "positive-slope-logit-affine",
            "slope": 2.0,
            "intercept": -0.1,
            "population": "fixture held-out calibration records",
        },
    }
    manifest_path = directory / "first_name_pattern_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _write_lookup_bundle(directory: Path) -> tuple[Path, Path]:
    table = pa.Table.from_pylist(
        [
            {
                "normalized_name": "priya",
                "female_label_record_count": 1100,
                "male_label_record_count": 300,
            },
            {
                "normalized_name": "rahul",
                "female_label_record_count": 0,
                "male_label_record_count": 1001,
            },
        ],
        schema=LOOKUP_TABLE_SCHEMA,
    )
    table_path = directory / "first_name_composition.parquet"
    pq.write_table(table, table_path, compression="zstd")
    manifest = {
        "schema_version": 1,
        "artifact_version": "v2_1k-global-binary-v1",
        "artifact_role": "global_first_name_label_composition_lookup",
        "reference_population": LOOKUP_REFERENCE_POPULATION,
        "label_source": LOOKUP_LABEL_SOURCE,
        "dataset": {
            "persistent_id": "doi:10.7910/DVN/WZGJBM",
            "dataset_version": "3.2",
            "file_id": 4965695,
            "file_version": 1,
            "filename": "in_rolls_state_year_fn_naampy_x1k.csv.gz",
            "license": "CC0-1.0",
        },
        "label_contract": {
            "included_source_labels": ["n_female", "n_male"],
            "excluded_source_labels": ["n_third_gender"],
            "target": "female and male electoral-roll source-label record counts",
            "unit": "represented electoral-roll registration record",
            "third_gender_exclusion": THIRD_GENDER_EXCLUSION,
            "interpretation": (
                "aggregate first-name composition in the reference data; not an "
                "individual's gender identity"
            ),
        },
        "privacy": {
            "classification": "public",
            "publication_intent": "public_release_candidate",
            "release_grain": "one_nonoverlapping_global_level",
            "rationale": (
                "only global name aggregates with at least 1000 included source-label "
                "records are released; state and birth-year hierarchies are absent"
            ),
        },
        "source": {
            "local_filename": "in_rolls_state_year_fn_naampy_x1k.csv.gz",
            "format": "gzip_csv",
            "size_bytes": 62406025,
            "sha256": (
                "2f72d8555ee6da837f94adb93fad6661a80fa141abc1eda7fa4e17f565fe4417"
            ),
            "md5": "822fa00e3f54ac606b6e578d27ef3904",
            "totals": {
                "row_count": 2,
                "state_count": 31,
                "minimum_birth_year": 1887,
                "maximum_birth_year": 2017,
                "distinct_source_name_count": 2,
                "female_label_record_count": 1100,
                "male_label_record_count": 1301,
                "represented_binary_label_record_count": 2401,
            },
        },
        "output": {
            "filename": table_path.name,
            "format": "parquet_zstd",
            "size_bytes": table_path.stat().st_size,
            "sha256": _sha256(table_path),
            "totals": {
                "row_count": 2,
                "female_label_record_count": 1100,
                "male_label_record_count": 1301,
                "represented_binary_label_record_count": 2401,
            },
        },
        "data_contract": {
            "unit_of_observation": "one globally aggregated normalized first name",
            "row_order": "normalized_name ascending by Unicode code point",
            "normalization": (
                "Unicode NFC, strip surrounding whitespace, Unicode casefold, then "
                "require nonempty ASCII letters a-z"
            ),
            "supported_scripts": ["Latn"],
            "aggregation": (
                "sum included female and male source-label counts across all state "
                "and birth-year cells after normalization"
            ),
            "geography": {
                "source_state_count": 31,
                "release_level": "global_only",
                "published_geography_columns": [],
            },
            "birth_year": {
                "source_minimum": 1887,
                "source_maximum": 2017,
                "release_level": "all_years_combined",
                "published_birth_year_columns": [],
            },
            "support": {
                "minimum_represented_binary_label_records": 1000,
                "support_fields": [
                    "female_label_record_count",
                    "male_label_record_count",
                ],
                "rule": (
                    "female_label_record_count + male_label_record_count must be "
                    "at least 1000"
                ),
                "normalized_name_count_before_support_filter": 2,
                "normalized_name_count_excluded_below_support": 0,
            },
        },
        "arrow_schema": [
            {"name": "normalized_name", "type": "string", "nullable": False},
            {
                "name": "female_label_record_count",
                "type": "int64",
                "nullable": False,
            },
            {
                "name": "male_label_record_count",
                "type": "int64",
                "nullable": False,
            },
        ],
        "source_code_sha256": {"model_training/export_lookup_table.py": "c" * 64},
        "software_versions": {"pyarrow": pa.__version__},
    }
    manifest_path = directory / "first_name_composition_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, table_path


@pytest.fixture
def configured_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_manifest_path, _ = _write_model_bundle(tmp_path)
    model_bundle = load_model_bundle(
        model_manifest_path,
        artifact_resolver=lambda filename: tmp_path / filename,
        repository="fixture/models",
        revision="a" * 40,
    )
    lookup_manifest_path, _ = _write_lookup_bundle(tmp_path)
    lookup_bundle = load_lookup_bundle(
        lookup_manifest_path,
        artifact_resolver=lambda filename: tmp_path / filename,
        repository="fixture/table",
        revision="b" * 40,
    )
    inference._cached_default_model_bundle.cache_clear()
    inference._cached_default_lookup_bundle.cache_clear()
    inference._load_composition_by_name.cache_clear()
    monkeypatch.setattr(inference, "load_default_model_bundle", lambda: model_bundle)
    monkeypatch.setattr(inference, "load_default_lookup_bundle", lambda: lookup_bundle)


def test_estimate_contract_and_abstention_precedence(configured_runtime: None) -> None:
    names = [
        None,
        "  ",
        "An Kit",
        "An-kit",
        "Ésha",
        "देवraj",
        "देव",
        "123",
        "ab",
        " abc ",
        "abcdefghijklmnopqrs",
        "abcdefghijklmnopqrst",
        "shaaan",
    ]
    estimates = estimate_first_name_pattern(names)

    assert list(estimates.columns) == list(inference.ESTIMATE_COLUMNS)
    assert estimates["abstention_reason"].tolist() == [
        "missing-name",
        "missing-name",
        "not-single-first-name",
        "unsupported-characters",
        "unsupported-characters",
        "unsupported-script",
        "unsupported-script",
        "unsupported-characters",
        "outside-training-scope",
        pd.NA,
        pd.NA,
        "outside-training-scope",
        "outside-training-scope",
    ]
    assert estimates["detected_script"].tolist()[4:8] == [
        "Latn",
        "mixed",
        "Deva",
        pd.NA,
    ]
    assert estimates["script_supported"].tolist()[4:8] == [True, False, False, pd.NA]
    assert bool(estimates.loc[2, "script_supported"]) is True
    assert estimates.loc[9, "normalized_name"] == "abc"
    assert estimates.loc[[9, 10], "female_label_score"].notna().all()
    assert estimates.loc[:8, "female_label_score"].isna().all()
    assert estimates.loc[[11, 12], "female_label_score"].isna().all()


def test_estimate_averages_probabilities_then_calibrates(
    configured_runtime: None,
) -> None:
    estimates = estimate_first_name_pattern("Priya")
    ensemble_mean = (0.5 + 0.8) / 2
    expected = 1 / (
        1 + math.exp(-(2 * math.log(ensemble_mean / (1 - ensemble_mean)) - 0.1))
    )

    assert estimates.loc[0, "input_name"] == "Priya"
    assert estimates.loc[0, "normalized_name"] == "priya"
    assert estimates.loc[0, "female_label_score"] == pytest.approx(expected)
    assert bool(estimates.loc[0, "abstained"]) is False
    assert "predicted_label" not in estimates
    assert "confidence" not in estimates
    assert estimates.loc[0, "model_revision"] == "a" * 40
    assert estimates.loc[0, "model_bundle_sha256"] == (
        "5a22c4c67cc84b521a5f3aa93c8c5f6ec7bbdb9041c561297f23f1f68204daea"
    )


def test_model_bundle_digest_changes_with_validated_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _write_model_bundle(tmp_path)
    original_bundle = load_model_bundle(
        manifest_path,
        artifact_resolver=lambda filename: tmp_path / filename,
        repository="fixture/models",
        revision="local-artifact-directory",
    )
    original_digest = inference._model_bundle_sha256(original_bundle)

    manifest["calibration"]["intercept"] = 0.2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed_bundle = load_model_bundle(
        manifest_path,
        artifact_resolver=lambda filename: tmp_path / filename,
        repository="fixture/models",
        revision="local-artifact-directory",
    )

    assert inference._model_bundle_sha256(changed_bundle) != original_digest
    assert len(original_digest) == 64
    assert set(original_digest) <= set("0123456789abcdef")


def test_local_overrides_report_local_artifact_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_directory = tmp_path / "model"
    lookup_directory = tmp_path / "lookup"
    model_directory.mkdir()
    lookup_directory.mkdir()
    _write_model_bundle(model_directory)
    _write_lookup_bundle(lookup_directory)
    monkeypatch.setenv("NAAMPY_MODEL_DIR", str(model_directory))
    monkeypatch.setenv("NAAMPY_LOOKUP_TABLE_DIR", str(lookup_directory))

    model_bundle = load_default_model_bundle()
    lookup_bundle = load_default_lookup_bundle()

    assert model_bundle.repository == "local-artifact-directory"
    assert model_bundle.revision == "local-artifact-directory"
    assert lookup_bundle.repository == "local-artifact-directory"
    assert lookup_bundle.revision == "local-artifact-directory"

    monkeypatch.setattr(inference, "load_default_model_bundle", lambda: model_bundle)
    monkeypatch.setattr(inference, "load_default_lookup_bundle", lambda: lookup_bundle)
    inference._cached_default_model_bundle.cache_clear()
    inference._cached_default_lookup_bundle.cache_clear()

    estimates = estimate_first_name_pattern("Priya")
    composition = lookup_first_name_composition("Priya")

    assert estimates.loc[0, "model_repository"] == "local-artifact-directory"
    assert estimates.loc[0, "model_revision"] == "local-artifact-directory"
    model_bundle_sha256 = estimates.loc[0, "model_bundle_sha256"]
    assert isinstance(model_bundle_sha256, str)
    assert len(model_bundle_sha256) == 64
    assert composition.loc[0, "lookup_artifact_repository"] == (
        "local-artifact-directory"
    )
    assert composition.loc[0, "lookup_artifact_revision"] == (
        "local-artifact-directory"
    )

    inference._cached_default_model_bundle.cache_clear()
    inference._cached_default_lookup_bundle.cache_clear()
    inference._load_composition_by_name.cache_clear()


def test_estimate_has_explicit_nullable_dtypes(configured_runtime: None) -> None:
    estimates = estimate_first_name_pattern(["priya", None])

    assert str(estimates["female_label_score"].dtype) == "Float64"
    assert str(estimates["abstained"].dtype) == "boolean"
    assert str(estimates["script_supported"].dtype) == "boolean"
    assert str(estimates["model_manifest_schema_version"].dtype) == "Int64"
    for column in estimates.columns:
        assert estimates[column].dtype != object


def test_lookup_is_exact_and_keeps_source_label_semantics(
    configured_runtime: None,
) -> None:
    composition = lookup_first_name_composition(
        [" Priya ", "unknown", "Ésha", "देव", None]
    )

    assert list(composition.columns) == list(inference.LOOKUP_COLUMNS)
    assert composition["lookup_status"].tolist() == [
        "matched",
        "not-found",
        "abstained",
        "abstained",
        "abstained",
    ]
    assert composition["lookup_reason"].tolist() == [
        pd.NA,
        "not-released",
        "unsupported-characters",
        "unsupported-script",
        "missing-name",
    ]
    assert composition.loc[0, "female_label_record_count"] == 1100
    assert composition.loc[0, "male_label_record_count"] == 300
    assert composition.loc[0, "represented_binary_label_record_count"] == 1400
    assert composition.loc[
        0, "female_label_share_among_binary_labels"
    ] == pytest.approx(11 / 14)
    assert composition.loc[0, "male_label_share_among_binary_labels"] == pytest.approx(
        3 / 14
    )
    assert "third_gender_label_record_count" not in composition
    assert composition.loc[0, "third_gender_exclusion"] == THIRD_GENDER_EXCLUSION
    assert "female_label_score" not in composition
    assert composition.loc[0, "lookup_artifact_revision"] == "b" * 40


def test_lookup_has_explicit_nullable_dtypes(configured_runtime: None) -> None:
    composition = lookup_first_name_composition(["priya", None])

    for column in (
        "female_label_record_count",
        "male_label_record_count",
        "represented_binary_label_record_count",
        "lookup_manifest_schema_version",
    ):
        assert str(composition[column].dtype) == "Int64"
    for column in (
        "female_label_share_among_binary_labels",
        "male_label_share_among_binary_labels",
    ):
        assert str(composition[column].dtype) == "Float64"
    assert str(composition["script_supported"].dtype) == "boolean"
    for column in composition.columns:
        assert composition[column].dtype != object


def test_empty_inputs_do_not_resolve_artifacts() -> None:
    with (
        patch.object(
            inference,
            "_cached_default_model_bundle",
            side_effect=AssertionError("model artifacts were resolved"),
        ),
        patch.object(
            inference,
            "_cached_default_lookup_bundle",
            side_effect=AssertionError("lookup artifacts were resolved"),
        ),
    ):
        estimates = estimate_first_name_pattern([])
        compositions = lookup_first_name_composition([])

    assert estimates.empty
    assert list(estimates.columns) == list(inference.ESTIMATE_COLUMNS)
    assert compositions.empty
    assert list(compositions.columns) == list(inference.LOOKUP_COLUMNS)


def test_character_encoder_never_drops_unsupported_characters() -> None:
    with pytest.raises(KeyError):
        encode_normalized_name("ésha")


def test_lookup_table_is_read_once_per_bundle(configured_runtime: None) -> None:
    with patch("naampy.inference.pd.read_parquet", wraps=pd.read_parquet) as read_table:
        lookup_first_name_composition("priya")
        lookup_first_name_composition("rahul")

    read_table.assert_called_once()


def test_model_bundle_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest_path, manifest = _write_model_bundle(tmp_path)
    manifest["ensemble"]["members"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        load_model_bundle(
            manifest_path,
            artifact_resolver=lambda filename: tmp_path / filename,
            repository="fixture/models",
            revision="a" * 40,
        )


def test_model_manifest_rejects_unknown_schema_keys(tmp_path: Path) -> None:
    manifest_path, manifest = _write_model_bundle(tmp_path)
    manifest["surprise"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="model manifest keys"):
        parse_model_manifest(manifest_path)


def test_model_bundle_rejects_checkpoint_schema_mismatch(tmp_path: Path) -> None:
    manifest_path, manifest = _write_model_bundle(tmp_path)
    first_member = manifest["ensemble"]["members"][0]
    first_path = tmp_path / first_member["filename"]
    save_file({"not_a_model_weight": torch.zeros(1)}, first_path)
    first_member["sha256"] = _sha256(first_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its architecture"):
        load_model_bundle(
            manifest_path,
            artifact_resolver=lambda filename: tmp_path / filename,
            repository="fixture/models",
            revision="a" * 40,
        )


def test_lookup_bundle_rejects_hash_mismatch(tmp_path: Path) -> None:
    manifest_path, _ = _write_lookup_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        load_lookup_bundle(
            manifest_path,
            artifact_resolver=lambda filename: tmp_path / filename,
            repository="fixture/table",
            revision="b" * 40,
        )


def test_runtime_lookup_contract_matches_exporter() -> None:
    from model_training import export_lookup_table

    assert LOOKUP_ARTIFACT_VERSION == export_lookup_table.ARTIFACT_VERSION
    assert (
        LOOKUP_MINIMUM_BINARY_LABEL_RECORD_COUNT
        == export_lookup_table.MINIMUM_NAME_SUPPORT
    )
    assert LOOKUP_TABLE_SCHEMA == export_lookup_table.LOOKUP_TABLE_SCHEMA


def test_lookup_manifest_requires_explicit_third_gender_exclusion(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_lookup_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_contract"]["third_gender_exclusion"] = "excluded"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source-label contract is not recognized"):
        load_lookup_bundle(
            manifest_path,
            artifact_resolver=lambda filename: tmp_path / filename,
            repository="fixture/table",
            revision="b" * 40,
        )


def test_lookup_manifest_requires_exact_release_support(tmp_path: Path) -> None:
    manifest_path, _ = _write_lookup_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_contract"]["support"]["minimum_represented_binary_label_records"] = (
        999
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must be exactly 1000"):
        load_lookup_bundle(
            manifest_path,
            artifact_resolver=lambda filename: tmp_path / filename,
            repository="fixture/table",
            revision="b" * 40,
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    "normalized_name": "rahul",
                    "female_label_record_count": 0,
                    "male_label_record_count": 1001,
                },
                {
                    "normalized_name": "priya",
                    "female_label_record_count": 1100,
                    "male_label_record_count": 300,
                },
            ],
            "must be ascending",
        ),
        (
            [
                {
                    "normalized_name": "Ésha",
                    "female_label_record_count": 1100,
                    "male_label_record_count": 300,
                }
            ],
            "lowercase ASCII a-z",
        ),
        (
            [
                {
                    "normalized_name": "priya",
                    "female_label_record_count": 11,
                    "male_label_record_count": 3,
                }
            ],
            "below minimum_represented_binary_label_records",
        ),
    ],
)
def test_lookup_bundle_rejects_invalid_public_table_contract(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    message: str,
) -> None:
    manifest_path, table_path = _write_lookup_bundle(tmp_path)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=LOOKUP_TABLE_SCHEMA),
        table_path,
        compression="zstd",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    female_total = sum(row["female_label_record_count"] for row in rows)
    male_total = sum(row["male_label_record_count"] for row in rows)
    manifest["output"]["size_bytes"] = table_path.stat().st_size
    manifest["output"]["sha256"] = _sha256(table_path)
    manifest["output"]["totals"] = {
        "row_count": len(rows),
        "female_label_record_count": female_total,
        "male_label_record_count": male_total,
        "represented_binary_label_record_count": female_total + male_total,
    }
    support = manifest["data_contract"]["support"]
    support["normalized_name_count_before_support_filter"] = len(rows)
    support["normalized_name_count_excluded_below_support"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_lookup_bundle(
            manifest_path,
            artifact_resolver=lambda filename: tmp_path / filename,
            repository="fixture/table",
            revision="b" * 40,
        )
