import json
import subprocess
import sys
from copy import deepcopy

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_training.export_training_data import (
    PARTITION_NAMES,
    TRAINING_DATA_SCHEMA,
    export_training_data,
    file_sha256,
    validate_training_data_export,
)


def _write_source_data(path):
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
    rows = []
    for index, name in enumerate(names):
        female_count = index + 1
        male_count = 1
        rows.extend(
            [
                {
                    "state": "first",
                    "birth_year": 1990,
                    "first_name": name,
                    "n_female": female_count,
                    "n_male": male_count,
                },
                {
                    "state": "second",
                    "birth_year": 1991,
                    "first_name": name,
                    "n_female": 1,
                    "n_male": 0,
                },
            ]
        )
    rows.extend(
        [
            {
                "state": "excluded",
                "birth_year": 1990,
                "first_name": "ab",
                "n_female": 50,
                "n_male": 50,
            },
            {
                "state": "excluded",
                "birth_year": 1990,
                "first_name": "Asha",
                "n_female": 50,
                "n_male": 50,
            },
            {
                "state": "excluded",
                "birth_year": 1990,
                "first_name": "raaan",
                "n_female": 50,
                "n_male": 50,
            },
            {
                "state": "excluded",
                "birth_year": 1990,
                "first_name": None,
                "n_female": 50,
                "n_male": 50,
            },
        ]
    )
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return names


def _export(tmp_path, stem, *, minimum_name_support=1):
    source_path = tmp_path / "source.csv.gz"
    if not source_path.exists():
        _write_source_data(source_path)
    parquet_path = tmp_path / f"{stem}.parquet"
    manifest_path = tmp_path / f"{stem}.json"
    manifest = export_training_data(
        source_path,
        parquet_path,
        manifest_path,
        minimum_name_support=minimum_name_support,
        privacy_classification="private",
        publication_intent="private_model_development",
    )
    return source_path, parquet_path, manifest_path, manifest


def test_export_has_exact_schema_counts_and_exhaustive_partitions(tmp_path):
    expected_names = _write_source_data(tmp_path / "source.csv.gz")
    source_path, parquet_path, manifest_path, manifest = _export(tmp_path, "training")

    table = pq.read_table(parquet_path)

    assert table.schema == TRAINING_DATA_SCHEMA
    assert table.schema == pa.schema(
        [
            pa.field("normalized_name", pa.string(), nullable=False),
            pa.field("female_label_record_count", pa.int64(), nullable=False),
            pa.field("male_label_record_count", pa.int64(), nullable=False),
            pa.field(
                "represented_binary_label_record_count", pa.int64(), nullable=False
            ),
            pa.field("partition", pa.string(), nullable=False),
        ]
    )
    assert table["normalized_name"].to_pylist() == sorted(expected_names)
    assert len(set(table["normalized_name"].to_pylist())) == len(expected_names)
    assert set(table["partition"].to_pylist()) == set(PARTITION_NAMES)
    assert sum(
        partition["row_count"]
        for partition in manifest["output"]["partition_totals"].values()
    ) == len(expected_names)
    assert table["represented_binary_label_record_count"].to_pylist() == list(
        range(3, 23)
    )
    assert manifest["output"]["totals"] == {
        "female_label_record_count": 230,
        "male_label_record_count": 20,
        "represented_binary_label_record_count": 250,
        "row_count": 20,
    }
    assert manifest["privacy"] == {
        "classification": "private",
        "declaration_source": "explicit_export_argument",
        "publication_intent": "private_model_development",
    }
    assert manifest["data_contract"]["row_order"] == (
        "normalized_name ascending by Unicode code point"
    )
    assert manifest["label_contract"] == {
        "excluded_label_reason": (
            "n_third_gender is too sparse in the retained source for this binary "
            "target and is not included in any artifact count"
        ),
        "excluded_source_labels": ["n_third_gender"],
        "included_source_labels": ["n_female", "n_male"],
        "interpretation": (
            "aggregate electoral-roll source-label composition; not an individual's "
            "gender identity"
        ),
    }
    assert manifest["source"]["provenance"] == {
        "construction_revisions": {
            "eroll_transliteration": ("262844fdaec6ee707a87160306e139e141a52bcd"),
            "naampy": "2b15840cf0c63ddf6b5b81bf9ecf068d65d7722d",
        },
        "dataverse": {"doi": "10.7910/DVN/WZGJBM", "license": "CC0-1.0"},
    }
    assert manifest["source"]["sha256"] == file_sha256(source_path)
    assert manifest["output"]["sha256"] == file_sha256(parquet_path)
    assert manifest["source"]["filename"] == source_path.name
    assert manifest["output"]["filename"] == parquet_path.name
    assert "path" not in manifest["source"]
    assert "path" not in manifest["output"]
    assert (
        validate_training_data_export(
            parquet_path, manifest_path, source_path=source_path
        )
        == manifest
    )


def test_support_filter_preserves_full_split_memberships_and_assignments(tmp_path):
    _write_source_data(tmp_path / "source.csv.gz")
    _, full_path, _, full_manifest = _export(tmp_path, "full")
    _, filtered_path, _, filtered_manifest = _export(
        tmp_path, "filtered", minimum_name_support=10
    )

    full_table = pq.read_table(full_path)
    filtered_table = pq.read_table(filtered_path)
    full_assignments = dict(
        zip(
            full_table["normalized_name"].to_pylist(),
            full_table["partition"].to_pylist(),
            strict=True,
        )
    )
    filtered_assignments = dict(
        zip(
            filtered_table["normalized_name"].to_pylist(),
            filtered_table["partition"].to_pylist(),
            strict=True,
        )
    )

    assert filtered_table.num_rows == 13
    assert filtered_assignments == {
        name: partition
        for name, partition in full_assignments.items()
        if name in set(filtered_table["normalized_name"].to_pylist())
    }
    assert (
        filtered_manifest["split"]["pre_filter_membership_sha256"]
        == (full_manifest["split"]["pre_filter_membership_sha256"])
    )
    assert (
        filtered_manifest["split"]["exported_membership_sha256"]
        != (full_manifest["split"]["exported_membership_sha256"])
    )

    with pytest.raises(ValueError, match="excludes every usable name"):
        _export(tmp_path, "stricter", minimum_name_support=100)
    assert not (tmp_path / "stricter.parquet").exists()
    assert not (tmp_path / "stricter.json").exists()


def test_export_is_byte_deterministic_apart_from_output_path(tmp_path):
    _write_source_data(tmp_path / "source.csv.gz")
    _, first_parquet, _, first_manifest = _export(tmp_path, "first")
    _, second_parquet, _, second_manifest = _export(tmp_path, "second")

    comparable_first = deepcopy(first_manifest)
    comparable_second = deepcopy(second_manifest)
    del comparable_first["output"]["filename"]
    del comparable_second["output"]["filename"]

    assert first_parquet.read_bytes() == second_parquet.read_bytes()
    assert comparable_first == comparable_second


def test_validator_rejects_source_and_parquet_hash_changes(tmp_path):
    source_path, parquet_path, manifest_path, _ = _export(tmp_path, "training")

    source_path.write_bytes(source_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source hash"):
        validate_training_data_export(
            parquet_path, manifest_path, source_path=source_path
        )

    _write_source_data(source_path)
    source_path, parquet_path, manifest_path, _ = _export(tmp_path, "replacement")
    parquet_path.write_bytes(parquet_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Parquet hash"):
        validate_training_data_export(
            parquet_path, manifest_path, source_path=source_path
        )


def test_validator_recomputes_exported_and_pre_filter_membership_hashes(tmp_path):
    source_path, parquet_path, manifest_path, manifest = _export(tmp_path, "training")
    changed_exported_hash = deepcopy(manifest)
    changed_exported_hash["split"]["exported_membership_sha256"]["training"] = "0" * 64
    manifest_path.write_text(json.dumps(changed_exported_hash))

    with pytest.raises(ValueError, match="exported partition-membership"):
        validate_training_data_export(parquet_path, manifest_path)

    _, parquet_path, manifest_path, manifest = _export(tmp_path, "replacement")
    changed_pre_filter_hash = deepcopy(manifest)
    changed_pre_filter_hash["split"]["pre_filter_membership_sha256"]["training"] = (
        "0" * 64
    )
    manifest_path.write_text(json.dumps(changed_pre_filter_hash))

    with pytest.raises(ValueError, match="pre-filter partition-membership"):
        validate_training_data_export(
            parquet_path, manifest_path, source_path=source_path
        )


def test_validator_rejects_noncanonical_row_order(tmp_path):
    _, parquet_path, manifest_path, manifest = _export(tmp_path, "training")
    table = pq.read_table(parquet_path)
    reversed_indices = pa.array(range(table.num_rows - 1, -1, -1))
    pq.write_table(table.take(reversed_indices), parquet_path)
    manifest["output"]["sha256"] = file_sha256(parquet_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical ascending order"):
        validate_training_data_export(parquet_path, manifest_path)


def test_export_command_requires_declared_privacy_metadata(tmp_path):
    source_path = tmp_path / "source.csv.gz"
    _write_source_data(source_path)

    completed_process = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "model_training.export_training_data",
            "--data",
            str(source_path),
            "--output",
            str(tmp_path / "training.parquet"),
            "--manifest",
            str(tmp_path / "training.json"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed_process.returncode != 0
    assert "--privacy-classification" in completed_process.stderr
    assert "--publication-intent" in completed_process.stderr


@pytest.mark.parametrize(
    ("privacy_classification", "publication_intent", "message"),
    [
        ("confidential", "private_model_development", "privacy_classification"),
        ("private", "experimental_release", "publication_intent"),
    ],
)
def test_export_rejects_unknown_privacy_metadata(
    tmp_path, privacy_classification, publication_intent, message
):
    source_path = tmp_path / "source.csv.gz"
    _write_source_data(source_path)

    with pytest.raises(ValueError, match=message):
        export_training_data(
            source_path,
            tmp_path / "training.parquet",
            tmp_path / "training.json",
            privacy_classification=privacy_classification,
            publication_intent=publication_intent,
        )


@pytest.mark.parametrize(
    ("privacy_classification", "publication_intent"),
    [
        ("private", "public_release_candidate"),
        ("restricted", "public_release_candidate"),
        ("public", "private_model_development"),
    ],
)
def test_export_rejects_contradictory_privacy_metadata(
    tmp_path, privacy_classification, publication_intent
):
    source_path = tmp_path / "source.csv.gz"
    _write_source_data(source_path)

    with pytest.raises(ValueError, match="contradict each other"):
        export_training_data(
            source_path,
            tmp_path / "training.parquet",
            tmp_path / "training.json",
            privacy_classification=privacy_classification,
            publication_intent=publication_intent,
        )


def test_validator_rejects_contradictory_privacy_metadata(tmp_path):
    _, parquet_path, manifest_path, _ = _export(tmp_path, "training")
    manifest = json.loads(manifest_path.read_text())
    manifest["privacy"]["publication_intent"] = "public_release_candidate"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="contradict each other"):
        validate_training_data_export(parquet_path, manifest_path)


def test_manifest_is_valid_json_and_records_source_code_hashes(tmp_path):
    _, _, manifest_path, manifest = _export(tmp_path, "training")

    assert json.loads(manifest_path.read_text()) == manifest
    assert set(manifest["source_code_sha256"]) == {
        "model_training/evaluation.py",
        "model_training/export_training_data.py",
    }
    assert all(len(digest) == 64 for digest in manifest["source_code_sha256"].values())
