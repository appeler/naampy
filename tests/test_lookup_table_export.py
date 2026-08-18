import csv
import gzip
import io
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_training import export_lookup_table as exporter
from model_training.export_lookup_table import (
    LOOKUP_TABLE_SCHEMA,
    export_lookup_table,
    file_checksums,
    validate_lookup_table_export,
)


def _source_rows():
    return [
        ["first", "1990.0", "Asha", "500", "100", "0", str(500 / 600)],
        ["second", "1991.0", " asha ", "300", "100", "0", "0.75"],
        ["first", "1992.0", "RAVI", "0", "1000", "2", "0.0"],
        ["second", "1993.0", "kiran", "300", "699", "1", "0.3"],
    ]


def _write_source(path, rows=None, header=exporter.SOURCE_COLUMNS):
    source_rows = _source_rows() if rows is None else rows
    with (
        path.open("wb") as raw_file,
        gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as gzip_file,
        io.TextIOWrapper(gzip_file, encoding="utf-8", newline="") as text_file,
    ):
        writer = csv.writer(text_file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(source_rows)
    return path


def _accept_test_source(monkeypatch, source_path):
    checksums = file_checksums(source_path)
    monkeypatch.setattr(exporter, "PUBLISHED_SOURCE_SHA256", checksums["sha256"])
    monkeypatch.setattr(exporter, "PUBLISHED_SOURCE_MD5", checksums["md5"])
    monkeypatch.setattr(
        exporter, "PUBLISHED_SOURCE_SIZE_BYTES", source_path.stat().st_size
    )


def _export(tmp_path, monkeypatch, stem="lookup"):
    source_path = tmp_path / "source.csv.gz"
    if not source_path.exists():
        _write_source(source_path)
    _accept_test_source(monkeypatch, source_path)
    parquet_path = tmp_path / f"{stem}.parquet"
    manifest_path = tmp_path / f"{stem}.json"
    manifest = export_lookup_table(source_path, parquet_path, manifest_path)
    return source_path, parquet_path, manifest_path, manifest


def _rewrite_parquet_and_manifest(table, parquet_path, manifest_path, manifest):
    pq.write_table(
        table,
        parquet_path,
        compression="zstd",
        version="2.6",
        data_page_version="2.0",
        use_dictionary=False,
        write_statistics=True,
        row_group_size=65_536,
    )
    changed_manifest = deepcopy(manifest)
    changed_manifest["output"]["sha256"] = file_checksums(parquet_path)["sha256"]
    changed_manifest["output"]["size_bytes"] = parquet_path.stat().st_size
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
    return changed_manifest


def test_export_has_exact_global_schema_aggregation_and_support(tmp_path, monkeypatch):
    source_path, parquet_path, manifest_path, manifest = _export(tmp_path, monkeypatch)

    table = pq.read_table(parquet_path)

    assert table.schema == LOOKUP_TABLE_SCHEMA
    assert table.schema == pa.schema(
        [
            pa.field("normalized_name", pa.string(), nullable=False),
            pa.field("female_label_record_count", pa.int64(), nullable=False),
            pa.field("male_label_record_count", pa.int64(), nullable=False),
        ]
    )
    assert table.to_pydict() == {
        "normalized_name": ["asha", "ravi"],
        "female_label_record_count": [800, 0],
        "male_label_record_count": [200, 1000],
    }
    assert manifest["output"]["totals"] == {
        "row_count": 2,
        "female_label_record_count": 800,
        "male_label_record_count": 1200,
        "represented_binary_label_record_count": 2000,
    }
    assert manifest["source"]["totals"] == {
        "row_count": 4,
        "state_count": 2,
        "minimum_birth_year": 1990,
        "maximum_birth_year": 1993,
        "distinct_source_name_count": 4,
        "female_label_record_count": 1100,
        "male_label_record_count": 1899,
        "represented_binary_label_record_count": 2999,
    }
    assert manifest["data_contract"]["support"] == {
        "minimum_represented_binary_label_records": 1000,
        "support_fields": [
            "female_label_record_count",
            "male_label_record_count",
        ],
        "rule": (
            "female_label_record_count + male_label_record_count must be at least 1000"
        ),
        "normalized_name_count_before_support_filter": 3,
        "normalized_name_count_excluded_below_support": 1,
    }
    assert manifest["data_contract"]["geography"] == {
        "source_state_count": 2,
        "release_level": "global_only",
        "published_geography_columns": [],
    }
    assert manifest["data_contract"]["birth_year"] == {
        "source_minimum": 1990,
        "source_maximum": 1993,
        "release_level": "all_years_combined",
        "published_birth_year_columns": [],
    }
    assert manifest["data_contract"]["supported_scripts"] == ["Latn"]
    assert manifest["artifact_version"] == "v2_1k-global-binary-v1"
    assert "electoral-roll registration records" in manifest["reference_population"]
    assert "category labels" in manifest["label_source"]
    assert manifest["label_contract"]["excluded_source_labels"] == ["n_third_gender"]
    assert "third_gender" not in " ".join(table.column_names)
    assert (
        validate_lookup_table_export(
            parquet_path, manifest_path, source_path=source_path
        )
        == manifest
    )


def test_export_is_byte_deterministic_and_manifest_paths_are_portable(
    tmp_path, monkeypatch
):
    source_path = _write_source(tmp_path / "source.csv.gz")
    _accept_test_source(monkeypatch, source_path)
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_manifest = export_lookup_table(
        source_path,
        first_directory / "lookup.parquet",
        first_directory / "lookup.json",
    )
    second_manifest = export_lookup_table(
        source_path,
        second_directory / "lookup.parquet",
        second_directory / "lookup.json",
    )

    assert (first_directory / "lookup.parquet").read_bytes() == (
        second_directory / "lookup.parquet"
    ).read_bytes()
    assert first_manifest == second_manifest
    serialized_manifest = json.dumps(first_manifest)
    assert str(tmp_path.resolve()) not in serialized_manifest
    assert first_manifest["source"]["local_filename"] == "source.csv.gz"
    assert first_manifest["output"]["filename"] == "lookup.parquet"


def test_export_records_both_source_checksums_and_code_hash(tmp_path, monkeypatch):
    source_path, _, _, manifest = _export(tmp_path, monkeypatch)

    assert manifest["source"]["sha256"] == file_checksums(source_path)["sha256"]
    assert manifest["source"]["md5"] == file_checksums(source_path)["md5"]
    assert manifest["dataset"] == {
        "persistent_id": "doi:10.7910/DVN/WZGJBM",
        "dataset_version": "3.2",
        "file_id": 4965695,
        "file_version": 1,
        "filename": "in_rolls_state_year_fn_naampy_x1k.csv.gz",
        "license": "CC0-1.0",
    }
    assert set(manifest["source_code_sha256"]) == {
        "model_training/export_lookup_table.py"
    }
    assert len(next(iter(manifest["source_code_sha256"].values()))) == 64


def test_export_rejects_source_that_no_longer_matches_published_hash(
    tmp_path, monkeypatch
):
    source_path = _write_source(tmp_path / "source.csv.gz")
    _accept_test_source(monkeypatch, source_path)
    source_path.write_bytes(source_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="source size"):
        export_lookup_table(
            source_path, tmp_path / "lookup.parquet", tmp_path / "lookup.json"
        )

    assert not (tmp_path / "lookup.parquet").exists()
    assert not (tmp_path / "lookup.json").exists()


@pytest.mark.parametrize(
    ("rows", "header", "message"),
    [
        (_source_rows(), ("first_name", "n_female"), "source columns"),
        (
            [["first", "1990.5", "asha", "1000", "0", "0", "1.0"]],
            exporter.SOURCE_COLUMNS,
            "birth_year must be integral",
        ),
        (
            [["first", "1990.0", "asha-b", "1000", "0", "0", "1.0"]],
            exporter.SOURCE_COLUMNS,
            "ASCII letters",
        ),
        (
            [["first", "1990.0", "asha", "-1", "1001", "0", "0.0"]],
            exporter.SOURCE_COLUMNS,
            "n_female must be a nonnegative integer",
        ),
        (
            [["first", "1990.0", "asha", "1000", "0", "0", "0.5"]],
            exporter.SOURCE_COLUMNS,
            "prop_female does not match",
        ),
    ],
)
def test_export_rejects_malformed_source(tmp_path, monkeypatch, rows, header, message):
    source_path = _write_source(tmp_path / "source.csv.gz", rows, header)
    _accept_test_source(monkeypatch, source_path)

    with pytest.raises(ValueError, match=message):
        export_lookup_table(
            source_path, tmp_path / "lookup.parquet", tmp_path / "lookup.json"
        )


def test_validator_rejects_hash_and_manifest_tampering(tmp_path, monkeypatch):
    source_path, parquet_path, manifest_path, manifest = _export(tmp_path, monkeypatch)
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Parquet SHA-256"):
        validate_lookup_table_export(parquet_path, manifest_path)

    _, parquet_path, manifest_path, manifest = _export(
        tmp_path, monkeypatch, stem="replacement"
    )
    changed_manifest = deepcopy(manifest)
    changed_manifest["source"]["md5"] = "0" * 32
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source checksums"):
        validate_lookup_table_export(
            parquet_path, manifest_path, source_path=source_path
        )


def test_validator_rejects_unsupported_manifest_script(tmp_path, monkeypatch):
    _, parquet_path, manifest_path, manifest = _export(tmp_path, monkeypatch)
    changed_manifest = deepcopy(manifest)
    changed_manifest["data_contract"]["supported_scripts"] = ["Latn", "Deva"]
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="supported scripts"):
        validate_lookup_table_export(parquet_path, manifest_path)


def test_validator_rejects_noncanonical_order_and_third_gender_column(
    tmp_path, monkeypatch
):
    _, parquet_path, manifest_path, manifest = _export(tmp_path, monkeypatch)
    table = pq.read_table(parquet_path)
    reversed_table = table.take(pa.array([1, 0]))
    _rewrite_parquet_and_manifest(reversed_table, parquet_path, manifest_path, manifest)
    with pytest.raises(ValueError, match="canonical ascending order"):
        validate_lookup_table_export(parquet_path, manifest_path)

    _, parquet_path, manifest_path, manifest = _export(
        tmp_path, monkeypatch, stem="replacement"
    )
    table = pq.read_table(parquet_path).append_column(
        pa.field("third_gender_label_record_count", pa.int64(), nullable=False),
        pa.array([0, 0], type=pa.int64()),
    )
    _rewrite_parquet_and_manifest(table, parquet_path, manifest_path, manifest)
    with pytest.raises(ValueError, match="Parquet schema"):
        validate_lookup_table_export(parquet_path, manifest_path)


def test_validator_rejects_support_violation_and_incorrect_aggregation(
    tmp_path, monkeypatch
):
    source_path, parquet_path, manifest_path, manifest = _export(tmp_path, monkeypatch)
    table = pq.read_table(parquet_path).set_column(
        2,
        LOOKUP_TABLE_SCHEMA.field("male_label_record_count"),
        pa.array([199, 1000], type=pa.int64()),
    )
    _rewrite_parquet_and_manifest(table, parquet_path, manifest_path, manifest)
    with pytest.raises(ValueError, match="below the support floor"):
        validate_lookup_table_export(parquet_path, manifest_path)

    _, parquet_path, manifest_path, manifest = _export(
        tmp_path, monkeypatch, stem="replacement"
    )
    table = pq.read_table(parquet_path).set_column(
        1,
        LOOKUP_TABLE_SCHEMA.field("female_label_record_count"),
        pa.array([801, 0], type=pa.int64()),
    )
    changed_manifest = _rewrite_parquet_and_manifest(
        table, parquet_path, manifest_path, manifest
    )
    changed_manifest["output"]["totals"]["female_label_record_count"] = 801
    changed_manifest["output"]["totals"]["represented_binary_label_record_count"] = 2001
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source aggregation"):
        validate_lookup_table_export(
            parquet_path, manifest_path, source_path=source_path
        )


@pytest.mark.parametrize(
    ("output_name", "manifest_name", "message"),
    [
        ("lookup.csv", "lookup.json", "parquet extension"),
        ("lookup.parquet", "lookup.txt", "json extension"),
    ],
)
def test_export_requires_typed_artifact_extensions(
    tmp_path, monkeypatch, output_name, manifest_name, message
):
    source_path = _write_source(tmp_path / "source.csv.gz")
    _accept_test_source(monkeypatch, source_path)

    with pytest.raises(ValueError, match=message):
        export_lookup_table(
            source_path, tmp_path / output_name, tmp_path / manifest_name
        )
