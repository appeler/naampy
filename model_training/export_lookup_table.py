"""Export Naampy's public global first-name lookup table.

The source is the published ``v2_1k`` electoral-roll table. The exported
artifact contains one global row per normalized first name and deliberately
contains no state, birth-year, or third-gender-label fields.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION: Final = 1
ARTIFACT_VERSION: Final = "v2_1k-global-binary-v1"
MINIMUM_NAME_SUPPORT: Final = 1_000
DATAVERSE_DOI: Final = "10.7910/DVN/WZGJBM"
DATAVERSE_FILE_ID: Final = 4_965_695
DATAVERSE_FILE_VERSION: Final = 1
DATAVERSE_DATASET_VERSION: Final = "3.2"
DATAVERSE_FILENAME: Final = "in_rolls_state_year_fn_naampy_x1k.csv.gz"
DATAVERSE_LICENSE: Final = "CC0-1.0"
PUBLISHED_SOURCE_SHA256: Final = (
    "2f72d8555ee6da837f94adb93fad6661a80fa141abc1eda7fa4e17f565fe4417"
)
PUBLISHED_SOURCE_MD5: Final = "822fa00e3f54ac606b6e578d27ef3904"
PUBLISHED_SOURCE_SIZE_BYTES: Final = 62_406_025

SOURCE_COLUMNS: Final = (
    "state",
    "birth_year",
    "first_name",
    "n_female",
    "n_male",
    "n_third_gender",
    "prop_female",
)
LOOKUP_TABLE_SCHEMA: Final = pa.schema(
    [
        pa.field("normalized_name", pa.string(), nullable=False),
        pa.field("female_label_record_count", pa.int64(), nullable=False),
        pa.field("male_label_record_count", pa.int64(), nullable=False),
    ]
)

_MAXIMUM_INT64 = 2**63 - 1


@dataclass(frozen=True, slots=True)
class SourceSummary:
    """Validated summaries of the published source table."""

    row_count: int
    state_count: int
    minimum_birth_year: int
    maximum_birth_year: int
    distinct_source_name_count: int
    female_label_record_count: int
    male_label_record_count: int

    @property
    def represented_binary_label_record_count(self) -> int:
        """Return the included female-plus-male source-label count."""
        return self.female_label_record_count + self.male_label_record_count


def file_checksums(path: Path) -> dict[str, str]:
    """Return SHA-256 and MD5 checksums for a file."""
    sha256_digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            sha256_digest.update(chunk)
            md5_digest.update(chunk)
    return {"sha256": sha256_digest.hexdigest(), "md5": md5_digest.hexdigest()}


def _require_published_source(source_path: Path) -> dict[str, str]:
    """Verify that a source file is the pinned public Dataverse artifact."""
    if not source_path.is_file():
        raise FileNotFoundError(f"source file does not exist: {source_path}")
    source_size = source_path.stat().st_size
    if source_size != PUBLISHED_SOURCE_SIZE_BYTES:
        raise ValueError(
            "source size does not match the published v2_1k artifact: "
            f"expected {PUBLISHED_SOURCE_SIZE_BYTES}, found {source_size}"
        )
    checksums = file_checksums(source_path)
    if checksums["sha256"] != PUBLISHED_SOURCE_SHA256:
        raise ValueError("source SHA-256 does not match the published v2_1k artifact")
    if checksums["md5"] != PUBLISHED_SOURCE_MD5:
        raise ValueError("source MD5 does not match the published v2_1k artifact")
    return checksums


def _parse_nonnegative_count(value: str, *, column: str, row_number: int) -> int:
    """Parse one canonical nonnegative source-label count."""
    if (
        not value
        or value.strip() != value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise ValueError(f"row {row_number}: {column} must be a nonnegative integer")
    parsed_value = int(value)
    if parsed_value > _MAXIMUM_INT64:
        raise ValueError(f"row {row_number}: {column} exceeds int64")
    return parsed_value


def _parse_birth_year(value: str, *, row_number: int) -> int:
    """Parse one finite integral source birth year."""
    try:
        decimal_year = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"row {row_number}: birth_year must be integral") from error
    if not decimal_year.is_finite() or decimal_year != decimal_year.to_integral_value():
        raise ValueError(f"row {row_number}: birth_year must be integral")
    birth_year = int(decimal_year)
    if not -(2**15) <= birth_year <= 2**15 - 1:
        raise ValueError(f"row {row_number}: birth_year exceeds int16")
    return birth_year


def _normalize_first_name(value: str, *, row_number: int) -> str:
    """Return one canonical lookup key or reject an unsupported source name."""
    normalized_name = unicodedata.normalize("NFC", value).strip().casefold()
    if (
        not normalized_name
        or not normalized_name.isascii()
        or not normalized_name.isalpha()
    ):
        raise ValueError(
            f"row {row_number}: first_name must normalize to ASCII letters a-z"
        )
    return normalized_name


def _validate_source_proportion(
    value: str,
    *,
    female_count: int,
    male_count: int,
    third_gender_count: int,
    row_number: int,
) -> None:
    """Verify the published source's female-share integrity field."""
    try:
        female_proportion = float(value)
    except ValueError as error:
        raise ValueError(f"row {row_number}: prop_female must be numeric") from error
    total_count = female_count + male_count + third_gender_count
    if total_count <= 0:
        raise ValueError(f"row {row_number}: source-label total must be positive")
    expected_proportion = female_count / total_count
    if not math.isfinite(female_proportion) or not math.isclose(
        female_proportion,
        expected_proportion,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"row {row_number}: prop_female does not match the source-label counts"
        )


def _add_without_int64_overflow(
    current_value: int, increment: int, *, field: str
) -> int:
    """Add source counts while preserving the output's signed-int64 contract."""
    updated_value = current_value + increment
    if updated_value > _MAXIMUM_INT64:
        raise ValueError(f"aggregated {field} exceeds int64")
    return updated_value


def _read_and_aggregate_source(
    source_path: Path,
) -> tuple[dict[str, tuple[int, int]], SourceSummary]:
    """Validate and aggregate state-year cells to global normalized names."""
    aggregated_counts: dict[str, tuple[int, int]] = {}
    source_names: set[str] = set()
    states: set[str] = set()
    minimum_birth_year: int | None = None
    maximum_birth_year: int | None = None
    female_total = 0
    male_total = 0
    row_count = 0

    with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source_file:
        reader = csv.reader(source_file)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError("source table is empty") from error
        if header != SOURCE_COLUMNS:
            raise ValueError(
                "source columns do not match the published v2_1k schema: "
                f"expected {SOURCE_COLUMNS}, found {header}"
            )

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(SOURCE_COLUMNS):
                raise ValueError(
                    f"row {row_number}: expected {len(SOURCE_COLUMNS)} columns, "
                    f"found {len(row)}"
                )
            state, raw_year, raw_name, raw_female, raw_male, raw_third, raw_share = row
            if not state or state.strip() != state:
                raise ValueError(
                    f"row {row_number}: state must be nonempty and trimmed"
                )
            birth_year = _parse_birth_year(raw_year, row_number=row_number)
            normalized_name = _normalize_first_name(raw_name, row_number=row_number)
            female_count = _parse_nonnegative_count(
                raw_female, column="n_female", row_number=row_number
            )
            male_count = _parse_nonnegative_count(
                raw_male, column="n_male", row_number=row_number
            )
            third_gender_count = _parse_nonnegative_count(
                raw_third, column="n_third_gender", row_number=row_number
            )
            _validate_source_proportion(
                raw_share,
                female_count=female_count,
                male_count=male_count,
                third_gender_count=third_gender_count,
                row_number=row_number,
            )

            current_female, current_male = aggregated_counts.get(
                normalized_name, (0, 0)
            )
            aggregated_counts[normalized_name] = (
                _add_without_int64_overflow(
                    current_female,
                    female_count,
                    field="female_label_record_count",
                ),
                _add_without_int64_overflow(
                    current_male,
                    male_count,
                    field="male_label_record_count",
                ),
            )
            female_total = _add_without_int64_overflow(
                female_total, female_count, field="source female-label total"
            )
            male_total = _add_without_int64_overflow(
                male_total, male_count, field="source male-label total"
            )
            states.add(state)
            source_names.add(raw_name)
            minimum_birth_year = (
                birth_year
                if minimum_birth_year is None
                else min(minimum_birth_year, birth_year)
            )
            maximum_birth_year = (
                birth_year
                if maximum_birth_year is None
                else max(maximum_birth_year, birth_year)
            )
            row_count += 1

    if row_count == 0 or minimum_birth_year is None or maximum_birth_year is None:
        raise ValueError("source table contains no data rows")
    summary = SourceSummary(
        row_count=row_count,
        state_count=len(states),
        minimum_birth_year=minimum_birth_year,
        maximum_birth_year=maximum_birth_year,
        distinct_source_name_count=len(source_names),
        female_label_record_count=female_total,
        male_label_record_count=male_total,
    )
    return aggregated_counts, summary


def _lookup_table(
    aggregated_counts: dict[str, tuple[int, int]],
) -> tuple[pa.Table, dict[str, int]]:
    """Build the supported global lookup rows and filtering summary."""
    retained_rows = [
        (name, female_count, male_count)
        for name, (female_count, male_count) in aggregated_counts.items()
        if female_count + male_count >= MINIMUM_NAME_SUPPORT
    ]
    retained_rows.sort(key=lambda row: row[0])
    if not retained_rows:
        raise ValueError("the support floor excludes every normalized name")
    table = pa.Table.from_pydict(
        {
            "normalized_name": [row[0] for row in retained_rows],
            "female_label_record_count": [row[1] for row in retained_rows],
            "male_label_record_count": [row[2] for row in retained_rows],
        },
        schema=LOOKUP_TABLE_SCHEMA,
    )
    return table, {
        "normalized_name_count_before_support_filter": len(aggregated_counts),
        "normalized_name_count_excluded_below_support": (
            len(aggregated_counts) - len(retained_rows)
        ),
    }


def _arrow_schema_manifest() -> list[dict[str, str | bool]]:
    """Return the exact Arrow schema in JSON-compatible form."""
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in LOOKUP_TABLE_SCHEMA
    ]


def _table_totals(table: pa.Table) -> dict[str, int]:
    """Return derived totals for a validated lookup table."""
    female_counts = cast("list[int]", table["female_label_record_count"].to_pylist())
    male_counts = cast("list[int]", table["male_label_record_count"].to_pylist())
    female_total = sum(female_counts)
    male_total = sum(male_counts)
    return {
        "row_count": table.num_rows,
        "female_label_record_count": female_total,
        "male_label_record_count": male_total,
        "represented_binary_label_record_count": female_total + male_total,
    }


def _source_summary_manifest(summary: SourceSummary) -> dict[str, int]:
    """Return source totals without publishing third-gender-label counts."""
    return {
        "row_count": summary.row_count,
        "state_count": summary.state_count,
        "minimum_birth_year": summary.minimum_birth_year,
        "maximum_birth_year": summary.maximum_birth_year,
        "distinct_source_name_count": summary.distinct_source_name_count,
        "female_label_record_count": summary.female_label_record_count,
        "male_label_record_count": summary.male_label_record_count,
        "represented_binary_label_record_count": (
            summary.represented_binary_label_record_count
        ),
    }


def _dataset_manifest() -> dict[str, Any]:
    """Return the pinned Dataverse identity and licensing metadata."""
    return {
        "persistent_id": f"doi:{DATAVERSE_DOI}",
        "dataset_version": DATAVERSE_DATASET_VERSION,
        "file_id": DATAVERSE_FILE_ID,
        "file_version": DATAVERSE_FILE_VERSION,
        "filename": DATAVERSE_FILENAME,
        "license": DATAVERSE_LICENSE,
    }


def _label_contract() -> dict[str, Any]:
    """Return the public lookup's exact source-label semantics."""
    return {
        "included_source_labels": ["n_female", "n_male"],
        "excluded_source_labels": ["n_third_gender"],
        "target": "female and male electoral-roll source-label record counts",
        "unit": "represented electoral-roll registration record",
        "third_gender_exclusion": (
            "n_third_gender is validated as a nonnegative source field but is not "
            "aggregated, zero-imputed, or published in the lookup artifact or totals"
        ),
        "interpretation": (
            "aggregate first-name composition in the reference data; not an "
            "individual's gender identity"
        ),
    }


def _data_contract(
    filter_summary: dict[str, int], source_summary: SourceSummary
) -> dict[str, Any]:
    """Return the row, normalization, aggregation, and support contract."""
    return {
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
            "source_state_count": source_summary.state_count,
            "release_level": "global_only",
            "published_geography_columns": [],
        },
        "birth_year": {
            "source_minimum": source_summary.minimum_birth_year,
            "source_maximum": source_summary.maximum_birth_year,
            "release_level": "all_years_combined",
            "published_birth_year_columns": [],
        },
        "support": {
            "minimum_represented_binary_label_records": MINIMUM_NAME_SUPPORT,
            "support_fields": [
                "female_label_record_count",
                "male_label_record_count",
            ],
            "rule": (
                "female_label_record_count + male_label_record_count must be "
                f"at least {MINIMUM_NAME_SUPPORT}"
            ),
            **filter_summary,
        },
    }


def _source_code_hashes() -> dict[str, str]:
    """Return the exporter source hash using a repository-portable path."""
    source_path = Path(__file__).resolve()
    return {
        "model_training/export_lookup_table.py": file_checksums(source_path)["sha256"]
    }


def _write_parquet_temporary(table: pa.Table, output_path: Path) -> Path:
    """Write deterministic Zstandard Parquet to the output directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            version="2.6",
            data_page_version="2.0",
            use_dictionary=False,
            write_statistics=True,
            row_group_size=65_536,
        )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_json_temporary(document: dict[str, Any], output_path: Path) -> Path:
    """Write canonical pretty JSON to the output directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(document, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        return Path(temporary_file.name)


def export_lookup_table(
    source_path: Path, parquet_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Export the public global lookup table and return its manifest."""
    resolved_paths = {
        source_path.resolve(),
        parquet_path.resolve(),
        manifest_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError("source, Parquet output, and manifest paths must be distinct")
    if parquet_path.suffix != ".parquet":
        raise ValueError("lookup output must use the .parquet extension")
    if manifest_path.suffix != ".json":
        raise ValueError("lookup manifest must use the .json extension")

    source_checksums = _require_published_source(source_path)
    aggregated_counts, source_summary = _read_and_aggregate_source(source_path)
    table, filter_summary = _lookup_table(aggregated_counts)
    temporary_parquet = _write_parquet_temporary(table, parquet_path)
    try:
        output_checksums = file_checksums(temporary_parquet)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "artifact_role": "global_first_name_label_composition_lookup",
            "reference_population": (
                "electoral-roll registration records represented in Naampy v2_1k "
                "across 31 Indian states and union territories and source birth "
                "years 1887 through 2017"
            ),
            "label_source": (
                "female and male category labels recorded in the source Indian "
                "electoral-roll tables"
            ),
            "dataset": _dataset_manifest(),
            "label_contract": _label_contract(),
            "privacy": {
                "classification": "public",
                "publication_intent": "public_release_candidate",
                "release_grain": "one_nonoverlapping_global_level",
                "rationale": (
                    "only global name aggregates with at least 1000 included "
                    "source-label records are released; state and birth-year "
                    "hierarchies are absent"
                ),
            },
            "source": {
                "local_filename": source_path.name,
                "format": "gzip_csv",
                "size_bytes": source_path.stat().st_size,
                **source_checksums,
                "totals": _source_summary_manifest(source_summary),
            },
            "output": {
                "filename": parquet_path.name,
                "format": "parquet_zstd",
                "size_bytes": temporary_parquet.stat().st_size,
                "sha256": output_checksums["sha256"],
                "totals": _table_totals(table),
            },
            "data_contract": _data_contract(filter_summary, source_summary),
            "arrow_schema": _arrow_schema_manifest(),
            "source_code_sha256": _source_code_hashes(),
            "software_versions": {"pyarrow": pa.__version__},
        }
        temporary_manifest = _write_json_temporary(manifest, manifest_path)
        try:
            temporary_parquet.replace(parquet_path)
            parquet_path.chmod(0o644)
            temporary_manifest.replace(manifest_path)
            manifest_path.chmod(0o644)
        finally:
            temporary_manifest.unlink(missing_ok=True)
    finally:
        temporary_parquet.unlink(missing_ok=True)
    return manifest


def _validate_fixed_manifest_contract(manifest: dict[str, Any]) -> None:
    """Validate manifest sections that must never vary between exports."""
    expected_top_level_keys = {
        "schema_version",
        "artifact_version",
        "artifact_role",
        "reference_population",
        "label_source",
        "dataset",
        "label_contract",
        "privacy",
        "source",
        "output",
        "data_contract",
        "arrow_schema",
        "source_code_sha256",
        "software_versions",
    }
    if set(manifest) != expected_top_level_keys:
        raise ValueError("manifest top-level fields do not match schema version 1")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest must use schema version {SCHEMA_VERSION}")
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("manifest artifact version is not recognized")
    if manifest.get("artifact_role") != "global_first_name_label_composition_lookup":
        raise ValueError("manifest artifact role is not recognized")
    if manifest.get("reference_population") != (
        "electoral-roll registration records represented in Naampy v2_1k across "
        "31 Indian states and union territories and source birth years 1887 "
        "through 2017"
    ):
        raise ValueError("manifest reference population is not recognized")
    if manifest.get("label_source") != (
        "female and male category labels recorded in the source Indian "
        "electoral-roll tables"
    ):
        raise ValueError("manifest label source is not recognized")
    if manifest.get("dataset") != _dataset_manifest():
        raise ValueError("manifest Dataverse provenance is not recognized")
    if manifest.get("label_contract") != _label_contract():
        raise ValueError("manifest source-label contract is not recognized")
    if manifest.get("arrow_schema") != _arrow_schema_manifest():
        raise ValueError("manifest Arrow schema does not match the exporter schema")
    if manifest.get("source_code_sha256") != _source_code_hashes():
        raise ValueError("source-code hash does not match the exporter")
    privacy = manifest.get("privacy", {})
    if privacy != {
        "classification": "public",
        "publication_intent": "public_release_candidate",
        "release_grain": "one_nonoverlapping_global_level",
        "rationale": (
            "only global name aggregates with at least 1000 included "
            "source-label records are released; state and birth-year hierarchies "
            "are absent"
        ),
    }:
        raise ValueError("manifest privacy classification is not recognized")
    source = manifest.get("source", {})
    if (
        set(source)
        != {
            "local_filename",
            "format",
            "size_bytes",
            "sha256",
            "md5",
            "totals",
        }
        or source.get("format") != "gzip_csv"
        or source.get("size_bytes") != PUBLISHED_SOURCE_SIZE_BYTES
        or source.get("sha256") != PUBLISHED_SOURCE_SHA256
        or source.get("md5") != PUBLISHED_SOURCE_MD5
    ):
        raise ValueError("manifest source checksums do not match the published source")
    output = manifest.get("output", {})
    if set(output) != {"filename", "format", "size_bytes", "sha256", "totals"}:
        raise ValueError("manifest output fields are not recognized")
    if output.get("format") != "parquet_zstd":
        raise ValueError("manifest output format is not recognized")
    data_contract = manifest.get("data_contract", {})
    support = data_contract.get("support", {})
    geography = data_contract.get("geography", {})
    birth_year = data_contract.get("birth_year", {})
    if data_contract.get("supported_scripts") != ["Latn"]:
        raise ValueError("manifest supported scripts are not recognized")
    if (
        support.get("minimum_represented_binary_label_records") != MINIMUM_NAME_SUPPORT
        or support.get("support_fields")
        != ["female_label_record_count", "male_label_record_count"]
        or support.get("rule")
        != (
            "female_label_record_count + male_label_record_count must be at "
            f"least {MINIMUM_NAME_SUPPORT}"
        )
    ):
        raise ValueError("manifest support contract is not recognized")
    if (
        geography.get("release_level") != "global_only"
        or geography.get("published_geography_columns") != []
    ):
        raise ValueError("manifest geography release contract is not recognized")
    if (
        birth_year.get("release_level") != "all_years_combined"
        or birth_year.get("published_birth_year_columns") != []
    ):
        raise ValueError("manifest birth-year release contract is not recognized")


def _validate_lookup_table(table: pa.Table, manifest: dict[str, Any]) -> None:
    """Validate schema, keys, counts, support, totals, and row ordering."""
    if table.schema != LOOKUP_TABLE_SCHEMA:
        raise ValueError("Parquet schema does not match the required Arrow schema")
    if table.num_rows == 0:
        raise ValueError("Parquet lookup table must not be empty")
    names = cast("list[str]", table["normalized_name"].to_pylist())
    female_counts = cast("list[int]", table["female_label_record_count"].to_pylist())
    male_counts = cast("list[int]", table["male_label_record_count"].to_pylist())
    if names != sorted(names):
        raise ValueError("normalized names must use canonical ascending order")
    if len(names) != len(set(names)):
        raise ValueError("normalized names must be unique")
    for row_number, (name, female_count, male_count) in enumerate(
        zip(names, female_counts, male_counts, strict=True), start=1
    ):
        if _normalize_first_name(name, row_number=row_number) != name:
            raise ValueError("normalized names must already be canonical")
        if female_count < 0 or male_count < 0:
            raise ValueError("label record counts must be nonnegative")
        if female_count + male_count < MINIMUM_NAME_SUPPORT:
            raise ValueError("Parquet contains a name below the support floor")
    if manifest.get("output", {}).get("totals") != _table_totals(table):
        raise ValueError("Parquet totals do not match the manifest")


def validate_lookup_table_export(
    parquet_path: Path,
    manifest_path: Path,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a lookup artifact, manifest, and optional published source."""
    manifest = cast(
        "dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    _validate_fixed_manifest_contract(manifest)
    output_manifest = manifest.get("output", {})
    if output_manifest.get("filename") != parquet_path.name:
        raise ValueError("manifest output filename does not match the Parquet path")
    if output_manifest.get("sha256") != file_checksums(parquet_path)["sha256"]:
        raise ValueError("Parquet SHA-256 does not match the manifest")
    if output_manifest.get("size_bytes") != parquet_path.stat().st_size:
        raise ValueError("Parquet size does not match the manifest")

    table = pq.read_table(parquet_path)
    _validate_lookup_table(table, manifest)
    if source_path is not None:
        source_checksums = _require_published_source(source_path)
        source_manifest = manifest.get("source", {})
        if source_manifest.get("local_filename") != source_path.name:
            raise ValueError("manifest source filename does not match the source path")
        if source_manifest.get("size_bytes") != source_path.stat().st_size:
            raise ValueError("source size does not match the manifest")
        if any(
            source_manifest.get(algorithm) != digest
            for algorithm, digest in source_checksums.items()
        ):
            raise ValueError("source checksums do not match the manifest")
        aggregated_counts, source_summary = _read_and_aggregate_source(source_path)
        expected_table, filter_summary = _lookup_table(aggregated_counts)
        if not table.equals(expected_table):
            raise ValueError("Parquet rows do not match the source aggregation")
        if source_manifest.get("totals") != _source_summary_manifest(source_summary):
            raise ValueError("source totals do not match the manifest")
        if manifest.get("data_contract") != _data_contract(
            filter_summary, source_summary
        ):
            raise ValueError("data contract does not match the source aggregation")
    return manifest


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Export the pinned public lookup artifact."""
    arguments = _parse_arguments()
    export_lookup_table(arguments.data, arguments.output, arguments.manifest)


if __name__ == "__main__":
    main()
