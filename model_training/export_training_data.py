"""Export the Naampy v3 model-development table as typed Parquet.

The exporter assigns the fixed seed-0 split before applying a minimum-support
filter. Raising the support floor therefore removes rows without changing any
usable name's partition.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from model_training import evaluation as evaluation_module
from model_training.evaluation import (
    name_membership_sha256,
    stratified_name_split,
)

SCHEMA_VERSION: Final = 1
SPLIT_SEED: Final = 0
PARTITION_NAMES: Final = ("training", "validation", "calibration", "test")
PRIVACY_CLASSIFICATIONS: Final = ("private", "restricted", "public")
PUBLICATION_INTENTS: Final = (
    "private_model_development",
    "public_release_candidate",
)
ALLOWED_PRIVACY_DECLARATIONS: Final = frozenset(
    {
        ("private", "private_model_development"),
        ("restricted", "private_model_development"),
        ("public", "public_release_candidate"),
    }
)
DATAVERSE_DOI: Final = "10.7910/DVN/WZGJBM"
DATAVERSE_LICENSE: Final = "CC0-1.0"
NAAMPY_CONSTRUCTION_REVISION: Final = "2b15840cf0c63ddf6b5b81bf9ecf068d65d7722d"
TRANSLITERATION_CONSTRUCTION_REVISION: Final = (
    "262844fdaec6ee707a87160306e139e141a52bcd"
)
PARTITION_FRACTIONS: Final = {
    "training": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "test": 0.10,
}
TRAINING_DATA_SCHEMA: Final = pa.schema(
    [
        pa.field("normalized_name", pa.string(), nullable=False),
        pa.field("female_label_record_count", pa.int64(), nullable=False),
        pa.field("male_label_record_count", pa.int64(), nullable=False),
        pa.field("represented_binary_label_record_count", pa.int64(), nullable=False),
        pa.field("partition", pa.string(), nullable=False),
    ]
)

_LOWERCASE_ASCII_NAME = re.compile(r"^[a-z]+$")
_REPEATED_CHARACTER = re.compile(r"(.)\1\1")
_MAXIMUM_INT64 = np.iinfo(np.int64).max


def _is_usable_name(name: object) -> bool:
    """Return whether a source value satisfies the model's name contract."""
    return (
        isinstance(name, str)
        and 2 < len(name) < 20
        and _LOWERCASE_ASCII_NAME.fullmatch(name) is not None
        and _REPEATED_CHARACTER.search(name) is None
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _label_contract() -> dict[str, Any]:
    """Return the binary source-label estimand contract."""
    return {
        "included_source_labels": ["n_female", "n_male"],
        "excluded_source_labels": ["n_third_gender"],
        "excluded_label_reason": (
            "n_third_gender is too sparse in the retained source for this "
            "binary target and is not included in any artifact count"
        ),
        "interpretation": (
            "aggregate electoral-roll source-label composition; not an "
            "individual's gender identity"
        ),
    }


def _source_provenance() -> dict[str, Any]:
    """Return immutable source and construction provenance."""
    return {
        "dataverse": {
            "doi": DATAVERSE_DOI,
            "license": DATAVERSE_LICENSE,
        },
        "construction_revisions": {
            "naampy": NAAMPY_CONSTRUCTION_REVISION,
            "eroll_transliteration": TRANSLITERATION_CONSTRUCTION_REVISION,
        },
    }


def _validated_label_counts(source_table: pd.DataFrame) -> pd.DataFrame:
    """Return source label counts as validated signed 64-bit integers."""
    validated = source_table.copy()
    for column_name in ("n_female", "n_male"):
        numeric_values = pd.to_numeric(validated[column_name], errors="raise")
        if pd.api.types.is_integer_dtype(numeric_values.dtype):
            if (numeric_values < 0).any() or (numeric_values > _MAXIMUM_INT64).any():
                raise ValueError(f"{column_name} must contain nonnegative int64 values")
        else:
            numeric_array = numeric_values.to_numpy(dtype=np.float64)
            if not np.isfinite(numeric_array).all():
                raise ValueError(f"{column_name} must contain only finite values")
            if (
                np.any(numeric_array < 0)
                or np.any(numeric_array > _MAXIMUM_INT64)
                or np.any(numeric_array != np.floor(numeric_array))
            ):
                raise ValueError(f"{column_name} must contain nonnegative int64 values")
        try:
            validated[column_name] = numeric_values.astype("int64")
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{column_name} values must fit in int64") from error
    return validated


def _load_usable_names(source_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate source cells and apply the documented model-input filters."""
    with gzip.open(source_path, "rt", encoding="utf-8") as compressed_file:
        source_table = pd.read_csv(
            compressed_file,
            usecols=["first_name", "n_female", "n_male"],
            dtype={"first_name": "string"},
        )
    source_table = _validated_label_counts(source_table)
    source_summary = {
        "row_count": len(source_table),
        "female_label_record_count": sum(map(int, source_table["n_female"])),
        "male_label_record_count": sum(map(int, source_table["n_male"])),
    }
    source_summary["represented_binary_label_record_count"] = (
        source_summary["female_label_record_count"]
        + source_summary["male_label_record_count"]
    )

    groupable_source = source_table[source_table["first_name"].notna()]
    groupable_female_total = sum(map(int, groupable_source["n_female"]))
    groupable_male_total = sum(map(int, groupable_source["n_male"]))
    grouped_names = cast(
        "pd.DataFrame",
        groupable_source.groupby("first_name", as_index=False, sort=True, dropna=True)[
            ["n_female", "n_male"]
        ].sum(),
    )
    if (
        (grouped_names[["n_female", "n_male"]] < 0).any().any()
        or sum(map(int, grouped_names["n_female"])) != groupable_female_total
        or sum(map(int, grouped_names["n_male"])) != groupable_male_total
    ):
        raise ValueError("aggregated label counts overflowed int64")
    if (grouped_names["n_female"] > _MAXIMUM_INT64 - grouped_names["n_male"]).any():
        raise ValueError("aggregated represented record counts exceed int64")
    represented_counts = grouped_names["n_female"] + grouped_names["n_male"]
    grouped_names = grouped_names[represented_counts > 0].copy()
    usable = grouped_names["first_name"].map(_is_usable_name)
    grouped_names = grouped_names[usable].copy()
    grouped_names.rename(
        columns={
            "first_name": "normalized_name",
            "n_female": "female_label_record_count",
            "n_male": "male_label_record_count",
        },
        inplace=True,
    )
    grouped_names["represented_binary_label_record_count"] = (
        grouped_names["female_label_record_count"]
        + grouped_names["male_label_record_count"]
    )
    grouped_names.sort_values("normalized_name", kind="stable", inplace=True)
    grouped_names.reset_index(drop=True, inplace=True)
    return grouped_names, source_summary


def _partition_labels(grouped_names: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Return fixed partition labels and full usable-name membership hashes."""
    names = cast("list[str]", grouped_names["normalized_name"].tolist())
    female_counts = grouped_names["female_label_record_count"].to_numpy(
        dtype=np.float64
    )
    record_counts = grouped_names["represented_binary_label_record_count"].to_numpy(
        dtype=np.float64
    )
    female_proportions = female_counts / record_counts
    split = stratified_name_split(
        female_proportions,
        record_counts,
        seed=SPLIT_SEED,
    )
    labels = [""] * len(names)
    membership_hashes: dict[str, str] = {}
    for partition_name, partition_indices in (
        ("training", split.training),
        ("validation", split.validation),
        ("calibration", split.calibration),
        ("test", split.test),
    ):
        membership_hashes[partition_name] = name_membership_sha256(
            names, partition_indices
        )
        for row_index in partition_indices:
            labels[int(row_index)] = partition_name
    if any(not label for label in labels):
        raise RuntimeError("split did not assign every usable name exactly once")
    return labels, membership_hashes


def _arrow_schema_manifest() -> list[dict[str, str | bool]]:
    """Return a JSON representation of the exact Arrow schema."""
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in TRAINING_DATA_SCHEMA
    ]


def _table_totals(table: pa.Table) -> dict[str, int]:
    """Return row and label-count totals for an Arrow table."""
    female_counts = cast("list[int]", table["female_label_record_count"].to_pylist())
    male_counts = cast("list[int]", table["male_label_record_count"].to_pylist())
    represented_counts = cast(
        "list[int]", table["represented_binary_label_record_count"].to_pylist()
    )
    return {
        "row_count": table.num_rows,
        "female_label_record_count": sum(female_counts),
        "male_label_record_count": sum(male_counts),
        "represented_binary_label_record_count": sum(represented_counts),
    }


def _partition_totals(table: pa.Table) -> dict[str, dict[str, int]]:
    """Return exported row and label-count totals by partition."""
    partition_values = cast("list[str]", table["partition"].to_pylist())
    return {
        partition_name: _table_totals(
            table.filter(
                pa.array([value == partition_name for value in partition_values])
            )
        )
        for partition_name in PARTITION_NAMES
    }


def _canonical_name_set_sha256(names: list[str]) -> str:
    """Return the canonical hash of a possibly empty name set."""
    encoded_names = json.dumps(
        sorted(names), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded_names).hexdigest()


def _exported_membership_hashes(table: pa.Table) -> dict[str, str]:
    """Return partition-membership hashes for the exported rows."""
    names = cast("list[str]", table["normalized_name"].to_pylist())
    partitions = cast("list[str]", table["partition"].to_pylist())
    return {
        partition_name: _canonical_name_set_sha256(
            [
                name
                for name, assigned_partition in zip(names, partitions, strict=True)
                if assigned_partition == partition_name
            ]
        )
        for partition_name in PARTITION_NAMES
    }


def _write_parquet_atomic(table: pa.Table, output_path: Path) -> None:
    """Write a deterministic Parquet artifact without exposing a partial file."""
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
        temporary_path.replace(output_path)
        output_path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_atomic(document: dict[str, Any], output_path: Path) -> None:
    """Write a JSON document without exposing a partial file."""
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
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(output_path)
    output_path.chmod(0o644)


def _source_code_hashes() -> dict[str, str]:
    """Return hashes of code that defines the export and split."""
    repository_root = Path(__file__).resolve().parents[1]
    code_paths = (
        Path(__file__).resolve(),
        Path(cast("str", evaluation_module.__file__)).resolve(),
    )
    return {
        str(path.relative_to(repository_root)): file_sha256(path) for path in code_paths
    }


def _validate_privacy_declaration(
    privacy_classification: object, publication_intent: object
) -> None:
    """Reject privacy classifications that contradict publication intent."""
    if privacy_classification not in PRIVACY_CLASSIFICATIONS:
        raise ValueError(
            "privacy_classification must be one of "
            f"{', '.join(PRIVACY_CLASSIFICATIONS)}"
        )
    if publication_intent not in PUBLICATION_INTENTS:
        raise ValueError(
            f"publication_intent must be one of {', '.join(PUBLICATION_INTENTS)}"
        )
    if (privacy_classification, publication_intent) not in (
        ALLOWED_PRIVACY_DECLARATIONS
    ):
        raise ValueError(
            "privacy_classification and publication_intent contradict each other: "
            "private or restricted artifacts are for private model development; "
            "public release candidates must be classified public"
        )


def export_training_data(
    source_path: Path,
    parquet_path: Path,
    manifest_path: Path,
    *,
    minimum_name_support: int = 1,
    privacy_classification: str,
    publication_intent: str,
) -> dict[str, Any]:
    """Export typed model-development data and return its manifest."""
    if minimum_name_support < 1:
        raise ValueError("minimum_name_support must be positive")
    _validate_privacy_declaration(privacy_classification, publication_intent)
    resolved_paths = {
        source_path.resolve(),
        parquet_path.resolve(),
        manifest_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError("source, Parquet output, and manifest paths must be distinct")

    grouped_names, source_totals = _load_usable_names(source_path)
    partition_labels, membership_hashes = _partition_labels(grouped_names)
    grouped_names["partition"] = partition_labels
    usable_totals = {
        "row_count": len(grouped_names),
        "female_label_record_count": sum(
            map(int, grouped_names["female_label_record_count"])
        ),
        "male_label_record_count": sum(
            map(int, grouped_names["male_label_record_count"])
        ),
        "represented_binary_label_record_count": sum(
            map(int, grouped_names["represented_binary_label_record_count"])
        ),
    }
    exported_names = grouped_names[
        grouped_names["represented_binary_label_record_count"] >= minimum_name_support
    ].copy()
    if exported_names.empty:
        raise ValueError("minimum_name_support excludes every usable name")

    table = pa.Table.from_pydict(
        {
            field.name: exported_names[field.name].tolist()
            for field in TRAINING_DATA_SCHEMA
        },
        schema=TRAINING_DATA_SCHEMA,
    )
    _write_parquet_atomic(table, parquet_path)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "name_pattern_model_training_data",
        "target": "female share among female and male electoral-roll labels",
        "reference_population": (
            "aggregated Indian electoral-roll registration records represented "
            "in the Naampy v3 construction"
        ),
        "label_contract": _label_contract(),
        "privacy": {
            "classification": privacy_classification,
            "publication_intent": publication_intent,
            "declaration_source": "explicit_export_argument",
        },
        "source": {
            "filename": source_path.name,
            "format": "gzip_csv",
            "sha256": file_sha256(source_path),
            "totals": source_totals,
            "provenance": _source_provenance(),
        },
        "output": {
            "filename": parquet_path.name,
            "format": "parquet",
            "sha256": file_sha256(parquet_path),
            "totals": _table_totals(table),
            "partition_totals": _partition_totals(table),
        },
        "data_contract": {
            "unit_of_observation": "one usable normalized first name",
            "row_order": "normalized_name ascending by Unicode code point",
            "normalization": (
                "retain the upstream-normalized first_name exactly; require lowercase "
                "ASCII letters a-z; do not transliterate or case-fold during export"
            ),
            "aggregation": (
                "sum female and male label counts across state and birth-year cells "
                "before filtering"
            ),
            "filters": {
                "minimum_length": 3,
                "maximum_length": 19,
                "allowed_characters": "lowercase ASCII a-z",
                "exclude_three_repeated_characters": True,
                "require_positive_female_plus_male_count": True,
                "minimum_name_support": minimum_name_support,
            },
            "usable_totals_before_support_filter": usable_totals,
        },
        "split": {
            "method": "stratified disjoint unique-name split",
            "membership_scope": "all usable names before minimum-support filtering",
            "seed": SPLIT_SEED,
            "fractions": PARTITION_FRACTIONS,
            "strata": {
                "record_count_rank_bins": 100,
                "female_proportion_bins": 10,
            },
            "pre_filter_membership_sha256": membership_hashes,
            "exported_membership_sha256": _exported_membership_hashes(table),
        },
        "arrow_schema": _arrow_schema_manifest(),
        "software_versions": {"pyarrow": pa.__version__},
        "source_code_sha256": _source_code_hashes(),
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def validate_training_data_export(
    parquet_path: Path,
    manifest_path: Path,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Validate hashes, schema, counts, partitions, and support constraints."""
    manifest = cast(
        "dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest must use schema version {SCHEMA_VERSION}")
    if manifest.get("arrow_schema") != _arrow_schema_manifest():
        raise ValueError("manifest Arrow schema does not match the exporter schema")
    if manifest.get("label_contract") != _label_contract():
        raise ValueError("manifest source-label contract is not recognized")
    if manifest.get("source", {}).get("provenance") != _source_provenance():
        raise ValueError("manifest source provenance is not recognized")
    privacy = manifest.get("privacy", {})
    _validate_privacy_declaration(
        privacy.get("classification"), privacy.get("publication_intent")
    )
    if manifest["output"]["sha256"] != file_sha256(parquet_path):
        raise ValueError("Parquet hash does not match the manifest")
    if source_path is not None and manifest["source"]["sha256"] != file_sha256(
        source_path
    ):
        raise ValueError("source hash does not match the manifest")
    if manifest.get("source_code_sha256") != _source_code_hashes():
        raise ValueError("source-code hashes do not match the manifest")

    table = pq.read_table(parquet_path)
    if table.schema != TRAINING_DATA_SCHEMA:
        raise ValueError("Parquet schema does not match the required Arrow schema")
    if manifest["output"]["totals"] != _table_totals(table):
        raise ValueError("Parquet totals do not match the manifest")
    if manifest["output"]["partition_totals"] != _partition_totals(table):
        raise ValueError("Parquet partition totals do not match the manifest")
    if manifest["split"]["exported_membership_sha256"] != (
        _exported_membership_hashes(table)
    ):
        raise ValueError("exported partition-membership hashes do not match Parquet")

    names = cast("list[str]", table["normalized_name"].to_pylist())
    partitions = cast("list[str]", table["partition"].to_pylist())
    female_counts = np.asarray(
        table["female_label_record_count"].to_pylist(), dtype=np.int64
    )
    male_counts = np.asarray(
        table["male_label_record_count"].to_pylist(), dtype=np.int64
    )
    represented_counts = np.asarray(
        table["represented_binary_label_record_count"].to_pylist(), dtype=np.int64
    )
    if len(names) != len(set(names)):
        raise ValueError("normalized names must be unique")
    if names != sorted(names):
        raise ValueError("normalized names must use canonical ascending order")
    if any(partition not in PARTITION_NAMES for partition in partitions):
        raise ValueError("every row must have exactly one recognized partition")
    if np.any(female_counts < 0) or np.any(male_counts < 0):
        raise ValueError("label counts must be nonnegative")
    if not np.array_equal(represented_counts, female_counts + male_counts):
        raise ValueError("represented record count must equal female plus male counts")
    minimum_support = int(manifest["data_contract"]["filters"]["minimum_name_support"])
    if np.any(represented_counts < minimum_support):
        raise ValueError("Parquet contains a name below the declared support floor")
    if source_path is not None:
        grouped_names, source_totals = _load_usable_names(source_path)
        _, pre_filter_membership_hashes = _partition_labels(grouped_names)
        if manifest["source"]["totals"] != source_totals:
            raise ValueError("source totals do not match the manifest")
        if manifest["split"]["pre_filter_membership_sha256"] != (
            pre_filter_membership_hashes
        ):
            raise ValueError(
                "pre-filter partition-membership hashes do not match the source"
            )
    return manifest


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--minimum-name-support", type=int, default=1)
    parser.add_argument(
        "--privacy-classification",
        choices=PRIVACY_CLASSIFICATIONS,
        required=True,
    )
    parser.add_argument(
        "--publication-intent",
        choices=PUBLICATION_INTENTS,
        required=True,
    )
    arguments = parser.parse_args()
    if arguments.minimum_name_support < 1:
        parser.error("--minimum-name-support must be positive")
    return arguments


def main() -> None:
    """Export the typed training artifact and its manifest."""
    arguments = _parse_arguments()
    export_training_data(
        arguments.data,
        arguments.output,
        arguments.manifest,
        minimum_name_support=arguments.minimum_name_support,
        privacy_classification=arguments.privacy_classification,
        publication_intent=arguments.publication_intent,
    )


if __name__ == "__main__":
    main()
