"""Typed, validated exact electoral-roll lookup bundles."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import _resources

if TYPE_CHECKING:
    from collections.abc import Callable

LOOKUP_MANIFEST_SCHEMA_VERSION = 1
LOOKUP_ARTIFACT_VERSION = "v2_1k-global-binary-v1"
LOOKUP_ARTIFACT_ROLE = "global_first_name_label_composition_lookup"
LOOKUP_MINIMUM_BINARY_LABEL_RECORD_COUNT = 1_000
LOOKUP_REFERENCE_POPULATION = (
    "electoral-roll registration records represented in Naampy v2_1k across "
    "31 Indian states and union territories and source birth years 1887 through 2017"
)
LOOKUP_LABEL_SOURCE = (
    "female and male category labels recorded in the source Indian electoral-roll "
    "tables"
)
THIRD_GENDER_EXCLUSION = (
    "n_third_gender is validated as a nonnegative source field but is not "
    "aggregated, zero-imputed, or published in the lookup artifact or totals"
)
LOOKUP_TABLE_SCHEMA = pa.schema(
    [
        pa.field("normalized_name", pa.string(), nullable=False),
        pa.field("female_label_record_count", pa.int64(), nullable=False),
        pa.field("male_label_record_count", pa.int64(), nullable=False),
    ]
)

_EXPECTED_DATASET = {
    "persistent_id": "doi:10.7910/DVN/WZGJBM",
    "dataset_version": "3.2",
    "file_id": 4965695,
    "file_version": 1,
    "filename": "in_rolls_state_year_fn_naampy_x1k.csv.gz",
    "license": "CC0-1.0",
}
_EXPECTED_PRIVACY = {
    "classification": "public",
    "publication_intent": "public_release_candidate",
    "release_grain": "one_nonoverlapping_global_level",
    "rationale": (
        "only global name aggregates with at least 1000 included source-label "
        "records are released; state and birth-year hierarchies are absent"
    ),
}
_EXPECTED_SOURCE_IDENTITY = {
    "format": "gzip_csv",
    "size_bytes": 62_406_025,
    "sha256": "2f72d8555ee6da837f94adb93fad6661a80fa141abc1eda7fa4e17f565fe4417",
    "md5": "822fa00e3f54ac606b6e578d27ef3904",
}
_EXPECTED_DATA_CONTRACT_TEXT = {
    "unit_of_observation": "one globally aggregated normalized first name",
    "row_order": "normalized_name ascending by Unicode code point",
    "normalization": (
        "Unicode NFC, strip surrounding whitespace, Unicode casefold, then "
        "require nonempty ASCII letters a-z"
    ),
    "aggregation": (
        "sum included female and male source-label counts across all state and "
        "birth-year cells after normalization"
    ),
}
_EXPECTED_LABEL_CONTRACT = {
    "included_source_labels": ["n_female", "n_male"],
    "excluded_source_labels": ["n_third_gender"],
    "target": "female and male electoral-roll source-label record counts",
    "unit": "represented electoral-roll registration record",
    "third_gender_exclusion": THIRD_GENDER_EXCLUSION,
    "interpretation": (
        "aggregate first-name composition in the reference data; not an "
        "individual's gender identity"
    ),
}
_EXPECTED_ARROW_SCHEMA = [
    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
    for field in LOOKUP_TABLE_SCHEMA
]


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Immutable identity of the source Dataverse file."""

    persistent_id: str
    dataset_version: str
    file_id: int
    file_version: int
    filename: str
    license: str


@dataclass(frozen=True, slots=True)
class LabelContract:
    """Included, excluded, and interpreted source-label semantics."""

    included_source_labels: tuple[str, str]
    excluded_source_labels: tuple[str]
    target: str
    unit: str
    third_gender_exclusion: str
    interpretation: str


@dataclass(frozen=True, slots=True)
class PrivacyContract:
    """Publication-level privacy declaration."""

    classification: str
    publication_intent: str
    release_grain: str
    rationale: str


@dataclass(frozen=True, slots=True)
class BinaryLabelTotals:
    """Female, male, and derived represented-record totals."""

    female_label_record_count: int
    male_label_record_count: int
    represented_binary_label_record_count: int


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """Validated transport-level source metadata."""

    local_filename: str
    format: str
    size_bytes: int
    sha256: str
    md5: str
    row_count: int
    state_count: int
    minimum_birth_year: int
    maximum_birth_year: int
    distinct_source_name_count: int
    label_totals: BinaryLabelTotals


@dataclass(frozen=True, slots=True)
class OutputArtifact:
    """Validated public Parquet metadata."""

    filename: str
    format: str
    size_bytes: int
    sha256: str
    row_count: int
    label_totals: BinaryLabelTotals


@dataclass(frozen=True, slots=True)
class SupportContract:
    """Release support threshold and its row-accounting audit."""

    minimum_represented_binary_label_records: int
    support_fields: tuple[str, str]
    rule: str
    normalized_name_count_before_support_filter: int
    normalized_name_count_excluded_below_support: int


@dataclass(frozen=True, slots=True)
class DataContract:
    """Global aggregation, script, time, geography, and support contract."""

    unit_of_observation: str
    row_order: str
    normalization: str
    supported_scripts: tuple[str]
    aggregation: str
    source_state_count: int
    source_minimum_birth_year: int
    source_maximum_birth_year: int
    support: SupportContract


@dataclass(frozen=True, slots=True)
class FirstNameCompositionManifest:
    """Validated canonical manifest for one public lookup artifact."""

    schema_version: int
    artifact_version: str
    artifact_role: str
    reference_population: str
    label_source: str
    dataset: DatasetIdentity
    label_contract: LabelContract
    privacy: PrivacyContract
    source: SourceArtifact
    output: OutputArtifact
    data_contract: DataContract
    source_code_sha256: str
    pyarrow_version: str

    @property
    def supported_scripts(self) -> frozenset[str]:
        """Return scripts covered by exact published keys."""
        return frozenset(self.data_contract.supported_scripts)


@dataclass(frozen=True, slots=True)
class FirstNameCompositionBundle:
    """A validated typed table and its immutable provenance."""

    manifest: FirstNameCompositionManifest
    table_path: Path
    repository: str
    revision: str


def load_default_lookup_bundle() -> FirstNameCompositionBundle:
    """Load the package's configured immutable exact-lookup bundle."""
    repository, revision = _resources.artifact_provenance(
        repository=_resources.LOOKUP_TABLE_REPOSITORY,
        revision=_resources.LOOKUP_TABLE_REVISION,
        local_directory_environment_variable=(
            _resources.LOOKUP_TABLE_DIRECTORY_ENVIRONMENT_VARIABLE
        ),
    )
    manifest_path = _resources.resolve_lookup_table_artifact(
        _resources.LOOKUP_TABLE_MANIFEST_FILENAME
    )
    return load_lookup_bundle(
        manifest_path,
        artifact_resolver=_resources.resolve_lookup_table_artifact,
        repository=repository,
        revision=revision,
    )


def load_lookup_bundle(
    manifest_path: str | Path,
    *,
    artifact_resolver: Callable[[str], str | Path],
    repository: str,
    revision: str,
) -> FirstNameCompositionBundle:
    """Load and validate a canonical public lookup manifest and artifact."""
    manifest = parse_lookup_manifest(manifest_path)
    table_path = Path(artifact_resolver(manifest.output.filename))
    if table_path.stat().st_size != manifest.output.size_bytes:
        raise ValueError("Lookup artifact size does not match the manifest")
    _verify_sha256(table_path, manifest.output.sha256)
    _validate_lookup_table(
        table_path,
        manifest=manifest,
    )
    return FirstNameCompositionBundle(
        manifest=manifest,
        table_path=table_path,
        repository=repository,
        revision=revision,
    )


def parse_lookup_manifest(path: str | Path) -> FirstNameCompositionManifest:
    """Parse and strictly validate the canonical rich lookup manifest."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read lookup manifest {path!s}") from error
    if not isinstance(payload, dict):
        raise ValueError("Lookup manifest must contain a JSON object")
    _require_exact_keys(
        payload,
        {
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
        },
        "lookup manifest",
    )
    schema_version = _required_integer(payload, "schema_version")
    if schema_version != LOOKUP_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported lookup manifest schema_version {schema_version!r}"
        )
    artifact_role = _required_text(payload, "artifact_role")
    if artifact_role != LOOKUP_ARTIFACT_ROLE:
        raise ValueError(f"Unsupported lookup artifact_role {artifact_role!r}")
    artifact_version = _required_text(payload, "artifact_version")
    if artifact_version != LOOKUP_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported lookup artifact_version {artifact_version!r}")
    reference_population = _required_text(payload, "reference_population")
    if reference_population != LOOKUP_REFERENCE_POPULATION:
        raise ValueError("Lookup reference_population is not recognized")
    label_source = _required_text(payload, "label_source")
    if label_source != LOOKUP_LABEL_SOURCE:
        raise ValueError("Lookup label_source is not recognized")

    dataset_payload = _required_mapping(payload, "dataset")
    _require_exact_keys(dataset_payload, set(_EXPECTED_DATASET), "dataset")
    if dataset_payload != _EXPECTED_DATASET:
        raise ValueError("Lookup dataset identity is not the published v2_1k source")
    dataset = DatasetIdentity(**dataset_payload)

    label_payload = _required_mapping(payload, "label_contract")
    _require_exact_keys(label_payload, set(_EXPECTED_LABEL_CONTRACT), "label_contract")
    if label_payload != _EXPECTED_LABEL_CONTRACT:
        raise ValueError("Lookup source-label contract is not recognized")
    label_contract = LabelContract(
        included_source_labels=tuple(label_payload["included_source_labels"]),
        excluded_source_labels=tuple(label_payload["excluded_source_labels"]),
        target=label_payload["target"],
        unit=label_payload["unit"],
        third_gender_exclusion=label_payload["third_gender_exclusion"],
        interpretation=label_payload["interpretation"],
    )

    privacy = _parse_privacy(_required_mapping(payload, "privacy"))
    source = _parse_source(_required_mapping(payload, "source"))
    output = _parse_output(_required_mapping(payload, "output"))
    data_contract = _parse_data_contract(
        _required_mapping(payload, "data_contract"), source=source, output=output
    )

    if payload["arrow_schema"] != _EXPECTED_ARROW_SCHEMA:
        raise ValueError("Lookup manifest Arrow schema is not recognized")
    source_code_payload = _required_mapping(payload, "source_code_sha256")
    _require_exact_keys(
        source_code_payload,
        {"model_training/export_lookup_table.py"},
        "source_code_sha256",
    )
    source_code_sha256 = _required_sha256(
        source_code_payload, "model_training/export_lookup_table.py"
    )
    software_payload = _required_mapping(payload, "software_versions")
    _require_exact_keys(software_payload, {"pyarrow"}, "software_versions")

    return FirstNameCompositionManifest(
        schema_version=schema_version,
        artifact_version=artifact_version,
        artifact_role=artifact_role,
        reference_population=reference_population,
        label_source=label_source,
        dataset=dataset,
        label_contract=label_contract,
        privacy=privacy,
        source=source,
        output=output,
        data_contract=data_contract,
        source_code_sha256=source_code_sha256,
        pyarrow_version=_required_text(software_payload, "pyarrow"),
    )


def _parse_privacy(payload: dict[str, Any]) -> PrivacyContract:
    _require_exact_keys(
        payload,
        {"classification", "publication_intent", "release_grain", "rationale"},
        "privacy",
    )
    if payload != _EXPECTED_PRIVACY:
        raise ValueError("Lookup privacy contract is not recognized")
    return PrivacyContract(
        classification=payload["classification"],
        publication_intent=payload["publication_intent"],
        release_grain=payload["release_grain"],
        rationale=payload["rationale"],
    )


def _parse_source(payload: dict[str, Any]) -> SourceArtifact:
    _require_exact_keys(
        payload,
        {"local_filename", "format", "size_bytes", "sha256", "md5", "totals"},
        "source",
    )
    if {key: payload.get(key) for key in _EXPECTED_SOURCE_IDENTITY} != (
        _EXPECTED_SOURCE_IDENTITY
    ):
        raise ValueError("Lookup source identity does not match published v2_1k")
    totals_payload = _required_mapping(payload, "totals")
    _require_exact_keys(
        totals_payload,
        {
            "row_count",
            "state_count",
            "minimum_birth_year",
            "maximum_birth_year",
            "distinct_source_name_count",
            "female_label_record_count",
            "male_label_record_count",
            "represented_binary_label_record_count",
        },
        "source totals",
    )
    label_totals = _parse_label_totals(totals_payload, "source totals")
    minimum_birth_year = _required_integer(totals_payload, "minimum_birth_year")
    maximum_birth_year = _required_integer(totals_payload, "maximum_birth_year")
    state_count = _required_positive_integer(totals_payload, "state_count")
    if state_count != 31 or minimum_birth_year != 1887 or maximum_birth_year != 2017:
        raise ValueError("Lookup source scope does not match published v2_1k")
    return SourceArtifact(
        local_filename=_required_text(payload, "local_filename"),
        format=payload["format"],
        size_bytes=payload["size_bytes"],
        sha256=payload["sha256"],
        md5=payload["md5"],
        row_count=_required_positive_integer(totals_payload, "row_count"),
        state_count=state_count,
        minimum_birth_year=minimum_birth_year,
        maximum_birth_year=maximum_birth_year,
        distinct_source_name_count=_required_positive_integer(
            totals_payload, "distinct_source_name_count"
        ),
        label_totals=label_totals,
    )


def _parse_output(payload: dict[str, Any]) -> OutputArtifact:
    _require_exact_keys(
        payload,
        {"filename", "format", "size_bytes", "sha256", "totals"},
        "output",
    )
    filename = _required_text(payload, "filename")
    if Path(filename).name != filename or not filename.endswith(".parquet"):
        raise ValueError("Lookup output filename must be a bare .parquet name")
    output_format = _required_text(payload, "format")
    if output_format != "parquet_zstd":
        raise ValueError("Lookup output format must be parquet_zstd")
    totals_payload = _required_mapping(payload, "totals")
    _require_exact_keys(
        totals_payload,
        {
            "row_count",
            "female_label_record_count",
            "male_label_record_count",
            "represented_binary_label_record_count",
        },
        "output totals",
    )
    return OutputArtifact(
        filename=filename,
        format=output_format,
        size_bytes=_required_positive_integer(payload, "size_bytes"),
        sha256=_required_sha256(payload, "sha256"),
        row_count=_required_positive_integer(totals_payload, "row_count"),
        label_totals=_parse_label_totals(totals_payload, "output totals"),
    )


def _parse_data_contract(
    payload: dict[str, Any], *, source: SourceArtifact, output: OutputArtifact
) -> DataContract:
    _require_exact_keys(
        payload,
        {
            "unit_of_observation",
            "row_order",
            "normalization",
            "supported_scripts",
            "aggregation",
            "geography",
            "birth_year",
            "support",
        },
        "data_contract",
    )
    if payload["supported_scripts"] != ["Latn"]:
        raise ValueError("Lookup supported_scripts must be exactly ['Latn']")
    for field, expected_value in _EXPECTED_DATA_CONTRACT_TEXT.items():
        if payload[field] != expected_value:
            raise ValueError(f"Lookup data_contract {field!r} is not recognized")

    geography = _required_mapping(payload, "geography")
    _require_exact_keys(
        geography,
        {"source_state_count", "release_level", "published_geography_columns"},
        "geography",
    )
    source_state_count = _required_positive_integer(geography, "source_state_count")
    if source_state_count != source.state_count:
        raise ValueError("Geography state count does not match source totals")
    if geography["release_level"] != "global_only":
        raise ValueError("Lookup geography release_level must be global_only")
    if geography["published_geography_columns"] != []:
        raise ValueError("Lookup must not publish geography columns")

    birth_year = _required_mapping(payload, "birth_year")
    _require_exact_keys(
        birth_year,
        {
            "source_minimum",
            "source_maximum",
            "release_level",
            "published_birth_year_columns",
        },
        "birth_year",
    )
    source_minimum = _required_integer(birth_year, "source_minimum")
    source_maximum = _required_integer(birth_year, "source_maximum")
    if (
        source_minimum != source.minimum_birth_year
        or source_maximum != source.maximum_birth_year
    ):
        raise ValueError("Birth-year bounds do not match source totals")
    if birth_year["release_level"] != "all_years_combined":
        raise ValueError("Lookup birth-year release_level must be all_years_combined")
    if birth_year["published_birth_year_columns"] != []:
        raise ValueError("Lookup must not publish birth-year columns")

    support = _parse_support(_required_mapping(payload, "support"), output=output)
    return DataContract(
        unit_of_observation=_required_text(payload, "unit_of_observation"),
        row_order=payload["row_order"],
        normalization=_required_text(payload, "normalization"),
        supported_scripts=("Latn",),
        aggregation=_required_text(payload, "aggregation"),
        source_state_count=source_state_count,
        source_minimum_birth_year=source_minimum,
        source_maximum_birth_year=source_maximum,
        support=support,
    )


def _parse_support(
    payload: dict[str, Any], *, output: OutputArtifact
) -> SupportContract:
    _require_exact_keys(
        payload,
        {
            "minimum_represented_binary_label_records",
            "support_fields",
            "rule",
            "normalized_name_count_before_support_filter",
            "normalized_name_count_excluded_below_support",
        },
        "support",
    )
    support_fields = payload["support_fields"]
    expected_fields = ["female_label_record_count", "male_label_record_count"]
    if support_fields != expected_fields:
        raise ValueError(f"Lookup support_fields must be {expected_fields!r}")
    minimum_support = _required_positive_integer(
        payload, "minimum_represented_binary_label_records"
    )
    if minimum_support != LOOKUP_MINIMUM_BINARY_LABEL_RECORD_COUNT:
        raise ValueError(
            "minimum_represented_binary_label_records must be exactly 1000"
        )
    expected_rule = (
        "female_label_record_count + male_label_record_count must be at least 1000"
    )
    if payload["rule"] != expected_rule:
        raise ValueError("Lookup support rule is not recognized")
    before_count = _required_positive_integer(
        payload, "normalized_name_count_before_support_filter"
    )
    excluded_count = _required_nonnegative_integer(
        payload, "normalized_name_count_excluded_below_support"
    )
    if before_count - excluded_count != output.row_count:
        raise ValueError("Lookup support-filter row accounting does not reconcile")
    return SupportContract(
        minimum_represented_binary_label_records=minimum_support,
        support_fields=("female_label_record_count", "male_label_record_count"),
        rule=payload["rule"],
        normalized_name_count_before_support_filter=before_count,
        normalized_name_count_excluded_below_support=excluded_count,
    )


def _parse_label_totals(payload: dict[str, Any], description: str) -> BinaryLabelTotals:
    female_count = _required_nonnegative_integer(payload, "female_label_record_count")
    male_count = _required_nonnegative_integer(payload, "male_label_record_count")
    represented_count = _required_nonnegative_integer(
        payload, "represented_binary_label_record_count"
    )
    if female_count + male_count != represented_count:
        raise ValueError(f"{description} binary-label totals do not reconcile")
    return BinaryLabelTotals(female_count, male_count, represented_count)


def _validate_lookup_table(
    path: Path, *, manifest: FirstNameCompositionManifest
) -> None:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if any(
        metadata.row_group(row_group).column(column).compression != "ZSTD"
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    ):
        raise ValueError("Lookup table must use Zstandard compression")
    actual_schema = pq.read_schema(path)
    if not actual_schema.equals(LOOKUP_TABLE_SCHEMA, check_metadata=False):
        raise ValueError(
            f"Lookup table has schema {actual_schema}, expected {LOOKUP_TABLE_SCHEMA}"
        )
    table = pq.read_table(path, schema=LOOKUP_TABLE_SCHEMA)
    normalized_names = table.column("normalized_name").to_pylist()
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("Lookup table normalized_name values must be unique")
    if normalized_names != sorted(normalized_names):
        raise ValueError("Lookup table normalized_name values must be ascending")
    for normalized_name in normalized_names:
        if (
            normalized_name
            != unicodedata.normalize("NFC", normalized_name).strip().casefold()
            or not normalized_name
            or not normalized_name.isascii()
            or not normalized_name.isalpha()
            or not normalized_name.islower()
        ):
            raise ValueError(
                "Lookup table normalized_name values must be lowercase ASCII a-z"
            )

    female_counts = table.column("female_label_record_count").to_numpy()
    male_counts = table.column("male_label_record_count").to_numpy()
    if np.any(female_counts < 0) or np.any(male_counts < 0):
        raise ValueError("Lookup table source-label record counts must be nonnegative")
    represented_counts = female_counts + male_counts
    minimum_support = (
        manifest.data_contract.support.minimum_represented_binary_label_records
    )
    if np.any(represented_counts < minimum_support):
        raise ValueError(
            "Lookup table contains a row below minimum_represented_binary_label_records"
        )
    output = manifest.output
    if table.num_rows != output.row_count:
        raise ValueError("Lookup table row count does not match the manifest")
    if int(female_counts.sum()) != output.label_totals.female_label_record_count:
        raise ValueError("Lookup table female total does not match the manifest")
    if int(male_counts.sum()) != output.label_totals.male_label_record_count:
        raise ValueError("Lookup table male total does not match the manifest")


def _verify_sha256(path: Path, expected_digest: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            f"Artifact hash mismatch for {path.name!r}: "
            f"expected {expected_digest}, found {actual_digest}"
        )


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], description: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{description} keys must be {sorted(expected)!r}; found {sorted(actual)!r}"
        )


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} must be a JSON object")
    return value


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be non-empty text")
    return value


def _required_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer")
    return value


def _required_positive_integer(payload: dict[str, Any], key: str) -> int:
    value = _required_integer(payload, key)
    if value <= 0:
        raise ValueError(f"{key!r} must be positive")
    return value


def _required_nonnegative_integer(payload: dict[str, Any], key: str) -> int:
    value = _required_integer(payload, key)
    if value < 0:
        raise ValueError(f"{key!r} must be nonnegative")
    return value


def _required_sha256(payload: dict[str, Any], key: str) -> str:
    digest = _required_text(payload, key)
    if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ValueError(f"{key!r} must be 64 lowercase hexadecimal characters")
    return digest


def _required_md5(payload: dict[str, Any], key: str) -> str:
    digest = _required_text(payload, key)
    if len(digest) != 32 or set(digest) - set("0123456789abcdef"):
        raise ValueError(f"{key!r} must be 32 lowercase hexadecimal characters")
    return digest
