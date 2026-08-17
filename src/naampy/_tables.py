"""Typed electoral-roll table storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

ROLL_COLUMNS = (
    "state",
    "birth_year",
    "first_name",
    "n_female",
    "n_male",
    "n_third_gender",
)

ROLL_SCHEMA = pa.schema(
    [
        pa.field("state", pa.string(), nullable=False),
        pa.field("birth_year", pa.int16(), nullable=False),
        pa.field("first_name", pa.string(), nullable=False),
        pa.field("n_female", pa.int64(), nullable=False),
        pa.field("n_male", pa.int64(), nullable=False),
        pa.field("n_third_gender", pa.int64(), nullable=False),
    ]
)

_CSV_TYPES = {
    "state": pa.string(),
    "birth_year": pa.float64(),
    "first_name": pa.string(),
    "n_female": pa.int64(),
    "n_male": pa.int64(),
    "n_third_gender": pa.int64(),
}


def _validated_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Validate and normalize one source batch."""
    for column in ("state", "birth_year", "n_female", "n_male", "n_third_gender"):
        if batch.column(column).null_count:
            raise ValueError(f"Electoral-roll column {column!r} contains nulls")

    years = batch.column("birth_year")
    year_values = years.to_numpy(zero_copy_only=False)
    if np.any(year_values != np.floor(year_values)):
        raise ValueError("Electoral-roll birth_year values must be whole years")
    if np.any((year_values < -32768) | (year_values > 32767)):
        raise ValueError("Electoral-roll birth_year values exceed int16 range")

    for column in ("n_female", "n_male", "n_third_gender"):
        values = batch.column(column).to_numpy(zero_copy_only=False)
        if np.any(values < 0):
            raise ValueError(f"Electoral-roll column {column!r} contains negatives")

    names = batch.column("first_name")
    keep = pa.array(
        [name is not None and bool(name.strip()) for name in names.to_pylist()]
    )
    batch = batch.filter(keep)

    arrays = [
        batch.column("state"),
        batch.column("birth_year").cast(pa.int16(), safe=True),
        batch.column("first_name"),
        batch.column("n_female"),
        batch.column("n_male"),
        batch.column("n_third_gender"),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=ROLL_SCHEMA)


def csv_to_parquet(source: str | Path, target: str | Path) -> Path:
    """Convert a Dataverse CSV transport file into an atomic typed cache.

    Args:
        source: Source CSV or CSV.gz path.
        target: Destination Parquet path.

    Returns:
        Path to the completed Parquet file.

    Raises:
        ValueError: If required values violate the electoral-roll schema.
    """
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    options = csv.ConvertOptions(
        column_types=_CSV_TYPES,
        include_columns=list(ROLL_COLUMNS),
        strings_can_be_null=True,
    )
    reader = csv.open_csv(source, convert_options=options)

    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    rows = 0
    try:
        with pq.ParquetWriter(temporary, ROLL_SCHEMA, compression="zstd") as writer:
            for batch in reader:
                clean = _validated_batch(batch)
                writer.write_batch(clean)
                rows += clean.num_rows
        if rows == 0:
            raise ValueError("Electoral-roll source contains no usable rows")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    return target


def validate_parquet(path: str | Path) -> Path:
    """Verify that a cached table has the canonical Arrow schema."""
    path = Path(path)
    actual = pq.read_schema(path)
    if not actual.equals(ROLL_SCHEMA, check_metadata=False):
        raise ValueError(
            f"Electoral-roll cache has schema {actual}, expected {ROLL_SCHEMA}"
        )
    return path
