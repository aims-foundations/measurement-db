"""Write validated registration tables into a benchmark dataset folder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .register_measurements import _locked_registration_rows
from .validate_measurement_tables import (
    PARQUET_SCHEMAS,
    _validated_registration_tables,
    validate_response_grades,
    validate_dataset,
    validate_table,
)


PARQUET_SCHEMA_VERSION = "3"
_ARROW_BATCH_ROWS = 10_000
_ARROW_TYPES = {
    "string": pa.string(),
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "binary": pa.binary(),
    "list[string]": pa.list_(pa.string()),
}


def canonical_arrow_schema(table_name: str) -> pa.Schema:
    """The complete physical schema shared by every benchmark's table."""
    return pa.schema([
        pa.field(column["name"], _ARROW_TYPES[column["type"]],
                 nullable=column.get("nullable", True))
        for column in PARQUET_SCHEMAS[table_name]["columns"]
    ])


def validate_parquet_schema(table_name: str, schema: pa.Schema) -> None:
    """Reject missing/extra/reordered columns or differing physical types."""
    expected = canonical_arrow_schema(table_name)
    if not schema.remove_metadata().equals(expected, check_metadata=False):
        raise ValueError(
            f"{table_name}.parquet must use the fixed canonical columns, types, "
            f"and nullability; expected {expected}, got {schema.remove_metadata()}"
        )


def write_parquet(
    table: pd.DataFrame, path: Path | str, *, item_id_version: str = "4",
    table_name: str | None = None, response_scale: dict | str | None = None,
    items: pd.DataFrame | None = None, **kwargs,
) -> None:
    """Validate and write canonical fields with fixed Arrow types.

    Infer the table name from its canonical filename unless supplied. All-null
    and empty columns use their declared types and nullability, just as populated
    columns do. Every written table contains exactly the full canonical schema,
    including derived benchmark statistics. No extension columns are permitted.
    Response writes require an explicit scale and, for mixed scales, the item
    table. Every table requires non-null, unique schema-declared primary keys.
    Responses and traces also require positive integer trials and unique
    observation keys. All validation and Arrow conversion finish before opening
    the file; duplicate rows are never silently removed.
    """
    name = table_name or Path(path).stem
    if name not in PARQUET_SCHEMAS:
        raise ValueError(f"Unknown measurement table {name!r}; supply table_name")
    validate_table(
        name, table, include_derived=True, context=str(path),
    )
    if name == "responses":
        if response_scale is None:
            raise ValueError("writing responses requires an explicit response_scale")
        validate_response_grades(table, response_scale, items=items)
    # Arrow's canonical string/binary types have 32-bit offsets per array.
    # Convert bounded row batches so a large trace corpus does not require one
    # array exceeding 2 GB. Keep complete values and finish conversion before
    # opening the output, including for an empty table. Materialize each batch:
    # slicing pandas' Arrow strings otherwise retains the oversized parent buffer.
    arrow = pa.concat_tables([
        pa.Table.from_pandas(
            table.iloc[start:start + _ARROW_BATCH_ROWS].astype(object),
            schema=canonical_arrow_schema(name), preserve_index=False, safe=True,
        )
        for start in range(0, max(len(table), 1), _ARROW_BATCH_ROWS)
    ])
    metadata = dict(arrow.schema.metadata or {})
    metadata[b"measurement_db.schema_version"] = PARQUET_SCHEMA_VERSION.encode()
    metadata[b"measurement_db.response_id_version"] = b"1"
    metadata[b"measurement_db.item_id_version"] = item_id_version.encode()
    kwargs.setdefault("row_group_size", _ARROW_BATCH_ROWS)
    pq.write_table(arrow.replace_schema_metadata(metadata), path, **kwargs)


def save(out_dir: Path | str, *, additional_tables: dict[str, pd.DataFrame] | None = None,
         expected_benchmark_id: str | None = None) -> None:
    """Validate and write this process's registered measurement tables.

    Non-empty ``subjects``, ``items``, and ``benchmarks`` tables are written
    beside the benchmark's ``responses.parquet``. Validation completes before
    the output directory is created, preventing partial output on failure.
    Completed builders supply their responses/traces/assets as additional_tables
    to validate the complete dataset before these registry tables are written.
    """
    output_directory = Path(out_dir)
    with _locked_registration_rows() as rows:
        tables = _validated_registration_tables(rows, output_directory.name, require_complete=True)
        if additional_tables is not None:
            if set(additional_tables) - {"responses", "traces", "assets"}:
                raise ValueError("additional_tables may only contain responses, traces, and assets")
            validate_dataset({**tables, **additional_tables},
                             expected_benchmark_id=expected_benchmark_id, context=str(output_directory))
        output_directory.mkdir(parents=True, exist_ok=True)
        for table_name, table in tables.items():
            write_parquet(table, output_directory / f"{table_name}.parquet")
