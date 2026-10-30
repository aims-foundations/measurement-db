"""Write validated registration tables into a benchmark dataset folder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .register_measurements import _locked_registration_rows
from .validate_measurement_tables import (
    PARQUET_SCHEMAS,
    _validated_registration_tables,
    validate_response_grades,
    validate_dataset,
    validate_table,
)


PARQUET_SCHEMA_VERSION = "4"
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
    items: pd.DataFrame | None = None, row_group_bytes: int | None = None, **kwargs,
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
    arrow = arrow.replace_schema_metadata(metadata)
    if row_group_bytes is None:
        kwargs.setdefault("row_group_size", _ARROW_BATCH_ROWS)
        pq.write_table(arrow, path, **kwargs)
    else:
        if row_group_bytes <= 0:
            raise ValueError("row_group_bytes must be positive")
        max_rows = kwargs.pop("row_group_size", _ARROW_BATCH_ROWS)
        # Long text has little useful min/max information. Omitting its
        # statistics also prevents large records from inflating the footer.
        if name in {"items", "attempts", "traces"}:
            large_fields = {"content", "grading_criterion", "verifier", "trace"}
            kwargs.setdefault("write_statistics", [
                field for field in arrow.column_names if field not in large_fields
            ])
        groups = list(_byte_bounded_groups(arrow, row_group_bytes, max_rows))
        with pq.ParquetWriter(path, arrow.schema, **kwargs) as output:
            for start, count in groups:
                output.write_table(arrow.slice(start, count), row_group_size=count)


def _byte_bounded_groups(table, budget, max_rows):
    """Bound decoded row groups; an oversized record occupies its own group."""
    start, count, size, position = 0, 0, 0, 0
    for batch in table.to_batches(max_chunksize=_ARROW_BATCH_ROWS):
        sizes = [0] * batch.num_rows
        for column in batch.columns:
            if pa.types.is_string(column.type) or pa.types.is_binary(column.type):
                lengths = pc.fill_null(pc.binary_length(column), 0).to_pylist()
                sizes = [a + b + 5 for a, b in zip(sizes, lengths)]
            else:
                sizes = [a + column.type.bit_width // 8 + 1 for a in sizes]
        for row_size in sizes:
            if count and (size + row_size > budget or count == max_rows):
                yield start, count
                start, count, size = position, 0, 0
            size += row_size
            count += 1
            position += 1
    if count:
        yield start, count


def save(out_dir: Path | str, *, additional_tables: dict[str, pd.DataFrame] | None = None,
         expected_benchmark_id: str | None = None) -> None:
    """Validate and write this process's registered measurement tables.

    Non-empty ``subjects``, ``items``, and ``benchmarks`` tables are written
    beside the benchmark's ``responses.parquet``. Validation completes before
    the output directory is created, preventing partial output on failure.
    Completed builders supply their responses/traces/attempts/assets as additional_tables
    to validate the complete dataset before these registry tables are written.
    """
    output_directory = Path(out_dir)
    with _locked_registration_rows() as rows:
        tables = _validated_registration_tables(rows, output_directory.name, require_complete=True)
        if additional_tables is not None:
            if set(additional_tables) - {"responses", "traces", "attempts", "assets"}:
                raise ValueError("additional_tables may only contain responses, traces, attempts, and assets")
            validate_dataset({**tables, **additional_tables},
                             expected_benchmark_id=expected_benchmark_id, context=str(output_directory))
        output_directory.mkdir(parents=True, exist_ok=True)
        for table_name, table in tables.items():
            write_parquet(table, output_directory / f"{table_name}.parquet",
                row_group_bytes=2 * 1024 * 1024 if table_name == 'items' else None)
