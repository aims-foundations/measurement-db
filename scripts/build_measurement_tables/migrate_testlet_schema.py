"""Upgrade canonical v3 tables to v4 without inferring testlets or changing IDs.

python -m scripts.build_measurement_tables.migrate_testlet_schema SOURCE OUTPUT

SOURCE is one formatted_tables directory. OUTPUT must not exist. The two new
nullable links start unrecorded; unchanged tables are copied byte-for-byte.
ProgramBench's finer-grained conversion uses its builder, not this migration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from .write_measurement_tables import canonical_arrow_schema, validate_parquet_schema

ADDED = {"items": "testlet_id", "responses": "attempt_id"}
BATCH_ROWS = 10_000


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def migrate_table(source: Path, output: Path) -> dict:
    name = source.stem
    parquet = pq.ParquetFile(source)
    added = ADDED.get(name)
    old = parquet.schema_arrow
    target = canonical_arrow_schema(name)
    if added and added not in old.names:
        expected = pa.schema([field for field in target if field.name != added])
        if not old.remove_metadata().equals(expected):
            raise ValueError(f"{source}: expected the canonical v3 schema")
        version = (old.metadata or {}).get(b"measurement_db.schema_version")
        if version != b"3":
            raise ValueError(f"{source}: expected schema version 3, got {version!r}")
        metadata = {**(old.metadata or {}), b"measurement_db.schema_version": b"4"}
        with pq.ParquetWriter(output, target.with_metadata(metadata), compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
                columns = {field: batch.column(field) for field in batch.schema.names}
                columns[added] = pa.nulls(batch.num_rows, type=pa.string())
                writer.write_batch(pa.RecordBatch.from_arrays(
                    [columns[field] for field in target.names], schema=target))
        # Read back every original value, including IDs, nulls, grades and order.
        migrated = pq.ParquetFile(output)
        validate_parquet_schema(name, migrated.schema_arrow)
        if migrated.metadata.num_rows != parquet.metadata.num_rows:
            raise ValueError(f"{source}: row count changed")
        before = parquet.iter_batches(batch_size=BATCH_ROWS)
        after = migrated.iter_batches(batch_size=BATCH_ROWS)
        for left, right in zip(before, after, strict=True):
            if right.column(added).null_count != right.num_rows:
                raise ValueError(f"{source}: migration invented a relationship")
            if not left.equals(right.select(old.names), check_metadata=False):
                raise ValueError(f"{source}: migration changed existing values")
    else:
        validate_parquet_schema(name, old)
        shutil.copyfile(source, output)
    return {"rows": parquet.metadata.num_rows, "added": added if added not in old.names else None,
            "source_sha256": digest(source), "output_sha256": digest(output)}


def migrate_dataset(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be separate from the source dataset")
    if output.exists():
        raise FileExistsError(output)
    files = sorted(source.glob("*.parquet"))
    if not {"items", "subjects", "benchmarks"} <= {p.stem for p in files}:
        raise ValueError("Source must contain canonical items, subjects and benchmarks tables")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".testlet-migration-", dir=output.parent) as temporary:
        stage = Path(temporary) / "tables"
        stage.mkdir()
        receipt = {p.stem: migrate_table(p, stage / p.name) for p in files}
        (stage / "migration.json").write_text(json.dumps(receipt, indent=2) + "\n")
        stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_dataset(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
