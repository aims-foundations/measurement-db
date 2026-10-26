"""Index flat canonical Parquet rows for exact, lazy browser reads.

Only IDs and page headers are read. Large trace strings stay in the source file.
This is a storage index: it does not interpret grades, answers, or display fields.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import struct
from pathlib import Path

import pyarrow.parquet as pq
from fastparquet.cencoding import NumpyIO, ThriftObject


def write_index(path: Path, value) -> None:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_bytes(gzip.compress(raw.encode(), mtime=0))


def export_parquet(path: Path, output: Path, key: str, url: str) -> dict:
    output.mkdir()
    table = pq.ParquetFile(path)
    buckets: list[dict] = [{} for _ in range(256)]
    for group in range(table.num_row_groups):
        ids = table.read_row_group(group, columns=[key]).column(key).to_pylist()
        for row, identity in enumerate(ids):
            if not isinstance(identity, str):
                raise ValueError(f"{path}: missing {key}")
            bucket = buckets[hashlib.sha256(identity.encode()).digest()[0]]
            if identity in bucket:
                raise ValueError(f"{path}: duplicate {key}: {identity}")
            bucket[identity] = [group, row]
    for number, bucket in enumerate(buckets):
        write_index(output / f"index-{number:02x}.json.gz", bucket)

    with path.open("rb") as stream:
        stream.seek(-8, 2)
        length = struct.unpack("<I", stream.read(4))[0]
        stream.seek(-length - 8, 2)
        metadata = ThriftObject.from_buffer(stream.read(length), "FileMetaData")
        for number, group in enumerate(metadata.row_groups):
            pages = {}
            for chunk in group.columns:
                column = chunk.meta_data
                field = ".".join(p.decode() if isinstance(p, bytes) else p
                                 for p in column.path_in_schema)
                offset = column.dictionary_page_offset or column.data_page_offset
                end = offset + column.total_compressed_size
                row, locations = 0, []
                while offset < end:
                    stream.seek(offset)
                    reader = NumpyIO(stream.read(min(65536, end - offset)))
                    header = ThriftObject.from_buffer(reader, "PageHeader")
                    size = reader.tell() + header.compressed_page_size
                    if header.type in (0, 3):
                        data = header.data_page_header if header.type == 0 else header.data_page_header_v2
                        locations.append([offset, size, row])
                        row += data.num_values
                    offset += size
                if row != group.num_rows or offset != end:
                    raise ValueError(f"{path}: unsupported flat page layout for {field}")
                pages[field] = locations
            footer = bytes(ThriftObject.from_fields(
                "FileMetaData", version=metadata.version, schema=metadata.schema,
                num_rows=group.num_rows, row_groups=[group], key_value_metadata=[]).to_bytes())
            footer += struct.pack("<I", len(footer)) + b"PAR1"
            write_index(output / f"group-{number}.json.gz", {
                "footer": base64.b64encode(footer).decode(), "pages": pages,
            })
    return {"key": key, "rows": table.metadata.num_rows, "pages": 0,
            "parquet": {"url": url, "bytes": path.stat().st_size}}
