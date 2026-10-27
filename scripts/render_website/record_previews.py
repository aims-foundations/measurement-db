"""Bounded, exact record reads when a source Parquet page is too large.

This optional storage cache is prepared separately from website builds. It contains
only records blocked by the original reader; canonical tables remain unchanged.
Every payload is independently compressed and checksummed. Large strings are split
at Unicode boundaries, so clients can read every part without loading the whole value.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from fastparquet.cencoding import NumpyIO, ThriftObject

SOURCE_LIMIT = 16 * 1024**2
RECORD_LIMIT = 2 * 1024**2
TEXT_CHARS = 32 * 1024  # At most 128 KiB UTF-8, even for four-byte characters.
PACK_LIMIT = 64 * 1024**2


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def source_ranges(path: Path) -> list[dict]:
    """Measure the current reader's downloaded-plus-decompressed bytes per row."""
    groups = []
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        length = struct.unpack("<I", stream.read(4))[0]
        stream.seek(-length - 8, 2)
        metadata = ThriftObject.from_buffer(stream.read(length), "FileMetaData")
        for group in metadata.row_groups:
            events, fixed = defaultdict(int), 0
            for chunk in group.columns:
                column = chunk.meta_data
                offset = column.dictionary_page_offset or column.data_page_offset
                end = offset + column.total_compressed_size
                row = 0
                while offset < end:
                    stream.seek(offset)
                    reader = NumpyIO(stream.read(min(65536, end - offset)))
                    header = ThriftObject.from_buffer(reader, "PageHeader")
                    size = reader.tell() + header.compressed_page_size
                    cost = size + (header.uncompressed_page_size if column.codec else 0)
                    if header.type in (0, 3):
                        page = header.data_page_header if header.type == 0 else header.data_page_header_v2
                        events[row] += cost
                        row += page.num_values
                        events[row] -= cost
                    else:
                        fixed += cost
                    offset += size
                if row != group.num_rows or offset != end:
                    raise ValueError(f"Unsupported flat page layout: {path}")
            cost, previous, ranges, maximum = fixed, 0, [], 0
            for position, delta in sorted(events.items()):
                if position > previous:
                    maximum = max(maximum, cost)
                    if cost > SOURCE_LIMIT:
                        if ranges and ranges[-1][1] == previous:
                            ranges[-1][1] = position
                        else:
                            ranges.append([previous, position])
                cost += delta
                previous = position
            groups.append({"rows": group.num_rows, "ranges": ranges, "maximum": maximum})
    return groups


class RecordPacks:
    def __init__(self, output: Path):
        self.output = output
        self.files: list[int] = []
        self.stream = None

    def put(self, raw: bytes) -> list:
        if len(raw) > 4 * 1024**2:
            raise ValueError("Record or index exceeds the browser payload limit")
        compressed = gzip.compress(raw, compresslevel=1, mtime=0)
        if self.stream is None or (self.files[-1] and self.files[-1] + len(compressed) > PACK_LIMIT):
            if self.stream:
                self.stream.close()
            self.files.append(0)
            self.stream = (self.output / f"pack-{len(self.files)-1}.bin").open("wb")
        offset = self.files[-1]
        self.stream.write(compressed)
        self.files[-1] += len(compressed)
        return [len(self.files)-1, offset, len(compressed), len(raw), hashlib.sha256(compressed).hexdigest()]

    def close(self):
        if self.stream:
            self.stream.close()


def export_previews(path: Path, output: Path, key: str, groups: list[dict] | None = None) -> dict:
    """Preserve every selected row, including nulls, IDs, and full text."""
    groups = source_ranges(path) if groups is None else groups
    expected = sum(end - start for group in groups for start, end in group["ranges"])
    output.mkdir(parents=True, exist_ok=False)
    packs = RecordPacks(output)
    buckets: list[dict] = [{} for _ in range(256)]
    count = large = 0
    checksum = hashlib.sha256()
    source = pq.ParquetFile(path)
    try:
        for gi, group in enumerate(groups):
            if not group["ranges"]:
                continue
            position = 0
            for batch in source.iter_batches(batch_size=32, row_groups=[gi]):
                stop = position + len(batch)
                intersections = [(max(start, position)-position, min(end, stop)-position)
                                 for start, end in group["ranges"] if start < stop and end > position]
                for start, end in intersections:
                    for record in batch.slice(start, end-start).to_pylist():
                        identity = record[key]
                        bucket = buckets[hashlib.sha256(identity.encode()).digest()[0]]
                        if identity in bucket:
                            raise ValueError(f"Duplicate {key}: {identity}")
                        raw = encoded(record)
                        checksum.update(hashlib.sha256(raw).digest())
                        if len(raw) > RECORD_LIMIT:
                            fields = {}
                            small = dict(record)
                            # Each original long field is still accessible in full, in order.
                            for name, value in record.items():
                                if isinstance(value, str) and len(value) > TEXT_CHARS:
                                    fields[name] = [packs.put(value[i:i+TEXT_CHARS].encode())
                                                    for i in range(0, len(value), TEXT_CHARS)]
                                    del small[name]
                            raw = encoded({"kind": "paged-record", "record": small, "fields": fields})
                            if len(raw) > RECORD_LIMIT or not fields:
                                raise ValueError(f"Record requires unsupported non-text paging: {identity}")
                            large += 1
                        else:
                            raw = encoded({"kind": "record", "record": record})
                        bucket[identity] = packs.put(raw)
                        count += 1
                position = stop
        if count != expected:
            raise ValueError(f"Expected {expected} records, exported {count}")
    finally:
        packs.close()
    # Pack the small ID indexes too: publication has hundreds of packs, not
    # tens of thousands of tiny files. No row-wide index enters the browser.
    packs.stream = None
    indexes = [packs.put(encoded(bucket)) for bucket in buckets]
    packs.close()
    with path.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"source_sha256": source_sha, "format": "record-preview-v1", "rows": count, "paged": large,
              "source_bytes": path.stat().st_size, "source_rows": source.metadata.num_rows,
              "record_checksum": checksum.hexdigest(), "packs": packs.files, "indexes": indexes}
    (output / "manifest.json").write_bytes(encoded(result))
    return result


def verify_previews(path: Path, directory: Path, key: str) -> dict:
    """Reconstruct the archive and compare its full-row digest to the source digest."""
    manifest = json.loads((directory / "manifest.json").read_text())
    handles = [(directory / f"pack-{i}.bin").open("rb") for i in range(len(manifest["packs"]))]
    def read(location):
        pack, offset, length, expanded, digest = location
        handle = handles[pack]
        handle.seek(offset)
        compressed = handle.read(length)
        if hashlib.sha256(compressed).hexdigest() != digest:
            raise ValueError("Record range checksum mismatch")
        raw = gzip.decompress(compressed)
        if len(raw) != expanded:
            raise ValueError("Record range size mismatch")
        return raw
    try:
        buckets = [json.loads(read(location)) for location in manifest["indexes"]]
        groups = source_ranges(path)
        source = pq.ParquetFile(path)
        digest = hashlib.sha256()
        count = 0
        for gi, group in enumerate(groups):
            if not group["ranges"]:
                continue
            ids = source.read_row_group(gi, columns=[key]).column(key).to_pylist()
            for start, end in group["ranges"]:
                for identity in ids[start:end]:
                    bucket = buckets[hashlib.sha256(identity.encode()).digest()[0]]
                    value = json.loads(read(bucket[identity]))
                    record = value["record"]
                    if value["kind"] == "paged-record":
                        for field, parts in value["fields"].items():
                            record[field] = "".join(read(part).decode() for part in parts)
                    if record[key] != identity:
                        raise ValueError("Record ID mismatch")
                    record = {name: record[name] for name in source.schema_arrow.names}
                    digest.update(hashlib.sha256(encoded(record)).digest())
                    count += 1
        if digest.hexdigest() != manifest["record_checksum"] or count != manifest["rows"]:
            raise ValueError("Reconstructed records differ from the original source rows")
        with path.open("rb") as stream:
            source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest.get("source_sha256") and source_sha != manifest["source_sha256"]:
            raise ValueError("Source table changed; prepare a new archive")
        return {"rows": count, "record_checksum": digest.hexdigest(), "source_sha256": source_sha}
    finally:
        for handle in handles:
            handle.close()
