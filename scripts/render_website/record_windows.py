"""Bounded pages of unchanged observations, grouped by item for comparison.

SQLite supplies a disk-backed sort; neither the exporter nor the browser holds
the full response table. Every item is indexed, including unobserved items.
An item too large for one page is split explicitly into partial pages.
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

MAX_ROWS = 4096
MAX_ITEMS = 256
MAX_BYTES = 2 * 1024 * 1024


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def write(path: Path, value) -> int:
    raw = encoded(value).encode()
    path.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    return len(raw)


def export_windows(directory: Path, output: Path, *, max_rows=MAX_ROWS,
                   max_items=MAX_ITEMS, max_bytes=MAX_BYTES) -> dict:
    output.mkdir()
    items = sorted(pq.read_table(directory / "items.parquet", columns=["item_id"])
                   .column("item_id").to_pylist())
    known_items = set(items)
    if len(known_items) != len(items):
        raise ValueError("Duplicate item IDs")
    subjects = set(pq.read_table(directory / "subjects.parquet", columns=["subject_id"])
                   .column("subject_id").to_pylist())
    buckets = [{} for _ in range(256)]
    page, count, largest = 0, 0, 0
    item_ids, observations, partial = [], [], []
    size = 128  # Reserve the JSON envelope and separators.

    def flush():
        nonlocal page, item_ids, observations, partial, size, largest
        if not item_ids:
            return
        raw = write(output / f"{page}.json.gz", {
            "itemIds": item_ids, "observations": observations, "partialItemIds": partial,
        })
        if raw > max_bytes or len(observations) > max_rows:
            raise ValueError("Observation page exceeds its declared budget")
        largest = max(largest, raw)
        for identity in item_ids:
            bucket = buckets[hashlib.sha256(identity.encode()).digest()[0]]
            if identity in bucket:
                bucket[identity][1] = page
            else:
                bucket[identity] = [page, page]
        page += 1
        item_ids, observations, partial, size = [], [], [], 128

    def parts(rows, identity):
        values, length = [], 0
        allowance = max_bytes - 256 - 4 * len(encoded(identity).encode())
        for _, payload in rows:
            n = len(payload.encode()) + 1
            if n > allowance:
                raise ValueError(f"Observation for {identity} exceeds the page byte budget")
            if values and (len(values) == max_rows or length + n > allowance):
                yield values, length
                values, length = [], 0
            values.append(json.loads(payload))
            length += n
        yield values, length

    with TemporaryDirectory(prefix="record-sort-") as temporary:
        with sqlite3.connect(str(Path(temporary) / "rows.sqlite")) as db:
            db.executescript("""
                PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                PRAGMA cache_size=-65536; PRAGMA temp_store=FILE;
                CREATE TABLE observations (response_id TEXT PRIMARY KEY NOT NULL, item_id TEXT, payload TEXT);
            """)
            path = directory / "responses.parquet"
            if path.exists():
                for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
                    rows = batch.to_pylist()
                    for row in rows:
                        if not isinstance(row["response_id"], str):
                            raise ValueError("Missing response_id")
                        if row["item_id"] not in known_items or row["subject_id"] not in subjects:
                            raise ValueError(f"Unknown observation reference: {row['response_id']}")
                    db.executemany("INSERT INTO observations VALUES (?, ?, ?)",
                                   ((r["response_id"], r["item_id"], encoded(r)) for r in rows))
                    count += len(rows)
                db.commit()
            groups = iter(itertools.groupby(db.execute(
                "SELECT item_id, payload FROM observations ORDER BY item_id, response_id"),
                key=lambda row: row[0]))
            current = next(groups, None)
            for identity in items:
                rows = current[1] if current and current[0] == identity else ()
                chunks = iter(parts(rows, identity))
                chunk = next(chunks)
                part = 0
                while chunk is not None:
                    following = next(chunks, None)
                    values, length = chunk
                    split = bool(part or following is not None)
                    added = length + len(encoded(identity).encode()) * (2 if split else 1) + 2
                    if item_ids and (split or len(item_ids) == max_items or
                                     len(observations) + len(values) > max_rows or size + added > max_bytes):
                        flush()
                    item_ids.append(identity)
                    observations.extend(values)
                    size += added
                    if split:
                        partial.append(identity)
                        flush()
                    chunk, part = following, part + 1
                if current and current[0] == identity:
                    current = next(groups, None)
            flush()
    for number, bucket in enumerate(buckets):
        write(output / f"index-{number:02x}.json.gz", bucket)
    return {"pages": page, "rows": count, "maxRows": max_rows, "maxItems": max_items,
            "maxBytes": max_bytes, "largestPageBytes": largest}
