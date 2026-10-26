"""Verify a raw gallery export against every canonical source row before packing."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def check(source: Path, output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    subjects = set(json.loads((output / "subjects/index.json").read_text()))
    items = set(json.loads((output / "items/index.json").read_text()))
    report = {}
    for name, descriptor in manifest["tables"].items():
        table = pq.ParquetFile(source / f"{name}.parquet")
        if table.metadata.num_rows != descriptor["rows"]:
            raise ValueError(f"{name}: source row count differs from export")
        key = descriptor["key"]
        directory = output / name
        if "parquet" in descriptor:
            buckets = [json.loads(gzip.decompress((directory / f"index-{i:02x}.json.gz").read_bytes()))
                       for i in range(256)]
            count = 0
            for group in range(table.num_row_groups):
                for row, identity in enumerate(table.read_row_group(group, columns=[key]).column(key).to_pylist()):
                    bucket = buckets[hashlib.sha256(identity.encode()).digest()[0]]
                    if bucket.get(identity) != [group, row]:
                        raise ValueError(f"{name}: wrong source position for {identity}")
                    count += 1
            if sum(map(len, buckets)) != count:
                raise ValueError(f"{name}: extra lookup identities")
            report[name] = {"rows": count, "exact_source_positions": True}
            del buckets
            continue

        index = json.loads((directory / "index.json").read_text())
        @lru_cache(maxsize=4)
        def page(number):
            return json.loads((directory / f"{number}.json").read_text())

        counts: Counter = Counter()
        subject_pages: dict[str, set[int]] = {}
        count = 0
        for batch in table.iter_batches():
            for expected in batch.to_pylist():
                identity = expected[key]
                number = index[identity]
                actual = page(number)[identity]
                if name == "assets":
                    actual = {**actual, "data": (directory / actual["data_url"]).read_bytes()}
                    del actual["data_url"]
                if actual != expected:
                    raise ValueError(f"{name}: changed source fields for {identity}")
                if name == "responses":
                    if expected["subject_id"] not in subjects or expected["item_id"] not in items:
                        raise ValueError(f"responses: unknown subject or item in {identity}")
                    counts[expected["response"]] += 1
                    subject_pages.setdefault(expected["subject_id"], set()).add(number)
                count += 1
        if len(index) != count:
            raise ValueError(f"{name}: extra lookup identities")
        report[name] = {"rows": count, "exact_records": True}
        if name == "responses":
            if json.loads((directory / descriptor["bySubject"]).read_text()) != {k: sorted(v) for k, v in subject_pages.items()}:
                raise ValueError("Response pages do not cover every subject")
            report[name]["grades"] = {"null" if k is None else str(k): v for k, v in counts.items()}
        page.cache_clear()
    return report
