"""Pack generated lookup buckets into deterministic, independently readable gzip ranges.

Small neighboring hash buckets are combined up to 512 KiB compressed / 4 MiB
expanded. Existing oversized buckets stay isolated. No source tables or fitted
values are changed. Only the preparation worker calls this module.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct

FORMAT = "gallery-lookup-pack-v1"
CHUNK_BYTES = 512 * 1024
CHUNK_RAW = 4 * 1024 * 1024
PACK_BYTES = 8 * 1024 * 1024
BUCKET = re.compile(r"[0-9a-f]{2}\.json\.gz")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def normalized(path: Path) -> bytes:
    """Strip timestamp/filename headers without retaining expanded large chunks."""
    output = io.BytesIO()
    with gzip.open(path, "rb") as source, gzip.GzipFile(
            fileobj=output, mode="wb", filename="", mtime=0, compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    return output.getvalue()


def family_chunks(directory: Path):
    sizes = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.is_symlink() or not BUCKET.fullmatch(path.name):
            raise ValueError(f"Unexpected lookup bucket: {path}")
        with path.open("rb") as stream:
            stream.seek(-4, 2)
            raw = struct.unpack("<I", stream.read(4))[0]
        sizes[int(path.name[:2], 16)] = (path, path.stat().st_size, raw)
    bits = 8
    for candidate in range(9):
        width = 256 >> candidate
        groups = [[v for k, v in sizes.items() if start <= k < start + width]
                  for start in range(0, 256, width)]
        if all(len(group) <= 1 or (sum(v[1] for v in group) <= CHUNK_BYTES
                                  and sum(v[2] for v in group) <= CHUNK_RAW) for group in groups):
            bits = candidate
            break

    def chunks():
        width = 256 >> bits
        for start in range(0, 256, width):
            members = [sizes[k] for k in sorted(sizes) if start <= k < start + width]
            if not members:
                continue
            if len(members) == 1:
                payload = normalized(members[0][0])
            else:
                entries = {}
                for path, _, _ in members:
                    with gzip.open(path, "rb") as stream:
                        raw = stream.read(CHUNK_RAW + 1)
                    if len(raw) > CHUNK_RAW:
                        raise ValueError("Expanded lookup bucket exceeds its declared merge budget")
                    values = json.loads(raw)
                    if entries.keys() & values.keys():
                        raise ValueError("Duplicate lookup key across buckets")
                    entries.update(values)
                payload = gzip.compress(encoded(entries), compresslevel=6, mtime=0)
            yield f"{start // width:02x}", payload
    return bits, chunks()


def pack_lookups(directory: Path) -> dict:
    destination = directory / "lookup-pack"
    destination.mkdir()  # Repacking an already packed bundle is an error.
    index = {"format": FORMAT, "families": {}, "packs": []}
    buffer = bytearray()

    def flush():
        if not buffer:
            return
        sha = digest(buffer)
        name = sha + ".bin"
        (destination / name).write_bytes(buffer)
        index["packs"].append({"file": name, "bytes": len(buffer), "sha256": sha})
        buffer.clear()

    families = [p for p in sorted(directory.iterdir()) if p.is_dir() and p != destination]
    for family in families:
        bits, chunks = family_chunks(family)
        indexed = {}
        index["families"][family.name] = {"bits": bits, "chunks": indexed}
        for bucket, chunk in chunks:
            if buffer and len(buffer) + len(chunk) > PACK_BYTES:
                flush()
            indexed[bucket] = [len(index["packs"]), len(buffer), len(chunk), digest(chunk)]
            buffer.extend(chunk)
    flush()
    raw_index = encoded(index)
    index_name = f"index.{digest(raw_index)}.json"
    (destination / index_name).write_bytes(raw_index)
    source_path = directory / "source.json.gz"
    source = json.loads(gzip.decompress(source_path.read_bytes()))
    source["lookupPack"] = {"index": "lookup-pack/" + index_name}
    source_path.write_bytes(gzip.compress(encoded(source), compresslevel=6, mtime=0))
    # Stabilize view/IRT bytes as well, so a rebuild with identical results does
    # not re-upload files solely because gzip recorded a new creation time.
    for path in directory.glob("*.json.gz"):
        path.write_bytes(normalized(path))
    for family in families:
        shutil.rmtree(family)
    return source
