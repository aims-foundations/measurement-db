"""Combine JSON pages into independently compressed ranges, without changing rows."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

PACK_BYTES = 8 * 1024 * 1024


def pack_records(directory: Path) -> dict:
    packs, entries, buffer = [], {}, bytearray()

    def flush():
        if not buffer:
            return
        name = f"pack-{len(packs)}.bin"
        (directory / name).write_bytes(buffer)
        packs.append({"file": name, "bytes": len(buffer)})
        buffer.clear()

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if not (path.name.endswith(".json") or path.name.endswith(".json.gz")):
            raise ValueError(f"Unexpected record page: {path}")
        payload = path.read_bytes()
        if path.name.endswith(".gz"):
            raw_size = len(gzip.decompress(payload))
            logical = str(path.relative_to(directory))[:-3]
        else:
            raw_size = len(payload)
            payload = gzip.compress(payload, compresslevel=6, mtime=0)
            logical = str(path.relative_to(directory))
        if buffer and len(buffer) + len(payload) > PACK_BYTES:
            flush()
        entries[logical] = [len(packs), len(buffer), len(payload), raw_size,
                            hashlib.sha256(payload).hexdigest()]
        buffer.extend(payload)
        path.unlink()
    flush()
    index = {"format": "canonical-record-pack-v1", "packs": packs, "entries": entries}
    (directory / "pack-index.json").write_text(json.dumps(index, separators=(",", ":")))
    return {"index": "pack-index.json"}
