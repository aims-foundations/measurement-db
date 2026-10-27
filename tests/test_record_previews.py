"""Exercise storage roundtrips and resource boundaries, not rendering choices."""
import gzip
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.render_website import record_previews as previews


def read(directory, part):
    pack, offset, length, size, digest = part
    with (directory / f"pack-{pack}.bin").open("rb") as stream:
        stream.seek(offset)
        compressed = stream.read(length)
    assert hashlib.sha256(compressed).hexdigest() == digest
    raw = gzip.decompress(compressed)
    assert len(raw) == size
    return raw


def test_large_shared_pages_and_exact_paged_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(previews, "SOURCE_LIMIT", 1024)
    monkeypatch.setattr(previews, "RECORD_LIMIT", 1024)
    monkeypatch.setattr(previews, "TEXT_CHARS", 500)
    source = tmp_path / "traces.parquet"
    rows = [{"response_id": str(i), "trace": "small" if i else "😺λ\\n" * 1000,
             "grade": None, "trial": i} for i in range(25)]
    pq.write_table(pa.Table.from_pylist(rows), source, compression="gzip", use_dictionary=False)
    groups = previews.source_ranges(source)
    assert groups[0]["ranges"] == [[0, 25]]
    output = tmp_path / "archive"
    manifest = previews.export_previews(source, output, "response_id", groups)
    assert manifest["rows"] == 25
    assert manifest["paged"] == 1
    assert previews.verify_previews(source, output, "response_id")["rows"] == 25
    for original in rows:
        index = json.loads(read(output, manifest["indexes"][hashlib.sha256(original["response_id"].encode()).digest()[0]]))
        envelope = json.loads(read(output, index[original["response_id"]]))
        record = envelope["record"]
        for field, parts in envelope.get("fields", {}).items():
            record[field] = "".join(read(output, part).decode() for part in parts)
        assert record == original
    # A changed byte is caught before it can become another source answer.
    pack = output / "pack-0.bin"
    data = bytearray(pack.read_bytes()); data[15] ^= 1; pack.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        previews.verify_previews(source, output, "response_id")


def test_only_selected_rows_are_cached(tmp_path):
    source = tmp_path / "items.parquet"
    pq.write_table(pa.Table.from_pylist([{"item_id":str(i), "content":"text"} for i in range(4)]),source)
    output = tmp_path / "archive"
    manifest = previews.export_previews(source, output, "item_id", [{"rows":4,"ranges":[[1,3]]}])
    assert manifest["rows"] == 2
    identities = set()
    for part in manifest["indexes"]:
        identities.update(json.loads(read(output, part)))
    assert identities == {"1","2"}


def test_resume_rejects_changed_source_with_identical_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(previews, "SOURCE_LIMIT", 0)
    source = tmp_path / "items.parquet"
    pq.write_table(pa.Table.from_pylist([{"item_id":"same", "content":"before"}]),source)
    output = tmp_path / "archive"
    previews.export_previews(source, output, "item_id")
    pq.write_table(pa.Table.from_pylist([{"item_id":"same", "content":"after"}]),source)
    with pytest.raises(ValueError, match="Source table changed"):
        previews.verify_previews(source, output, "item_id")
