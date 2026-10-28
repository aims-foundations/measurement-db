"""Testlets change column organization, never source observations or page budgets."""
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_gallery_new import fixture
from scripts.render_website.record_windows import export_windows
from scripts.ci.check_gallery_records import check_windows


def grouped_fixture(tmp_path):
    source, data = fixture(tmp_path)
    # Deliberately interleaved physical rows, including independent items.
    for i, row in enumerate(data["items"]):
        row["testlet_id"] = "task-b" if i % 3 == 0 else "task-a" if i % 3 == 1 else None
    tables = source / "formatted_tables"
    pq.write_table(pa.Table.from_pylist(data["items"]), tables / "items.parquet")
    output = tmp_path / "export"
    output.mkdir()
    descriptor = export_windows(tables, output / "windows", max_rows=7, max_items=3, max_bytes=1900)
    return tables, output, descriptor, data


def test_grouping_preserves_each_binary_observation_and_bounded_pages(tmp_path):
    source, output, descriptor, data = grouped_fixture(tmp_path)
    report = check_windows(source, output, descriptor, {row["item_id"] for row in data["items"]})
    assert report["exact_testlet_membership"] and report["testlets"] == 2
    assert report["rows"] == 48
    seen, observations, partial = [], [], []
    for page in range(descriptor["pages"]):
        raw = gzip.decompress((output / "windows" / f"{page}.json.gz").read_bytes())
        window = json.loads(raw)
        assert len(raw) <= 1900
        assert len(window["itemIds"]) <= 3
        assert len(window["observations"]) <= 7
        seen.extend(window["itemIds"])
        observations.extend(window["observations"])
        if "i0" in window["partialItemIds"]:
            partial.append(window["testlets"][0])
    assert list(dict.fromkeys(seen)) == ["i1", "i4", "i7", "i10", "i13", "i0", "i3", "i6", "i9", "i12", "i11", "i2", "i5", "i8"]
    assert sorted(observations, key=lambda r: r["response_id"]) == sorted(data["responses"], key=lambda r: r["response_id"])
    assert len(partial) > 1
    assert not partial[0]["continuedBefore"]
    assert all(group["continuedBefore"] for group in partial[1:])
    assert all(group["continuedAfter"] for group in partial)


@pytest.mark.parametrize("field,value,error", [
    ("totalItems", 999, "total item count"),
    ("itemIds", ["i0"], "membership"),
    ("continuedBefore", True, "continuation"),
])
def test_corrupt_testlet_metadata_is_rejected(tmp_path, field, value, error):
    source, output, descriptor, data = grouped_fixture(tmp_path)
    path = output / "windows/0.json.gz"
    window = json.loads(gzip.decompress(path.read_bytes()))
    window["testlets"][0][field] = value
    path.write_bytes(gzip.compress(json.dumps(window).encode()))
    with pytest.raises(ValueError, match=error):
        check_windows(source, output, descriptor, {row["item_id"] for row in data["items"]})


def test_legacy_and_null_membership_have_identical_pages(tmp_path):
    source, data = fixture(tmp_path)
    tables = source / "formatted_tables"
    first, second = tmp_path / "legacy", tmp_path / "null"
    a = export_windows(tables, first, max_rows=7, max_items=3)
    for row in data["items"]:
        row["testlet_id"] = None
    pq.write_table(pa.Table.from_pylist(data["items"]), tables / "items.parquet")
    b = export_windows(tables, second, max_rows=7, max_items=3)
    assert a == b and "testlets" not in b
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
