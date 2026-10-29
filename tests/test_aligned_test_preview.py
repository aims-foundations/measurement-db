import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.render_website.prepare_aligned_test_preview import prepare


def fixture(tmp_path, *, repeated=False, nonbinary=False):
    tables = tmp_path / "tables"
    tables.mkdir()
    content = json.dumps({"messages": [{"role": "user", "content": "Build it"}],
                          "environment": {"image": "example:v1", "cwd": "/workspace"}})
    items = [dict(item_id=id, raw_item_id=f"task:{test}", testlet_id="task", content_hash="prompt",
                  content=content, verifier=json.dumps({"spec": json.dumps({"grader_version": grader})}))
             for id, test, grader in [("a", "branch1/test", "1"), ("b", "branch1/test", "2"),
                                      ("c", "branch2/test", "2")]]
    subjects = [dict(subject_id=s, display_name=s) for s in ("old", "new")]
    attempts = [dict(attempt_id=s, subject_id=s, raw_attempt_id=f"run-{s}/task") for s in ("old", "new")]
    responses = [dict(subject_id=s, item_id=i, attempt_id=s, response=v)
                 for s, i, v in [("old", "a", 1), ("new", "b", 0), ("new", "c", 1)]]
    if repeated:
        responses.append(responses[0].copy())
    if nonbinary:
        responses[0]["response"] = 0.5
    for name, rows in [("items", items), ("subjects", subjects), ("attempts", attempts), ("responses", responses)]:
        pq.write_table(pa.Table.from_pylist(rows), tables / f"{name}.parquet")
    return tables


def test_aligns_versions_without_merging_branch_identifiers_or_mutating_items(tmp_path):
    tables = fixture(tmp_path)
    before = (tables / "items.parquet").read_bytes()
    output = tmp_path / "preview"
    result = prepare(tables, output, "revision")
    assert result["columns"] == 2
    assert result["canonicalItems"] == 3
    assert result["observations"] == 3
    assert list(gzip.decompress((output / "scores.bin.gz").read_bytes())) == [2, 0, 1, 2]
    index = json.loads((output / "pack-index.json").read_text())
    pack, start, length, *_ = index["entries"]["tests/0/0.json"]
    raw = (output / index["packs"][pack]["file"]).read_bytes()[start:start + length]
    tests = json.loads(gzip.decompress(raw))["tests"]
    assert tests == [{"name": "branch1/test", "items": ["a", "b"]},
                     {"name": "branch2/test", "items": [None, "c"]}]
    assert (tables / "items.parquet").read_bytes() == before


@pytest.mark.parametrize("option, message", [("repeated", "Multiple observations"), ("nonbinary", "non-binary")])
def test_refuses_lossy_alignment(tmp_path, option, message):
    tables = fixture(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=message):
        prepare(tables, tmp_path / "preview", "revision")
