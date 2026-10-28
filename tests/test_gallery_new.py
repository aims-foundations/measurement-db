"""The new gallery transports canonical records without inventing display data."""

import hashlib
import gzip
import json
import sys
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_website import generate_benchmark_gallery_new as gallery
from scripts.ci.check_gallery_records import check
from scripts.render_website.record_windows import export_windows


def fixture(root, *, kind="binary"):
    source = root / kind
    tables = source / "formatted_tables"
    tables.mkdir(parents=True)
    scale = {
        "kind": "discrete",
        "values": [0, 1],
        "meanings": {"0": "unsafe", "1": "safe"},
    }
    if kind == "pairwise":
        scale = {
            "kind": "discrete",
            "values": [0, 0.5, 1],
            "meanings": {"0": "loss", "0.5": "tie", "1": "win"},
        }
    if kind == "mixed":
        scale = {"kind": "mixed"}
    response_type = {"pairwise": "win_rate", "mixed": "mixed"}.get(kind, "binary")
    metadata = {
        "benchmark_id": kind,
        "name": kind.title(),
        "description": "Exact released observations",
        "granularity": "not_released" if kind == "item_bank" else "item",
        "response_type": response_type,
        "response_scale": json.dumps(scale),
    }
    (source / "metadata.yaml").write_text(
        yaml.safe_dump({"benchmark": {**metadata, "response_scale": scale}})
    )
    criterion = {"reference_answer": "Reference", "rule": "Released grade"}
    if kind == "mixed":
        criterion["response_scale"] = {"kind": "interval", "min": 0, "max": 100}
    # One exact attachment, and deliberately hostile prompt text to test text rendering.
    image = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c020000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
    )
    asset_id = hashlib.sha256(image).hexdigest()
    data = {
        "benchmarks": [metadata],
        "subjects": [
            {"subject_id": f"s{i}", "display_name": f"Model {i}", "harness": f"h{i}"}
            for i in range(3)
        ],
        "items": [
            {
                "item_id": f"i{i}",
                "content": f"Task {i} </script><script>window.injected=true</script>",
                "grading_criterion": json.dumps(criterion),
                "item_features": f"language={i}",
                "asset_manifest": json.dumps(
                    [
                        {
                            "asset_id": asset_id,
                            "path": "stimulus.png",
                            "media_type": "image/png",
                            "role": "input",
                        }
                    ]
                )
                if i == 0
                else None,
            }
            for i in range(14)
        ],
        "responses": [
            {
                "response_id": f"r{s}-{trial}",
                "subject_id": f"s{s}",
                "item_id": "i0",
                "response": None
                if trial == 1
                else 37.5
                if kind == "mixed"
                else float(s),
                "trial": trial,
                "test_condition": "temperature=0",
                "interactors": f"opponent=s{1 - s}" if kind == "pairwise" else None,
            }
            for s in range(2)
            for trial in range(1, 25)
        ],
        "assets": [{"asset_id": asset_id, "data": image, "byte_size": len(image)}],
    }
    # Reverse physical trace order: only response_id is a valid join.
    data["traces"] = [
        {**record, "trace": f"Answer {record['response_id']}"}
        for record in reversed(data["responses"])
    ]
    if kind == "item_bank":
        del data["responses"], data["traces"]
    for name, records in data.items():
        pq.write_table(
            pa.Table.from_pylist(records), tables / f"{name}.parquet", row_group_size=3
        )
    return source, data


def read_table(output, name):
    index = json.loads((output / name / "index.json").read_text())
    pages = {
        page: json.loads((output / name / f"{page}.json").read_text())
        for page in set(index.values())
    }
    return {identity: pages[page][identity] for identity, page in index.items()}


def test_shared_attempt_answers_are_exported_once_and_keep_response_links(tmp_path):
    source, data = fixture(tmp_path)
    tables = source / "formatted_tables"
    attempts = [{"attempt_id": f"a{s}", "subject_id": f"s{s}", "trace": f"Shared program {s}"}
                for s in range(2)]
    responses = [{**row, "attempt_id": row["subject_id"].replace("s", "a")} for row in data["responses"]]
    pq.write_table(pa.Table.from_pylist(attempts), tables / "attempts.parquet")
    pq.write_table(pa.Table.from_pylist(responses), tables / "responses.parquet")
    (tables / "traces.parquet").unlink()
    output = tmp_path / "gallery"
    gallery.build([source], output, page_size=5)
    manifest = json.loads((output / "0/manifest.json").read_text())
    assert manifest["tables"]["attempts"]["rows"] == 2
    assert "traces" not in manifest["tables"]
    answers = read_table(output / "0", "attempts")
    observed = read_table(output / "0", "responses")
    assert len(observed) == 48
    for response in observed.values():
        assert answers[response["attempt_id"]]["subject_id"] == response["subject_id"]
    lazy = tmp_path / "lazy"
    gallery.build([source], lazy, parquet_base="https://fixture.invalid/pinned")
    descriptor = json.loads((lazy / "0/manifest.json").read_text())["tables"]["attempts"]
    assert descriptor["key"] == "attempt_id"
    assert descriptor["parquet"]["url"].endswith("/attempts.parquet")
    check(tables, lazy / "0")


@pytest.mark.parametrize("kind", ["binary", "pairwise", "mixed", "item_bank"])
def test_every_record_retains_ids_grades_metadata_and_exact_answers(tmp_path, kind):
    source, data = fixture(tmp_path, kind=kind)
    output = tmp_path / "gallery"
    gallery.build([source], output, page_size=5)
    manifest = json.loads((output / "0/manifest.json").read_text())
    assert manifest["benchmark"] == data["benchmarks"][0]
    for table in ["subjects", "items", "responses", "traces"]:
        if table not in data:
            assert table not in manifest["tables"]
            continue
        key = gallery.PRIMARY_KEYS[table]
        rows = read_table(output / "0", table)
        assert rows == {row[key]: row for row in data[table]}
    if "responses" in data:
        responses, traces = (
            read_table(output / "0", table) for table in ["responses", "traces"]
        )
        assert len(responses) == 48  # All 24 trials for both subjects, including nulls.
        for identity, response in responses.items():
            assert traces[identity]["trace"] == f"Answer {identity}"
            assert traces[identity]["trial"] == response["trial"]
    asset = next(iter(read_table(output / "0", "assets").values()))
    assert (output / "0/assets" / asset["data_url"]).read_bytes() == data["assets"][0][
        "data"
    ]
    assert not (output / "0/assets" / asset["data_url"]).is_symlink()


def test_output_is_relocatable_and_inputs_are_unchanged(tmp_path):
    source, _ = fixture(tmp_path)
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "gallery"
    gallery.build([source], output)
    catalog = json.loads((output / "index.json").read_text())
    assert (output / catalog[0]["manifest"]).is_file()
    assert not any(output.rglob("*.html"))
    assert not any(output.rglob("*.js"))
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    with pytest.raises(FileExistsError):
        gallery.build([source], output)


def test_stale_yaml_does_not_override_exported_grades(tmp_path):
    source, _ = fixture(tmp_path)
    path = source / "metadata.yaml"
    metadata = yaml.safe_load(path.read_text())
    metadata["benchmark"]["response_scale"]["meanings"]["1"] = "incorrect"
    path.write_text(yaml.safe_dump(metadata))
    output = tmp_path / "gallery"
    with pytest.raises(ValueError, match="rebuild them first"):
        gallery.build([source], output)
    assert not output.exists()


def test_duplicate_ids_fail_without_publishing_partial_output(tmp_path):
    source, data = fixture(tmp_path)
    data["responses"][1]["response_id"] = data["responses"][0]["response_id"]
    pq.write_table(
        pa.Table.from_pylist(data["responses"]),
        source / "formatted_tables/responses.parquet",
    )
    output = tmp_path / "gallery"
    with pytest.raises(ValueError, match="duplicate response_id"):
        gallery.build([source], output, page_size=1)
    assert not output.exists()


def test_missing_item_level_responses_is_an_error(tmp_path):
    source, _ = fixture(tmp_path)
    (source / "formatted_tables/responses.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        gallery.build([source], tmp_path / "gallery")


def test_table_directory_without_yaml_uses_exported_metadata(tmp_path):
    source, _ = fixture(tmp_path)
    (source / "metadata.yaml").unlink()
    gallery.build([source / "formatted_tables"], tmp_path / "gallery")


def test_subject_page_index_and_lazy_trace_positions_cover_every_source_row(tmp_path):
    source, _ = fixture(tmp_path)
    output = tmp_path / "gallery"
    gallery.build([source], output, page_size=5, parquet_base="https://fixture.invalid")
    report = check(source / "formatted_tables", output / "0")
    assert report["responses"]["rows"] == 48
    assert report["responses"]["grades"]["null"] == 2
    assert report["traces"] == {"rows": 48, "exact_source_positions": True}
    assert not list((output / "0/traces").glob("[0-9]*.json"))
    page = output / "0/windows/0.json.gz"
    rows = json.loads(gzip.decompress(page.read_bytes()))
    rows["observations"][0]["response"] = 1
    page.write_bytes(gzip.compress(json.dumps(rows).encode()))
    with pytest.raises(ValueError, match="changed source fields"):
        check(source / "formatted_tables", output / "0")


def test_ci_preparation_packages_verified_records_and_server_authenticated_traces(tmp_path):
    import gzip
    from scripts.ci.canonical_gallery import prepare

    source, data = fixture(tmp_path)
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "binary").symlink_to(source / "formatted_tables", target_is_directory=True)
    web = tmp_path / "website"
    config = web / "content/curated"
    config.mkdir(parents=True)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    entry = prepare("binary", {"revision": "a" * 40}, tables, bundle, web)
    assert entry["base"] == "/benchmark-data/binary/records"
    assert entry["manifest"]["pack"]
    assert entry["manifest"]["tables"]["traces"]["parquet"]["url"] == "/api/gallery-source/binary/traces"
    assert json.loads((bundle / "records/manifest.json").read_text()) == entry["manifest"]
    index = json.loads((bundle / "records/pack-index.json").read_text())
    number, offset, length, _, _ = index["entries"]["windows/0.json"]
    packed = (bundle / "records" / index["packs"][number]["file"]).read_bytes()
    rows = json.loads(gzip.decompress(packed[offset:offset + length]))["observations"]
    assert sorted(rows, key=lambda r: r["response_id"]) == sorted(data["responses"], key=lambda r: r["response_id"])
    assert json.loads((bundle / "records-validation.json").read_text())["responses"]["grades"]["null"] == 2


def test_windows_bound_rows_and_utf8_bytes_without_losing_attempts_or_unobserved_items(tmp_path):
    source, data = fixture(tmp_path, kind="pairwise")
    output = tmp_path / "windows"
    descriptor = export_windows(source / "formatted_tables", output, max_rows=7, max_items=3, max_bytes=1700)
    actual, items, partial = [], set(), set()
    for page in range(descriptor["pages"]):
        raw = gzip.decompress((output / f"{page}.json.gz").read_bytes())
        assert len(raw) <= 1700
        window = json.loads(raw)
        assert len(window["observations"]) <= 7
        assert len(window["itemIds"]) <= 3
        actual.extend(window["observations"])
        items.update(window["itemIds"])
        partial.update(window["partialItemIds"])
    assert sorted(actual, key=lambda r: r["response_id"]) == sorted(data["responses"], key=lambda r: r["response_id"])
    assert items == {r["item_id"] for r in data["items"]}
    assert partial == {"i0"}
    # A single model with many attempts follows the same limits.
    for row in data["responses"]:
        row["subject_id"] = "s0"
        row["test_condition"] = "context=" + "数学" * 70
    pq.write_table(pa.Table.from_pylist(data["responses"]), source / "formatted_tables/responses.parquet")
    second = tmp_path / "unicode-windows"
    descriptor = export_windows(source / "formatted_tables", second, max_rows=100, max_bytes=1700)
    assert descriptor["pages"] > 10
    assert all(len(gzip.decompress(p.read_bytes())) <= 1700 for p in second.glob('[0-9]*.json.gz'))


if __name__ == "__main__":
    root = Path(sys.argv[1])
    sources = [
        fixture(root, kind=kind)[0]
        for kind in ["binary", "pairwise", "mixed", "item_bank"]
    ]
    gallery.build(sources, root / "site", page_size=5)
    gallery.build(sources, root / "lazy", page_size=5, parquet_base="http://fixture.invalid")
    gallery.write_json(root / "expected-traces.json", {
        str(i): pq.read_table(source / "formatted_tables/traces.parquet").to_pylist()
        for i, source in enumerate(sources) if (source / "formatted_tables/traces.parquet").exists()
    })
    from scripts.render_website.record_packs import pack_records
    expected = {}
    for i, source in enumerate(sources):
        expected[str(i)] = {}
        for name in ("items", "traces", "assets"):
            path = source / "formatted_tables" / f"{name}.parquet"
            if path.exists():
                rows = pq.read_table(path).to_pylist()
                if name == "assets":
                    rows = [{**row, "data": list(row["data"])} for row in rows]
                expected[str(i)][name] = rows
        directory = root / "packed" / str(i)
        shutil.copytree(root / "lazy" / str(i), directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["pack"] = pack_records(directory)
        gallery.write_json(directory / "manifest.json", manifest)
    gallery.write_json(root / "expected-records.json", expected)
