"""Discover and download both public HF layouts without traversing raw data."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.render_website import generate_benchmark_gallery as gallery


@pytest.fixture
def hub(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    calls = []

    def tree(repo, *, repo_type, revision, path_in_repo=""):
        calls.append((path_in_repo, revision))
        return [gallery.RepoFolder(path=str(p.relative_to(remote)), oid="0" * 40)
                if p.is_dir() else SimpleNamespace(path=str(p.relative_to(remote)))
                for p in (remote / path_in_repo).iterdir()]

    def download(repo, filename, *, repo_type, revision, **kwargs):
        calls.append((filename, revision))
        path = remote / filename
        if not path.is_file():
            raise gallery.EntryNotFoundError("Missing fixture")
        return str(path)

    info = Mock(return_value=SimpleNamespace(sha="pinned"))
    monkeypatch.setattr(gallery, "HfApi", lambda: SimpleNamespace(
        dataset_info=info, list_repo_tree=tree))
    monkeypatch.setattr(gallery, "hf_hub_download", download)
    monkeypatch.setattr(gallery, "HF_REVISION", "migration/test")
    monkeypatch.setattr(gallery, "HIDDEN_PATH", tmp_path / "hidden.json")
    gallery.source_revision.cache_clear()
    gallery.source_tables.cache_clear()

    def write(path, content="table"):
        target = remote / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    yield SimpleNamespace(write=write, cache=tmp_path / "cache", calls=calls, info=info)
    gallery.source_revision.cache_clear()
    gallery.source_tables.cache_clear()


def test_discovers_both_layouts_and_excludes_raw_archives_and_withheld(hub):
    for slug, folder, release in [("legacy", "", "public"),
                                  ("modern", "formatted_tables/", "public"),
                                  ("withheld", "formatted_tables/", "withheld"),
                                  ("hidden", "", "public")]:
        hub.write(f"{slug}/{folder}benchmarks.parquet")
        hub.write(f"{slug}/metadata.yaml", json.dumps({"benchmark": {"release": release}}))
    hub.write("archive/raw/old/benchmarks.parquet")
    hub.write("unfinished/formatted_tables/items.parquet")
    gallery.HIDDEN_PATH.write_text('["hidden"]')
    assert gallery.list_slugs() == ["legacy", "modern"]
    assert all("raw/" not in path for path, _ in hub.calls)
    assert {revision for _, revision in hub.calls} == {"pinned"}
    hub.info.assert_called_once_with(gallery.HF_REPO, revision="migration/test")


def test_modern_tables_take_precedence_and_keep_local_analysis_paths(hub):
    hub.write("fixture/benchmarks.parquet", "old metadata")
    hub.write("fixture/responses.parquet", "old response")
    hub.write("fixture/formatted_tables/benchmarks.parquet", "new metadata")
    hub.write("fixture/formatted_tables/responses.parquet", "new response")
    hub.write("fixture/formatted_tables/traces.parquet", "new traces")
    path = gallery.fetch("fixture/responses.parquet", hub.cache)
    assert path == hub.cache / "fixture/responses.parquet"
    assert path.read_text() == "new response"
    assert gallery.fetch("fixture/traces.parquet", hub.cache).read_text() == "new traces"
    assert gallery.source_tables("fixture").directory == "fixture/formatted_tables"


def test_incomplete_modern_tables_do_not_fall_back_to_stale_flat_tables(hub):
    hub.write("fixture/responses.parquet", "stale")
    hub.write("fixture/formatted_tables/benchmarks.parquet")
    with pytest.raises(SystemExit):
        gallery.fetch("fixture/responses.parquet", hub.cache)
    assert not (hub.cache / "fixture/responses.parquet").exists()


def test_legacy_singular_response_name_still_works(hub):
    hub.write("fixture/benchmarks.parquet")
    hub.write("fixture/response.parquet", "legacy response")
    assert gallery.fetch("fixture/responses.parquet", hub.cache).read_text() == "legacy response"
