"""Scientific-output parity, modular preparation, and content-based reuse."""

import copy
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from scripts.ci import gallery_artifacts as artifacts
from scripts.ci import gallery_prepare as prepare
from scripts.render_website import generate_benchmark_gallery as gallery
from scripts.render_website import generate_model_timeline as timeline
from scripts.analyze_measurements import fit_rasch_models as rasch
from test_gallery_output_parity import write_fixture


REVISION = "a" * 40


def fixture(root, slug="binary"):
    tables, web = root / "tables", root / "website"
    overrides = write_fixture(tables, slug)
    subjects_path = tables / slug / "subjects.parquet"
    subjects = pd.read_parquet(subjects_path)
    subjects["normalized_name"] = subjects["display_name"]
    subjects["provider"], subjects["release_date"] = "Provider", "2025-01-01"
    subjects.to_parquet(subjects_path, index=False)
    for name, payload in {"benchmark-overrides.json": overrides, "dynamic-item-benchmarks.json": [],
                          "benchmark-affiliations.json": {"benchmarks": {}, "institutions": {}}, "hidden-benchmarks.json": [],
                          "benchmark-attribution-overrides.json": {}}.items():
        artifacts.write_json(web / "content/curated" / name, payload)
    source = {"revision": REVISION, "directory": f"{slug}/formatted_tables", "metadata_sha256": "b" * 64,
              "tables": {path.name: {"path": f"{slug}/formatted_tables/{path.name}",
                                     "size": path.stat().st_size, "sha256": artifacts.file_digest(path)}
                         for path in (tables / slug).glob("*.parquet")}}
    return tables, web, source


def prepare_fixture_process(task):
    root = Path(task["fixture_root"])
    def download(repo, path, **kwargs):
        return root / "tables" / task["slug"] / Path(path).name
    with patch.object(prepare, "ROOT", root), \
         patch("huggingface_hub.hf_hub_download", side_effect=download), \
         patch.object(prepare, "FIT_OPTIONS", {**prepare.FIT_OPTIONS, "max_iter": 50}):
        return prepare.prepare_one(task)


class GalleryPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # render_bundle sets process-local state, as it does in disposable CI workers.
        self.patchers = [patch.object(gallery, name, getattr(gallery, name))
                         for name in ("LOCAL_DATA_DIR", "LOCAL_SOURCES", "HF_REVISION")]
        self.patchers += [patch.object(rasch, name, getattr(rasch, name)) for name in ("gallery", "fetch")]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(gallery.source_tables.cache_clear)
        self.addCleanup(gallery.source_revision.cache_clear)

    def test_listing_retries_transient_errors_but_never_missing_or_forbidden_data(self):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError
        def error(status):
            return HfHubHTTPError("fixture", response=httpx.Response(
                status, request=httpx.Request("GET", "https://huggingface.co/fixture")))
        api = Mock()
        api.list_repo_tree.side_effect = [error(504), ["pinned-result"]]
        with patch.object(prepare.time, "sleep") as sleep:
            self.assertEqual(prepare.source_tree(api, REVISION, "ceval"), ["pinned-result"])
        sleep.assert_called_once_with(1)
        self.assertTrue(all(call.kwargs["revision"] == REVISION for call in api.list_repo_tree.call_args_list))
        for status in (401, 403, 404):
            api.reset_mock()
            api.list_repo_tree.side_effect = error(status)
            with patch.object(prepare.time, "sleep") as sleep, self.assertRaises(HfHubHTTPError):
                prepare.source_tree(api, REVISION, "ceval")
            sleep.assert_not_called()
            self.assertEqual(api.list_repo_tree.call_count, 1)
        api.reset_mock()
        api.list_repo_tree.side_effect = error(503)
        with patch.object(prepare.time, "sleep") as sleep, self.assertRaises(HfHubHTTPError):
            prepare.source_tree(api, REVISION, "ceval")
        self.assertEqual(api.list_repo_tree.call_count, 5)
        self.assertEqual(sleep.call_count, 4)

    def test_new_global_hf_revision_reuses_unchanged_bytes_but_local_inputs_invalidate(self):
        _, web, source = fixture(self.root)
        before = prepare.benchmark_inputs("binary", source, {"code": "renderer"}, web)
        updated = {**source, "revision": "f" * 40}
        self.assertEqual(artifacts.input_key(before), artifacts.input_key(
            prepare.benchmark_inputs("binary", updated, {"code": "renderer"}, web)))
        for key, value in (("metadata_sha256", "c" * 64), ("directory", "binary")):
            self.assertNotEqual(before, prepare.benchmark_inputs("binary", {**source, key: value}, {"code": "renderer"}, web))
        artifacts.write_json(web / "content/curated/dynamic-item-benchmarks.json", ["binary"])
        self.assertNotEqual(artifacts.input_key(before), artifacts.input_key(
            prepare.benchmark_inputs("binary", source, {"code": "renderer"}, web)))
        after = copy.deepcopy(before)
        after["tables"]["items.parquet"]["sha256"] = "d" * 64
        self.assertNotEqual(artifacts.input_key(before), artifacts.input_key(after))

    def test_explicit_dynamic_policy_skips_item_serialization_and_preserves_indexes(self):
        tables, _, source = fixture(self.root)
        gallery.LOCAL_DATA_DIR, gallery.LOCAL_SOURCES = tables, {"binary": source}
        gallery.HF_REVISION = REVISION
        output = self.root / "lookups"
        output.mkdir()
        with patch.object(gallery, "static_item", side_effect=AssertionError("must not serialize dynamic prompts")), \
             patch.object(gallery, "hf_hub_download", side_effect=AssertionError("no runtime download during local render")):
            self.assertEqual(gallery.write_website_lookups("binary", tables, output, True, dynamic_items=True), 0)
        descriptor = json.loads(gzip.decompress((output / "source.json.gz").read_bytes()))
        self.assertTrue(descriptor["dynamicItems"])
        self.assertFalse((output / "item").exists())
        self.assertTrue(list((output / "item-rows").glob("*.gz")))
        self.assertTrue(list((output / "answer").glob("*.gz")))

    def test_validated_local_source_never_falls_back_to_hf(self):
        tables, _, source = fixture(self.root)
        gallery.LOCAL_DATA_DIR, gallery.LOCAL_SOURCES = tables, {"binary": source}
        with patch.object(gallery, "hf_hub_download", side_effect=AssertionError("network fallback")):
            with self.assertRaises(SystemExit):
                gallery.fetch("binary/undeclared.parquet", tables)
            (tables / "binary/items.parquet").unlink()
            with self.assertRaises(FileNotFoundError):
                gallery.fetch("binary/items.parquet", tables)

    def test_parallel_timeline_reduction_matches_existing_weighted_model_registry(self):
        tables, web, _ = fixture(self.root)
        other = tables / "other"
        other.mkdir()
        pd.DataFrame([
            {"normalized_name": "Model 0", "provider": "Other", "release_date": "2024-01-01"},
            {"normalized_name": "Model 0", "provider": "Other", "release_date": "2024-01-01"},
            {"normalized_name": "Model 1", "provider": "Earlier", "release_date": "2023-01-01"},
            {"normalized_name": None, "provider": "Ignored", "release_date": None},
        ]).to_parquet(other / "subjects.parquet", index=False)
        cards = web / "content/generated/benchmark-cards.json"
        artifacts.write_json(cards, [{"slug": "binary"}, {"slug": "other"}])
        baseline = self.root / "timeline.json"
        timeline.main(baseline, tables, cards)
        summaries = {slug: {"timeline": prepare.timeline_counts(tables / slug / "subjects.parquet")}
                     for slug in ("binary", "other")}
        self.assertEqual(prepare.combine_timelines(summaries), json.loads(baseline.read_text()))

    def test_website_fit_omits_response_csv_without_changing_fits_or_summaries(self):
        tables, _, _ = fixture(self.root)
        frame = rasch.binary_frame(pd.read_parquet(tables / "binary/responses.parquet").fillna({"test_condition": ""}))
        result = rasch.fit(frame, **{**prepare.FIT_OPTIONS, "max_iter": 50}, device="cpu")
        full, website = self.root / "full", self.root / "web-fit"
        expected = rasch.write_outputs("binary", frame, result, {}, full, curves=False)
        actual = rasch.write_outputs("binary", frame, result, {}, website, curves=False, write_responses=False)
        self.assertEqual(actual, expected)
        self.assertFalse((website / "responses.csv").exists())
        self.assertTrue((full / "responses.csv").exists())
        for name in ("subjects.csv", "items.csv", "summary.json", "history.csv"):
            self.assertEqual((website / name).read_bytes(), (full / name).read_bytes())

    def test_cold_preparation_and_warm_reuse_do_not_refit_or_download_source(self):
        tables, _, source = fixture(self.root)
        task = {"slug": "binary", "source": source, "gallery_revision": "gallery-commit",
                "inputs": prepare.benchmark_inputs("binary", source, {"code": "fixture"}, self.root / "website"),
                "candidate": str(self.root / "cold"), "previous": None}

        def download(repo, path, **kwargs):
            self.assertEqual(repo, artifacts.SOURCE_REPO)
            self.assertEqual(kwargs["revision"], REVISION)
            return tables / "binary" / Path(path).name

        with patch.object(prepare, "ROOT", self.root), \
             patch("huggingface_hub.hf_hub_download", side_effect=download), \
             patch.object(prepare, "FIT_OPTIONS", {**prepare.FIT_OPTIONS, "max_iter": 50}):
            slug, entry, summary, timings = prepare.prepare_one(task)
        self.assertFalse(timings["cache_hit"])
        self.assertIsNotNone(summary["irt"])
        task.update(previous=entry, candidate=str(self.root / "warm"))
        archive = self.root / "cold" / entry["path"]
        with patch("huggingface_hub.hf_hub_download", return_value=archive) as fetch, \
             patch.object(prepare, "render_bundle", side_effect=AssertionError("cache hit must not render")):
            warm = prepare.prepare_one(task)
        self.assertEqual(warm[:3], (slug, entry, summary))
        self.assertTrue(warm[3]["cache_hit"])
        self.assertEqual(fetch.call_args.args, (artifacts.GALLERY_REPO, "v2/" + entry["path"]))

    def test_unreleased_item_bank_has_static_view_and_no_rasch_fit(self):
        tables, web, source = fixture(self.root, "item_bank")
        with patch.object(prepare, "ROOT", self.root), \
             patch.object(rasch, "fit", side_effect=AssertionError("no responses to fit")):
            output = prepare.render_bundle("item_bank", source, tables, web, self.root)
        summary = json.loads((output["data"] / "summary.json").read_text())
        self.assertIsNone(summary["irt"])
        self.assertFalse(summary["card"]["itemResponses"])
        self.assertTrue((output["data"] / "view.json.gz").exists())

    def test_spawned_workers_produce_isolated_bundles_and_complete_combined_catalog(self):
        tasks = []
        for slug in ("binary", "continuous"):
            root = self.root / slug
            _, web, source = fixture(root, slug)
            tasks.append({"slug": slug, "source": source,
                          "inputs": prepare.benchmark_inputs(slug, source, {"code": "fixture"}, web),
                          "fixture_root": str(root), "candidate": str(self.root / "parallel"),
                          "previous": None, "gallery_revision": "unused"})
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                                 max_tasks_per_child=1) as pool:
            results = list(pool.map(prepare_fixture_process, tasks))
        summaries = {slug: summary for slug, _, summary, _ in results}
        generated = self.root / "catalog"
        with patch.object(prepare, "ROOT", self.root / "binary"):
            prepare.assemble(summaries, generated)
        self.assertEqual({p.name for p in generated.iterdir()}, artifacts.GENERATED_FILES)
        cards = json.loads((generated / "benchmark-cards.json").read_text())
        self.assertEqual([card["slug"] for card in cards], ["binary", "continuous"])
        irt = json.loads((generated / "benchmark-irt.json").read_text())
        self.assertEqual(set(irt), {"binary"})
        for slug, entry, _, _ in results:
            data = self.root / "restored" / slug
            artifacts.unpack(self.root / "parallel" / entry["path"], data, entry,
                             {"slug": slug, "key": entry["key"]})
            with gzip.open(data / "source.json.gz", "rt") as stream:
                self.assertEqual(json.load(stream)["directory"], f"{slug}/formatted_tables")


if __name__ == "__main__":
    unittest.main()
