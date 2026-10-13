"""Exercise real handoff archives with warm, missing and corrupt runner caches."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from scripts.ci import gallery_artifacts as artifacts, gallery_cache as cache, gallery_release as release
from scripts.ci.gallery_prepare import stage_handoff
from test_gallery_artifacts import candidate_fixture, COMMIT, TREE, REVISION


class GalleryHandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate, self.manifest = candidate_fixture(self.root)
        self.entries = [self.manifest["catalog"], *self.manifest["benchmarks"].values()]
        environment = patch.dict(os.environ, GALLERY_CACHE_DIR=str(self.root / "cache"))
        environment.start()
        self.addCleanup(environment.stop)
        self.downloads = []
        self.legacy = False
        self.omit = set()

    def github(self, path, *, destination=None):
        if path == "actions/runs/123":
            return {"id": 123, "run_attempt": 2, "path": release.WORKFLOW, "event": "pull_request",
                    "repository": {"full_name": release.REPOSITORY},
                    "head_repository": {"full_name": release.REPOSITORY},
                    "status": "completed", "conclusion": "success"}
        if "artifacts?" in path:
            return {"artifacts": [{"id": 1, "name": "gallery-manifest-2", "expired": False},
                                  {"id": 2, "name": "gallery-candidate-2", "expired": False}]}
        self.downloads.append(path)
        with zipfile.ZipFile(destination, "w") as zipped:
            zipped.writestr("manifest.json", artifacts.encoded(self.manifest))
            if not self.legacy or path == "actions/artifacts/2/zip":
                zipped.write(self.candidate / "validation.json", "validation.json")
            if path == "actions/artifacts/2/zip":
                for entry in self.entries:
                    if entry["path"] not in self.omit:
                        zipped.write(self.candidate / entry["path"], entry["path"])

    def download(self):
        target = self.root / "restored"
        with patch.object(release, "git", side_effect=[COMMIT, TREE]), \
             patch.object(release, "github", side_effect=self.github):
            self.assertEqual(release.download(target, "123"), self.manifest)
        return json.loads((target / "handoff-timing.json").read_text())

    def test_warm_handoff_never_downloads_large_github_artifact(self):
        for entry in self.entries:
            cache.store(self.candidate / entry["path"], entry)
        timing = self.download()
        self.assertEqual(self.downloads, ["actions/artifacts/1/zip"])
        self.assertEqual(timing["cached_archives"], 2)
        self.assertFalse(timing["github_candidate_downloaded"])

    def test_cold_delta_uses_pinned_hf_for_accepted_benchmarks_only(self):
        entry = self.manifest["benchmarks"]["alpha"]
        self.manifest["base_gallery"] = {"revision": REVISION, "paths": [entry["path"]]}
        self.omit.add(entry["path"])
        # The HF SDK normally returns a snapshot symlink into its blob cache.
        snapshot = self.root / "hf-snapshot.tar"
        snapshot.symlink_to(self.candidate / entry["path"])
        with patch("huggingface_hub.hf_hub_download", return_value=snapshot) as fetch:
            timing = self.download()
        fetch.assert_called_once_with(artifacts.GALLERY_REPO, "v2/" + entry["path"],
                                      revision=REVISION, repo_type="dataset")
        self.assertEqual(timing["hf_archives"], 1)
        self.assertTrue(timing["github_candidate_downloaded"])

    def test_corrupt_cache_falls_back_to_portable_archive(self):
        for entry in self.entries:
            cache.store(self.candidate / entry["path"], entry)
        cache.object_path(self.entries[0]).write_bytes(b"corrupt")
        self.assertEqual(self.download()["cached_archives"], 1)
        self.assertTrue(cache.matches(cache.object_path(self.entries[0]), self.entries[0]))

    def test_stale_receipt_is_rejected_even_with_all_archives_cached(self):
        for entry in self.entries:
            cache.store(self.candidate / entry["path"], entry)
        (self.candidate / "validation.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "receipt checksum"):
            self.download()
        self.assertEqual(self.downloads, ["actions/artifacts/1/zip"])

    def test_legacy_manifest_only_artifact_remains_supported(self):
        self.legacy = True
        self.assertTrue(self.download()["github_candidate_downloaded"])

    def test_missing_unpublished_bundle_fails_instead_of_using_hf(self):
        self.omit.add(self.entries[0]["path"])
        with patch("huggingface_hub.hf_hub_download") as fetch:
            with self.assertRaisesRegex(ValueError, "Portable handoff is missing"):
                self.download()
        fetch.assert_not_called()

    def test_portable_delta_omits_only_objects_already_accepted_on_hf(self):
        entry = self.manifest["benchmarks"]["alpha"]
        self.manifest["base_gallery"] = {"revision": REVISION, "paths": [entry["path"]]}
        artifacts.write_json(self.candidate / "manifest.json", self.manifest)
        artifacts.write_json(self.candidate / "timings.json", {})
        stage_handoff(self.candidate, self.manifest)
        staged = self.candidate / "handoff"
        self.assertFalse((staged / entry["path"]).exists())
        self.assertTrue((staged / self.manifest["catalog"]["path"]).exists())
        self.assertTrue(cache.matches(cache.object_path(entry), entry))
