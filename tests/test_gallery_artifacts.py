"""Release integrity and provenance, with real archives and no network writes."""

import gzip
import io
import json
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts.ci import gallery_artifacts as artifacts
from scripts.ci import gallery_release as release

TREE, COMMIT, REVISION = "a" * 40, "b" * 40, "c" * 40


def candidate_fixture(root):
    candidate = root / "candidate"
    bundle = root / "bundle"
    bundle.mkdir()
    source = {"revision": REVISION, "dynamicItems": False, "tables": {}}
    artifacts.write_json(bundle / "summary.json", {"source": source})
    (bundle / "source.json.gz").write_bytes(gzip.compress(artifacts.encoded(source)))
    (bundle / "view.json.gz").write_bytes(gzip.compress(b'{"hello":"world"}'))
    inputs = {"slug": "alpha", "tables": {"items.parquet": {"sha256": "d" * 64, "size": 123}}}
    key = artifacts.input_key(inputs)
    archive = root / "bundle.tar"
    meta = artifacts.pack(bundle, archive, {"slug": "alpha", "key": key})
    meta.update(key=key, inputs=inputs, path=f"benchmarks/alpha/{key}/{meta['sha256']}.tar")
    target = candidate / meta["path"]
    target.parent.mkdir(parents=True)
    archive.rename(target)
    generated = root / "generated"
    for name in artifacts.GENERATED_FILES:
        value = ([{"slug": "alpha"}] if name == "benchmark-cards.json" else
                 {"alpha": source} if name == "gallery-sources.json" else {})
        artifacts.write_json(generated / name, value)
    archive = root / "catalog.tar"
    catalog = artifacts.pack(generated, archive, {"git_tree": TREE})
    catalog["path"] = f"catalog/{catalog['sha256']}.tar"
    (candidate / "catalog").mkdir()
    archive.rename(candidate / catalog["path"])
    report = {"head_sha": COMMIT, "hf_revision": REVISION, "run_id": "123", "run_attempt": 2,
              "warnings": [], "results": [{"benchmark": "alpha", "status": "passed", "tables": inputs["tables"]}]}
    artifacts.write_json(candidate / "validation.json", report)
    manifest = {"schema_version": 2, "source_repo": artifacts.SOURCE_REPO, "source_revision": REVISION,
                "git_tree": TREE, "git_sha": COMMIT, "benchmarks": {"alpha": meta}, "catalog": catalog,
                "validation": {"run_id": "123", "run_attempt": 2,
                               "report_sha256": artifacts.file_digest(candidate / "validation.json")}}
    artifacts.write_json(candidate / "manifest.json", manifest)
    return candidate, manifest


class GalleryArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, GALLERY_CACHE_DIR="")
        environment.start()
        self.addCleanup(environment.stop)
        self.candidate, self.manifest = candidate_fixture(self.root)

    def test_verified_release_restores_static_files_and_removes_obsolete_benchmarks(self):
        web = self.root / "website"
        artifacts.write_json(web / "public/benchmark-data/obsolete/view.json", {})
        release.verify_candidate(self.candidate, tree=TREE)
        artifacts.restore(self.candidate, web, self.manifest, tree=TREE)
        self.assertEqual({p.name for p in (web / "content/generated").iterdir()}, artifacts.GENERATED_FILES)
        data = web / "public/benchmark-data"
        self.assertEqual([p.name for p in data.iterdir()], ["alpha"])
        self.assertEqual(json.loads(gzip.decompress((data / "alpha/view.json.gz").read_bytes())), {"hello": "world"})
        self.assertFalse((data / "alpha/summary.json").exists())

    def test_corrupt_bundle_never_replaces_existing_website(self):
        archive = self.candidate / self.manifest["catalog"]["path"]
        archive.write_bytes(archive.read_bytes() + b"tampered")
        web = self.root / "website"
        sentinel = web / "public/benchmark-data/old.json"
        artifacts.write_json(sentinel, {"old": True})
        with self.assertRaisesRegex(ValueError, "checksum"):
            artifacts.restore(self.candidate, web, self.manifest, tree=TREE)
        self.assertTrue(sentinel.exists())

    def test_rejects_stale_tree_receipt_and_hf_byte_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Git tree"):
            release.verify_candidate(self.candidate, tree="f" * 40)
        report_path = self.candidate / "validation.json"
        report = json.loads(report_path.read_text())
        report["run_attempt"] = 1
        artifacts.write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "receipt checksum"):
            release.verify_candidate(self.candidate, tree=TREE)
        self.manifest["validation"]["report_sha256"] = artifacts.file_digest(report_path)
        artifacts.write_json(self.candidate / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "run_attempt"):
            release.verify_candidate(self.candidate, tree=TREE)
        report["run_attempt"] = 2
        report["results"][0]["tables"]["items.parquet"]["sha256"] = "e" * 64
        artifacts.write_json(report_path, report)
        self.manifest["validation"]["report_sha256"] = artifacts.file_digest(report_path)
        artifacts.write_json(self.candidate / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "byte-validated"):
            release.verify_candidate(self.candidate, tree=TREE)

    def test_failed_missing_reference_and_unvalidated_results_are_rejected(self):
        report = json.loads((self.candidate / "validation.json").read_text())
        for status in ("failed", "reference_error", "pending", "passed"):
            with self.subTest(status=status):
                report["results"][0] = {"benchmark": "alpha", "status": status}
                with self.assertRaisesRegex(ValueError, "byte-comparison"):
                    artifacts.validate_report(report, revision=REVISION, sha=COMMIT, run_id="123", attempt=2)

    def test_archives_reject_paths_links_and_unindexed_members_even_with_valid_archive_hash(self):
        for name, kind, indexed in (("../escape", tarfile.REGTYPE, True),
                                    ("link", tarfile.SYMTYPE, True),
                                    ("unindexed", tarfile.REGTYPE, False)):
            with self.subTest(name=name):
                archive = self.root / "hostile.tar"
                index = artifacts.encoded({"schema_version": 2, "identity": {},
                    "files": {name: {"size": 0, "sha256": "0" * 64}} if indexed else {}})
                with tarfile.open(archive, "w") as output:
                    member = tarfile.TarInfo("artifact-index.json")
                    member.size = len(index)
                    output.addfile(member, io.BytesIO(index))
                    member = tarfile.TarInfo(name)
                    member.type, member.linkname = kind, "../outside"
                    output.addfile(member)
                descriptor = {"sha256": artifacts.file_digest(archive), "size": archive.stat().st_size}
                with self.assertRaises(ValueError):
                    artifacts.unpack(archive, self.root / "extracted", descriptor, {})
        self.assertFalse((self.root / "escape").exists())

    def test_archive_member_hash_detects_tampering_even_if_whole_archive_hash_is_updated(self):
        entry = self.manifest["benchmarks"]["alpha"]
        path = self.candidate / entry["path"]
        payload = path.read_bytes().replace(b'"dynamicItems":false', b'"dynamicItems":true ')
        self.assertNotEqual(payload, path.read_bytes())
        path.write_bytes(payload)
        descriptor = {"sha256": artifacts.file_digest(path), "size": path.stat().st_size}
        with self.assertRaisesRegex(ValueError, "Member checksum"):
            artifacts.unpack(path, self.root / "extracted", descriptor, {"slug": "alpha", "key": entry["key"]})

    def test_zip_download_rejects_path_traversal(self):
        archive = self.root / "candidate.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("../manifest.json", "{}")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            release.unzip(archive, self.root / "extracted")

    def test_fork_push_and_unsuccessful_candidates_are_rejected(self):
        good = {"id": 123, "path": release.WORKFLOW, "event": "pull_request",
                "repository": {"full_name": release.REPOSITORY},
                "head_repository": {"full_name": release.REPOSITORY},
                "status": "completed", "conclusion": "success"}
        release.checked_run(good)
        for changes in ({"head_repository": {"full_name": "someone/fork"}},
                        {"event": "push"}, {"conclusion": "failure"}, {"path": "other.yml"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                release.checked_run({**good, **changes})
        with patch.dict(os.environ, GITHUB_RUN_ID="123"):
            release.checked_run({**good, "status": "in_progress", "conclusion": None}, allow_current=True)
            with self.assertRaises(ValueError):
                release.checked_run({**good, "status": "in_progress", "conclusion": None})

    def test_failed_preview_retry_reuses_one_complete_prior_preparation_attempt(self):
        run = {"id": 123, "run_attempt": 3}
        rows = [{"id": 21, "name": "gallery-manifest-2", "expired": False},
                {"id": 22, "name": "gallery-candidate-2", "expired": False},
                {"id": 11, "name": "gallery-manifest-1", "expired": False},
                {"id": 12, "name": "gallery-candidate-1", "expired": False}]
        def api(path, *, destination=None):
            if destination is None:
                return {"artifacts": rows}
            self.assertEqual(path, "actions/artifacts/21/zip")
            with zipfile.ZipFile(destination, "w") as zipped:
                zipped.writestr("manifest.json", artifacts.encoded(self.manifest))
        with patch.object(release, "github", side_effect=api):
            self.assertEqual(release.manifest_from_run(run, TREE), self.manifest)
            self.assertEqual(run["preparation_attempt"], 2)
            self.assertEqual(release.find_artifact(run, "gallery-candidate")["id"], 22)
            rows[1]["expired"] = True
            with self.assertRaisesRegex(ValueError, "Missing/expired"):
                release.find_artifact(run, "gallery-candidate")

    def test_publisher_writes_only_gallery_and_promotes_pointer_after_objects(self):
        from huggingface_hub.errors import RemoteEntryNotFoundError
        import httpx
        api = Mock()
        api.dataset_info.return_value.sha = "parent"
        api.create_commit.side_effect = [SimpleNamespace(oid="objects"), SimpleNamespace(oid="published")]
        with patch.dict(os.environ, GITHUB_REF="refs/heads/main", GITHUB_EVENT_NAME="push", HF_TOKEN="fixture"), \
             patch.object(release, "git", side_effect=[COMMIT, TREE]), \
             patch.object(release, "github", return_value={"object": {"sha": COMMIT}}), \
             patch("huggingface_hub.HfApi", return_value=api) as api_factory, \
             patch("huggingface_hub.hf_hub_download", side_effect=RemoteEntryNotFoundError(
                 "no v2 yet", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/fixture")))):
            self.assertEqual(release.publish(self.candidate), "published")
        api_factory.assert_called_once_with(token="fixture")
        calls = [call.kwargs for call in api.create_commit.call_args_list]
        self.assertEqual([c["repo_id"] for c in calls], [artifacts.GALLERY_REPO] * 2)
        self.assertTrue(all(op.path_in_repo.endswith(".tar") for op in calls[0]["operations"]))
        final_paths = [op.path_in_repo for op in calls[-1]["operations"]]
        self.assertEqual(final_paths[-1], "v2/latest.json")
        self.assertTrue(final_paths[0].startswith("v2/receipts/"))
        self.assertEqual(calls[-1]["parent_commit"], "objects")

    def test_stale_main_and_pr_cannot_publish(self):
        with patch.dict(os.environ, GITHUB_REF="refs/pull/42/merge", GITHUB_EVENT_NAME="pull_request"), \
             self.assertRaisesRegex(ValueError, "main-branch"):
            release.publish(self.candidate)
        with patch.dict(os.environ, GITHUB_REF="refs/heads/main", GITHUB_EVENT_NAME="push", HF_TOKEN="fixture"), \
             patch.object(release, "git", side_effect=[COMMIT, TREE]), \
             patch.object(release, "github", return_value={"object": {"sha": "new-main"}}), \
             self.assertRaisesRegex(ValueError, "stale"):
            release.publish(self.candidate)

    def test_published_hf_release_restores_without_github_artifacts_or_rendering(self):
        def download(repo, path, **kwargs):
            self.assertEqual(repo, artifacts.GALLERY_REPO)
            self.assertEqual(kwargs["revision"], "pinned-gallery")
            if path == "v2/latest.json":
                return self.candidate / "manifest.json"
            if path.startswith("v2/receipts/"):
                return self.candidate / "validation.json"
            return self.candidate / path.removeprefix("v2/")

        api = Mock()
        api.dataset_info.return_value.sha = "pinned-gallery"
        with patch("huggingface_hub.HfApi", return_value=api), \
             patch("huggingface_hub.hf_hub_download", side_effect=download):
            self.assertIsNone(release.download_published(self.root / "wrong-tree", "f" * 40))
            restored = self.root / "restored"
            manifest = release.download_published(restored, TREE)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(release.verify_candidate(restored, tree=TREE), self.manifest)


if __name__ == "__main__":
    unittest.main()
