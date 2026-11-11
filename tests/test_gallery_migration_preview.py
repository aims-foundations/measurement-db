"""Unpromoted galleries cannot enter the production restoration path."""

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.ci import gallery_artifacts as artifacts, gallery_preview as preview
from test_gallery_artifacts import candidate_fixture, COMMIT, TREE, REVISION


@pytest.mark.parametrize("mode,ref", [("production", "refs/heads/feature"), ("preview", "refs/heads/main")])
def test_rejects_production_before_network_or_writes(tmp_path, mode, ref):
    with patch.dict(os.environ, DEPLOY_ENV=mode, GITHUB_REF=ref), \
         patch("huggingface_hub.hf_hub_download") as download:
        with pytest.raises(ValueError, match="non-main preview"):
            preview.restore(REVISION, tmp_path / "candidate", tmp_path / "website")
        download.assert_not_called()
        assert not list(tmp_path.iterdir())


def test_pinned_preview_restores_verified_files_and_rejects_wrong_commit_or_corruption(tmp_path):
    candidate, manifest = candidate_fixture(tmp_path)
    manifest["preview_only"] = True
    artifacts.write_json(candidate / "preview.json", manifest)

    def download(repo, path, *, revision, repo_type):
        assert revision == REVISION
        assert repo == artifacts.GALLERY_REPO
        return str(candidate / ("preview.json" if path == "preview/manifest.json" else path.removeprefix("v2/")))

    with patch.dict(os.environ, DEPLOY_ENV="preview", GITHUB_REF="refs/heads/feature", GALLERY_CACHE_DIR="", GITHUB_STEP_SUMMARY=""), \
         patch.object(preview, "git", side_effect=lambda *args: TREE if args[-1] == "HEAD^{tree}" else COMMIT), \
         patch("huggingface_hub.hf_hub_download", side_effect=download):
        preview.restore(REVISION, tmp_path / "restore", tmp_path / "website")
        assert (tmp_path / "website/public/benchmark-data/alpha/view.json.gz").is_file()
        for field, value in [("git_sha", "f" * 40), ("git_tree", "f" * 40), ("preview_only", False)]:
            changed = copy.deepcopy(manifest)
            changed[field] = value
            with pytest.raises(ValueError):
                preview.check_manifest(changed, REVISION)
        with pytest.raises(ValueError, match="immutable"):
            preview.check_manifest(manifest, "main")
        archive = candidate / manifest["catalog"]["path"]
        archive.write_bytes(archive.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="checksum"):
            preview.restore(REVISION, tmp_path / "corrupt", tmp_path / "corrupt-web")
        assert not (tmp_path / "corrupt-web").exists()
