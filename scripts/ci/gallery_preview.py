"""Restore a pinned, explicitly unpromoted gallery for a non-main preview.

The ordinary release path still requires a successful reproduction receipt.
This path lets a schema migration be reviewed before its tables are published
to the canonical dataset. It cannot publish or deploy production data.
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import shutil

from . import gallery_artifacts as artifacts
from . import gallery_cache as cache
from .gallery_release import git


def check_manifest(manifest, revision):
    if os.environ.get("DEPLOY_ENV") != "preview" or os.environ.get("GITHUB_REF") == "refs/heads/main":
        raise ValueError("Unpromoted galleries require a non-main preview")
    if not artifacts.COMMIT.fullmatch(revision):
        raise ValueError("Preview gallery revision must be an immutable commit")
    artifacts.validate_manifest(manifest, tree=git("rev-parse", "HEAD^{tree}"))
    if manifest.get("preview_only") is not True or manifest["git_sha"] != git("rev-parse", "HEAD"):
        raise ValueError("Preview gallery must identify this exact code commit")


def restore(revision, candidate, web_dir):
    from huggingface_hub import hf_hub_download

    # Check the environment before any network requests or filesystem changes.
    if os.environ.get("DEPLOY_ENV") != "preview" or os.environ.get("GITHUB_REF") == "refs/heads/main":
        raise ValueError("Unpromoted galleries require a non-main preview")
    if not artifacts.COMMIT.fullmatch(revision):
        raise ValueError("Preview gallery revision must be an immutable commit")
    path = hf_hub_download(artifacts.GALLERY_REPO, "preview/manifest.json",
                           revision=revision, repo_type="dataset")
    manifest = json.loads(Path(path).read_text())
    check_manifest(manifest, revision)

    def fetch(entry):
        target = candidate / entry["path"]
        if cache.restore(entry, target):
            return
        path = hf_hub_download(artifacts.GALLERY_REPO, "v2/" + entry["path"],
                               revision=revision, repo_type="dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        cache.store(target, entry)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, [manifest["catalog"], *manifest["benchmarks"].values()]))
    artifacts.restore(candidate, web_dir, manifest, tree=manifest["git_tree"])
    message = (f"Unpromoted preview gallery: {artifacts.GALLERY_REPO}@{revision}; "
               f"code: {manifest['git_sha']}; candidate data: {manifest['source_revision']}. "
               "Canonical datasets and accepted production galleries were not changed.")
    print(message, flush=True)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as output:
            output.write("\n### Unpromoted migration preview\n\n" + message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--web-dir", type=Path, default=Path("website"))
    args = parser.parse_args()
    restore(args.revision, args.candidate, args.web_dir)
