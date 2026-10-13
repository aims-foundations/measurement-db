"""Promote validated PR artifacts and restore static files for Vercel.

This module intentionally uses no NumPy, Parquet, or PyTorch imports. It never
writes to the canonical measurement-db dataset and never regenerates a gallery.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from tempfile import TemporaryDirectory
import time
import urllib.parse
import urllib.request
import zipfile

from . import gallery_artifacts as artifacts
from . import gallery_cache as cache

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "aims-foundations/measurement-db"
WORKFLOW = ".github/workflows/benchmark-reproduction.yml"


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirect = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirect and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            redirect.remove_header("Authorization")
        return redirect


def github(path: str, *, destination: Path | None = None):
    url = f"https://api.github.com/repos/{REPOSITORY}/{path}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "measurement-db-gallery"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.build_opener(SafeRedirect()).open(request, timeout=120) as response:
        if destination is None:
            return json.load(response)
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def checked_run(run: dict, *, allow_current: bool = False) -> None:
    current = allow_current and str(run["id"]) == os.environ.get("GITHUB_RUN_ID")
    if (run.get("path") != WORKFLOW or run.get("event") not in {"pull_request", "workflow_dispatch"}
            or run.get("repository", {}).get("full_name") != REPOSITORY
            or run.get("head_repository", {}).get("full_name") != REPOSITORY):
        raise ValueError("Candidate must come from the repository's own validation workflow")
    if not current and (run.get("status") != "completed" or run.get("conclusion") != "success"):
        raise ValueError("Candidate validation workflow has not succeeded")


def find_artifact(run: dict, prefix: str) -> dict:
    attempt = run.get("preparation_attempt", run["run_attempt"])
    name = f"{prefix}-{attempt}"
    matches = []
    page = 1
    while True:
        rows = github(f"actions/runs/{run['id']}/artifacts?per_page=100&page={page}")["artifacts"]
        for row in rows:
            match = re.fullmatch(re.escape(prefix) + r"-([1-9][0-9]*)", row["name"])
            if match and not row["expired"]:
                number = int(match[1])
                # A failed-preview-only rerun does not recreate preparation.
                # Select the latest successful manifest, then require its exact
                # candidate attempt. Never mix artifacts across attempts.
                if number == attempt or (prefix == "gallery-manifest" and number < attempt):
                    matches.append((number, row))
        if len(rows) < 100:
            break
        page += 1
    if not matches:
        raise ValueError(f"Missing/expired {name}; rerun gallery preparation")
    newest = max(number for number, _ in matches)
    selected = [row for number, row in matches if number == newest]
    if len(selected) != 1:
        raise ValueError(f"Ambiguous artifact: {prefix}-{newest}")
    return selected[0]


def unzip(path: Path, target: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set()
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = artifacts.safe_path(member.filename)
            if name in names or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Duplicate or symlinked candidate file")
            names.add(name)
            if name not in {"manifest.json", "validation.json", "timings.json"} and not (
                    name.endswith(".tar") and name.startswith(("benchmarks/", "catalog/"))):
                raise ValueError(f"Unexpected candidate file: {name}")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def manifest_from_run(run: dict, tree: str, receipt: Path | None = None) -> dict:
    with TemporaryDirectory(prefix="gallery-manifest-") as temporary:
        archive = Path(temporary) / "manifest.zip"
        artifact = find_artifact(run, "gallery-manifest")
        github(f"actions/artifacts/{artifact['id']}/zip", destination=archive)
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            if (set(names) not in ({"manifest.json"}, {"manifest.json", "validation.json"})
                    or len(names) != len(set(names))
                    or any(zipped.getinfo(name).file_size > 32_000_000 for name in names)):
                raise ValueError("Invalid candidate manifest artifact")
            manifest = json.loads(zipped.read("manifest.json"))
            report = zipped.read("validation.json") if "validation.json" in names else None
    artifacts.validate_manifest(manifest, tree=tree)
    attempt = int(artifact["name"].rsplit("-", 1)[1])
    if manifest["validation"]["run_id"] != str(run["id"]) or manifest["validation"]["run_attempt"] != attempt:
        raise ValueError("Candidate manifest belongs to a different validation run")
    run["preparation_attempt"] = attempt
    if receipt is not None and report is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_bytes(report)
    return manifest


def candidate_runs(sha: str) -> list[dict]:
    # GitHub records PR run head_sha as the contributor's head, while checkout
    # builds the synthetic merge commit. Match the final tree separately so this
    # also works for squash merges and rejects a stale base branch.
    heads = {sha}
    for pr in github(f"commits/{sha}/pulls?per_page=100"):
        if pr.get("merged_at") and pr["base"]["ref"] == "main" and pr["head"]["repo"] and pr["head"]["repo"]["full_name"] == REPOSITORY:
            heads.add(pr["head"]["sha"])
    runs = {}
    for head in sorted(heads):
        query = urllib.parse.urlencode({"head_sha": head, "status": "completed", "per_page": 100})
        for run in github(f"actions/workflows/benchmark-reproduction.yml/runs?{query}")["workflow_runs"]:
            if run["event"] in {"pull_request", "workflow_dispatch"} and run["conclusion"] == "success":
                runs[run["id"]] = run
    return sorted(runs.values(), key=lambda run: (run["id"], run["run_attempt"]), reverse=True)


def verify_candidate(candidate: Path, *, tree: str) -> dict:
    manifest = json.loads((candidate / "manifest.json").read_text())
    artifacts.validate_manifest(manifest, tree=tree)
    verify_receipt(candidate / "validation.json", manifest)
    for entry in [manifest["catalog"], *manifest["benchmarks"].values()]:
        if not cache.matches(candidate / entry["path"], entry):
            raise ValueError(f"Candidate archive checksum mismatch: {entry['path']}")
    return manifest


def verify_receipt(report_path: Path, manifest: dict) -> None:
    if artifacts.file_digest(report_path) != manifest["validation"]["report_sha256"]:
        raise ValueError("Validation receipt checksum mismatch")
    receipts = artifacts.validate_report(json.loads(report_path.read_text()),
        revision=manifest["source_revision"], sha=manifest["git_sha"],
        run_id=manifest["validation"]["run_id"], attempt=manifest["validation"]["run_attempt"])
    for slug, receipt in receipts.items():
        if slug in manifest["benchmarks"]:
            inputs = manifest["benchmarks"][slug]["inputs"]["tables"]
            expected = {name: {"size": row["size"], "sha256": row["sha256"]} for name, row in inputs.items()}
            if receipt != expected:
                raise ValueError(f"{slug}: release differs from byte-validated tables")


def download(candidate: Path, run_id: str | None = None) -> dict:
    started = time.monotonic()
    if candidate.exists() and any(candidate.iterdir()):
        raise ValueError("Candidate directory must be empty")
    sha, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    if not run_id and (published := download_published(candidate, tree)) is not None:
        return published
    runs = [github(f"actions/runs/{int(run_id)}")] if run_id else candidate_runs(sha)
    selected, errors = None, []
    for run in runs:
        try:
            checked_run(run, allow_current=True)
            manifest = manifest_from_run(run, tree, candidate / "validation.json")
            selected = run
            break
        except ValueError as exc:
            errors.append(f"run {run['id']}: {exc}")
    if selected is None:
        raise ValueError("No successful prepared candidate matches this tree. Update/revalidate the PR, "
                         "or dispatch Benchmark table reproduction with prepare_gallery=true. " + "; ".join(errors))
    entries = [manifest["catalog"], *manifest["benchmarks"].values()]
    cached = 0
    portable_download = False
    if (candidate / "validation.json").exists():
        verify_receipt(candidate / "validation.json", manifest)
        for entry in entries:
            cached += cache.restore(entry, candidate / entry["path"])
    missing = [entry for entry in entries if not (candidate / entry["path"]).is_file()]
    base = manifest.get("base_gallery", {"paths": []})
    if any(entry["path"] not in base["paths"] for entry in missing) or not (candidate / "validation.json").exists():
        portable_download = True
        with TemporaryDirectory(prefix="gallery-candidate-") as temporary:
            archive = Path(temporary) / "candidate.zip"
            artifact = find_artifact(selected, "gallery-candidate")
            github(f"actions/artifacts/{artifact['id']}/zip", destination=archive)
            staged = Path(temporary) / "files"
            unzip(archive, staged)
            if json.loads((staged / "manifest.json").read_text()) != manifest:
                raise ValueError("Candidate archive and manifest artifact disagree")
            verify_receipt(staged / "validation.json", manifest)
            candidate.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged / "validation.json", candidate / "validation.json")
            for entry in missing:
                source = staged / entry["path"]
                if source.exists():
                    if not cache.matches(source, entry):
                        raise ValueError(f"Candidate archive checksum mismatch: {entry['path']}")
                    target = candidate / entry["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    missing = [entry for entry in entries if not (candidate / entry["path"]).is_file()]
    if missing:
        from huggingface_hub import hf_hub_download
        def fetch(entry):
            if entry["path"] not in base["paths"]:
                raise ValueError(f"Portable handoff is missing {entry['path']}")
            source = Path(hf_hub_download(artifacts.GALLERY_REPO, "v2/" + entry["path"],
                                        revision=base["revision"], repo_type="dataset"))
            if not cache.matches(source, entry):
                raise ValueError("Pinned HF fallback archive checksum mismatch")
            target = candidate / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, missing))
    artifacts.write_json(candidate / "manifest.json", manifest)
    verify_candidate(candidate, tree=tree)
    for entry in entries:
        cache.store(candidate / entry["path"], entry)
    timing = {"seconds": round(time.monotonic() - started, 3), "cached_archives": cached,
              "hf_archives": len(missing), "github_candidate_downloaded": portable_download,
              "archive_bytes": sum(entry["size"] for entry in entries)}
    artifacts.write_json(candidate / "handoff-timing.json", timing)
    print("Gallery handoff: " + json.dumps(timing), flush=True)
    print(f"Verified gallery candidate from run {selected['id']}, preparation attempt {selected['preparation_attempt']}")
    return manifest


def download_published(candidate: Path, tree: str) -> dict | None:
    """Durable redeploy path, pinned to one HF commit and this exact Git tree."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    revision = HfApi().dataset_info(artifacts.GALLERY_REPO).sha
    try:
        path = hf_hub_download(artifacts.GALLERY_REPO, "v2/latest.json", revision=revision,
                               repo_type="dataset")
    except RemoteEntryNotFoundError:
        return None
    manifest = json.loads(Path(path).read_text())
    artifacts.validate_manifest(manifest)
    if manifest["git_tree"] != tree:
        return None

    def fetch(pair):
        remote, local, entry = pair
        target = candidate / local
        if entry and cache.restore(entry, target):
            return
        path = hf_hub_download(artifacts.GALLERY_REPO, remote, revision=revision, repo_type="dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    files = [("v2/" + entry["path"], entry["path"], entry)
             for entry in [manifest["catalog"], *manifest["benchmarks"].values()]]
    files.append((f"v2/receipts/{manifest['validation']['report_sha256']}.json", "validation.json", None))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    artifacts.write_json(candidate / "manifest.json", manifest)
    verify_candidate(candidate, tree=tree)
    for entry in [manifest["catalog"], *manifest["benchmarks"].values()]:
        cache.store(candidate / entry["path"], entry)
    print(f"Downloaded accepted static gallery from {artifacts.GALLERY_REPO}@{revision}")
    return manifest


def publish(candidate: Path) -> str:
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    if os.environ.get("GITHUB_REF") != "refs/heads/main" or os.environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}:
        raise ValueError("Only a main-branch release can publish gallery artifacts")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Configure HF_TOKEN with write access to measurement-db-gallery")
    sha, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    if github("git/ref/heads/main")["object"]["sha"] != sha:
        raise ValueError("A newer main commit exists; refusing to publish a stale release")
    manifest = verify_candidate(candidate, tree=tree)
    api = HfApi(token=token)
    parent = api.dataset_info(artifacts.GALLERY_REPO).sha
    known = set()
    try:
        old = json.loads(Path(hf_hub_download(artifacts.GALLERY_REPO, "v2/latest.json", revision=parent,
                                             repo_type="dataset", token=token)).read_text())
        artifacts.validate_manifest(old)
        known = {entry["path"] for entry in [old["catalog"], *old["benchmarks"].values()]}
    except RemoteEntryNotFoundError:
        pass
    entries = [manifest["catalog"], *manifest["benchmarks"].values()]
    pending = [entry for entry in entries if entry["path"] not in known]
    # Stage immutable objects in bounded commits; expose the release pointer only
    # after every archive has uploaded. Leave all historical gallery files alone.
    for start in range(0, len(pending), 32):
        operations = [CommitOperationAdd(path_in_repo="v2/" + entry["path"],
                                         path_or_fileobj=candidate / entry["path"])
                      for entry in pending[start:start + 32]]
        commit = api.create_commit(repo_id=artifacts.GALLERY_REPO, repo_type="dataset",
            operations=operations, parent_commit=parent, num_threads=4,
            commit_message=f"Stage prepared gallery objects for {sha[:12]} ({start // 32 + 1})")
        parent = commit.oid
    if github("git/ref/heads/main")["object"]["sha"] != sha:
        raise ValueError("Main changed during upload; staged objects were not promoted")
    release = f"v2/releases/{tree}/{artifacts.digest(manifest)}.json"
    operations = [CommitOperationAdd(
        path_in_repo=f"v2/receipts/{manifest['validation']['report_sha256']}.json",
        path_or_fileobj=candidate / "validation.json")]
    operations += [CommitOperationAdd(path_in_repo=path, path_or_fileobj=artifacts.encoded(manifest))
                  for path in (release, "v2/latest.json")]
    commit = api.create_commit(repo_id=artifacts.GALLERY_REPO, repo_type="dataset",
        operations=operations, parent_commit=parent,
        commit_message=f"Publish validated gallery for {sha[:12]}")
    print(f"Published {artifacts.GALLERY_REPO}@{commit.oid}")
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as output:
            output.write(f"revision={commit.oid}\n")
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--run-id", default=None)
    for name in ("publish", "restore"):
        commands.add_parser(name)
    for command in commands.choices.values():
        command.add_argument("--candidate", type=Path, required=True)
    commands.choices["restore"].add_argument("--web-dir", type=Path, default=ROOT / "website")
    args = parser.parse_args()
    if args.command == "download":
        download(args.candidate, args.run_id or None)
    elif args.command == "publish":
        publish(args.candidate)
    else:
        tree = git("rev-parse", "HEAD^{tree}")
        manifest = verify_candidate(args.candidate, tree=tree)
        artifacts.restore(args.candidate, args.web_dir, manifest, tree=tree)
        print(f"Restored {len(manifest['benchmarks'])} static benchmark bundles; no rendering or model fitting")


if __name__ == "__main__":
    main()
