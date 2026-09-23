"""Restore and verify raw inputs using the file tree of an immutable HF commit.

The Hub tree supplies sizes and content hashes; metadata contains no per-file
inventory. Existing cached bytes are checked, never silently accepted or repaired.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from .load_source_files import SourceDataError

# Published pilot inputs are pinned once with the loader, not in every metadata file.
DEFAULT_SOURCE_REPOSITORY = "aims-foundations/measurement-db"
DEFAULT_SOURCE_REVISION = "c969fabbf60c44694dae0e6f4d521de022b21d84"


def snapshot_location(benchmark_dir: str | Path) -> dict[str, str]:
    """Resolve shared defaults or explicit runtime overrides for another snapshot."""
    slug = Path(benchmark_dir).name
    local_manifest = os.environ.get("MEASUREMENT_DB_SOURCE_MANIFEST")
    if local_manifest:
        return {"manifest": str(Path(local_manifest).resolve()), "benchmark": slug}
    repository = os.environ.get("MEASUREMENT_DB_SOURCE_REPO", DEFAULT_SOURCE_REPOSITORY)
    revision = os.environ.get("MEASUREMENT_DB_SOURCE_REVISION",
                              DEFAULT_SOURCE_REVISION if repository == DEFAULT_SOURCE_REPOSITORY else "")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", slug):
        raise SourceDataError(f"invalid benchmark directory name: {slug}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repository):
        raise SourceDataError("MEASUREMENT_DB_SOURCE_REPO must be an owner/dataset identifier")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SourceDataError("MEASUREMENT_DB_SOURCE_REVISION must be an immutable 40-character commit")
    return {"repo_id": repository, "revision": revision, "path": f"{slug}/raw"}


def snapshot_artifacts(archive: dict) -> list[dict]:
    if "manifest" in archive:
        # Development builds use a frozen local inventory until raw inputs have
        # a published Hub snapshot. This never changes benchmark metadata.
        manifest = json.loads(Path(archive["manifest"]).read_text())
        if manifest.get("format_version") != 1:
            raise SourceDataError("unsupported local source manifest version")
        artifacts = manifest.get("benchmarks", {}).get(archive["benchmark"])
        if not isinstance(artifacts, list) or not artifacts:
            raise SourceDataError("local source manifest has no inputs for this benchmark")
        seen = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {"file", "size", "hash_kind", "digest", "url"}:
                raise SourceDataError("invalid local source artifact fields")
            name = artifact["file"]
            if (not isinstance(name, str) or not name or name in seen
                    or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                    or "\\" in name or str(PurePosixPath(name)) != name):
                raise SourceDataError("invalid or duplicate local source path")
            seen.add(name)
            if (type(artifact["size"]) is not int or artifact["size"] < 0
                    or artifact["hash_kind"] != "sha256"
                    or not isinstance(artifact["digest"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", artifact["digest"])
                    or not isinstance(artifact["url"], str) or not artifact["url"].strip()):
                raise SourceDataError("invalid local source digest, size, or locator")
        return sorted(artifacts, key=lambda artifact: artifact["file"])

    from huggingface_hub import HfApi, hf_hub_url

    prefix = archive["path"] + "/"
    artifacts = []
    for entry in HfApi().list_repo_tree(
        archive["repo_id"], repo_type="dataset", revision=archive["revision"],
        path_in_repo=archive["path"], recursive=True,
    ):
        if not hasattr(entry, "blob_id"):
            continue
        if not entry.path.startswith(prefix):
            raise SourceDataError(f"unexpected snapshot path: {entry.path}")
        relative = entry.path[len(prefix):]
        parts = PurePosixPath(relative).parts
        if not relative or ".." in parts or "\\" in relative or relative.startswith("/"):
            raise SourceDataError(f"invalid snapshot path: {entry.path}")
        lfs = entry.lfs
        digest = (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256) if lfs else entry.blob_id
        artifacts.append({
            "file": relative, "size": entry.size,
            "hash_kind": "sha256" if lfs else "git_sha1", "digest": digest,
            "url": hf_hub_url(archive["repo_id"], entry.path, repo_type="dataset",
                              revision=archive["revision"]),
        })
    if not artifacts:
        raise SourceDataError(f"archive contains no raw inputs: {archive}")
    return sorted(artifacts, key=lambda artifact: artifact["file"])


def verify_snapshot_file(path: Path, artifact: dict) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SourceDataError(f"cannot stat {path}: {exc}") from exc
    if size != artifact["size"]:
        raise SourceDataError(f"{path}: expected {artifact['size']} bytes, found {size}")
    digest = (hashlib.sha256() if artifact["hash_kind"] == "sha256"
              else hashlib.sha1(b"blob %d\0" % size))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact["digest"]:
        raise SourceDataError(f"{path}: content differs from the pinned source")


def restore_snapshot(archive: dict, raw: Path, artifacts: list[dict]) -> None:
    from huggingface_hub import hf_hub_download

    root = raw.resolve()
    for artifact in artifacts:
        target = raw / artifact["file"]
        if not target.resolve().is_relative_to(root):
            raise SourceDataError(f"snapshot input escapes raw/: {target}")
        if target.exists():
            verify_snapshot_file(target, artifact)
            continue
        if "manifest" in archive:
            raise SourceDataError(f"frozen local input is absent: {target}")
        cached = Path(hf_hub_download(
            archive["repo_id"], f"{archive['path']}/{artifact['file']}",
            repo_type="dataset", revision=archive["revision"],
        ))
        verify_snapshot_file(cached, artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(cached, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
