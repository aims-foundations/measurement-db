"""Restore and verify raw inputs using the file tree of an immutable HF commit.

The Hub tree supplies sizes and content hashes; metadata contains no per-file
inventory. Existing cached bytes are checked, never silently accepted or repaired.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from .load_source_files import SourceDataError


def snapshot_artifacts(archive: dict) -> list[dict]:
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
        raise SourceDataError(f"{path}: content differs from the pinned archive")


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
