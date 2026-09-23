"""Load and validate raw benchmark source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from typing import Any


class SourceDataError(RuntimeError):
    """A downloaded source is absent, corrupt, or structurally unreadable."""


def upstream_artifacts(sources: list[dict], names: tuple[str, ...]) -> list[dict]:
    """Resolve named upstream selections to pinned files, without downloading them.

    Repository trees supply the file hashes. HTTP endpoints instead declare their
    expected bytes in metadata. No MeasurementDB archive is consulted.
    """
    named = {source["name"]: source for source in sources if "name" in source}
    if names == ("*",):
        names = tuple(named)
    if len(named) != sum("name" in source for source in sources):
        raise SourceDataError("Duplicate upstream source names")
    if not names or len(set(names)) != len(names) or set(names) - named.keys():
        raise SourceDataError(f"Select distinct declared upstream source names: {names}")
    artifacts, destinations = [], set()
    for name in names:
        source = named[name]
        url = source["url"]
        if "file" in source:
            selected = [dict(file=source["file"], url=url, size=source["size"],
                             hash_kind="sha256", digest=source["sha256"])]
        else:
            revision = source["revision"]
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise SourceDataError(f"{name}: pin the upstream repository to a full commit SHA")
            location = urlparse(url)
            entries = []
            if location.netloc == "github.com":
                repository = location.path.strip("/")
                if len(repository.split("/")) != 2:
                    raise SourceDataError(f"{name}: expected a GitHub repository URL")
                request = Request(f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1",
                                  headers={"User-Agent": "measurement-db"})
                with urlopen(request, timeout=120) as response:
                    tree = json.load(response)
                if tree.get("truncated"):
                    raise SourceDataError(f"{name}: upstream GitHub tree is truncated")
                for entry in tree["tree"]:
                    if entry["type"] == "blob":
                        entries.append(dict(path=entry["path"], size=entry["size"],
                            hash_kind="git_sha1", digest=entry["sha"],
                            url=f"https://raw.githubusercontent.com/{repository}/{revision}/{quote(entry['path'], safe='/')}"))
            elif location.netloc == "huggingface.co" and location.path.startswith("/datasets/"):
                from huggingface_hub import HfApi, hf_hub_url
                repository = location.path.removeprefix("/datasets/").rstrip("/")
                if len(repository.split("/")) != 2:
                    raise SourceDataError(f"{name}: expected a Hugging Face dataset URL")
                for entry in HfApi().list_repo_tree(repository, repo_type="dataset", revision=revision, recursive=True):
                    if not hasattr(entry, "blob_id"):
                        continue
                    lfs = entry.lfs
                    digest = (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256) if lfs else entry.blob_id
                    entries.append(dict(path=entry.path, size=entry.size,
                        hash_kind="sha256" if lfs else "git_sha1", digest=digest,
                        url=hf_hub_url(repository, entry.path, repo_type="dataset", revision=revision),
                        hf_repo=repository, hf_revision=revision, hf_path=entry.path))
            else:
                raise SourceDataError(f"{name}: unsupported repository URL {url}")
            selected = []
            for rule in source["files"]:
                matches = 0
                for entry in sorted(entries, key=lambda row: row["path"]):
                    match = re.fullmatch(rule["match"], entry["path"])
                    if match is None:
                        continue
                    matches += 1
                    destination = rule["path"].format(path=entry["path"], **match.groupdict())
                    # Keep established cache filenames for punctuation in run names.
                    destination = re.sub(r"[^A-Za-z0-9._/-]", lambda m: f"_x{ord(m[0]):02x}_", destination)
                    selected.append({**entry, "file": destination})
                if not matches:
                    raise SourceDataError(f"{name}: no upstream files match {rule['match']!r}")
        for artifact in selected:
            path = Path(artifact["file"])
            if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
                raise SourceDataError(f"{name}: unsafe raw destination {path}")
            if path.as_posix() in destinations:
                raise SourceDataError(f"Duplicate upstream destination: {path}")
            destinations.add(path.as_posix())
            artifacts.append(artifact)
    return sorted(artifacts, key=lambda artifact: artifact["file"])


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceDataError(f"cannot read {path}: {exc}") from None
    return digest.hexdigest()


def verify_file(
    path: Path,
    *,
    minimum_size: int | None = None,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Validate one source file and return its path.

    ``minimum_size`` is inclusive. Use ``expected_size`` when a provider's
    pinned byte count is known; supplying both is an authoring error.
    """
    if minimum_size is not None and expected_size is not None:
        raise ValueError("minimum_size and expected_size are mutually exclusive")
    for name, value in (
        ("minimum_size", minimum_size),
        ("expected_size", expected_size),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None")
    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal digits")

    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise SourceDataError(f"cannot stat {path}: {exc}") from None

    if minimum_size is not None and actual_size < minimum_size:
        raise SourceDataError(
            f"{path}: expected at least {minimum_size} bytes, found {actual_size}"
        )
    if expected_size is not None and actual_size != expected_size:
        raise SourceDataError(
            f"{path}: expected {expected_size} bytes, found {actual_size}"
        )
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256.lower():
            raise SourceDataError(
                f"{path}: expected SHA-256 {expected_sha256.lower()}, "
                f"found {actual_sha256}"
            )
    return path


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read nonblank JSONL lines, requiring one JSON object per physical line."""
    records: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SourceDataError(
                        f"{path}:{line_number}:{exc.colno}: invalid JSON: {exc.msg}"
                    ) from None
                if not isinstance(value, dict):
                    raise SourceDataError(
                        f"{path}:{line_number}: expected a JSON object, "
                        f"found {type(value).__name__}"
                    )
                records.append(value)
    except SourceDataError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        location = f"{path}:{line_number}" if line_number else str(path)
        raise SourceDataError(f"cannot read {location}: {exc}") from None
    return records
