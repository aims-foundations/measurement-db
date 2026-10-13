"""Optional shared runner cache. Manifests and content hashes remain authoritative.

GALLERY_CACHE_DIR may point to a shared disk on the trusted runner pool. Losing
this directory affects speed only: GitHub handoff artifacts and pinned accepted
HF objects remain the portable recovery path. Writes use atomic replacement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from . import gallery_artifacts as artifacts


def root() -> Path | None:
    value = os.environ.get("GALLERY_CACHE_DIR")
    return Path(value) if value else None


def matches(path: Path, entry: dict) -> bool:
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == entry["size"]
            and artifacts.file_digest(path) == entry["sha256"])


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=".writing-", delete=False) as output:
        temporary = Path(output.name)
        try:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def object_path(entry: dict) -> Path | None:
    artifacts.descriptor(entry)
    directory = root()
    return directory / "objects" / (entry["sha256"] + ".tar") if directory else None


def store(source: Path, entry: dict) -> None:
    target = object_path(entry)
    if target is None:
        return
    if not matches(source, entry):
        raise ValueError("Refusing to cache an unverified archive")
    if not matches(target, entry):
        atomic_copy(source, target)


def restore(entry: dict, target: Path) -> bool:
    source = object_path(entry)
    if source is None or not matches(source, entry):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def remember(slug: str, entry: dict, archive: Path) -> None:
    if root() is None:
        return
    if not artifacts.SLUG.fullmatch(slug) or artifacts.input_key(entry["inputs"]) != entry["key"]:
        raise ValueError("Invalid cache input identity")
    store(archive, entry)
    target = root() / "inputs" / slug / (entry["key"] + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=".writing-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(artifacts.encoded(entry))
            stream.flush()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def lookup(slug: str, inputs: dict) -> dict | None:
    if root() is None or not artifacts.SLUG.fullmatch(slug):
        return None
    key = artifacts.input_key(inputs)
    path = root() / "inputs" / slug / (key + ".json")
    try:
        if path.is_symlink():
            return None
        entry = json.loads(path.read_text())
        if entry["key"] != key or entry["inputs"] != inputs:
            return None
        if entry["path"] != f"benchmarks/{slug}/{key}/{entry['sha256']}.tar":
            return None
        source = object_path(entry)
        return entry if matches(source, entry) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
