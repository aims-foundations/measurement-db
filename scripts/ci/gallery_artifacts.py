"""Versioned gallery artifacts. No network access or rendering dependencies."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

SOURCE_REPO = "aims-foundations/measurement-db"
GALLERY_REPO = "aims-foundations/measurement-db-gallery"
SCHEMA_VERSION = 2
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
SLUG = re.compile(r"[a-z][a-z0-9_]*")
GENERATED_FILES = {
    "benchmark-cards.json", "benchmark-details.json", "gallery-sources.json",
    "chart-marginals.json", "model-timeline.json", "benchmark-attributions.json",
    "benchmark-saturation.json", "benchmark-irt.json",
}


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if not name or name == "." or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError(f"Unsafe artifact path: {name!r}")
    return name


def input_key(inputs: dict) -> str:
    # Revision/PR/run IDs deliberately stay outside this identity. Unchanged
    # table bytes and renderer inputs can be reused across unrelated HF commits.
    return digest(inputs)


def pack(root: Path, destination: Path, identity: dict) -> dict:
    """Use tar, not another gzip layer: all bulky viewer files are already gzipped."""
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact contains a symlink: {path}")
        if path.is_file():
            name = safe_path(path.relative_to(root).as_posix())
            if name == "artifact-index.json":
                raise ValueError("Reserved archive index name")
            files[name] = {"size": path.stat().st_size, "sha256": file_digest(path)}
    index = encoded({"schema_version": SCHEMA_VERSION, "identity": identity, "files": files})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("artifact-index.json")
        info.size, info.mode = len(index), 0o644
        archive.addfile(info, io.BytesIO(index))
        for name, meta in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = meta["size"], 0o644
            with (root / name).open("rb") as stream:
                archive.addfile(info, stream)
    return {"size": destination.stat().st_size, "sha256": file_digest(destination)}


def unpack(path: Path, destination: Path, descriptor: dict, identity: dict) -> None:
    """Verify archive and members while extracting only regular, indexed files."""
    if path.stat().st_size != descriptor["size"] or file_digest(path) != descriptor["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {path.name}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"Extraction requires an empty directory: {destination}")
    with tarfile.open(path, "r:") as archive:
        first = archive.next()
        if first is None or first.name != "artifact-index.json" or not first.isfile() or first.size > 32_000_000:
            raise ValueError("Missing or oversized artifact index")
        with archive.extractfile(first) as stream:
            index = json.load(stream)
        if index.get("schema_version") != SCHEMA_VERSION or index.get("identity") != identity:
            raise ValueError("Artifact identity mismatch")
        expected, seen = index["files"], set()
        for member in archive:
            # tarfile iteration includes the first member returned by next().
            if member is first:
                continue
            name = safe_path(member.name)
            if not member.isfile() or name not in expected or name in seen:
                raise ValueError(f"Unindexed, duplicate, or non-file archive member: {name}")
            meta = expected[name]
            if member.size != meta["size"]:
                raise ValueError(f"Member size mismatch: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            hashed = hashlib.sha256()
            with archive.extractfile(member) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    hashed.update(chunk)
                    output.write(chunk)
            if hashed.hexdigest() != meta["sha256"]:
                raise ValueError(f"Member checksum mismatch: {name}")
            seen.add(name)
        if seen != set(expected):
            raise ValueError("Archive is missing indexed files")


def validate_report(report: dict, *, revision: str, sha: str, run_id: str, attempt: int) -> dict:
    expected = {"head_sha": sha, "run_id": run_id, "run_attempt": attempt, "hf_revision": revision}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Validation report {key} does not match this preparation run")
    if report.get("warnings"):
        raise ValueError("Validation report has incomplete or inconsistent results")
    result = {}
    for row in report["results"]:
        slug = row["benchmark"]
        if not SLUG.fullmatch(slug) or slug in result:
            raise ValueError("Invalid or duplicate validation result")
        if row["status"] == "skipped_withheld":
            result[slug] = None
            continue
        if row["status"] != "passed" or not row.get("tables"):
            raise ValueError(f"{slug}: no successful byte-comparison receipt")
        result[slug] = row["tables"]
    return result


def validate_manifest(manifest: dict, *, tree: str | None = None) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("source_repo") != SOURCE_REPO:
        raise ValueError("Unsupported gallery manifest")
    if not COMMIT.fullmatch(manifest.get("source_revision", "")):
        raise ValueError("Gallery source must be an immutable HF commit")
    if not COMMIT.fullmatch(manifest.get("git_tree", "")) or (tree and manifest["git_tree"] != tree):
        raise ValueError("Prepared gallery does not match the merged Git tree; rerun preparation on the updated PR")
    if not COMMIT.fullmatch(manifest.get("git_sha", "")):
        raise ValueError("Missing preparation commit")
    if not manifest.get("benchmarks"):
        raise ValueError("Empty gallery release")
    for slug, entry in manifest["benchmarks"].items():
        if not SLUG.fullmatch(slug) or not SHA.fullmatch(entry.get("key", "")):
            raise ValueError("Invalid benchmark artifact identity")
        if input_key(entry["inputs"]) != entry["key"] or entry["inputs"]["slug"] != slug:
            raise ValueError(f"{slug}: inputs do not match cache key")
        descriptor(entry)
        expected_path = f"benchmarks/{slug}/{entry['key']}/{entry['sha256']}.tar"
        if entry["path"] != expected_path:
            raise ValueError(f"{slug}: invalid archive location")
    descriptor(manifest["catalog"])
    if manifest["catalog"]["path"] != f"catalog/{manifest['catalog']['sha256']}.tar":
        raise ValueError("Invalid catalog archive location")
    if base := manifest.get("base_gallery"):
        paths = base.get("paths")
        allowed = {entry["path"] for entry in manifest["benchmarks"].values()}
        if (not COMMIT.fullmatch(base.get("revision", "")) or not isinstance(paths, list)
                or any(not isinstance(p, str) for p in paths)
                or len(paths) != len(set(paths)) or not set(paths) <= allowed):
            raise ValueError("Invalid pinned gallery fallback")


def descriptor(entry: dict) -> None:
    safe_path(entry["path"])
    if not SHA.fullmatch(entry.get("sha256", "")) or type(entry.get("size")) is not int or entry["size"] <= 0:
        raise ValueError("Invalid archive digest or size")


def restore(candidate: Path, web_dir: Path, manifest: dict, *, tree: str) -> None:
    """Restore verified static files without importing any rendering/ML libraries."""
    from tempfile import TemporaryDirectory

    validate_manifest(manifest, tree=tree)
    with TemporaryDirectory(prefix="gallery-restore-", dir=web_dir.parent) as temporary:
        staging = Path(temporary)
        data = staging / "benchmark-data"
        data.mkdir()
        for slug, entry in sorted(manifest["benchmarks"].items()):
            unpack(candidate / entry["path"], data / slug, entry, {"slug": slug, "key": entry["key"]})
            (data / slug / "summary.json").unlink()
        catalog = manifest["catalog"]
        generated = staging / "generated"
        unpack(candidate / catalog["path"], generated, catalog, {"git_tree": tree})
        if {p.name for p in generated.iterdir()} != GENERATED_FILES:
            raise ValueError("Incomplete or unexpected generated catalog files")
        cards = json.loads((generated / "benchmark-cards.json").read_text())
        sources = json.loads((generated / "gallery-sources.json").read_text())
        if {c["slug"] for c in cards} != set(manifest["benchmarks"]) or set(sources) != set(manifest["benchmarks"]):
            raise ValueError("Catalog and benchmark archives disagree")
        import gzip
        for slug in sources:
            with gzip.open(data / slug / "source.json.gz", "rt") as stream:
                if json.load(stream) != sources[slug]:
                    raise ValueError(f"{slug}: source catalog does not match the static bundle")
        # Do not modify the checkout until the complete release has verified.
        target = web_dir / "public/benchmark-data"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        data.rename(target)
        target_generated = web_dir / "content/generated"
        target_generated.mkdir(parents=True, exist_ok=True)
        for name in GENERATED_FILES:
            shutil.copyfile(generated / name, target_generated / name)
