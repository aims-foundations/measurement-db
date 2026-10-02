"""Rebuild changed benchmarks and compare their published Parquet bytes."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HF_REPOSITORY = "aims-foundations/measurement-db"
HF_BRANCH = "migration/tabular-builders-20260924"
ZERO_SHA = "0" * 40


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root).decode()


def validate_slug(slug: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", slug):
        raise ValueError(f"Invalid benchmark folder: {slug!r}")
    return slug


def all_benchmarks(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / "benchmarks").glob("*/build.py")
                  if p.parent.name != "_template")


def changed_benchmarks(root: Path, event_name: str, event: dict) -> list[str]:
    """Diff the entire PR or push, including both sides of folder renames."""
    if event_name == "workflow_dispatch":
        slug = event.get("inputs", {}).get("benchmark", "").strip()
        return [validate_slug(slug)] if slug else all_benchmarks(root)
    if event_name == "pull_request":
        pr = event["pull_request"]
        head = pr["head"]["sha"]
        base = git(root, "merge-base", pr["base"]["sha"], head).strip()
    elif event_name == "push":
        if event.get("deleted"):
            return []
        base, head = event["before"], event["after"]
        if base == ZERO_SHA:
            # A new branch has no previous tree: check all its builders.
            return all_benchmarks(root)
    else:
        raise ValueError(f"Unsupported event: {event_name}")
    paths = git(root, "diff", "--name-only", "--no-renames", "-z", base, head, "--", "benchmarks/")
    slugs = set()
    for path in paths.split("\0"):
        parts = PurePosixPath(path).parts
        if len(parts) >= 3 and parts[0] == "benchmarks":
            if parts[1] == "_template":
                return all_benchmarks(root)
            slugs.add(validate_slug(parts[1]))
    # Keep deleted folders/builders: verification must fail, never silently pass.
    return sorted(slugs)


def build_matrix(slugs: list[str]) -> dict:
    # GitHub permits at most 256 matrix jobs. Usually there is one builder per
    # job; large migrations use small batches and still check every benchmark.
    size = max(1, math.ceil(len(slugs) / 256))
    return {"include": [{"benchmarks": slugs[i:i + size]}
                        for i in range(0, len(slugs), size)]}


def resolve_revision(revision: str) -> str:
    from huggingface_hub import HfApi

    sha = HfApi().repo_info(HF_REPOSITORY, repo_type="dataset", revision=revision).sha
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"HF did not resolve {revision!r} to an immutable commit")
    return sha


def reference_files(api, slug: str, revision: str) -> dict[str, str]:
    """Prefer the migration layout; support the older flat release explicitly."""
    entries = list(api.list_repo_tree(HF_REPOSITORY, repo_type="dataset",
                                     revision=revision, path_in_repo=slug))
    folder = f"{slug}/formatted_tables"
    modern = any(entry.path == folder for entry in entries)
    if modern:
        entries = list(api.list_repo_tree(HF_REPOSITORY, repo_type="dataset",
                                         revision=revision, path_in_repo=folder, recursive=True))
    prefix = (folder if modern else slug) + "/"
    files = {}
    for entry in entries:
        if not hasattr(entry, "blob_id") or not entry.path.endswith(".parquet"):
            continue
        if not entry.path.startswith(prefix):
            raise ValueError(f"Unexpected HF table path: {entry.path}")
        relative = entry.path[len(prefix):]
        parts = PurePosixPath(relative).parts
        if not relative or ".." in parts or "\\" in relative or relative.startswith("/"):
            raise ValueError(f"Unsafe HF table path: {entry.path}")
        # Historical releases used this spelling; the bytes are never rewritten.
        name = "responses.parquet" if not modern and relative == "response.parquet" else relative
        if name in files:
            raise ValueError(f"Ambiguous published table: {name}")
        files[name] = entry.path
    if not files:
        raise ValueError(f"No published Parquet tables for {slug} at {revision}")
    return files


def copy_build_source(root: Path, destination: Path) -> None:
    """Copy current source edits without reusing local raw inputs or outputs."""
    paths = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    for name in paths.split("\0"):
        if not name:
            continue
        path = Path(name)
        if "raw" in path.parts or "formatted_tables" in path.parts:
            continue
        source = root / path
        if source.is_file():
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_tables(generated: Path, expected: dict[str, Path]) -> None:
    actual = {str(path.relative_to(generated)): path for path in generated.rglob("*.parquet")}
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    problems = []
    if missing:
        problems.append(f"Missing generated tables: {', '.join(missing)}")
    if extra:
        problems.append(f"Unexpected generated tables: {', '.join(extra)}")
    if not expected:
        problems.append("No reference tables; an empty comparison cannot pass")
    for name in sorted(expected.keys() & actual.keys()):
        # Compare bytes, including Parquet metadata, row order and compression.
        # Hashes are diagnostics only; equality does not rely on a hash alone.
        identical = filecmp.cmp(actual[name], expected[name], shallow=False)
        actual_hash, expected_hash = sha256(actual[name]), sha256(expected[name])
        print(f"{name}: {'PASS' if identical else 'FAIL'} "
              f"generated_sha256={actual_hash} hf_sha256={expected_hash}", flush=True)
        if not identical:
            problems.append(f"Byte mismatch: {name}")
    if problems:
        raise ValueError("\n".join(problems))


def verify_benchmark(root: Path, slug: str, revision: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    validate_slug(slug)
    if not (root / "benchmarks" / slug / "build.py").is_file():
        raise ValueError(f"Missing builder: benchmarks/{slug}/build.py")
    files = reference_files(HfApi(), slug, revision)
    with tempfile.TemporaryDirectory(prefix=f"measurement-db-{slug}-") as temporary:
        scratch = Path(temporary)
        workspace = scratch / "measurement_db"  # Builders import this package name.
        copy_build_source(root, workspace)
        expected = {
            name: Path(hf_hub_download(HF_REPOSITORY, remote, repo_type="dataset",
                                      revision=revision, local_dir=scratch / "reference"))
            for name, remote in sorted(files.items())
        }
        env = os.environ.copy()
        # HF supplies expected tables only. Archive overrides would bypass the
        # author's upstream sources declared in each benchmark's metadata.yaml.
        for key in ("MEASUREMENT_DB_SOURCE_REPO", "MEASUREMENT_DB_SOURCE_REVISION",
                    "MEASUREMENT_DB_SOURCE_MANIFEST"):
            env.pop(key, None)
        env.update(PYTHONPATH=str(scratch), PYTHONHASHSEED="0")
        command = [sys.executable, f"benchmarks/{slug}/build.py"]
        print(f"Rebuilding {slug} from metadata.yaml upstream sources: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=workspace, env=env, check=True)
        compare_tables(workspace / "benchmarks" / slug / "formatted_tables", expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select", help="Emit the changed-benchmark matrix for a GitHub event")
    resolve = commands.add_parser("resolve", help="Resolve the HF reference once for all jobs")
    resolve.add_argument("--revision", default=HF_BRANCH)
    verify = commands.add_parser("verify", help="Build and compare one or more benchmarks")
    verify.add_argument("benchmarks", nargs="*")
    verify.add_argument("--revision", default=HF_BRANCH)
    args = parser.parse_args()
    if args.command == "select":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        slugs = changed_benchmarks(ROOT, os.environ["GITHUB_EVENT_NAME"], event)
        values = {"matrix": json.dumps(build_matrix(slugs), separators=(",", ":")),
                  "has_benchmarks": str(bool(slugs)).lower()}
        print(json.dumps({"benchmarks": slugs, **values}, indent=2))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            for name, value in values.items():
                output.write(f"{name}={value}\n")
        return 0
    revision = resolve_revision(args.revision)
    print(f"HF reference: {HF_REPOSITORY}@{revision}", flush=True)
    if args.command == "resolve":
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"revision={revision}\n")
        return 0
    slugs = args.benchmarks or json.loads(os.environ.get("BENCHMARKS_JSON", "[]"))
    if not isinstance(slugs, list) or not slugs:
        parser.error("Specify at least one benchmark")
    failures = []
    for slug in slugs:
        print(f"\n=== {slug} ===", flush=True)
        try:
            verify_benchmark(ROOT, slug, revision)
        except Exception as exc:
            print(f"FAIL {slug}: {exc}", file=sys.stderr, flush=True)
            failures.append(slug)
    if failures:
        print(f"Failed benchmarks: {', '.join(failures)}", file=sys.stderr)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
