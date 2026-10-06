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
import time

if __package__:
    from . import benchmark_reproduction_report as reports
    from .benchmark_release import load_release
else:
    import benchmark_reproduction_report as reports
    from benchmark_release import load_release


ROOT = Path(__file__).resolve().parents[2]
HF_REPOSITORY = "aims-foundations/measurement-db"
HF_BRANCH = "main"
ZERO_SHA = "0" * 40


class MissingReferenceError(ValueError):
    """The upstream build succeeded, but published comparison tables are absent."""


class WithheldBenchmark(ValueError):
    """Publication policy excludes this benchmark from reproduction CI."""


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
        selection = event.get("inputs", {}).get("benchmark", "").strip()
        if not selection:
            return all_benchmarks(root)
        return sorted({validate_slug(slug.strip()) for slug in selection.split(",")})
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


def python_versions(root: Path, slugs: list[str]) -> dict[str, str]:
    """Honor benchmark-specific interpreter pins without importing builders."""
    import yaml

    versions = {}
    for slug in slugs:
        path = root / "benchmarks" / slug / "metadata.yaml"
        metadata = yaml.safe_load(path.read_text())
        runtime = metadata.get("build", {}).get("parameters", {}).get("runtime", {})
        version = runtime.get("python_version", "3.11")
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
            raise ValueError(f"{path}: runtime.python_version must be a quoted Python version")
        versions[slug] = version
    return versions


def build_matrix(slugs: list[str], versions: dict[str, str] | None = None) -> dict:
    # GitHub permits at most 256 matrix jobs. Usually there is one builder per
    # job; large migrations batch only benchmarks using the same interpreter.
    groups: dict[str, list[str]] = {}
    for slug in slugs:
        groups.setdefault((versions or {}).get(slug, "3.11"), []).append(slug)
    if len(groups) > 256:
        raise ValueError("Too many Python versions for GitHub's 256-job matrix limit")
    size = max(1, math.ceil(len(slugs) / 256))
    while sum(math.ceil(len(group) / size) for group in groups.values()) > 256:
        size += 1
    return {"include": [{"benchmarks": group[i:i + size], "python_version": version}
                        for version, group in groups.items()
                        for i in range(0, len(group), size)]}


def select_releases(root: Path, slugs: list[str]) -> tuple[list[str], dict[str, str]]:
    public, withheld = [], {}
    for slug in slugs:
        validate_slug(slug)
        directory = root / "benchmarks" / slug
        if not (directory / "build.py").is_file():
            raise ValueError(f"Missing builder: benchmarks/{slug}/build.py")
        release, reason = load_release(directory / "metadata.yaml")
        if release == "withheld":
            withheld[slug] = reason
        else:
            public.append(slug)
    return public, withheld


def resolve_revision(revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
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
        raise MissingReferenceError(f"No published Parquet tables for {slug} at {revision}")
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


def verify_benchmark(root: Path, slug: str, revision: str, *, progress=None) -> None:
    """Build from upstream first, then fetch references and compare exact bytes."""
    progress = progress or (lambda phase, revision=None: None)
    validate_slug(slug)
    if not (root / "benchmarks" / slug / "build.py").is_file():
        raise ValueError(f"Missing builder: benchmarks/{slug}/build.py")
    release, reason = load_release(root / "benchmarks" / slug / "metadata.yaml")
    if release == "withheld":
        raise WithheldBenchmark(reason)
    progress("build")
    with tempfile.TemporaryDirectory(prefix=f"measurement-db-{slug}-") as temporary:
        scratch = Path(temporary)
        workspace = scratch / "measurement_db"  # Builders import this package name.
        copy_build_source(root, workspace)
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
        generated = workspace / "benchmarks" / slug / "formatted_tables"
        if not any(generated.rglob("*.parquet")):
            raise ValueError(f"Builder produced no Parquet tables for {slug}")
        print(f"BUILD PASS {slug}: upstream build completed.", flush=True)
        progress("reference")
        if not revision:
            raise ValueError("HF reference could not be pinned; see the workflow's reference resolution step.")
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        revision = resolve_revision(revision)
        progress("reference", revision)
        print(f"Comparing {slug} against {HF_REPOSITORY}@{revision}", flush=True)
        try:
            files = reference_files(HfApi(), slug, revision)
            expected = {
                name: Path(hf_hub_download(HF_REPOSITORY, remote, repo_type="dataset",
                                          revision=revision, local_dir=scratch / "reference"))
                for name, remote in sorted(files.items())
            }
        except RemoteEntryNotFoundError as exc:
            raise MissingReferenceError(f"Published reference missing for {slug} at {revision}: {exc}") from exc
        progress("comparison")
        compare_tables(generated, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select", help="Emit the changed-benchmark matrix for a GitHub event")
    resolve = commands.add_parser("resolve", help="Resolve the HF reference once for all jobs")
    resolve.add_argument("--revision", default=HF_BRANCH)
    verify = commands.add_parser("verify", help="Build and compare one or more benchmarks")
    verify.add_argument("benchmarks", nargs="*")
    verify.add_argument("--revision", default=HF_BRANCH)
    verify.add_argument("--report", type=Path, help="Checkpoint individual benchmark outcomes as JSON")
    args = parser.parse_args()
    if args.command == "select":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        slugs = changed_benchmarks(ROOT, os.environ["GITHUB_EVENT_NAME"], event)
        public, withheld = select_releases(ROOT, slugs)
        values = {"matrix": json.dumps(build_matrix(public, python_versions(ROOT, public)), separators=(",", ":")),
                  "withheld": json.dumps(withheld, separators=(",", ":")),
                  "has_benchmarks": str(bool(public)).lower()}
        print(json.dumps({"benchmarks": slugs, **values}, indent=2))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            for name, value in values.items():
                output.write(f"{name}={value}\n")
        return 0
    if args.command == "resolve":
        revision = resolve_revision(args.revision)
        print(f"HF reference: {HF_REPOSITORY}@{revision}", flush=True)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"revision={revision}\n")
        return 0
    slugs = args.benchmarks or json.loads(os.environ.get("BENCHMARKS_JSON", "[]"))
    if not isinstance(slugs, list) or not slugs:
        parser.error("Specify at least one benchmark")
    report = reports.new_batch(slugs, args.revision)
    if args.report:
        reports.save(args.report, report)
    failures = []
    for slug, row in zip(slugs, report["results"]):
        print(f"\n=== {slug} ===", flush=True)
        started = time.monotonic()
        row.update(status="running", phase="metadata", started_at=reports.utc_now(),
                   message="Checking benchmark release metadata.")
        if args.report:
            reports.save(args.report, report)

        def progress(phase, revision=None):
            row.update(phase=phase, build_status="running" if phase == "build" else "passed",
                       comparison_status="not_run" if phase == "build" else "running")
            row["message"] = {
                "build": "Building from declared upstream sources.",
                "reference": "Upstream build passed; preparing comparison references.",
                "comparison": "Upstream build passed; comparing every table byte for byte.",
            }[phase]
            if revision is not None:
                report["hf_revision"] = revision
            if args.report:
                reports.save(args.report, report)

        try:
            verify_benchmark(ROOT, slug, report["hf_revision"], progress=progress)
        except WithheldBenchmark as exc:
            row.update(reports.withheld_result(slug, str(exc)))
            print(f"SKIPPED (WITHHELD) {slug}: {exc}", flush=True)
        except Exception as exc:
            if row["phase"] == "metadata":
                status = "failed"
            elif row["phase"] == "build":
                status = "build_failed"
                row["build_status"] = "failed"
            elif row["phase"] == "reference":
                status = "reference_missing" if isinstance(exc, MissingReferenceError) else "reference_failed"
                row["comparison_status"] = status
            else:
                status = "comparison_failed"
                row["comparison_status"] = status
            prefix = "Upstream build passed. " if row["build_status"] == "passed" else ""
            message = reports.safe_message(f"{prefix}{type(exc).__name__}: {exc}")
            print(f"{reports.STATUSES[status]} {slug}: {message}", file=sys.stderr, flush=True)
            row.update(status=status, message=message)
            failures.append(slug)
        except (KeyboardInterrupt, SystemExit):
            row.update(status="interrupted", message="Verification was interrupted before completion.")
            for field in ("build_status", "comparison_status"):
                if row[field] == "running":
                    row[field] = "interrupted"
            raise
        else:
            row.update(status="passed", build_status="passed", comparison_status="passed",
                       message="Upstream build passed; every published table matched byte for byte.")
        finally:
            row.update(duration_seconds=round(time.monotonic() - started, 3), finished_at=reports.utc_now())
            if args.report:
                reports.save(args.report, report)
    if failures:
        print(f"Failed benchmarks: {', '.join(failures)}", file=sys.stderr)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
