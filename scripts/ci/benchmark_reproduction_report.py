"""Checkpoint benchmark results and publish one report for the workflow run.

Only the standard library is needed, including when dependency installation fails.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import urllib.request


STATUSES = {
    "passed": "PASS", "failed": "FAIL", "interrupted": "INTERRUPTED",
    "not_run": "NOT RUN", "not_reported": "NOT REPORTED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_message(message: object) -> str:
    text = str(message)
    # GitHub masks logs, but does not redact the contents of artifact files.
    for name in ("HF_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        if value := os.environ.get(name):
            text = text.replace(value, "***")
    return text[:4000]


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def new_batch(slugs: list[str], revision: str = "") -> dict:
    if not slugs or len(set(slugs)) != len(slugs) or any(
        not re.fullmatch(r"[a-z][a-z0-9_]*", slug) for slug in slugs
    ):
        raise ValueError("Expected a nonempty list of distinct benchmark folder names")
    return {
        "schema_version": 1,
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        "head_sha": os.environ.get("GITHUB_SHA", ""),
        "hf_revision": revision,
        "runner": os.environ.get("RUNNER_NAME", ""),
        "job_name": "Reproduce " + ", ".join(slugs),
        "created_at": utc_now(),
        "results": [{"benchmark": slug, "status": "not_run", "duration_seconds": None,
                     "message": "Verification did not start."} for slug in slugs],
    }


def finalize_batch(data: dict, job_status: str) -> dict:
    data["job_status"] = job_status
    for row in data["results"]:
        if row["status"] == "running":
            row.update(status="interrupted", message="Verification started but did not record a result.")
        elif row["status"] == "not_run":
            row["message"] = safe_message(f"Verification did not start (job: {job_status}). " + row["message"])
    return data


def github_jobs() -> list[dict]:
    """Include prior attempts so successful jobs survive a rerun of failed jobs."""
    if not os.environ.get("GITHUB_TOKEN"):
        return []
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    repository, run_id = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
    jobs = []
    for page in range(1, 100):
        request = urllib.request.Request(
            f"{base}/repos/{repository}/actions/runs/{run_id}/jobs?filter=all&per_page=100&page={page}",
            headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                     "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)["jobs"]
        jobs.extend(batch)
        if len(batch) < 100:
            return jobs
    raise ValueError("Job metadata exceeded the pagination limit")


def collect(matrix: dict, directory: Path, revision: str, jobs: list[dict],
            changes_result: str = "success") -> dict:
    groups = [entry["benchmarks"] for entry in matrix["include"]]
    expected = [slug for group in groups for slug in group]
    if expected:
        new_batch(expected)  # Validate names and reject duplicate coverage.
    warnings = []
    documents = []
    for path in sorted(directory.glob("**/results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["schema_version"] != 1:
                raise ValueError("unsupported report schema")
            for key, variable in (("run_id", "GITHUB_RUN_ID"), ("head_sha", "GITHUB_SHA")):
                if os.environ.get(variable) and data[key] != os.environ[variable]:
                    raise ValueError(f"{key} does not match this run")
            if data["hf_revision"] != revision:
                raise ValueError("HF revision does not match this run")
            data["run_attempt"] = int(data["run_attempt"])
            if data["run_attempt"] > int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")):
                raise ValueError("report is from a later attempt")
            checked = new_batch([row["benchmark"] for row in data["results"]])
            if data["job_name"] != checked["job_name"] or not isinstance(data["runner"], str):
                raise ValueError("invalid job identity")
            if any(row["status"] not in {*STATUSES, "running"} for row in data["results"]):
                raise ValueError("unrecognized benchmark status")
            for row in data["results"]:
                row["message"] = safe_message(row.get("message", ""))
                seconds = row.get("duration_seconds")
                if seconds is not None and (not isinstance(seconds, (int, float)) or
                                            not math.isfinite(seconds) or seconds < 0):
                    raise ValueError("invalid duration")
            documents.append(data)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            warnings.append(f"Could not use {path.parent.name}/{path.name}: {safe_message(exc)}")

    latest_jobs = {}
    for job in jobs:
        if job["id"] > latest_jobs.get(job["name"], {}).get("id", -1):
            latest_jobs[job["name"]] = job
    rows = []
    for group in groups:
        name = "Reproduce " + ", ".join(group)
        job = latest_jobs.get(name, {})
        candidates = [data for data in documents if data["job_name"] == name and
                      [row["benchmark"] for row in data["results"]] == group]
        latest = max(candidates, key=lambda data: data["run_attempt"], default=None)
        # Never use an older success to hide a newer job whose artifact is missing.
        if latest and int(job.get("run_attempt", 1)) > latest["run_attempt"]:
            latest = None
        if latest:
            result_rows = finalize_batch(latest, job.get("conclusion") or latest.get("job_status", "unknown"))["results"]
        else:
            status = "not_run" if changes_result != "success" else "not_reported"
            message = (f"Selection/reference job: {changes_result}; verification did not start."
                       if status == "not_run" else
                       f"No result artifact was received (job: {job.get('conclusion') or 'unknown'}).")
            result_rows = [{"benchmark": slug, "status": status, "duration_seconds": None,
                            "message": message} for slug in group]
        for result in result_rows:
            row = dict(result)
            row.update(job_url=job.get("html_url", ""), runner=latest["runner"] if latest else job.get("runner_name", ""),
                       run_attempt=latest["run_attempt"] if latest else job.get("run_attempt"), hf_revision=revision)
            rows.append(row)
    counts = Counter(row["status"] for row in rows)
    if changes_result != "success":
        warnings.append(f"Benchmark selection/reference job ended with: {changes_result}.")
    return {"schema_version": 1, "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
            "head_sha": os.environ.get("GITHUB_SHA", ""), "hf_revision": revision,
            "generated_at": utc_now(), "counts": {status: counts[status] for status in STATUSES},
            "warnings": warnings, "results": sorted(rows, key=lambda row: row["benchmark"])}


def cell(value: object) -> str:
    text = html.escape(str(value))
    for character in ("\\", "`", "[", "]", "*", "_"):
        text = text.replace(character, "\\" + character)
    return text.replace("|", "&#124;").replace("\n", "<br>").replace("\r", "")


def markdown(data: dict) -> str:
    rows = data["results"]
    counts = Counter(row["status"] for row in rows)
    lines = ["# Benchmark reproduction report", "",
             f"**{len(rows)} benchmarks:** " + " · ".join(
                 f"{counts[status]} {label}" for status, label in STATUSES.items()), ""]
    if data.get("hf_revision"):
        lines.extend([f"HF reference: `{data['hf_revision']}`", ""])
    if data.get("head_sha"):
        lines.extend([f"Source commit: `{data['head_sha']}`", ""])
    for warning in data.get("warnings", []):
        lines.extend([f"**Report note:** {cell(warning)}", ""])
    if not rows:
        lines.append("No benchmarks were selected." if not data.get("warnings") else
                     "No benchmark results are available; see the report notes above.")
    else:
        lines.extend(["| Benchmark | Result | Duration | Details | Job log |",
                      "| --- | --- | ---: | --- | --- |"])
        for row in rows:
            seconds = row.get("duration_seconds")
            duration = f"{seconds:.1f}s" if seconds is not None else "—"
            message = row.get("message", "")
            detail = cell(message[:320] + ("…" if len(message) > 320 else ""))
            link = f"[Logs]({row['job_url']})" if row.get("job_url") else "—"
            lines.append(f"| `{row['benchmark']}` | {STATUSES[row['status']]} | {duration} | {detail} | {link} |")
        lines.extend(["", "NOT RUN means verification did not start. NOT REPORTED means no usable result artifact",
                      "was received; it does not imply that the benchmark failed. Full details are in the JSON/CSV artifact."])
    return "\n".join(lines) + "\n"


def publish_summary(text: str) -> None:
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as output:
            output.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--report", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--report", type=Path, required=True)
    finalize.add_argument("--job-status", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--reports-dir", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        save(args.report, new_batch(json.loads(os.environ["BENCHMARKS_JSON"]), os.environ.get("HF_REVISION", "")))
        return 0
    if args.command == "finalize":
        data = finalize_batch(json.loads(args.report.read_text(encoding="utf-8")), args.job_status)
        save(args.report, data)
        publish_summary(markdown(data))
        return 0
    warnings = []
    try:
        jobs = github_jobs()
    except Exception as exc:
        jobs = []
        warnings.append("Job log links unavailable: " + safe_message(exc))
    matrix = json.loads(os.environ.get("MATRIX_JSON") or '{"include": []}')
    data = collect(matrix, args.reports_dir, os.environ.get("HF_REVISION", ""), jobs,
                   os.environ.get("CHANGES_RESULT", "success"))
    data["warnings"].extend(warnings)
    if os.environ.get("DOWNLOAD_OUTCOME", "success") not in ("success", "skipped"):
        data["warnings"].append("Artifact download did not complete; some results may be unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(args.output_dir / "results.json", data)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["benchmark", "status", "duration_seconds", "message", "job_url", "runner", "run_attempt", "hf_revision"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["results"])
    summary = markdown(data)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    publish_summary(summary)
    print(summary)
    return int(any(row["status"] != "passed" for row in data["results"]) or
               os.environ.get("CHANGES_RESULT", "success") != "success" or
               os.environ.get("DOWNLOAD_OUTCOME", "success") not in ("success", "skipped"))


if __name__ == "__main__":
    raise SystemExit(main())
