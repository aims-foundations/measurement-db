"""Small output-writing helpers; builders continue to read upstream formats."""
import argparse
import io
import json
from pathlib import Path
import tarfile


def arguments():
    parser = argparse.ArgumentParser(description="Preserve saved pilot results in the upstream input layout; no model calls")
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def prepare(benchmark, run, output):
    run = run.resolve(strict=True)
    output = (output or run / "native").resolve()
    if output.is_relative_to(benchmark / "raw") or output == run:
        raise ValueError("Export must not replace captured inputs or run records")
    saved = run / "procedure/reproduction_checks/luna.json"
    config = json.loads(saved.read_text())
    result = json.loads((run / "fresh_results.json").read_text())
    tasks = result["tasks"]
    if not tasks or any(t.get("status") != "graded" or not t.get("api_attempts")
                        or t.get("response") not in (0, 1) for t in tasks):
        raise ValueError("Export requires graded fresh attempts; controls/replays are not observations")
    output.mkdir(parents=True, exist_ok=False)
    return run, output, config, tasks


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def archive(path, members):
    with tarfile.open(path, "w:gz") as stream:
        for name, content in sorted(members.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(content), 0o644
            stream.addfile(entry, io.BytesIO(content))


def settings(output, config, **features):
    features.update({key: config[key] for key in (
        "reasoning_effort", "max_calls_per_task", "max_output_tokens",
        "task_timeout_seconds", "max_request_bytes") if key in config})
    (output / "subject_settings.json").write_bytes(encoded({config["model"]: features}))


def receipt(output, run):
    (output / "export.json").write_bytes(encoded({
        "derived_from_run": str(run), "format": "upstream result layout",
        "observations": "fresh attempts only; saved-output regrading is excluded"}))
    print(output)
