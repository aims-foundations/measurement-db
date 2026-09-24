"""Independent source-to-table checks for the third private migration batch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import ast
import csv
import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter

import pandas as pd
import yaml

from measurement_db.scripts.curate_benchmarks.appworld_bundle import open_bundle


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None


def _features(text):
    return dict(
        part.split("=", 1) for part in str(text or "").split(";") if "=" in part
    )


def _tables(directory, tables_directory=None):
    root = Path(tables_directory) if tables_directory is not None else directory / "formatted_tables"
    return {p.stem: pd.read_parquet(p) for p in root.glob("*.parquet")}


def _check(actual, expected, description):
    if actual != expected:
        raise ValueError(
            f"{description}: observations differ from captured provider evidence"
        )


def _actual(tables, key):
    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = {
        r.response_id: _digest(r.trace)
        for r in tables.get("traces", pd.DataFrame()).itertuples()
    }
    return Counter(
        (
            *key(subjects[r.subject_id], items[r.item_id], r),
            None if pd.isna(r.response) else r.response,
            traces.get(r.response_id),
        )
        for r in tables["responses"].itertuples()
    )


def _appworld(directory, tables):
    expected = Counter()
    items = tables["items"].set_index("raw_item_id")
    with open_bundle(directory / "raw/data-0.1.0.bundle") as tasks:
        specs = {
            name.split("/")[-2]: json.loads(tasks.read(name))
            for name in tasks.namelist()
            if name.startswith("data/tasks/") and name.endswith("/specs.json")
        }
        for task, item in items.iterrows():
            _check(item.content, specs[task]["instruction"], "AppWorld instruction")
            _check(
                json.loads(item.verifier)["spec"],
                tasks.read(f"data/tasks/{task}/ground_truth/evaluation.py").decode(),
                "AppWorld evaluator",
            )
            criterion = json.loads(json.loads(item.grading_criterion)["rule"])
            _check(
                criterion["test_data"],
                json.loads(
                    tasks.read(f"data/tasks/{task}/ground_truth/test_data.json")
                ),
                "AppWorld test data",
            )
    for path in sorted((directory / "raw/bundles").glob("*.bundle")):
        with open_bundle(path) as bundle:
            name = path.stem
            metadata = json.loads(bundle.read(f"{name}/metadata.json"))
            model = metadata["llm"].get("tooltip") or metadata["llm"]["name"]
            method = metadata["method"]["name"]
            split = metadata["dataset"]
            outcomes = json.loads(bundle.read(f"{name}/evaluations/{split}.json"))[
                "individual"
            ]
            for task, result in outcomes.items():
                if not isinstance(result["success"], bool):
                    raise TypeError("Nonboolean AppWorld result")
                trace_path = f"{name}/tasks/{task}/logs/environment_io.md"
                trace = (
                    bundle.read(trace_path).decode("utf-8", "replace")
                    if trace_path in bundle.namelist()
                    else None
                )
                if trace is not None and not trace.strip():
                    trace = None
                expected[
                    model, method, task, float(result["success"]), _digest(trace)
                ] += 1
    actual = _actual(
        tables, lambda s, i, r: (s["display_name"], s["harness"], i["raw_item_id"])
    )
    _check(actual, expected, "AppWorld source observations and traces")
    return {
        "source_responses": sum(expected.values()),
        "source_successes": int(sum(k[-2] * n for k, n in expected.items())),
        "source_items": len({k[2] for k in expected}),
        "source_configurations": len({k[:2] for k in expected}),
        "source_traces": sum(n for k, n in expected.items() if k[-1] is not None),
    }


_ALGOTUNE_TRANSCRIPTS = {
    "claude-opus-4-1-20250805": "anthropic_claude-opus-4-1-20250805",
    "claude-opus-4-20250514": "claude-opus-4-20250514",
    "claude-opus-4.5": "openrouter_anthropic_claude-opus-4.5",
    "claude-opus-4.6": "openrouter_anthropic_claude-opus-4.6",
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
    "deepseek-reasoner": "deepseek_deepseek-reasoner",
    "gemini-2.5-pro": "gemini_gemini-2.5-pro",
    "gemini-3-pro-preview": "openrouter_google_gemini-3-pro-preview",
    "gemini-3.1-pro-preview": "openrouter_google_gemini-3.1-pro-preview",
    "glm-4.5": "openrouter_z-ai_glm-4.5",
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-pro (medium)": "gpt-5-pro",
    "gpt-5.2": "openrouter_openai_gpt-5.2",
    "gpt-5.4": "openrouter_openai_gpt-5.4",
    "gpt-oss-120b": "gpt-oss-120b",
    "o4-mini": "o4-mini",
    "qwen3-coder": "openrouter_qwen_qwen3-coder",
}


def _algotune(directory, tables):
    raw = directory / "raw"
    summary = json.loads((raw / "agent_summary.json").read_text())
    expected = Counter()
    for task, models in summary.items():
        for model, result in models.items():
            speedup = result["final_speedup"] if isinstance(result, dict) else result
            if speedup == "N/A" or speedup is None:
                grade = 0.0
            else:
                grade = float(float(speedup) >= 1.0)
            path = raw / "transcripts" / f"{task}__{_ALGOTUNE_TRANSCRIPTS[model]}.txt"
            trace = (
                (path.read_text(errors="replace").strip() or None)
                if path.exists()
                else None
            )
            expected[model, task, grade, _digest(trace)] += 1
    for item in tables["items"].itertuples():
        task = item.raw_item_id
        _check(
            item.content,
            (raw / "descriptions" / f"{task}.txt").read_text(errors="replace").strip(),
            "AlgoTune task description",
        )
        _check(
            json.loads(item.verifier)["spec"],
            (raw / "task_code" / f"{task}.py").read_text(errors="replace").strip(),
            "AlgoTune verifier",
        )
    _check(
        _actual(tables, lambda s, i, r: (s["display_name"], i["raw_item_id"])),
        expected,
        "AlgoTune source outcomes and transcripts",
    )
    return {
        "source_responses": sum(expected.values()),
        "source_successes": int(sum(k[-2] * n for k, n in expected.items())),
        "source_items": len(summary),
        "source_traces": sum(n for k, n in expected.items() if k[-1] is not None),
    }


def _agentdojo(directory, tables):
    expected = Counter()
    metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
    release = next(source for source in metadata["sources"]["upstream"] if source.get("name") == "release")
    with tarfile.open(directory / "raw" / release["file"]) as archive:
        for member in archive:
            if (
                not member.isfile()
                or not member.name.split("/", 1)[1].startswith("runs/")
                or not member.name.endswith(".json")
            ):
                continue
            parts = Path(member.name).parts
            record = json.load(archive.extractfile(member))
            model, suite, task, attack = parts[2:6]
            suite, task = (
                record.get("suite_name", suite),
                record.get("user_task_id", task),
            )
            attack = record.get("attack_type") or (None if attack == "none" else attack)
            trace = (
                json.dumps(record["messages"], ensure_ascii=False)
                if record.get("messages")
                else None
            )
            for metric in ["utility", "security"]:
                if metric == "security" and not attack:
                    continue
                outcome = record.get(metric)
                if outcome is not None and not isinstance(outcome, bool):
                    raise TypeError("Unexpected AgentDojo source grade")
                task_id = f"{suite}::{task}"
                if metric == "security":
                    task_id += "::" + (record.get("injection_task_id") or "none")
                expected[
                    model,
                    task_id,
                    metric,
                    f"attacker={attack}" if attack else None,
                    None if outcome is None else float(outcome),
                    _digest(trace),
                ] += 1

    def key(subject, item, response):
        features = _features(subject["subject_features_extra"])
        model = features["source_model_variant"]
        if features.get("defense"):
            model += "-" + features["defense"]
        metric = json.loads(json.loads(item["verifier"])["spec"])["field"]
        scale = json.loads(item["grading_criterion"])["response_scale"]
        _check(
            scale["direction"],
            "higher_is_better" if metric == "utility" else "lower_is_better",
            "AgentDojo score direction",
        )
        return (
            model,
            item["raw_item_id"],
            metric,
            None if pd.isna(response.interactors) else response.interactors,
        )

    _check(_actual(tables, key), expected, "AgentDojo outcomes and trace associations")
    return {
        "source_responses": sum(expected.values()),
        "source_successes": int(sum(k[-2] * n for k, n in expected.items() if k[-2] is not None)),
        "source_ungraded_observations": sum(n for k, n in expected.items() if k[-2] is None),
        "source_traces": sum(n for k, n in expected.items() if k[-1] is not None),
        "source_utility_observations": sum(
            n for k, n in expected.items() if k[2] == "utility"
        ),
        "source_attacker_success_observations": sum(
            n for k, n in expected.items() if k[2] == "security"
        ),
    }


def _workbench(directory, tables):
    csv.field_size_limit(sys.maxsize)  # Released full-response fields exceed the CSV default.
    raw = directory / "raw/WorkBench"
    report = json.loads((raw / "retro/data/model_results.json").read_text())
    _check(
        {entry["ground_truth_version"] for entry in report["models"].values()},
        {"v2"},
        "WorkBench captured ground-truth version",
    )
    bank = {}
    for path in (raw / "data/processed/tasks_and_outcomes").glob("*.csv"):
        tool = path.name.removesuffix("_tasks_and_outcomes.csv")
        for row in pd.read_csv(path, dtype=str).itertuples():
            bank[tool, row.task] = str(ast.literal_eval(row.outcome))
    for item in tables["items"].itertuples():
        tool, task = item.raw_item_id.split("::", 1)
        _check(item.content, task, "WorkBench source task text")
        _check(
            json.loads(item.grading_criterion)["reference_answer"],
            bank[tool, task],
            "WorkBench reference actions",
        )
        verifier = json.loads(json.loads(item.verifier)["spec"])
        _check(verifier["kind"], "deterministic_sandbox", "WorkBench verifier kind")
        _check(
            verifier["revision"],
            "da6f8ee8d9f8efc87b2f5f3cc9edc1befdac726e",
            "WorkBench scorer revision",
        )
        _check(
            verifier["entrypoint"],
            "src.evals.metrics.compute_metrics",
            "WorkBench scorer entrypoint",
        )
    # Independently join CSV records using native run paths and task text, not
    # the builder's DataFrame operations or its output identifiers.
    with gzip.open(directory / "raw/item_level_results.csv.gz", "rt", newline="") as handle:
        verdict_rows = [row for row in csv.DictReader(handle) if row["run_group"] == "revisited_2026"]
    verdicts = {(row["results_file"], row["task"]): row for row in verdict_rows}
    _check(len(verdicts), len(verdict_rows), "WorkBench unique native verdicts")
    expected = Counter()
    seen = set()
    for label, entry in report["models"].items():
        # Presentation aliases do not change the native model/run mapping.
        model = "Claude " + label if label.startswith(("Haiku ", "Sonnet ", "Opus ", "Fable ")) else label
        model = model.replace("Mistral-Small-4", "Mistral Small 4").replace("Mistral-Medium-3.5", "Mistral Medium 3.5")
        successes = 0
        for tool, path in entry["sources"].items():
            opener = gzip.open if path.endswith(".gz") else open
            with opener(raw / path, "rt", newline="") as handle:
                source = list(csv.DictReader(handle))
            group_successes = 0
            for row in source:
                key = path, row["task"]
                if key in seen or key not in verdicts:
                    raise ValueError("Duplicate WorkBench attempt or absent released verdict")
                seen.add(key)
                verdict = verdicts[key]
                _check((verdict["model"], verdict["domain"], verdict["ground_truth_version"]),
                       (entry["model_name"], tool, "v2"), "WorkBench native run identity")
                if verdict["correct"] not in ("True", "False"):
                    raise ValueError("WorkBench correctness is not a released boolean")
                _check(verdict["error"], row["error"], "WorkBench native error")
                _check(ast.literal_eval(verdict["predicted_actions"]),
                       [action.replace("\n", "\\n") for action in ast.literal_eval(row["function_calls"])],
                       "WorkBench native predicted actions")
                _check(verdict["ground_truth_actions"], bank[tool, row["task"]], "WorkBench native reference actions")
                grade = float(verdict["correct"] == "True")
                expected[model, f"{tool}::{row['task']}", grade, _digest(row["full_response"] or None)] += 1
                group_successes += int(grade)
            _check(len(source), entry["per_tool"][tool]["total"], "WorkBench per-tool coverage")
            _check(group_successes, entry["per_tool"][tool]["correct"], "WorkBench per-tool correctness")
            successes += group_successes
        _check(successes, entry["correct"], "WorkBench published model accuracy")
    _check(seen, set(verdicts), "WorkBench complete exported run selection")
    _check(_actual(tables, lambda subject, item, response: (subject["display_name"], item["raw_item_id"])),
           expected, "WorkBench per-attempt outcomes and complete transcripts")
    return {
        "source_responses": sum(expected.values()),
        "source_successes": sum(e["correct"] for e in report["models"].values()),
        "source_configurations": len(report["models"]),
        "source_traces": sum(n for key, n in expected.items() if key[-1] is not None),
    }


def _swe_live(directory, tables):
    raw = directory / "raw"
    expected = Counter()
    language_directories = {
        "cs",
        "java",
        "js_ts",
        "js",
        "ts",
        "go",
        "rust",
        "c",
        "cpp",
        "csharp",
        "tsjs",
        "all_languages",
    }
    result_files = list((raw / "submissions").rglob("results.json"))
    for path in result_files:
        parts = path.parent.relative_to(raw / "submissions").parts
        configuration = "/".join(
            re.sub(r"^\d{8}-", "", part)
            for part in parts[1:]
            if part not in language_directories
        )
        track = {"lite": "python", "multilang": "multilang", "windows": "windows"}[
            parts[0]
        ]
        report = json.loads(path.read_text())
        known = set()
        for key, values in report.items():
            if key.endswith("_ids") and isinstance(values, list):
                known.update(values)
        positive = set(report.get("resolved_ids", report.get("success_ids", [])))
        negative = set(report.get("unresolved_ids", report.get("failure_ids", [])))
        if positive & negative:
            raise ValueError(
                "Conflicting terminal outcomes in captured SWE-bench-Live results"
            )
        # Nonterminal error/empty flags can remain after a later graded result.
        # Only explicit final success/failure determines a non-null grade.
        grades = {iid: None for iid in known}
        grades.update(dict.fromkeys(negative, 0.0))
        grades.update(dict.fromkeys(positive, 1.0))
        predictions = {}
        pred_path = path.parent / "preds.json"
        if pred_path.exists():
            data = json.loads(pred_path.read_text())
            if isinstance(data, dict):
                for iid, value in data.items():
                    if isinstance(value, dict) and value.get("model_patch"):
                        predictions[iid] = value["model_patch"]
            else:
                for value in data:
                    if value.get("model_patch"):
                        predictions[value["instance_id"]] = value["model_patch"]
        for iid, grade in grades.items():
            expected[
                configuration,
                f"{track}/{iid}",
                report.get("related_upstream_evaluator_fix"),
                grade,
                _digest(predictions.get(iid)),
            ] += 1
    banks = {}
    historical_items = set()
    for item in tables["items"].itertuples():
        features = _features(item.item_features)
        filename = features["task_definition_file"]
        if filename not in banks:
            banks[filename] = pd.read_parquet(raw / filename).set_index("instance_id")
        record = banks[filename].loc[features["source_instance_id"]]
        _check(item.content, record.problem_statement, "SWE-bench-Live issue text")
        criterion = json.loads(item.grading_criterion)
        patch = record.get("patch")
        if isinstance(patch, str) and patch:
            _check(
                criterion["reference_answer"], patch, "SWE-bench-Live reference patch"
            )
        checks = json.loads(criterion["rule"])["checks"]
        _check(checks["repo"], record.repo, "SWE-bench-Live task repository")
        _check(
            checks["base_commit"],
            record.base_commit,
            "SWE-bench-Live repository revision",
        )
        if isinstance(record.get("test_patch"), str):
            _check(
                checks["test_patch"], record.test_patch, "SWE-bench-Live grading tests"
            )
        if "_history/" in filename:
            historical_items.add(item.raw_item_id)

    def key(subject, item, response):
        configuration = _features(subject["subject_features_extra"])[
            "submission_configuration"
        ]
        verifier = json.loads(json.loads(item["verifier"])["spec"])
        return (
            configuration,
            item["raw_item_id"],
            verifier["historical_evaluator_revision"],
        )

    _check(
        _actual(tables, key),
        expected,
        "SWE-bench-Live released attempts and patch associations",
    )
    return {
        "source_responses": sum(expected.values()),
        "source_successes": sum(n for k, n in expected.items() if k[-2] == 1),
        "source_ungraded_attempts": sum(
            n for k, n in expected.items() if k[-2] is None
        ),
        "source_traces": sum(n for k, n in expected.items() if k[-1] is not None),
        "source_result_files": len(result_files),
        "historical_task_definitions": len(historical_items),
    }


def verify_batch3(directory, tables_directory=None):
    directory = Path(directory)
    functions = {
        'appworld': _appworld,
        'algotune': _algotune,
        'agentdojo': _agentdojo,
        'workbench_revisited': _workbench,
        'swebench_live': _swe_live,
    }
    return functions[directory.name](directory, _tables(directory, tables_directory))
