"""Independent source reconciliation for the September 2026 second migration batch.

Read captured provider files directly; never import a benchmark builder or derive
source expectations from curated outputs. Generated tables are compared against
these source observations after source selection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import ast
import gzip
import math
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict

import pandas as pd
import yaml


def _json(path):
    return json.loads(path.read_text())


def _features(value):
    return dict(
        part.split("=", 1) for part in str(value or "").split(";") if "=" in part
    )


def _cells(directory):
    return (
        pd.read_parquet(directory / "responses.parquet")
        .merge(
            pd.read_parquet(directory / "subjects.parquet"),
            on="subject_id",
            validate="many_to_one",
        )
        .merge(
            pd.read_parquet(directory / "items.parquet"),
            on="item_id",
            validate="many_to_one",
            suffixes=("", "_item"),
        )
    )


def _equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(
            f"{label}: {sum((expected - actual).values())} missing and "
            f"{sum((actual - expected).values())} extra observations"
        )


def verify_batch2(directory, tables_directory=None):
    directory = Path(directory)
    raw, slug = directory / "raw", directory.name
    tables_directory = Path(tables_directory) if tables_directory is not None else directory / "formatted_tables"
    cells = _cells(tables_directory)
    items = pd.read_parquet(tables_directory / "items.parquet")
    if slug == "hcast":
        instructions = {}
        for p in raw.glob("task_info_*.json"):
            blob = _json(p)
            family = blob["org.metr.inspect-task-bridge/task-family/name"]
            for task, content in blob[
                "org.metr.inspect-task-bridge/task-family/setup-data"
            ]["instructions"].items():
                instructions[f"{family}/{task}"] = content.strip()
        expected = Counter()
        seen = set()
        for p in sorted(raw.glob("runs_*.jsonl")):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if (
                    r.get("task_source") != "HCAST"
                    or r["task_id"] not in instructions
                    or not r.get("alias")
                    or r["alias"] == "human"
                    or "fake" in str(r.get("run_id", "")).lower()
                    or r.get("score_binarized") is None
                ):
                    continue
                assert r["run_id"] not in seen, (
                    "A run was republished; it must not be counted twice"
                )
                seen.add(r["run_id"])
                harness = str(r.get("scaffold")).removeprefix("mtb/start_metr_task,")
                harness = {
                    "modular-public": "modular",
                    "metr_agents/react": "react",
                    "triframe_inspect/triframe_agent": "triframe",
                    "None": "unspecified",
                }.get(harness, harness)
                alias = r["alias"].replace(" (Inspect)", "").strip()
                expected[alias, harness, r["task_id"], float(r["score_binarized"])] += 1
        actual = Counter(
            (
                _features(r.subject_features_extra)["model_variant"],
                r.harness,
                r.raw_item_id,
                r.response,
            )
            for r in cells.itertuples()
        )
        _equal(actual, expected, slug)
        for r in items.itertuples():
            assert instructions[r.raw_item_id] in r.content
        return {
            "source_responses": len(seen),
            "source_items": len({k[2] for k in expected}),
            "source_successes": sum(k[-1] * v for k, v in expected.items()),
        }
    if slug == "editbench":
        expected = Counter()
        released_traces = {}
        model_apis = {}
        prompt_template = None
        with tarfile.open(next(raw.glob("editbench-*.tar.gz")), "r:gz") as archive:
            for member in archive:
                path = Path(*Path(member.name).parts[1:])
                if not member.isfile():
                    continue
                if path.as_posix() == "prompts/whole_file.txt":
                    prompt_template = archive.extractfile(member).read().decode().replace("\r\n", "\n").replace("\r", "\n")
                if path.as_posix() in {"examples/openai_experiment.py", "examples/openrouter_experiment.py"}:
                    for node in ast.parse(archive.extractfile(member).read()).body:
                        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                            continue
                        if node.targets[0].id in {"GPT_MAP", "OPENROUTER_NAME_MAP"}:
                            api = "OpenAI API" if node.targets[0].id == "GPT_MAP" else "OpenRouter"
                            for model in ast.literal_eval(node.value).values():
                                model_apis[model] = api
                if (
                    path.parent.as_posix() == "results/whole_file"
                    and path.suffix == ".json"
                ):
                    data = json.load(archive.extractfile(member))
                    for key, value in data.items():
                        if key.startswith("question_"):
                            expected[
                                path.stem, str(int(key[9:])), float(float(value) == 1)
                            ] += 1
                if (
                    path.parent.as_posix() == "generations/whole_file/gpt-o3-mini"
                    and path.name.isdigit()
                ):
                    released_traces[path.stem] = (
                        archive.extractfile(member).read().decode("utf-8")
                    )
        actual = Counter(
            (r.display_name, r.raw_item_id, r.response) for r in cells.itertuples()
        )
        _equal(actual, expected, slug)
        subjects = pd.read_parquet(tables_directory / "subjects.parquet")
        for subject in subjects.itertuples():
            model = subject.display_name.removesuffix("-high")
            if _features(subject.subject_features_extra).get("inference_api") != model_apis[model]:
                raise ValueError(f"EDIT-Bench API attribution differs from the released adapter: {model}")
        core = {
            str(json.loads(line)["problem_id"])
            for line in next(raw.glob("core-*.jsonl")).read_text().splitlines()
        }
        max_error = 0.0
        for subset in ("complete", "core"):
            reported = (
                pd.read_csv(next(raw.glob(f"leaderboard-{subset}-*.csv")))
                .set_index("Model")["Pass Rate (All)"]
                .sort_index()
            )
            scoped = cells[cells.raw_item_id.isin(core)] if subset == "core" else cells
            actual = scoped.groupby("display_name").response.mean().sort_index()
            assert list(actual.index) == list(reported.index)
            error = float((actual - reported).abs().max())
            assert error <= 0.00005000001
            max_error = max(max_error, error)
        source_items = {
            str(r["problem_id"]): r
            for r in map(
                json.loads, next(raw.glob("complete-*.jsonl")).read_text().splitlines()
            )
        }
        assert set(items.raw_item_id) == set(source_items)
        for r in items.itertuples():
            source = source_items[r.raw_item_id]
            assert r.content == prompt_template.format(
                lang=source["programming_language"], original_code=source["original_code"],
                instruction=source["instruction"], highlighted_code=source["highlighted_code"],
            )
            payload = json.loads(json.loads(r.verifier)["spec"])
            for field in ("problem_id", "pair_id", "programming_language", "python_version",
                          "original_code", "requirements", "test_code", "test_harness"):
                assert payload[field] == source.get(field), f"EDIT-Bench changed source field {field}"
            if source["programming_language"] == "python":
                assert payload["execution"]["commands"][0][-1] == source["python_version"]
        trace_cells = cells.merge(
            pd.read_parquet(tables_directory / "traces.parquet"), on="response_id"
        )
        expected_traces = Counter(
            ("gpt-o3-mini", key, value) for key, value in released_traces.items()
        )
        _equal(
            Counter(
                (r.display_name, r.raw_item_id, r.trace)
                for r in trace_cells.itertuples()
            ),
            expected_traces,
            "EDIT traces",
        )
        return {
            "source_responses": sum(expected.values()),
            "source_items": len(source_items),
            "source_successes": sum(k[-1] * v for k, v in expected.items()),
            "source_api_attributions": len(subjects),
            "leaderboard_max_absolute_error": max_error,
        }
    if slug == "corebench":
        root = raw / "core-bench"
        from measurement_db.scripts.build_measurement_tables.load_source_files import read_gpg_json
        # Upstream documents this public password in its README. Independently
        # read the captured archive and encrypted test bank, without the builder.
        test = read_gpg_json(root / "benchmark/dataset/core_test.json.gpg",
                             password="reproducibility", scratch_dir=tables_directory.parent)
        tasks = {r["capsule_id"]: r for r in _json(root / "benchmark/dataset/core_train.json") + test}
        names = {
            "gpt4o": "GPT-4o",
            "gpt4o-mini": "GPT-4o-mini",
            "o1": "o1",
            "o1-mini": "o1-mini",
            "o1-preview": "o1-preview",
            "claude_35_sonnet": "Claude-3.5-Sonnet",
        }
        expected = Counter()
        source_traces = Counter()
        skip = {
            "20250112-052839_codeocean_hard.json",
            "20250112-065033_codeocean_hard.json",
        }
        with tarfile.open(root / "agent_results.tar.gz") as archive:
            for member in archive:
                path = Path(member.name)
                if not member.isfile() or path.suffix != ".json" or path.name in skip:
                    continue
                parts = path.parent.name.split("_")
                model = "_".join(parts[2:-1])
                agent = {"coreagent": "CORE-Agent", "autogpt": "AutoGPT"}[parts[1]]
                level = next(x for x in ("easy", "medium", "hard") if "codeocean_" + x in path.name)
                for r in json.load(archive.extractfile(member))["capsule_results"]:
                    total = r["total_written_questions"] + r["total_vision_questions"]
                    assert total > 0
                    grade = (r["correct_written_answers"] + r["correct_vision_answers"]) / total
                    key = (names[model], agent, parts[-1], r["capsule_id"] + "__" + level, grade)
                    expected[key] += 1
                    if isinstance(r.get("result_report"), dict) and r["result_report"]:
                        source_traces[key + (json.dumps(r["result_report"], ensure_ascii=False),)] += 1
        actual = Counter(
            (
                r.display_name,
                r.harness,
                _features(r.subject_features_extra)["cost_limit"],
                r.raw_item_id,
                r.response,
            )
            for r in cells.itertuples()
        )
        _equal(actual, expected, slug)
        traces = cells.merge(
            pd.read_parquet(tables_directory / "traces.parquet"), on="response_id"
        )
        actual_traces = Counter(
            (
                r.display_name,
                r.harness,
                _features(r.subject_features_extra)["cost_limit"],
                r.raw_item_id,
                r.response,
                r.trace,
            )
            for r in traces.itertuples()
        )
        _equal(actual_traces, source_traces, "CORE traces")
        templates = _json(root / "benchmark/benchmark_prompts.json")
        for r in items.itertuples():
            capsule, level = r.raw_item_id.split("__")
            task = tasks[capsule]
            prompt = templates["codeocean_" + level].replace("{task_prompt}", task["task_prompt"])
            prompt = prompt.replace("{json_fields}", str(task["results"][0].keys()))
            prompt += "\n\nScientific code capsule: " + capsule + "\nCapsule DOI: " + task["capsule_doi"]
            assert r.content == prompt
            assert json.loads(json.loads(r.grading_criterion)["reference_answer"]) == task["results"]
        return {
            "source_responses": sum(expected.values()),
            "source_capsules": len(tasks),
            "source_traces": sum(source_traces.values()),
        }
    if slug == "theagentcompany":
        metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
        selection = metadata["build"]["parameters"]
        expected, native_traces = Counter(), Counter()
        task_names = set()
        trace_files = 0
        for run, model in selection["models"].items():
            harness = selection["harnesses"][run]
            run_dir = raw / "experiments/evaluation/1.0.0" / run
            grades = {}
            for path in sorted((run_dir / "results").glob("eval_*.json")):
                score = _json(path)["final_score"]
                earned, available = float(score["result"]), float(score["total"])
                assert math.isfinite(earned) and math.isfinite(available) and 0 <= earned <= available and available > 0
                task = path.stem.removeprefix("eval_").removesuffix("-image")
                assert task not in grades
                grades[task] = earned / available
                expected[model, harness, task, grades[task]] += 1
                task_names.add(task)
            seen = {}
            for path in sorted((run_dir / "trajectories").glob("traj_*")):
                filename = path.name.removeprefix("traj_")
                if filename.endswith(".json.gz"):
                    task = filename[:-8].removesuffix("-image")
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        payload = handle.read()
                else:
                    assert path.suffix in (".json", ".txt")
                    task = filename[:-len(path.suffix)].removesuffix("-image")
                    payload = path.read_text()
                if task not in grades:
                    grades[task] = None
                    expected[model, harness, task, None] += 1
                    task_names.add(task)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                trace_files += 1
                if task in seen:
                    assert seen[task] == digest, f"Conflicting native trajectory copies: {path}"
                    continue
                seen[task] = digest
                native_traces[model, harness, task, grades[task], digest] += 1
            assert set(seen) == set(grades), f"Pinned native release has an untraced attempt: {run}"
        _equal(Counter((r.display_name, r.harness, r.raw_item_id, None if pd.isna(r.response) else r.response)
                       for r in cells.itertuples()),
               expected, "TheAgentCompany native per-attempt scores and subject configurations")
        assert set(items.raw_item_id) == task_names
        for item in items.itertuples():
            assert (raw / "tasks" / item.raw_item_id / "task.md").read_text().strip() == item.content
            verifier = json.loads(json.loads(item.verifier)["spec"])
            assert verifier["grading_code"] == (raw / "tasks" / item.raw_item_id / "evaluator.py").read_text()
            assert verifier["source_revision"] == "98b68ef82a47690c316f42fddb05baafaab56851"
            assert verifier["kind"] == "provider_result_mapping"
        traces = cells.merge(pd.read_parquet(tables_directory / "traces.parquet"), on="response_id", validate="one_to_one")
        actual_traces = Counter((r.display_name, r.harness, r.raw_item_id, None if pd.isna(r.response) else r.response,
                                 hashlib.sha256(r.trace.encode()).hexdigest()) for r in traces.itertuples())
        _equal(actual_traces, native_traces, "TheAgentCompany complete native trajectories")
        return {"source_responses": sum(expected.values()), "source_items": len(task_names),
                "source_traces": sum(native_traces.values()), "source_trace_files": trace_files,
                "source_ungraded_observations": sum(count for key, count in expected.items() if key[-1] is None)}
    raise ValueError(f"No source audit for {slug}")
