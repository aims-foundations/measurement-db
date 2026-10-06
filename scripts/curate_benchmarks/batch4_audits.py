"""Read-only checks against independently parsed captured provider releases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
import re
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq


def _read(path):
    return json.loads(path.read_text())


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_batch4(directory, tables_directory=None):
    directory = Path(directory)
    raw, slug = directory / "raw", directory.name
    tables_directory = Path(tables_directory) if tables_directory is not None else directory / "formatted_tables"
    count = successes = 0
    extra = {}
    if slug == "confagents":
        source = _read(raw / "ConfAgents.json")
        for row in source:
            gold = str(row["reference_answer"]).strip()
            _check(gold in row["options"], "Medical reference is not a released option")
            for method in ("conformal", "colacare", "mdagents", "medagent"):
                value = row.get(method, {}).get("final_answer")
                if value is not None:
                    count += 1
                    successes += str(value).strip() == gold
        extra["source_questions"] = len(source)
    elif slug == "os_harm":
        excluded = 0
        for path in raw.glob("batch_*.json"):
            for key, verdict in _read(path)["results"].items():
                parts = key.split("/")
                task = "/".join(parts[4:])
                grades = [
                    v
                    for k, v in verdict.items()
                    if k in {"safety", "success"} and isinstance(v, bool)
                ]
                if task == "libreoffice_impress/_smear_campaign_aug":
                    spec = _read(raw / "gh_tasks" / f"{task}.json")
                    _check(
                        spec["instruction"] == "",
                        "Revisit exclusion: task instruction is now present",
                    )
                    excluded += len(grades)
                    continue
                count += len(grades)
                successes += sum(grades)
        extra["excluded_empty_instruction_assessments"] = excluded
        items = pd.read_parquet(tables_directory / "items.parquet")
        _check(
            ~items.content.str.startswith(("[task:", "[base task:")).any(),
            "OS-Harm identifier placeholder remains",
        )
    elif slug == "swe_together":
        text = (raw / "metrics.js").read_text()
        result = json.JSONDecoder().raw_decode(text.split("window.METRICS = ", 1)[1])[0]
        missing = 0
        for task in result.values():
            for cell in task["models"].values():
                for grade in cell["trials"]:
                    if grade is None:
                        missing += 1
                    else:
                        _check(
                            not isinstance(grade, bool) and 0 <= grade <= 1,
                            "Invalid SWE-Together source grade",
                        )
                        count += 1
                        successes += grade >= 0.85
        extra.update(source_tasks=len(result), missing_replicates=missing)
        items = pd.read_parquet(tables_directory / "items.parquet")
        bank = pd.read_parquet(raw / "task_specs.parquet").set_index("task_id")
        for item in items.itertuples():
            _check(
                str(bank.loc[item.raw_item_id, "instruction"]).strip() in item.content,
                "SWE-Together task text changed",
            )
            _check("Language: nan" not in item.content, "Missing language became text")
    elif slug == "scivisagentbench":
        sources = _read(raw / "capture_sources.json")["reports"]
        reports = [
            name
            for name in sources
            if name.startswith(("site__eval_reports", "history__"))
        ]
        for name in reports:
            text = (raw / name).read_text()
            percentages = re.findall(
                r'<div class="case-score">[\d.]+/[\d.]+ \(([\d.]+)%\)</div>', text
            )
            sections = re.findall(r'<section class="case-section[^\"]*"', text)
            _check(len(percentages) == len(sections), "Unparsed SciVis report case")
            count += len(percentages)
            successes += sum(float(v) >= 50 for v in percentages)
        extra["source_reports"] = len(reports)
    elif slug == "live_agent_risk":
        games = 0
        for path in (raw / "bundle").rglob("end_game_results.json"):
            record = _read(path)
            players = [
                p["name"]
                for p in record.get("players", [])
                if p.get("name")
                and p["name"].lower()
                not in {"alpha", "bravo", "charlie", "delta", "echo"}
            ]
            if not players:
                continue
            _check(
                record["winner"] in players,
                "Risk winner does not identify a released seat",
            )
            games += 1
            count += len(players)
            successes += 1
        extra["source_games"] = games
    elif slug == "wikihow_agent":
        configurations = 0
        for path in sorted(raw.glob("T-*_corrected.json")):
            configurations += 1
            for record in _read(path)["total_conversations"]:
                value = record.get("evaluation", {}).get("Completion Achieved")
                if value is None:
                    continue
                _check(value in (0, 1), "Unexpected WikiHow completion value")
                transcript = record.get("conversation") or []
                _check(
                    bool(value) == ("FINISHED" in json.dumps(transcript)),
                    "WikiHow flag disagrees with completion marker",
                )
                count += 1
                successes += value
        extra["source_workflow_configurations"] = configurations
    elif slug == "agentic_review_perturbation":
        from measurement_db.scripts.curate_benchmarks.batch3_audits import _actual, _features
        tables = {p.stem: pd.read_parquet(p) for p in tables_directory.glob("*.parquet")}
        items = tables["items"].set_index("raw_item_id")
        root = raw / "extracted/benchmarks/experimental_perturbations"
        groups, expected = Counter(), Counter()
        for path in root.glob("results_*/**/score/llm/*_score.json"):
            record = _read(path)
            detected, missed = record["detected"], record["missed"]
            _check(len(set(detected + missed)) == len(detected + missed) == record["n_injected"],
                   "Conflicting or duplicate perturbation scores")
            _check(len(detected) == record["n_detected"], "Detection count disagrees with source summary")
            parts = path.relative_to(root).parts
            tree, method, model = parts[:3]
            paper = next(p for p in parts if p.startswith("paper_"))
            stem = path.name.removesuffix("_score.json")
            kept = _read(root / tree / method / "perturb/experimental" / paper / (stem + "_kept_perturbations.json"))
            rubric = {p["perturbation_id"]: p for p in kept["perturbations"]}
            paper_paths = list((root / "perturbation_results").glob(
                f"*/all/{paper.removeprefix('paper_')}/experimental/{stem}_recorrupted.md"))
            _check(len(paper_paths) == 1, "Ambiguous review paper")
            text = paper_paths[0].read_text().strip()
            for pid in detected + missed:
                raw_item_id = f"{paper}/{stem}/{pid}"
                item = items.loc[raw_item_id]
                _check(item.content == text, "Review item is not its captured paper")
                criterion = json.loads(json.loads(item.grading_criterion)["rule"])
                _check(criterion["injected_error"] == rubric[pid], "Changed per-error grading rubric")
                expected[model, method, tree.removeprefix("results_"), raw_item_id, float(pid in detected), None] += 1
            count += len(detected + missed)
            successes += len(detected)
            groups[tree] += len(detected + missed)
        actual = _actual(tables, lambda s, i, r: (
            s["display_name"], s["harness"], _features(s["subject_features_extra"])["review_prompt"], i["raw_item_id"],
        ))
        _check(actual == expected, "Review source-to-response associations differ")
        _check(set(items.index) == {row[3] for row in expected}, "Missing or extra review items")
        extra.update({"source_" + k: v for k, v in groups.items()})
    elif slug == "metaagent":
        selections = {
            "gpt4o_direct.log": {"direct"},
            "gpt4o_multi.log": {"cot", "llm_debate", "self_refine"},
            "gpt4o_cotsc.log": {"cot_sc"},
            "gpt4o_spp.log": {"spp"},
            "gpt35_dircot.log": {"direct", "cot"},
            "gpt35_cotsc.log": {"cot_sc"},
        }
        blocks = []
        for name, methods in selections.items():
            text = (raw / name).read_text()
            for method, body in re.findall(
                r"=====ANSWER \(([a-z_]+)\)=====(.*?)(?="
                + re.escape("=" * 25)
                + r"|=====ANSWER \(|\Z)",
                text,
                re.DOTALL,
            ):
                if method in methods:
                    blocks.append(body)
        blocks.extend(
            re.findall(
                r"=====ANSWER=====(.*?)=====\\ANSWER=====",
                (raw / "metaagent_gpt35.log").read_text(),
                re.DOTALL,
            )
        )
        for body in blocks:
            gold = re.search(r"Correct answer:.*?\(\(([a-dA-D])\)\)", body, re.DOTALL)
            response = re.search(r"Model response:(.*)$", body, re.DOTALL)
            if (
                not gold
                or "Question:" not in body
                or not response
                or not response[1].strip()
            ):
                continue
            answer = response[1].strip()
            if "<|submit|>" in answer:
                # Boundary-aware independent check also catches matching a letter inside a word.
                match = re.match(
                    r"\s*\(?([a-dA-D])\)?(?:\s|[.,:]|$)", answer.split("<|submit|>")[-1]
                )
                letters = [match[1]] if match else []
            else:
                letters = re.findall(r"\(([a-dA-D])\)", answer) or re.findall(
                    r"answer\s*(?:is|:)?\s*\(?([a-dA-D])\)?", answer, re.IGNORECASE
                )
            count += 1
            successes += bool(letters and letters[-1].lower() == gold[1].lower())
    elif slug == "swe_smith":
        runs, represented = {}, 0
        for path in sorted(raw.glob("*.parquet")):
            if not path.name.startswith(("tool-", "ticks-", "xml-")):
                continue
            for record in pq.read_table(
                path, columns=["traj_id", "instance_id", "model", "resolved"]
            ).to_pylist():
                represented += 1
                key = record["instance_id"], record["model"], record["resolved"]
                tid = record["traj_id"]
                _check(
                    tid not in runs or runs[tid] == key,
                    "Conflicting SWE-smith trajectory grade",
                )
                runs[tid] = key
        count = len(runs)
        successes = sum(v[2] for v in runs.values())
        extra["source_serialized_rows"] = represented
        extra["duplicate_renderings_removed"] = represented - count
    else:
        raise ValueError(f"No batch4 source audit for {slug}")
    return {"source_responses": int(count), "source_successes": int(successes), **extra}
