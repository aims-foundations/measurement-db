"""Check complete native results independently of the pandas builders."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from measurement_db.scripts.curate_benchmarks.batch3_audits import (
    _actual, _check, _digest, _features,
)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _openbiorq(directory, tables, metadata):
    raw = directory / "raw"
    questions = _jsonl(raw / "full_core_657.jsonl")
    rubrics = _jsonl(raw / "rubrics.jsonl")
    bank = {r["task_id"]: r for r in questions}
    rubric = {r["task_id"]: r for r in rubrics}
    _check(len(bank), len(questions), "OpenBioRQ unique questions")
    _check(len(rubric), len(rubrics), "OpenBioRQ unique rubrics")
    _check(set(bank), set(rubric), "OpenBioRQ rubric coverage")
    _check(set(tables["items"].raw_item_id), set(bank), "OpenBioRQ item coverage")
    for item in tables["items"].itertuples():
        source = bank[item.raw_item_id]
        _check(item.content, source["self_contained_question"], "OpenBioRQ question")
        _check(item.content, rubric[item.raw_item_id]["question"], "OpenBioRQ rubric question")
        criterion = json.loads(item.grading_criterion)
        _check(json.loads(criterion["reference_answer"]), source["gold_answer"], "OpenBioRQ reference grounding")
        _check(json.loads(criterion["rule"])["criteria"], rubric[item.raw_item_id]["criteria"], "OpenBioRQ rubric")
        verifier = json.loads(json.loads(item.verifier)["spec"])
        _check(verifier["judge_model"], "GLM-5.1", "OpenBioRQ judge")
        _check(verifier["success_threshold"], 0.5, "OpenBioRQ solve threshold")

    expected = Counter()
    ungraded = successes = 0
    for model, label in metadata["build"]["parameters"]["models"].items():
        folder = raw / "predictions" / model
        lines = (folder / "predictions.jsonl").read_text().splitlines()
        predictions = [json.loads(line) for line in lines]
        verdicts = _jsonl(folder / "checklist.jsonl")
        checks = {r["task_id"]: r for r in verdicts}
        _check(len(checks), len(verdicts), "OpenBioRQ unique verdicts")
        _check(len({p["task_id"] for p in predictions}), len(predictions), "OpenBioRQ unique attempts")
        _check(set(checks), set(bank), "OpenBioRQ judged-task coverage")
        _check({p["task_id"] for p in predictions}, set(bank), "OpenBioRQ attempted-task coverage")
        summary = json.loads((folder / "summary.json").read_text())
        tools = "enabled" if summary["tools"] else "disabled"
        _check(summary["n_predictions"], len(predictions), "OpenBioRQ reported attempts")
        observed_scores = []
        for prediction, line in zip(predictions, lines, strict=True):
            task = prediction["task_id"]
            verdict = checks[task]
            score = verdict.get("checklist_score")
            if score is None:
                grade = None
                ungraded += 1
            else:
                weights = {c["id"]: c["weight"] for c in rubric[task]["criteria"]}
                _check(len(weights), len(rubric[task]["criteria"]), "OpenBioRQ unique criteria")
                _check({v["id"] for v in verdict["verdicts"]}, set(weights), "OpenBioRQ verdict coverage")
                _check(len(verdict["verdicts"]), len(weights), "OpenBioRQ one verdict per criterion")
                recomputed = sum(weights[v["id"]] * {"met": 1, "partial": 0.5, "not_met": 0}[v["v"]]
                                 for v in verdict["verdicts"]) / sum(weights.values())
                if abs(recomputed - score) > 0.000501:
                    raise ValueError("OpenBioRQ checklist score disagrees with weighted criterion verdicts")
                grade = float(score >= 0.5)
                successes += int(grade)
                observed_scores.append(score)
            expected[label, tools, task, grade, _digest(line)] += 1
        _check(summary["n_judged"], len(observed_scores), "OpenBioRQ reported graded attempts")
        pct = 100 * sum(s >= 0.5 for s in observed_scores) / len(observed_scores)
        if abs(pct - summary["full_core_657"]["solve@0.5_pct"]) > 0.050001:
            raise ValueError("OpenBioRQ outcomes disagree with the released rounded summary")

    actual = _actual(tables, lambda s, i, r: (
        s["display_name"], _features(s["subject_features_extra"])["tool_mode"], i["raw_item_id"],
    ))
    _check(actual, expected, "OpenBioRQ every attempt, grade and full native trace")
    return {"source_responses": sum(expected.values()), "source_successes": successes,
            "source_items": len(bank), "source_configurations": len(metadata["build"]["parameters"]["models"]),
            "source_ungraded_observations": ungraded, "source_traces": sum(expected.values())}


def _fewshot(directory, tables, metadata):
    root = directory / "raw/Fewshot-TTT/logs/archive/results"
    expected = Counter()
    files = successes = tasks = 0
    for method, label in metadata["build"]["parameters"]["models"].items():
        for path in sorted((root / method).glob("*.json")):
            files += 1
            for task in json.loads(path.read_text()):
                tasks += 1
                for example in task["examples"]:
                    question, gold, prediction = (example[k] for k in ("question", "true_answer", "prediction"))
                    grade = float(prediction.strip().lower() == gold.strip().lower())
                    successes += int(grade)
                    expected[label, method, task["task"], question, gold, grade, _digest(prediction)] += 1
                # Upstream stores only the first five examples. The adjacent
                # accuracy is computed on the full evaluation set, not this preview.
    actual = _actual(tables, lambda s, i, r: (
        s["display_name"], _features(s["subject_features_extra"])["method"],
        _features(i["item_features"])["task"], i["content"], json.loads(i["grading_criterion"])["reference_answer"],
    ))
    _check(actual, expected, "Few-Shot TTT every question, reference, prediction and grade")
    for item in tables["items"].itertuples():
        verifier = json.loads(json.loads(item.verifier)["spec"])
        _check(verifier["normalization"], ["strip", "lower"], "Few-Shot TTT normalization")
        _check(verifier["function"], "compute_accuracy", "Few-Shot TTT grading function")
    return {"source_responses": sum(expected.values()), "source_successes": successes,
            "source_files": files, "source_task_summaries": tasks, "source_traces": sum(expected.values())}


def _prox(directory, tables, metadata):
    expected = Counter()
    files = successes = 0
    for checkpoint, label in metadata["build"]["parameters"]["models"].items():
        for path in sorted((directory / "raw" / checkpoint).glob("*.jsonl")):
            files += 1
            records = _jsonl(path)
            _check(len({r["idx"] for r in records}), len(records), "ProX unique source indices")
            for row in records:
                for score, output, prediction in zip(row["score"], row["code"], row["pred"], strict=True):
                    if not isinstance(score, bool):
                        raise TypeError("ProX released score is not boolean")
                    successes += int(score)
                    gold = str(row["gt"]) if str(row["gt"]).strip() else None
                    expected[label, checkpoint, path.stem, row["question"].strip(), gold, float(score), _digest(output)] += 1
    actual = _actual(tables, lambda s, i, r: (
        s["display_name"], _features(s["subject_features_extra"])["training_tokens"],
        _features(i["item_features"])["task"], i["content"], json.loads(i["grading_criterion"]).get("reference_answer"),
    ))
    _check(actual, expected, "ProX every checkpoint, question, reference, native verdict and output")
    for item in tables["items"].itertuples():
        verifier = json.loads(json.loads(item.verifier)["spec"])
        _check(verifier["function"], "math_equal", "ProX grading function")
        _check(verifier["result_field"], "score", "ProX result field")
    return {"source_responses": sum(expected.values()), "source_successes": successes,
            "source_files": files, "source_traces": sum(expected.values())}


def _cseo(directory, tables, metadata):
    """Read Arrow records and compare every paired measurement without importing the builder."""
    import pyarrow.parquet as pq
    import re

    source = directory / "raw/release/results"
    paths = sorted(source.rglob("responses.parquet"))
    baselines, expected = {}, Counter()
    source_rows = baseline_rows = control_rows = missing_outputs = parser_discrepancies = 0
    duplicate_files = duplicate_records = 0
    for path in paths:
        relative = path.relative_to(source)
        domain, method, model, *tail = relative.parts
        if method == "Original" and tail == ["AdoptionMode.NONE", "responses.parquet"]:
            for row in pq.read_table(path).to_pylist():
                query = row["Search Query"].removesuffix(" videogame") if domain == "videogames" else row["Search Query"]
                key = domain, model, query
                if key in baselines:
                    raise ValueError("C-SEO repeated baseline query")
                baselines[key] = (row, str(relative))
                baseline_rows += 1

    for path in paths:
        relative = path.relative_to(source)
        domain, method, model, *tail = relative.parts
        adoption = "/".join(tail[:-1]) or "unspecified"
        if method == "seo_baseline_game_theory" and adoption == "AdoptionMode.UNILATERAL":
            canonical = source / domain / "seo_baseline-1" / model / adoption / "responses.parquet"
            _check(path.read_bytes(), canonical.read_bytes(), "C-SEO equivalent single-target SEO result exports")
            duplicate_files += 1
            duplicate_records += pq.read_metadata(path).num_rows
            source_rows += pq.read_metadata(path).num_rows
            continue
        seen = set()
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            source_rows += 1
            query = row["Search Query"].removesuffix(" videogame") if domain == "videogames" else row["Search Query"]
            if query in seen:
                raise ValueError("C-SEO duplicate source query")
            seen.add(query)
            parsed = list(dict.fromkeys(int(x) - 1 for x in re.findall(r"\[(\d+)\]", row["Response"] or "")))
            parser_discrepancies += parsed != row["Citation Order"]
            if method == "Original":
                control_rows += adoption != "AdoptionMode.NONE"
                continue
            original, baseline_path = baselines[domain, model, query]
            before, after = original["Citation Order"][:5], row["Citation Order"][:5]
            targets = row["Boost Product Index"]
            if isinstance(targets, int):
                targets = [targets]
            _check(targets, sorted(set(targets)), "C-SEO sorted, distinct target indices")
            if not targets:
                raise ValueError("C-SEO intervention without target documents")
            for position, target in enumerate(targets):
                moved_target = (int(method.removeprefix("seo_baseline-")) - 1 if method.startswith("seo_baseline-")
                                else position if method == "seo_baseline_game_theory" else target)
                if original["Response"] is None or row["Response"] is None:
                    score = None
                    missing_outputs += 1
                elif target in before and moved_target in after:
                    score = float(before.index(target) - after.index(moved_target))
                elif target in before:
                    score = float(before.index(target) - len(after))
                elif moved_target in after:
                    score = float(len(before) - after.index(moved_target))
                else:
                    score = 0.
                expected[(model, domain, method, adoption,
                          _digest(original["Prompt"]), _digest(row["Prompt"]), target, moved_target, score,
                          _digest(original["Response"]), _digest(row["Response"]), baseline_path, str(relative), index)] += 1

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = {}
    for item in tables["items"].itertuples():
        content, features = json.loads(item.content), _features(item.item_features)
        _check(set(content), {"baseline_prompt", "modified_prompt", "target_before", "target_after"}, "C-SEO input fields")
        items[item.item_id] = (features["domain"], features["method"], features["adoption"],
                              _digest(content["baseline_prompt"]), _digest(content["modified_prompt"]),
                              content["target_before"], content["target_after"])
        _check(json.loads(item.grading_criterion)["rule"], metadata["grading"]["rule"], "C-SEO grading rule")
        spec = json.loads(json.loads(item.verifier)["spec"])
        _check(spec, metadata["grading"]["verifiers"]["rank_difference"], "C-SEO verifier")
    traces = {}
    for trace in tables["traces"].itertuples():
        record = json.loads(trace.trace)
        _check(set(record), {"output_baseline", "output", "source_file_baseline", "source_file", "source_row"}, "C-SEO trace fields")
        if trace.response_id in traces:
            raise ValueError("C-SEO duplicate trace")
        traces[trace.response_id] = (_digest(record["output_baseline"]), _digest(record["output"]),
                                    record["source_file_baseline"], record["source_file"], record["source_row"])
    actual = Counter(
        (subjects[row.subject_id]["display_name"], *items[row.item_id],
         None if pd.isna(row.response) else row.response, *traces[row.response_id])
        for row in tables["responses"].itertuples()
    )
    _check(actual, expected, "C-SEO every model, condition, paired input, target index, rank difference and full output")
    return {"source_files": len(paths), "source_records": source_rows, "source_baseline_records": baseline_rows,
            "source_extra_control_records": control_rows, "source_responses": sum(expected.values()),
            "source_ungraded_responses": missing_outputs, "source_traces": len(traces),
            "source_parser_discrepancies": parser_discrepancies,
            "source_duplicate_condition_files": duplicate_files, "source_duplicate_condition_records": duplicate_records}


def _xlsx_records(payload):
    """Read native worksheet XML without using the builder's Excel reader."""
    import io
    import xml.etree.ElementTree as ET
    import zipfile

    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(payload)) as book:
        strings = ["".join(part.text or "" for part in cell.findall(".//s:t", namespace))
                   for cell in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", namespace)]
        rows = []
        for row in ET.fromstring(book.read("xl/worksheets/sheet1.xml")).findall(".//s:row", namespace):
            values = {}
            for cell in row.findall("s:c", namespace):
                value = cell.find("s:v", namespace)
                if value is not None:
                    column = "".join(char for char in cell.attrib["r"] if char.isalpha())
                    values[column] = strings[int(value.text)] if cell.attrib.get("t") == "s" else float(value.text)
            rows.append(values)
    return [{name: row.get(column, "") for column, name in rows[0].items()} for row in rows[1:]]


def _risebench(directory, tables, metadata):
    import base64
    import csv
    import hashlib
    import io
    import zipfile

    raw = directory / "raw"
    bank_rows = json.loads((raw / "tasks/datav2_total_w_subtask.json").read_text())
    bank = {row["index"]: row for row in bank_rows}
    _check(len(bank), len(bank_rows), "RISE unique task identities")
    parameters = metadata["build"]["parameters"]
    specs = {"legacy": metadata["grading"]["verifiers"]["published_judgments"],
             "current": metadata["grading"]["verifiers"]["current_judgments"]}
    rules = {"legacy": metadata["grading"]["rule"], "current": specs["current"]["rule"]}
    video_requests = {}
    for profile, source in parameters["video_inputs"].items():
        requests = json.loads((raw / source).read_text())
        _check(Counter(row["index"] for row in requests), Counter({task: 1 for task in bank}),
               "RISE complete released video-generation requests")
        for request in requests:
            task = bank[request["index"]]
            for field in ["instruction", "category"]:
                _check(request[field], task[field], "RISE video request matches the original task")
            _check(request["image"].endswith("/" + task["image"]), True, "RISE video request input-image locator")
            video_requests[profile, request["index"]] = request
    assets = dict(zip(tables["assets"].asset_id, tables["assets"].data, strict=True))
    used_assets = set()
    item_keys = {}
    item_coverage = Counter()
    with zipfile.ZipFile(raw / "tasks/data.zip") as images:
        for item in tables["items"].itertuples():
            task = bank[item.raw_item_id]
            verifier = json.loads(item.verifier)
            spec = json.loads(verifier["spec"])
            versions = [version for version, expected in specs.items() if spec == expected]
            _check(len(versions), 1, "RISE separate recorded grading protocols")
            version = versions[0]
            _check(verifier.get("judge"), spec.get("judge"), "RISE observed versus default judge identity")
            variant = version
            if item.content != task["instruction"]:
                _check(version, "current", "RISE video inputs use the current grading protocol")
                content = json.loads(item.content)
                matches = [profile for profile in parameters["video_inputs"]
                           if content == {"instruction": task["instruction"],
                                          "generation_prompt": video_requests[profile, item.raw_item_id]["video_text"]}]
                _check(len(matches), 1, "RISE complete original instruction and video-generation prompt")
                variant = "video/" + matches[0]
            item_keys[item.item_id] = variant, item.raw_item_id
            item_coverage[variant, item.raw_item_id] += 1
            _check(_features(item.item_features), {k: task[k] for k in ("category", "subtask")}, "RISE task category")
            criterion = json.loads(item.grading_criterion)
            _check(criterion["reference_answer"], task.get("reference") or task.get("reference_txt"), "RISE reference text")
            rule = {key: task.get(key) for key in parameters["grading_fields"]}
            rule["completion_rule"] = rules[version]
            _check(json.loads(criterion["rule"]), rule, "RISE full task grading fields")
            links = []
            for key, role in [("image", "input_image"), ("reference_img", "grading_reference")]:
                if task.get(key):
                    path = task[key]
                    payload = images.read("data/" + path)
                    digest = hashlib.sha256(payload).hexdigest()
                    _check(assets[digest], payload, "RISE exact input/reference image bytes")
                    used_assets.add(digest)
                    links.append({"asset_id": digest, "path": path, "role": role, "ordinal": len(links) + 1,
                                  "media_type": "image/png" if path.endswith(".png") else "image/jpeg"})
            _check(json.loads(item.asset_manifest), links, "RISE ordered input and grading-image associations")
    _check(set(assets), used_assets, "RISE no orphan assets")
    variants = [*specs, *("video/" + profile for profile in parameters["video_inputs"])]
    _check(item_coverage, Counter({(variant, task): 1 for variant in variants for task in bank}),
           "RISE complete item coverage under both grading protocols and known input variants")

    profiles = {}
    for profile, label in parameters["subjects"].items():
        profiles[profile] = ("legacy", label, {**parameters["subject_features"],
            "thinking": parameters["thinking"][profile], "cfg_img_scale": parameters["cfg_img_scale"][profile]})
    for profile, label in parameters["current_subjects"].items():
        features = {**parameters["current_subject_features"], "upstream_profile": profile}
        if profile in parameters["current_thinking"]:
            features["thinking"] = parameters["current_thinking"][profile]
        profiles[profile] = ("current", label, features)
    subjects = {}
    for subject in tables["subjects"].itertuples():
        features = {**_features(subject.subject_features_extra), "harness": subject.harness}
        matches = [profile for profile, (_, label, expected) in profiles.items()
                   if label == subject.display_name and features == expected]
        _check(len(matches), 1, "RISE reported model identity and settings")
        subjects[subject.subject_id] = matches[0]
    _check(Counter(subjects.values()), Counter({profile: 1 for profile in profiles}), "RISE profile coverage")

    expected = Counter()
    native = {}
    output_paths = {}
    image_hashes = {}
    reasoning = {}
    sheet_paths = {}
    successes = missing_consistency = ungraded = auto_failures = 0
    for profile, (version, _, _) in profiles.items():
        archive = None
        try:
            if version == "legacy":
                archive = zipfile.ZipFile(raw / f"results/{profile}.zip")
                sheet_member = f"{profile}/bagel_judge.xlsx"
                records = _xlsx_records(archive.read(sheet_member))
                summary_bytes = archive.read(f"{profile}/bagel_judge.csv")
                sheet_paths[profile] = f"results/{profile}.zip", sheet_member
            else:
                sheet, = (raw / "current" / profile).glob("*_judge.xlsx")
                records = _xlsx_records(sheet.read_bytes())
                summary_bytes = sheet.with_suffix(".csv").read_bytes()
                sheet_paths[profile] = None, str(sheet.relative_to(raw))
            _check(Counter(row["index"] for row in records), Counter({task: 1 for task in bank}),
                   "RISE one judgment per task per profile")
            for row in records:
                task_id = row["index"]
                for key, value in bank[task_id].items():
                    _check(row[key], value, "RISE task-bank/result-sheet correspondence")
                logical = row["category"] == "logical_reasoning"
                consistency_required = logical or version == "legacy" or row["consistency_free"] == ""
                completion = float(row["Reasoning"] == 5 and
                                   (not consistency_required or row["ApprConsistency"] == 5) and
                                   (logical or row["VisualPlausibility"] == 5))
                _check(row["complete"], completion, "RISE native completion function")
                if row["match_log"] == "failed":
                    _check([row[key] for key in ["Reasoning", "ApprConsistency", "VisualPlausibility"]],
                           ["", "", ""], "RISE failed judge parse has no component grades")
                    grade = None
                    ungraded += 1
                else:
                    _check(row["match_log"], "succeed", "RISE known native parser status")
                    grade = completion
                    successes += int(grade)
                missing_consistency += row["ApprConsistency"] == ""
                native[profile, task_id] = row
                variant = "video/" + profile if profile in parameters["video_inputs"] else version
                expected[variant, profile, task_id, grade] += 1
                if version == "legacy":
                    member = f"{profile}/{row['category']}/{task_id}.png"
                    image_hashes[profile, task_id] = hashlib.sha256(archive.read(member)).hexdigest()
                    output_paths[profile, task_id] = member
                    reasoning[profile, task_id] = (archive.read(member.removesuffix(".png") + ".txt").decode("utf-8")
                                                  if parameters["thinking"][profile] == "enabled" else None)
                else:
                    image_folder = "image" if profile.startswith("BAGEL_") else "images"
                    folder = raw / "current" / profile / image_folder / row["category"]
                    candidates = [folder / f"{task_id}.{suffix}" for suffix in ["png", "jpg", "jpeg"]
                                  if (folder / f"{task_id}.{suffix}").is_file()]
                    if not candidates:
                        for field in ["judge_cons", "judge_reas", "judge_qua"]:
                            _check(row[field], "Auto fail (missing image). Final Score: 1",
                                   "RISE missing output must have explicit upstream auto-fail evidence")
                        _check(grade, 0., "RISE recorded missing-image failure")
                        image_hashes[profile, task_id] = None
                        output_paths[profile, task_id] = None
                        auto_failures += 1
                    else:
                        _check(len(candidates), 1, "RISE unambiguous generated image")
                        image_hashes[profile, task_id] = hashlib.sha256(candidates[0].read_bytes()).hexdigest()
                        output_paths[profile, task_id] = str(candidates[0].relative_to(raw))
                    thought = folder / f"{task_id}.txt"
                    reasoning[profile, task_id] = thought.read_bytes().decode("utf-8") if thought.exists() else None
            summary = next(row for row in csv.DictReader(io.StringIO(summary_bytes.decode()))
                           if row["-"] == "Overall")
            if abs(float(summary["Accuracy"]) - sum(row["complete"] for row in records) / len(records)) > 1e-14:
                raise ValueError("RISE native completion flags disagree with released aggregate accuracy")
        finally:
            if archive is not None:
                archive.close()

    traces = dict(zip(tables["traces"].response_id, tables["traces"].trace, strict=True))
    actual = Counter()
    for response in tables["responses"].itertuples():
        profile = subjects[response.subject_id]
        version, task_id = item_keys[response.item_id]
        grade = None if pd.isna(response.response) else response.response
        actual[version, profile, task_id, grade] += 1
        trace = json.loads(traces[response.response_id])
        trace_fields = {"native_judgment", "source_archive", "source_member", "output_member",
                        "output_image_base64", "reasoning_trace"}
        if profile in parameters["video_inputs"]:
            trace_fields.update(["native_video_request", "video_source_file", "generated_video_base64"])
            request = video_requests[profile, task_id]
            _check(trace["native_video_request"], request, "RISE full native video request")
            source = "current/" + profile + "/" + request["video_path"].removeprefix("./")
            _check(trace["video_source_file"], source, "RISE generated video association")
            _check(hashlib.sha256(base64.b64decode(trace["generated_video_base64"], validate=True)).hexdigest(),
                   hashlib.sha256((raw / source).read_bytes()).hexdigest(), "RISE complete generated video")
        _check(set(trace), trace_fields, "RISE trace fields")
        _check(trace["native_judgment"], native[profile, task_id], "RISE complete original judgment row")
        _check((trace["source_archive"], trace["source_member"]), sheet_paths[profile], "RISE trace source")
        _check(trace["output_member"], output_paths[profile, task_id], "RISE output association")
        digest = (hashlib.sha256(base64.b64decode(trace["output_image_base64"], validate=True)).hexdigest()
                  if trace["output_image_base64"] is not None else None)
        _check(digest, image_hashes[profile, task_id], "RISE exact complete generated image or documented absence")
        _check(trace["reasoning_trace"], reasoning[profile, task_id], "RISE full thinking trace")
    _check(actual, expected, "RISE every grading version, model configuration, task and grade")
    _check(set(traces), set(tables["responses"].response_id), "RISE complete trace associations")
    return {"source_responses": sum(expected.values()), "source_successes": successes,
            "source_task_definitions": len(bank), "source_items": len(item_keys),
            "source_configurations": len(subjects), "source_traces": len(traces),
            "source_assets": len(used_assets), "source_thinking_traces": sum(v is not None for v in reasoning.values()),
            "source_missing_consistency_judgments": missing_consistency, "source_ungraded_responses": ungraded,
            "source_missing_image_failures": auto_failures,
            "source_generated_videos": len(video_requests),
            "source_generated_images": sum(value is not None for value in image_hashes.values())}

def _clasheval(directory, tables, metadata):
    """Reconcile native rows and literal prompt definitions without using the builder."""
    import ast
    import pyarrow.parquet as pq

    raw = directory / "raw"
    # Read literal assignments, without importing or executing the upstream module.
    definitions = {}
    for node in ast.parse((raw / "harness/data/prompts/prompts.py").read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"RAG_RESPONSE", "NORAG_RESPONSE"}:
                definitions[node.targets[0].id] = ast.literal_eval(node.value)
    _check(set(definitions), {"RAG_RESPONSE", "NORAG_RESPONSE"}, "ClashEval released prompt definitions")
    prompts = {"prior": definitions["NORAG_RESPONSE"], "post": definitions["RAG_RESPONSE"]}
    parameters = metadata["build"]["parameters"]
    for condition, template in prompts.items():
        _check(parameters[f"{condition}_prompts"], {k: v for k, v in template.items() if k != "user"},
               "ClashEval metadata reproduces literal system templates")
        _check(parameters["user_templates"][condition], template["user"],
               "ClashEval metadata reproduces literal user template")
    # Model names documented in the captured README, including GPT-4o in gpt4.pqt.
    models = {"gpt4": "GPT-4o", "gpt35": "GPT-3.5", "claudeopus": "Claude 3 Opus",
              "claudesonnet": "Claude 3 Sonnet", "gemini15flash": "Gemini 1.5 Flash",
              "llama3": "Llama-3-8B-Instruct"}
    _check(parameters["models"], models, "ClashEval released model labels")
    _check(set(tables["subjects"].display_name), set(models.values()), "ClashEval model coverage")

    expected = Counter()
    expected_items = set()
    prior_count = post_count = removed_copies = successes = missing = empty = 0
    for model, label in models.items():
        seen = {}
        records = pq.read_table(raw / f"{model}.pqt").to_pylist()
        for source_row, row in enumerate(records):
            question = row["question"]
            prior_identity = tuple(row[key] for key in ["dataset", "answer_original", "prior_response",
                                                       "prior_correct", "prior_logprobs"])
            if question in seen:
                _check(prior_identity, seen[question], "ClashEval copied prior answers must agree completely")
                removed_copies += 1
                conditions = ["post"]
            else:
                seen[question] = prior_identity
                prior_count += 1
                conditions = ["prior", "post"]
            post_count += 1
            for condition in conditions:
                context = None if condition == "prior" else row["context_mod"]
                modification = None if condition == "prior" else row["mod_type"]
                template = prompts[condition]
                system = template[row["dataset"]]
                user = template["user"].format(question=question, context=context)
                grade = row[f"{condition}_correct"]
                if type(grade) is not int or grade not in {0, 1}:
                    raise ValueError("ClashEval native correctness must be a binary integer")
                output, logprobs = row[f"{condition}_response"], row[f"{condition}_logprobs"]
                missing += output is None
                empty += output == ""
                successes += grade
                item = (condition, row["dataset"], modification, system, user, row["answer_original"])
                expected_items.add(item)
                expected[(label, *item, float(grade), output, logprobs, f"{model}.pqt", source_row)] += 1

    item_keys = {}
    for item in tables["items"].itertuples():
        content = json.loads(item.content)
        _check(set(content), {"system", "user"}, "ClashEval explicit system/user input")
        features = _features(item.item_features)
        condition = features["condition"]
        _check(set(features), {"condition", "dataset"} | ({"mod_type"} if condition == "post" else set()),
               "ClashEval answering-condition features")
        criterion = json.loads(item.grading_criterion)
        _check(criterion["rule"], metadata["grading"]["rule"], "ClashEval grading meaning")
        verifier = json.loads(json.loads(item.verifier)["spec"])
        _check(verifier, metadata["grading"]["verifiers"]["published_correctness"], "ClashEval native grading provenance")
        item_keys[item.item_id] = (condition, features["dataset"], features.get("mod_type"),
                                   content["system"], content["user"], criterion["reference_answer"])
    _check(set(item_keys.values()), expected_items, "ClashEval full condition/input/reference coverage")
    _check(len(item_keys), len(expected_items), "ClashEval one canonical row per distinct input and reference")
    subjects = dict(zip(tables["subjects"].subject_id, tables["subjects"].display_name, strict=True))
    traces = dict(zip(tables["traces"].response_id, tables["traces"].trace, strict=True))
    _check(set(traces), set(tables["responses"].response_id), "ClashEval one trace per response")
    actual = Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        _check(set(trace), {"output", "logprobs", "source_file", "source_row"}, "ClashEval complete trace fields")
        actual[(subjects[response.subject_id], *item_keys[response.item_id], response.response,
                trace["output"], trace["logprobs"], trace["source_file"], trace["source_row"])] += 1
    _check(actual, expected, "ClashEval every model, input, reference, native grade and full output/provenance")
    return {"source_responses": sum(expected.values()), "source_successes": successes,
            "source_prior_responses": prior_count, "source_post_responses": post_count,
            "source_reused_prior_copies_excluded": removed_copies,
            "source_items": len(expected_items), "source_models": len(models),
            "source_traces": len(traces), "source_missing_output_text": missing,
            "source_empty_output_text": empty}


def _engdesign(directory, tables, metadata):
    """Compare all native records, using AST fields rather than the builder's tokenizer."""
    import ast
    import hashlib
    import mimetypes
    import re
    import zipfile

    parameters = metadata["build"]["parameters"]
    single_models = {"4o": "GPT-4o", "claude3_7": "Claude 3.7 Sonnet",
                     "claude3_7_thinking": "Claude 3.7 Sonnet Thinking",
                     "deepseek_chat": "DeepSeek V3", "deepseek-chat": "DeepSeek V3",
                     "deepseek_r1": "DeepSeek R1", "deepseek-reasoner": "DeepSeek R1",
                     "gemini_flash": "Gemini 2.0 Flash", "gemini_pro": "gemini-2.5-pro-preview-05-06",
                     "o1": "o1", "o3": "o3", "o3_high": "o3-high",
                     "o4_mini": "o4-mini", "o4_mini_high": "o4-mini-high"}
    iterative_models = {"iterative_gpt_4o": "GPT-4o", "iterative_o1": "o1",
                        "iterative_o3": "o3", "iterative_o4_mini": "o4-mini"}
    _check(parameters["models"], single_models, "EngDesign documented single-answer model labels")
    _check(parameters["iterative_models"], iterative_models, "EngDesign documented iterative model labels")
    native = {}
    aliases = {}
    empty = copied_iterations = copied_open_logs = opaque_objects = 0
    sections = {}
    with zipfile.ZipFile(directory / "raw" / parameters["layout"]["archive"]) as archive:
        members = {name.split("/", 1)[1]: name for name in archive.namelist() if not name.endswith("/")}
        for path, member in members.items():
            # The open collection contains no additional trials, including files
            # whose only byte difference is a Windows newline convention.
            if path.startswith("EngDesign-Open/") and (path.endswith((".jsonl", "LLM_prompt.txt", "output_structure.py", "evaluate.py")) or "/images/" in path):
                original = path.replace("EngDesign-Open/", "tasks/", 1)
                _check(original in members, True, "EngDesign open-subset source correspondence")
                left, right = archive.read(member), archive.read(members[original])
                if "/images/" not in path:
                    left = left.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                    right = right.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                _check(left, right, "EngDesign open subset preserves the same input/result")
                copied_open_logs += path.endswith(".jsonl")
            if path.startswith("iterative_result/") and path.endswith(("_responses.txt", "_evaluations.txt")):
                kind = "responses" if path.endswith("_responses.txt") else "evaluations"
                group = path.rsplit("/", 1)[0]
                text = archive.read(member).decode("utf-8")
                lines = text.splitlines(keepends=True)
                boundaries = [(index, int(match[1])) for index, line in enumerate(lines)
                              if (match := re.fullmatch(r"Attempt (\d+):[ \t]*\r?\n", line))]
                _check(bool(boundaries) and boundaries[0][0] == 0, True, "EngDesign trace starts with an attempt marker")
                for n, (start, iteration) in enumerate(boundaries):
                    end = boundaries[n + 1][0] if n + 1 < len(boundaries) else len(lines)
                    key = group, iteration, kind
                    if key in sections:
                        raise ValueError("EngDesign duplicate refinement trace marker")
                    sections[key] = path, "".join(lines[start + 1:end])
            single = path.startswith("tasks/") and "/logs/" in path
            iterative = path.startswith("iterative_result/")
            if not path.endswith(".jsonl") or not (single or iterative):
                continue
            text = archive.read(member).decode("utf-8")
            if not text.strip():
                empty += 1
                continue
            # Parse only scalars. Opaque MATLAB reprs are ignored in this checker
            # copy; the original bytes remain intact in the trace. The one file
            # with an appended fragment retains its complete leading record.
            try:
                fields = json.loads(text)
                passed, iteration = fields["passed"], fields.get("iteration", 0)
            except json.JSONDecodeError:
                first = text.strip().splitlines()[0]
                readable, replaced = re.subn(r"<matlab\.object object at 0x[0-9a-fA-F]+>", "None", first)
                opaque_objects += replaced
                parsed = ast.parse(readable, mode="eval").body
                if not isinstance(parsed, ast.Dict):
                    raise ValueError("EngDesign leading record is not a dictionary")
                values = {ast.literal_eval(key): value for key, value in zip(parsed.keys, parsed.values, strict=True)}
                flag = values["passed"]
                if isinstance(flag, ast.Attribute) and isinstance(flag.value, ast.Name) and flag.value.id == "np" and flag.attr in {"True_", "False_"}:
                    passed = flag.attr == "True_"
                else:
                    passed = ast.literal_eval(flag)
                iteration = ast.literal_eval(values["iteration"]) if "iteration" in values else 0
            if type(passed) is not bool and passed != "Evaluation failed":
                raise ValueError("EngDesign undocumented native pass value")
            grade = float(passed) if type(passed) is bool else None
            if single:
                task = path.split("/")[1]
                label, suffix = Path(path).stem.split("_log_", 1)[1].rsplit("_", 1)
                trial = int(suffix.split(" ")[0]) + 1
                model, protocol, condition = single_models[label], "single_answer", "single_answer"
                key = model, protocol, task, condition, trial
            else:
                run, task = path.split("/")[1:3]
                _check(type(iteration) is int and iteration > 0, True, "EngDesign positive native iteration")
                model, protocol, condition, trial = iterative_models[run], "iterative", f"iteration={iteration}", 1
                key = model, protocol, task, condition, trial
            if key in native:
                _check(protocol, "iterative", "EngDesign duplicate provenance belongs to a refinement copy")
                _check(native[key][1:], (text, grade, path.rsplit("/", 1)[0], iteration), "EngDesign repeated native iteration is an identical record")
                copied_iterations += 1
                aliases[key].append(path)
                continue
            native[key] = path, text, grade, path.rsplit("/", 1)[0], iteration
            aliases[key] = [path]

        tasks = {key[2] for key in native}
        _check(set(tables["items"].raw_item_id), tasks, "EngDesign complete task coverage")
        assets = dict(zip(tables["assets"].asset_id, tables["assets"].data, strict=True))
        used_assets = set()
        image_links = 0
        for item in tables["items"].itertuples():
            prefix = f"tasks/{item.raw_item_id}/"
            prompt = archive.read(members[prefix + "LLM_prompt.txt"]).decode().replace("\r\n", "\n").replace("\r", "\n").strip()
            structure = archive.read(members[prefix + "output_structure.py"]).decode().replace("\r\n", "\n").replace("\r", "\n")
            evaluator = archive.read(members[prefix + "evaluate.py"]).decode().replace("\r\n", "\n").replace("\r", "\n")
            _check(json.loads(item.content), {"prompt": prompt, "output_structure": structure}, "EngDesign complete prompt and response schema")
            criterion = json.loads(item.grading_criterion)
            _check(json.loads(criterion["rule"]), {"interpretation": metadata["grading"]["rule"], "released_evaluator": evaluator}, "EngDesign complete task-specific grading definition")
            spec = metadata["grading"]["verifiers"]["native_pass"]
            expected_spec = {k: v for k, v in spec.items() if k != "source_template"}
            expected_spec["source"] = spec["source_template"].format(task=item.raw_item_id)
            _check(json.loads(json.loads(item.verifier)["spec"]), expected_spec, "EngDesign grading provenance")
            links = []
            for path, member in members.items():
                if not path.startswith(prefix + "images/") or len(path[len(prefix + "images/"):].split("/")) != 1:
                    continue
                if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}:
                    continue
                data = archive.read(member)
                digest = hashlib.sha256(data).hexdigest()
                _check(assets[digest], data, "EngDesign exact input-image bytes")
                used_assets.add(digest)
                links.append({"path": path, "asset_id": digest, "role": "input", "ordinal": len(links) + 1,
                              "media_type": mimetypes.guess_type(path)[0]})
            image_links += len(links)
            _check(json.loads(item.asset_manifest) if isinstance(item.asset_manifest, str) else [], links, "EngDesign image-to-task associations")
        _check(set(assets), used_assets, "EngDesign asset coverage")

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = dict(zip(tables["items"].item_id, tables["items"].raw_item_id, strict=True))
    traces = dict(zip(tables["traces"].response_id, tables["traces"].trace, strict=True))
    _check(set(traces), set(tables["responses"].response_id), "EngDesign complete trace associations")
    actual = Counter()
    for response in tables["responses"].itertuples():
        subject = subjects[response.subject_id]
        _check(subject["harness"], "EngDesign", "EngDesign subject harness")
        effort = subject["reasoning_effort"]
        _check(None if pd.isna(effort) else effort, parameters["reasoning_effort"].get(subject["display_name"]), "EngDesign reported reasoning configuration")
        protocol = _features(subject["subject_features_extra"])["protocol"]
        key = subject["display_name"], protocol, items[response.item_id], response.test_condition, response.trial
        if key not in native:
            raise ValueError("EngDesign subject/task/condition/trial not in source")
        _, text, grade, group, iteration = native[key]
        actual[key + (None if pd.isna(response.response) else response.response,)] += 1
        trace = json.loads(traces[response.response_id])
        expected_trace = {"source_files": aliases[key], "native_record": text}
        for kind in ["responses", "evaluations"]:
            source, section = sections.get((group, iteration, kind), (None, None))
            expected_trace[kind], expected_trace[kind + "_source"] = section, source
        _check(trace, expected_trace, "EngDesign full native record, model output, feedback and source associations")
    expected = Counter({key + (row[2],): 1 for key, row in native.items()})
    _check(actual, expected, "EngDesign every source observation and grade")
    _check(len(subjects), len({key[:2] for key in native}), "EngDesign model/configuration coverage")
    return {"source_responses": len(native), "source_successes": sum(row[2] == 1 for row in native.values()),
            "source_ungraded_observations": sum(row[2] is None for row in native.values()),
            "source_single_answer_trials": sum(key[1] == "single_answer" for key in native),
            "source_iterative_outcomes": sum(key[1] == "iterative" for key in native),
            "source_iterative_chains": len({(key[0], key[2]) for key in native if key[1] == "iterative"}),
            "source_empty_files": empty, "source_open_subset_log_copies": copied_open_logs,
            "source_repeated_iteration_copies": copied_iterations, "source_items": len(tasks),
            "source_configurations": len(subjects), "source_traces": len(traces),
            "source_assets": len(assets), "source_image_links": image_links,
            "source_opaque_matlab_object_representations": opaque_objects}


def _phyblock(directory, tables, metadata):
    """Compare all conversions with the reviewed author's deterministic checker."""
    import ast
    import hashlib
    import re
    import tempfile
    import zipfile

    parameters = metadata["build"]["parameters"]
    archive_path = directory / "raw" / parameters["layout"]["archive"]
    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = dict(zip(tables["traces"].response_id, tables["traces"].trace, strict=True))
    assets = dict(zip(tables["assets"].asset_id, tables["assets"].data, strict=True))
    _check(set(traces), set(tables["responses"].response_id), "PhyBlock complete trace associations")
    expected, prompts, observations, image_data = {}, {}, {}, {}
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=directory) as temp:
        scratch = Path(temp)
        members = {n.split("/", 1)[1]: n for n in archive.namelist() if not n.endswith("/")}
        source = archive.read(members["evaluate_block_construction.py"])
        # Only these reviewed definitions are evaluated, never imports, main(), or
        # model output. A changed upstream grader requires a new source review.
        _check(hashlib.sha256(source).hexdigest(),
               "512171b70c0eefa1ce61b8b39ca5e4a3cd00eb11310dc990cfe2f5e43739ff85",
               "PhyBlock reviewed author checker")
        functions = {"extract_message_content", "are_blocks_equal", "is_place_legal",
                     "validate_block_placement", "validate_block_placement_level1"}
        definitions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in functions]
        _check({n.name for n in definitions}, functions, "PhyBlock author functions")
        author = {"json": json, "Path": Path}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "reviewed_phyblock_checker", "exec"), author)

        goals, candidates = {}, {}
        for name, member in members.items():
            if name.startswith("data/SCENEs_400_Goal_Jsons/") and name.endswith(".json"):
                scene = Path(name).stem
                goals[scene] = json.loads(archive.read(member))
                (scratch / f"{scene}.json").write_bytes(archive.read(member))
                _check([b["order"] for b in goals[scene]["blocks"]],
                       list(range(1, len(goals[scene]["blocks"]) + 1)), "PhyBlock ordered reference blocks")
                candidates[scene] = json.loads(archive.read(members[f"data/SCENEs_400_Cand_Jsons/{scene}_16_cand_blocks.json"]))
                for folder, suffix in [("Goal_Imgs", "_1.png"), ("Cand_Imgs", "_16_cand.png")]:
                    image_path = f"data/SCENEs_400_{folder}/{scene}{suffix}"
                    image_data[image_path] = archive.read(members[image_path])
        for name, member in members.items():
            if not name.startswith("outputs/") or not name.endswith(".json"):
                continue
            _, model, filename = name.split("/")
            scene = Path(filename).stem
            text = archive.read(member).decode("utf-8")
            record = json.loads(text)
            observations[name] = record, text, model, scene
            if "instruction" in record:
                _check(record["image"], [f"SCENEs/{scene}_1.png", f"SCENEs_cand/{scene}_16_cand.png"],
                       "PhyBlock recorded input references")
                prompts[name] = record["instruction"], "native_record"
            else:
                prompts[name] = parameters["instructions"]["paper_definition"], "paper_definition_not_recorded_request"
            message = author["extract_message_content"](record)
            indices = [int(n) for n in re.findall(r"block with index (\d+)", message)]
            selected = [candidates[scene][n - 1] for n in indices if 0 < n <= len(candidates[scene])]
            pred = scratch / "prediction.json"
            pred.write_text(json.dumps({"blocks": selected}))
            level, tp, fp, fn, _, _, f1 = author["validate_block_placement"](scratch / f"{scene}.json", pred)
            expected[name, "pose_constrained"] = {"condition": "pose_constrained", "TP": tp, "FP": fp, "FN": fn, "F1": round(f1, 3)}
            if level == 2 and len(goals[scene]["blocks"]) < 7:
                _, _, tp, fp, fn, _, _, f1 = author["validate_block_placement_level1"](scratch / f"{scene}.json", pred)
                expected[name, "topology_only"] = {"condition": "topology_only", "TP": tp, "FP": fp, "FN": fn, "F1": round(f1, 3)}

    used_assets, actual_items = set(), set()
    for item in items.values():
        scene = item["raw_item_id"]
        features = _features(item["item_features"])
        condition = features["grading_condition"]
        _check(features["scene"], scene, "PhyBlock scene features")
        _check(int(features["source_level"]), goals[scene]["level"], "PhyBlock scene level")
        reference = json.loads(item["grading_criterion"])
        _check(json.loads(reference["reference_answer"]), {"goal": goals[scene], "candidates": candidates[scene]},
               "PhyBlock exact reference geometry and candidate ordering")
        _check(reference["rule"], metadata["grading"]["rule"], "PhyBlock derived-score definition")
        spec = json.loads(json.loads(item["verifier"])["spec"])
        _check(spec, metadata["grading"]["verifiers"][condition], "PhyBlock verifier provenance")
        _check(spec["grade_origin"], "recomputed_from_released_model_output", "PhyBlock explicit grade derivation")
        _check(spec["matching_fields"], ["type", "color", "euler"] if condition == "pose_constrained" else ["type", "color"],
               "PhyBlock grading fields")
        _check(spec["check_dependencies"], condition == "pose_constrained", "PhyBlock dependency rule")
        links = []
        for path in [f"data/SCENEs_400_Goal_Imgs/{scene}_1.png", f"data/SCENEs_400_Cand_Imgs/{scene}_16_cand.png"]:
            digest = hashlib.sha256(image_data[path]).hexdigest()
            used_assets.add(digest)
            _check(assets[digest], image_data[path], "PhyBlock exact task-image bytes")
            links.append({"path": path, "asset_id": digest, "role": "input", "ordinal": len(links) + 1, "media_type": "image/png"})
        _check(json.loads(item["asset_manifest"]), links, "PhyBlock ordered image associations")
        actual_items.add((scene, item["content"], condition, features["prompt_source"]))
    _check(set(assets), used_assets, "PhyBlock asset coverage")
    expected_items, actual = set(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        name, condition = trace["source_file"], response.test_condition
        key = name, condition
        if key not in expected:
            raise ValueError("PhyBlock observation not supported by the released source")
        record, text, model, scene = observations[name]
        subject, item = subjects[response.subject_id], items[response.item_id]
        _check(trace, {"source_file": name, "native_record": text, "derived_grade": expected[key]},
               "PhyBlock complete output and native-checker grade")
        _check(response.response, expected[key]["F1"], "PhyBlock all per-scene F1 grades")
        _check(response.trial, 1, "PhyBlock one released trial per condition")
        _check(subject["display_name"], record.get("model", record.get("model_version", model)), "PhyBlock exact returned model ID")
        subject_features = _features(subject["subject_features_extra"])
        _check(subject_features["native_model_directory"], model, "PhyBlock subject associations")
        _check(subject_features["planning_strategy"], "one_time", "PhyBlock planning protocol")
        _check(subject["harness"], "PhyBlock", "PhyBlock harness")
        _check(None if pd.isna(subject["reasoning_effort"]) else subject["reasoning_effort"],
               "extended" if model == "claude-3-7-thinking" else None, "PhyBlock thinking configuration")
        content, provenance = prompts[name]
        _check(subject_features["prompt_source"], provenance, "PhyBlock subject prompt provenance")
        _check((item["raw_item_id"], item["content"], _features(item["item_features"])["grading_condition"],
                _features(item["item_features"])["prompt_source"]), (scene, content, condition, provenance),
               "PhyBlock exact prompt/scene/condition association")
        expected_items.add((scene, content, condition, provenance))
        actual[key] += 1
    _check(actual, Counter({key: 1 for key in expected}), "PhyBlock exhaustive observation coverage")
    _check(actual_items, expected_items, "PhyBlock prompt/protocol item coverage")
    _check(len(subjects), len({row[2] for row in observations.values()}), "PhyBlock subject coverage")
    return {"source_responses": len(expected), "source_native_outputs": len(observations),
            "source_pose_constrained_grades": sum(key[1] == "pose_constrained" for key in expected),
            "source_topology_only_grades": sum(key[1] == "topology_only" for key in expected),
            "source_scenes": len(goals), "source_items": len(items), "source_configurations": len(subjects),
            "source_recorded_prompts": sum(v[1] == "native_record" for v in prompts.values()),
            "source_unrecorded_requests": sum(v[1] != "native_record" for v in prompts.values()),
            "source_traces": len(traces), "source_assets": len(assets), "source_image_links": len(items) * 2}


def _scigym(directory, tables, metadata):
    """Reconcile every native trajectory using a separate XML reader and set algebra."""
    import math
    import re
    from xml.dom import minidom

    import pyarrow.parquet as pq

    raw = directory / "raw"
    records = pq.read_table(raw / "results.parquet").to_pylist()
    bank_rows = pq.read_table(raw / "tasks.parquet").to_pylist()
    bank = {row["folder_name"]: row for row in bank_rows}
    _check(len(bank), len(bank_rows), "SciGym unique task-bank IDs")
    native, positions = {}, {}
    for index, row in enumerate(records):
        key = row["model_name"], row["biomodel_id"]
        if key in native:
            _check(row, native[key], "SciGym exact copies, not unrecorded independent trials")
        native[key] = row
        positions.setdefault(key, []).append(index)
    _check({key[1] for key in native}, set(bank), "SciGym complete evaluated task bank")

    signatures = {}
    def reaction_sets(xml):
        if xml in signatures:
            return signatures[xml]
        document = minidom.parseString(xml)
        model = document.getElementsByTagNameNS("*", "model")
        _check(len(model), 1, "SciGym one SBML model per document")
        containers = [node for node in model[0].childNodes if node.localName == "listOfReactions"]
        with_modifiers, without_modifiers = set(), set()
        if containers:
            _check(len(containers), 1, "SciGym one reaction list")
            for reaction in containers[0].childNodes:
                if reaction.localName != "reaction":
                    continue
                roles = []
                for name, tag in [("listOfReactants", "speciesReference"), ("listOfProducts", "speciesReference"),
                                  ("listOfModifiers", "modifierSpeciesReference")]:
                    groups = [child for child in reaction.childNodes if child.localName == name]
                    species = frozenset(child.getAttribute("species") for group in groups for child in group.childNodes
                                        if child.localName == tag)
                    roles.append(species)
                with_modifiers.add(tuple(roles))
                without_modifiers.add(tuple(roles[:2]))
        document.unlink()
        signatures[xml] = {"rpm": with_modifiers, "rp": without_modifiers}
        return signatures[xml]

    expected, prompts, iterations = {}, {}, {}
    for key, row in native.items():
        task = bank[key[1]]
        marker = "Format your response according to the instructions in the system message."
        prefix, separator, _ = row["chat_history"].partition(marker)
        _check(separator, marker, "SciGym initial prompt boundary")
        prompts[key] = prefix + separator
        if task["partial"] not in prompts[key]:
            raise ValueError("SciGym initial input does not contain the exact task-bank model")
        iterations[key] = int(re.search(r"## Max iterations\s+(\d+)", prompts[key]).group(1))
        true, partial, predicted = [reaction_sets(xml) for xml in [task["truth_xml"], task["partial"], row["final_model"]]]
        for condition in ["rp", "rpm"]:
            missing, added = true[condition] - partial[condition], predicted[condition] - partial[condition]
            found = len(missing.intersection(added))
            precision = found / len(added) if added else 0.0
            recall = found / len(missing) if missing else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            for statistic, value in [("precision", precision), ("recall", recall), ("f1", f1)]:
                expected[key + (condition + "_" + statistic,)] = value
        expected[key + ("ste",)] = None

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = dict(zip(tables["traces"].response_id, tables["traces"].trace, strict=True))
    _check(set(traces), set(tables["responses"].response_id), "SciGym complete trace associations")
    _check(len(tables.get("assets", [])), 0, "SciGym text/XML-only records")
    item_metrics = {}
    expected_items = set()
    for identifier, item in items.items():
        task_id = item["raw_item_id"]
        task = bank[task_id]
        verifier = json.loads(json.loads(item["verifier"])["spec"])
        metric = verifier["native_metric"]
        item_metrics[identifier] = metric
        specification = metadata["grading"]["verifiers"]["trajectory_error" if metric == "ste" else "reaction_recovery"]
        _check(verifier, {**specification, "native_metric": metric}, "SciGym metric-specific verifier")
        criterion = json.loads(item["grading_criterion"])
        _check(json.loads(criterion["reference_answer"]), {name: task[name] for name in ["truth_xml", "partial", "truth_sedml"]},
               "SciGym complete hidden model, starting model and simulation specification")
        _check(criterion["rule"], metadata["grading"]["rule"], "SciGym grading rule")
        _check(criterion["response_scale"], {"kind": "interval", "min": 0.0, "max": 1.0,
                                            "direction": "lower_is_better" if metric == "ste" else "higher_is_better"},
               "SciGym explicit metric domains and directions")
        _check(_features(item["item_features"]), {"split": "small"}, "SciGym item split")
        expected_items.add((task_id, metric))
    _check(expected_items, {(key[1], key[2]) for key in expected}, "SciGym metric/task item coverage")
    _check(len(items), len(expected_items), "SciGym unique metric/task items")

    actual = Counter()
    for response in tables["responses"].itertuples():
        subject, item = subjects[response.subject_id], items[response.item_id]
        key = subject["display_name"], item["raw_item_id"]
        metric = item_metrics[response.item_id]
        if key + (metric,) not in expected:
            raise ValueError("SciGym result has no matching native trajectory and metric")
        row = native[key]
        _check(item["content"], prompts[key], "SciGym full initial user prompt")
        _check(subject["harness"], "SciGym", "SciGym subject harness")
        features = _features(subject["subject_features_extra"])
        _check(int(features["max_iterations"]), iterations[key], "SciGym recorded action allowance")
        _check(features["interaction"], "iterative", "SciGym iterative configuration")
        _check(response.trial, 1, "SciGym copied records are not separate trials")
        if not pd.isna(response.test_condition):
            raise ValueError("SciGym export does not identify an additional trial condition")
        wanted = expected[key + (metric,)]
        if wanted is None:
            if not pd.isna(response.response):
                raise ValueError("SciGym unreleased trajectory error must remain null")
        elif not math.isclose(response.response, wanted, rel_tol=0, abs_tol=1e-15):
            raise ValueError("SciGym reaction score differs from the released final model")
        trace = json.loads(traces[response.response_id])
        _check(trace, {"source_rows": positions[key], "chat_history": row["chat_history"], "final_model": row["final_model"],
                       "native_metric": metric, "grade_origin": "not_released_not_rerun" if metric == "ste"
                                                                else "recomputed_from_released_final_model"},
               "SciGym complete native trajectory, output, aliases and grade provenance")
        actual[key + (metric,)] += 1
    _check(actual, Counter({key: 1 for key in expected}), "SciGym exhaustive trajectory/metric coverage")
    _check(len(subjects), len({key[0] for key in native}), "SciGym model coverage")
    return {"source_responses": len(expected), "source_native_trajectories": len(native),
            "source_export_rows": len(records), "source_duplicate_copies": len(records) - len(native),
            "source_tasks": len(bank), "source_items": len(items), "source_configurations": len(subjects),
            "source_derived_reaction_scores": sum(value is not None for value in expected.values()),
            "source_ungraded_observations": sum(value is None for value in expected.values()),
            "source_traces": len(traces), "source_longest_native_history": max(len(row["chat_history"]) for row in native.values())}


def _legal_rag(directory, tables, metadata):
    """Reconcile native answers, retrieved text and each distinct grading protocol."""
    import hashlib

    raw = directory / "raw"
    layout = metadata["build"]["parameters"]["paths"]
    native = _jsonl(raw / "results.jsonl")
    questions = {str(row["id"]): row for row in _jsonl(raw / layout["questions"])}
    corpus_bytes = (raw / layout["corpus"]).read_bytes()
    corpus = {row["id"]: row for row in _jsonl(raw / layout["corpus"])}
    assets = tables["assets"].set_index("asset_id").to_dict("index")
    digest = hashlib.sha256(corpus_bytes).hexdigest()
    _check(set(assets), {digest}, "Legal RAG complete corpus asset")
    _check(assets[digest]["data"], corpus_bytes, "Legal RAG unmodified corpus bytes")
    for source in native:
        question = questions[str(source["question_id"])]
        for observed, expected in [("question", "question"), ("gold_answer", "answer"), ("gold_id", "relevant_passage_id")]:
            _check(source[observed], question[expected], "Legal RAG original question/reference")
        _check(len(source["context"]), 5, "Legal RAG five retrieved documents")
        for document in source["context"]:
            reference = corpus[document["metadata"]["id"]]
            _check(document["page_content"], reference["text"], "Legal RAG retrieved passage text")
            _check(document["metadata"]["chunk_title"], reference["title"], "Legal RAG passage title")
            _check(document["metadata"]["footnotes"], reference["footnotes"] or "", "Legal RAG passage footnotes")
        retrieved = any(source["gold_id"] in doc["metadata"].get("id", "") for doc in source["context"])
        _check(source["gold_id_in_context"], retrieved, "Legal RAG native retrieval predicate")

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    models = {"gemini3": ("Gemini 3.1 Pro", "gemini-3.1-pro-preview"), "gpt52": ("GPT-5.2", "gpt-5.2")}
    embeddings = {"kanon2": "kanon-2-embedder", "te3l": "text-embedding-3-large", "gemini001": "gemini-embedding-001"}
    actual, successes = Counter(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        index = trace["source_row"]
        source = native[index]
        _check(trace, {"source_file": "results.jsonl", "source_row": index, "record": source}, "Legal RAG complete source record")
        subject, item = subjects[response.subject_id], items[response.item_id]
        label, model_id = models[source["generative_model"]]
        _check(subject["display_name"], label, "Legal RAG generator")
        _check(subject["harness"], "Legal RAG Bench (LangChain FAISS)", "Legal RAG pipeline harness")
        _check(_features(subject["subject_features_extra"]), {
            "embedding_model": embeddings[source["embedding_model"]], "generator_identifier": model_id,
            "retrieval_k": "5"}, "Legal RAG pipeline configuration")
        _check(item["content"], source["question"], "Legal RAG task input")
        criterion = json.loads(item["grading_criterion"])
        rule = json.loads(criterion["rule"])
        kind = rule["criterion"]
        _check(rule, {"criterion": kind, "rule": metadata["build"]["parameters"]["criteria"][kind],
                      "relevant_passage_id": source["gold_id"]}, "Legal RAG criterion and gold passage")
        _check(criterion["reference_answer"], source["gold_answer"], "Legal RAG expert answer")
        _check(item["raw_item_id"], str(source["question_id"]) + ":" + kind, "Legal RAG criterion-specific item")
        verifier = json.loads(item["verifier"])
        expected_spec = metadata["grading"]["verifiers"]["retrieval"] if kind == "retrieval" else {
            **metadata["grading"]["verifiers"]["llm"], "criterion": kind}
        _check(json.loads(verifier["spec"]), expected_spec, "Legal RAG grading instrument")
        if kind != "retrieval":
            _check(verifier["judged_by"], "llm", "Legal RAG judgment type")
        expected = source["gold_id_in_context"] if kind == "retrieval" else source["judge_verdict"][{
            "correctness": "correct", "groundedness": "grounded"}[kind]]
        _check(response.response, float(expected), "Legal RAG unchanged native verdict")
        _check(response.trial, 1, "Legal RAG one recorded answer per pipeline/question")
        _check(json.loads(item["asset_manifest"]), [{"asset_id": digest, "path": "corpus.jsonl",
            "media_type": "application/x-ndjson", "role": "retrieval_corpus", "ordinal": 1}], "Legal RAG corpus association")
        actual[index, kind] += 1
        successes[kind] += int(expected)
    _check(actual, Counter({(index, kind): 1 for index in range(len(native))
                           for kind in ["correctness", "groundedness", "retrieval"]}), "Legal RAG every grade exactly once")
    _check(len(items), 3 * len(questions), "Legal RAG item/protocol coverage")
    _check(len(subjects), len({row["model_name"] for row in native}), "Legal RAG pipeline coverage")
    _check(len(traces), 3 * len(native), "Legal RAG trace coverage")
    return {"source_responses": len(actual), "source_answers": len(native), "source_questions": len(questions),
            "source_items": len(items), "source_configurations": len(subjects), "source_traces": len(traces),
            "source_retrieved_passages": sum(len(row["context"]) for row in native),
            "source_corpus_passages": len(corpus), "criterion_successes": dict(successes)}


def _advprompter(directory, tables, metadata):
    """Compare every native JSON record and its two different prompt roles."""
    raw = directory / "raw"
    sources = {
        "advprompter_vicuna-7b_LlamaGuard3.json": "none",
        "safedecoding__advprompter_vicuna-7b_LlamaGuard3.json": "safedecoding",
    }
    native = {}
    judgments = Counter()
    for path, defense in sources.items():
        records = json.loads((raw / path).read_text())
        for index, record in enumerate(records):
            if type(record["final_score"]) is not int or record["final_score"] not in (0, 1):
                raise ValueError("AdvPrompter unexpected released judgment")
            native[path, index] = record
            judgments[defense, record["final_score"]] += 1

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    _check(len(subjects), 2, "AdvPrompter defense configurations")
    _check(len(items), len({(r["query"], r["final_query"]) for r in native.values()}), "AdvPrompter complete prompts")
    _check(len(traces), len(native), "AdvPrompter full trace coverage")
    actual = Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_file"], trace["source_row"]
        _check(key in native, True, "AdvPrompter source position")
        record = native[key]
        _check(trace, {**record, "source_file": key[0], "source_row": key[1]}, "AdvPrompter every native field")
        subject, item = subjects[response.subject_id], items[response.item_id]
        _check(subject["display_name"], "Vicuna-7B-v1.5", "AdvPrompter target model")
        _check(_features(subject["subject_features_extra"]), {"defense": sources[key[0]]}, "AdvPrompter defense identity")
        _check(item["content"], record["final_query"], "AdvPrompter actual target input")
        _check(item["raw_item_id"], f"harmbench_50:{key[1]}", "AdvPrompter source item position")
        criterion = json.loads(item["grading_criterion"])
        _check(criterion.get("reference_answer"), None, "AdvPrompter attacker prefix is not a reference")
        _check(json.loads(criterion["rule"]), {"rule": metadata["grading"]["rule"], "judge_query": record["query"]}, "AdvPrompter judge input")
        verifier = json.loads(item["verifier"])
        _check(verifier["judged_by"], "llm", "AdvPrompter verifier type")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["llama_guard"], "AdvPrompter verifier definition")
        _check(response.response, record["final_score"], "AdvPrompter unchanged native grade")
        _check(response.interactors, "attacker=AdvPrompter", "AdvPrompter attack method")
        _check(response.trial, 1, "AdvPrompter observed attempt count")
        actual[key] += 1
    _check(actual, Counter({key: 1 for key in native}), "AdvPrompter every source observation exactly once")
    scale = json.loads(tables["benchmarks"].iloc[0].response_scale)
    _check(scale["direction"], "lower_is_better", "AdvPrompter subject safety direction")
    return {"source_responses": len(native), "source_items": len(items), "source_configurations": len(subjects),
            "source_unsafe_judgments": sum(r["final_score"] for r in native.values()),
            "source_traces": len(traces), "unsafe_by_defense": {d: judgments[d, 1] for d in sources.values()}}


def _nester(directory, tables, metadata):
    """Read XLSX XML independently of pandas/openpyxl and check every original cell."""
    import xml.etree.ElementTree as ET
    from zipfile import ZipFile

    models = {"gpt4.xlsx": "GPT-4", "gpt4-o.xlsx": "GPT-4o", "gpt4o-mini.xlsx": "GPT-4o mini"}
    columns = {"correctness": "Correctness", "explainability": "explainability ",
               "factuality": "factuality", "consistency": "consistency "}
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    native, clean = {}, {}
    annotation = metadata["build"]["parameters"]["annotation"]
    for filename in models:
        with ZipFile(directory / "raw" / filename) as archive:
            strings = ["".join(s.itertext()) for s in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        _check(sheet.findall(".//m:f", ns), [], "NESTER no unevaluated spreadsheet formulas")
        headers = None
        for row in sheet.findall("m:sheetData/m:row", ns):
            cells = dict.fromkeys("ABCDEFGH", "")
            for cell in row:
                column = cell.attrib["r"].rstrip("0123456789")
                value = cell.find("m:v", ns)
                if value is not None:
                    value = strings[int(value.text)] if cell.get("t") == "s" else int(value.text)
                    cells[column] = value
            if row.attrib["r"] == "1":
                headers = [v if v else f"Unnamed: {i}" for i, v in enumerate(cells.values())]
                _check(headers, ["id", "prompt", "gpt4", *columns.values(), "Unnamed: 7"], "NESTER original headers")
                continue
            record = dict(zip(headers, cells.values(), strict=True))
            key = filename, int(row.attrib["r"])
            native[key] = record
            if annotation["marker"] not in record["prompt"]:
                if record["id"] in clean:
                    _check(clean[record["id"]], record["prompt"], "NESTER clean prompts agree across models")
                clean[record["id"]] = record["prompt"]

    reconstructed = [key for key, row in native.items() if annotation["marker"] in row["prompt"]]
    _check(len(reconstructed), 1, "NESTER one documented prompt annotation")
    affected = reconstructed[0]
    _check((affected[0], native[affected]["id"]), (annotation["source_file"], annotation["affected_id"]),
           "NESTER documented annotated source row")
    _check(native[affected]["prompt"].split(annotation["marker"], 1)[0].rstrip(),
           clean[native[affected]["id"]].rstrip(), "NESTER annotated prefix matches independently released input")
    _check(native[affected]["gpt4"], "", "NESTER unavailable primary output remains blank")
    trials, counts = {}, Counter()
    for key, row in native.items():
        counts[key[0], clean[row["id"]]] += 1
        trials[key] = counts[key[0], clean[row["id"]]]
        for column in columns.values():
            _check(type(row[column]) is int and row[column] in (0, 1), True, "NESTER native binary annotation")

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    actual, successes = Counter(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_file"], trace["source_excel_row"]
        _check(key in native, True, "NESTER source row association")
        record = native[key]
        _check(trace, {"source_file": key[0], "source_excel_row": key[1], "record": record,
                       "input_reconstructed": key == affected}, "NESTER every native cell, blank output and annotation")
        subject, item = subjects[response.subject_id], items[response.item_id]
        _check(subject["display_name"], models[key[0]], "NESTER model attribution")
        _check(item["content"], clean[record["id"]], "NESTER complete unannotated task input")
        criterion = json.loads(item["grading_criterion"])
        rule = json.loads(criterion["rule"])
        kind = rule["criterion"]
        _check(kind in columns, True, "NESTER known grading criterion")
        _check(criterion.get("reference_answer"), None, "NESTER unlabeled extra cells are not reference answers")
        _check(rule, {"criterion": kind, "rule": metadata["grading"]["verifiers"]["human"]["criteria"][kind]},
               "NESTER criterion-specific grading rule")
        source_id, suffix = item["raw_item_id"].rsplit(":", 1)
        _check((clean[source_id], suffix), (item["content"], kind), "NESTER canonical item source identity")
        verifier = json.loads(item["verifier"])
        _check(verifier["judged_by"], "human", "NESTER human verifier")
        _check(json.loads(verifier["spec"]), {**metadata["grading"]["verifiers"]["human"], "criterion": kind},
               "NESTER grading instrument")
        _check(response.response, float(record[columns[kind]]), "NESTER unchanged human grade")
        _check(response.trial, trials[key], "NESTER distinct recorded rows for repeated prompts")
        actual[key, kind] += 1
        successes[models[key[0]], kind] += record[columns[kind]]
    _check(actual, Counter({(key, kind): 1 for key in native for kind in columns}), "NESTER every grade exactly once")
    _check(len(subjects), len(models), "NESTER model coverage")
    _check(len(items), len(set(clean.values())) * len(columns), "NESTER prompt/protocol coverage")
    _check(len(traces), len(native) * len(columns), "NESTER complete trace coverage")
    return {"source_responses": len(actual), "source_rows": len(native), "source_ids": len(clean),
            "source_prompts": len(set(clean.values())), "source_items": len(items), "source_configurations": len(subjects),
            "source_traces": len(traces), "source_annotated_prompts": len(reconstructed),
            "source_blank_outputs": sum(row["gpt4"] == "" for row in native.values()),
            "source_unlabeled_extra_cells": sum(row["Unnamed: 7"] != "" for row in native.values()),
            "criterion_successes": {f"{model}/{kind}": successes[model, kind]
                                    for model in models.values() for kind in columns}}


def _engibench(directory, tables, metadata):
    """Reconcile saved notebook outputs against literal task inputs, without execution."""
    import ast
    import math

    root = directory / "raw" / metadata["build"]["parameters"]["layout"]["root"]
    models = {"example_easy_model.ipynb": "EngiOpt Conditional GAN",
              "example_hard_model.ipynb": "EngiOpt Conditional Diffusion"}
    headings = {"Initial optimality gaps": "initial_gap", "Cumulative optimality gaps": "cumulative_gap",
                "Final optimality gaps": "final_gap"}
    native, conditions, expected = {}, {}, {}
    for filename in models:
        notebook = json.loads((root / filename).read_text())
        native[filename] = notebook
        for index, cell in enumerate(notebook["cells"]):
            code = "".join(cell.get("source", []))
            if code.startswith("n_samples = "):
                assignment = next(node for node in ast.parse(code).body if isinstance(node, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == "conditions" for t in node.targets))
                _check(filename in conditions, False, "EngiBench one explicit task batch")
                conditions[filename] = ast.literal_eval(assignment.value)
            for output in cell.get("outputs", []):
                for line in "".join(output.get("text", [])).splitlines():
                    line = line.removeprefix("\x1b[96m").removesuffix("\x1b[0m")
                    heading, separator, payload = line.partition(": ")
                    if heading not in headings:
                        continue
                    _check(separator, ": ", "EngiBench score separator")
                    values = ast.literal_eval(payload)
                    for position, grade in enumerate(values):
                        _check(type(grade) in (float, int) and math.isfinite(grade), True, "EngiBench finite printed grade")
                        key = filename, headings[heading], position
                        _check(key in expected, False, "EngiBench nonduplicated native output")
                        expected[key] = index, float(grade)
    _check(conditions["example_easy_model.ipynb"], conditions["example_hard_model.ipynb"], "EngiBench shared design conditions")
    for filename, criterion, position in expected:
        _check(position < len(conditions[filename]), True, "EngiBench output position maps to a recorded condition")

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    actual = Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        filename, position = trace["source_file"], trace["design_index"]
        item = items[response.item_id]
        criterion = json.loads(item["grading_criterion"])
        rule = json.loads(criterion["rule"])
        kind = rule["criterion"]
        key = filename, kind, position
        _check(key in expected, True, "EngiBench observed source tuple")
        cell, grade = expected[key]
        _check(trace, {"source_file": filename, "source_cell": cell, "design_index": position,
                       "notebook": native[filename]}, "EngiBench complete native notebook, outputs and rendering data")
        _check(subjects[response.subject_id]["display_name"], models[filename], "EngiBench generative method")
        _check(subjects[response.subject_id]["harness"], "EngiOpt Beams2D v0", "EngiBench recorded problem version")
        _check(_features(subjects[response.subject_id]["subject_features_extra"]), {"source_run": filename}, "EngiBench source run")
        _check(json.loads(item["content"]), {"problem": {"name": "Beams2D", "version": "v0"},
                                            "conditions": conditions[filename][position]}, "EngiBench exact numerical task conditions")
        _check(item["raw_item_id"], f"beams2d:example:{position}:{kind}", "EngiBench item identity")
        _check(criterion.get("reference_answer"), None, "EngiBench no invented reference design")
        _check(rule, {"criterion": kind, "rule": metadata["build"]["parameters"]["rules"][kind]}, "EngiBench protocol definition")
        verifier = json.loads(item["verifier"])
        _check(verifier["class"], "exact_matcher", "EngiBench numerical grading")
        _check(json.loads(verifier["spec"]), {**metadata["grading"]["verifiers"]["native_gap"], "criterion": kind},
               "EngiBench grading instrument")
        _check(response.response, grade, "EngiBench unchanged signed native grade")
        _check(response.trial, 1, "EngiBench one recorded demonstration per method/condition")
        actual[key] += 1
    _check(actual, Counter({key: 1 for key in expected}), "EngiBench every native score exactly once")
    _check(len(items), len(conditions[next(iter(models))]) * len(headings), "EngiBench item coverage")
    _check(len(subjects), len(models), "EngiBench method coverage")
    _check(len(traces), len(expected), "EngiBench full trace coverage")
    _check(json.loads(tables["benchmarks"].iloc[0].response_scale),
           {"kind": "interval", "min": None, "max": None, "direction": "lower_is_better"}, "EngiBench unbounded signed scale")
    return {"source_responses": len(expected), "source_items": len(items), "source_conditions": len(conditions[next(iter(models))]),
            "source_designs": sum(map(len, conditions.values())), "source_configurations": len(subjects),
            "source_traces": len(traces), "source_negative_scores": sum(grade < 0 for _, grade in expected.values())}


def _llmail_source_records(directory, metadata):
    """Read original JSON lines independently, retaining exact input identity and record digests."""
    parameters = metadata["build"]["parameters"]
    raw = directory / "raw"
    root = raw / parameters["layout"]["root"]
    levels = json.loads((root / "data/levels_descriptions.json").read_text())
    scenarios = json.loads((root / "data/scenarios.json").read_text())
    objective_names = {"email.retrieved", "defense.undetected", "exfil.sent", "exfil.destination", "exfil.content"}
    expected, inputs, trials, counts, jobs, targets = {}, {}, Counter(), Counter(), set(), set()
    for phase in ("phase1", "phase2"):
        path = root / f"data/raw_submissions_{phase}.jsonl"
        filename = str(path.relative_to(raw))
        with path.open() as stream:
            for position, line in enumerate(stream):
                row = json.loads(line)
                job = row["job_id"]
                _check(job in jobs, False, "LLMail unique native job identity")
                _check(row["RowKey"], job, "LLMail native row/job association")
                jobs.add(job)
                counts["source_jobs"] += 1
                flags = json.loads(row["objectives"])
                if row["started_time"] is None:
                    _check((flags, row["completed_time"], row["output"]), ({}, None, None), "LLMail unstarted job")
                    counts["source_unstarted_jobs"] += 1
                    continue
                if row["output"] == "Scenario not found.":
                    _check(flags, {}, "LLMail nonexistent scenario has no verdict")
                    counts["source_nonexistent_scenarios"] += 1
                    continue
                if flags:
                    _check(set(flags), objective_names, "LLMail five native objectives")
                    _check(all(type(value) is bool for value in flags.values()), True, "LLMail boolean native flags")
                    grade = float(all(flags.values()))
                    _check(row["output"].startswith("Processed example output for prompt:"), True, "LLMail output is a native echo")
                    counts["source_graded"] += 1
                    counts["source_attack_successes"] += int(grade)
                else:
                    _check(row["output"].startswith("Job failed to process after multiple attempts."), True,
                           "LLMail empty verdict denotes a processing failure")
                    grade = None
                    counts["source_ungraded_observations"] += 1
                level = row["scenario"]
                _check(len(level) == 7 and level.startswith("level") and level[5] in "1234", True, "LLMail scenario syntax")
                scenario, letter = f"scenario_{level[5]}", level[-1]
                target = phase, letter
                targets.add(target)
                if letter not in levels[phase]:
                    _check((phase, letter, job), ("phase2", "h", "2216014b-fcb2-420a-ae98-26a042f75d1a"),
                           "LLMail documented unresolved legacy target")
                    counts["source_unresolved_configuration_observations"] += 1
                key = scenario, row["subject"], row["body"]
                inputs.setdefault(key, job)
                trials[target, key] += 1
                if "\x00" in row["subject"] or "\x00" in row["body"]:
                    counts["source_records_with_nul"] += 1
                expected[filename, position] = (target, key, grade, trials[target, key],
                                                _digest(json.dumps(row, ensure_ascii=True, sort_keys=True)))
    counts.update(source_responses=len(expected), source_items=len(inputs), source_configurations=len(targets),
                  source_traces=len(expected), source_assets=len(scenarios))
    return expected, inputs, levels, scenarios, dict(counts)


def _llmail_inject(directory, tables, metadata, source_records=None):
    """Compare every released attempt, including NUL-bearing inputs, failures and repetitions."""
    expected, inputs, levels, scenarios, counts = (
        _llmail_source_records(directory, metadata) if source_records is None else source_records)
    subjects, items = {}, {}
    for subject in tables["subjects"].to_dict("records"):
        features = _features(subject["subject_features_extra"])
        phase, letter = features["challenge_phase"], features["level_letter"]
        if letter in levels[phase]:
            family, _, defense = levels[phase][letter].partition(" with ")
            canonical = {"Phi-3": "Microsoft Phi-3 Medium", "GPT": "OpenAI GPT-4o mini"}[family]
            model_id = {"Phi-3": "microsoft/Phi-3-medium-128k-instruct", "GPT": "gpt-4o-mini"}[family]
            _check(subject["normalized_name"], canonical, "LLMail underlying model")
            _check(features, {"challenge_phase": phase, "level_letter": letter,
                              "defense": defense, "model_identifier": model_id}, "LLMail phase and defense identity")
        else:
            _check((phase, letter), ("phase2", "h"), "LLMail opaque legacy subject")
            _check(subject["display_name"], "LLMail-Inject legacy target h (configuration unreported)", "LLMail unresolved target label")
            _check(pd.isna(subject["normalized_name"]), True, "LLMail no inferred legacy model")
            _check(features, {"challenge_phase": phase, "level_letter": letter, "configuration_status": "unreported"},
                   "LLMail unresolved deployment settings")
        _check(subject["harness"], "LLMail-Inject", "LLMail harness identity")
        subjects[subject["subject_id"]] = phase, letter
    _check(len(set(subjects.values())), counts["source_configurations"], "LLMail complete distinct target configurations")
    _check(len(subjects), counts["source_configurations"], "LLMail subject coverage")

    assets = tables["assets"].set_index("asset_id").data.to_dict()
    for item in tables["items"].itertuples():
        content = json.loads(item.content)
        key = content["scenario_key"], content["email_subject"], content["email_body"]
        _check(key in inputs, True, "LLMail exact email/scenario input")
        scenario = scenarios[key[0]]
        _check(content, {"scenario_key": key[0], "user_query": scenario["user_query"],
                         "released_insertion_position": scenario["position"], "email_subject": key[1], "email_body": key[2]},
               "LLMail complete source stimulus")
        _check(item.raw_item_id, inputs[key], "LLMail first native job as item alias")
        criterion = json.loads(item.grading_criterion)
        _check(criterion.get("reference_answer"), None, "LLMail no invented reference answer")
        _check(json.loads(criterion["rule"]), {"attack_objective": scenario["task"], "success_rule": metadata["grading"]["rule"]},
               "LLMail scenario-specific grading rule")
        verifier = json.loads(item.verifier)
        _check(verifier["class"], "exact_matcher", "LLMail released objective grading")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["objectives"], "LLMail verifier provenance")
        manifest = json.loads(item.asset_manifest)
        _check(len(manifest), 1, "LLMail released corpus attachment")
        asset = manifest[0]
        _check({k: v for k, v in asset.items() if k != "asset_id"},
               {"path": key[0] + "_background_emails.json", "role": "released_scenario_corpus",
                "media_type": "application/json", "ordinal": 1}, "LLMail attachment describes a pool, not a per-attempt retrieval")
        _check(json.loads(assets[asset["asset_id"]]), scenario["emails"], "LLMail every released background email")
        items[item.item_id] = key
    _check(len(items), counts["source_items"], "LLMail distinct input coverage")
    _check(set(items.values()), set(inputs), "LLMail complete input identity including control characters")
    _check(len(assets), counts["source_assets"], "LLMail released corpus count")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    actual = Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        _check(set(trace), {"source_file", "source_row", "record"}, "LLMail native trace fields")
        key = trace["source_file"], trace["source_row"]
        _check(key in expected, True, "LLMail attempted source record association")
        grade = None if pd.isna(response.response) else response.response
        observed = (subjects[response.subject_id], items[response.item_id], grade, response.trial,
                    _digest(json.dumps(trace["record"], ensure_ascii=True, sort_keys=True)))
        _check(observed, expected[key], "LLMail target, exact input, grade, trial and complete native record")
        actual[key] += 1
    _check(actual, Counter({key: 1 for key in expected}), "LLMail every eligible job exactly once")
    _check(len(traces), counts["source_traces"], "LLMail full trace coverage")
    scale = json.loads(tables["benchmarks"].iloc[0].response_scale)
    _check((scale["kind"], scale["values"], scale["direction"]), ("discrete", [0, 1], "lower_is_better"),
           "LLMail attack-success grade direction")
    return counts


def _safeagentbench(directory, tables, metadata):
    """Audit native rejection decisions independently of the tabular transformation."""
    parameters = metadata["build"]["parameters"]
    task_root = directory / "raw" / parameters["layout"]["tasks"]
    result_root = directory / "raw" / parameters["layout"]["results"]
    tasks, aliases = {}, {}
    categories = {"long_horizon": "long_horizon", "unsafe_detailed": "detailed_unsafe",
                  "abstract": "abstract", "safe_detailed": "detailed_safe"}
    for category in categories:
        path = task_root / f"{category}.jsonl"
        for position, task in enumerate(_jsonl(path)):
            instruction = task["instruction"][0] if category == "abstract" else task["instruction"]
            key = f"t{len(tasks)}"
            tasks[key] = {"instruction": instruction, "category": categories[category], "record": task,
                          "file": str(path.relative_to(directory / "raw")), "position": position}
            aliases.setdefault(instruction, key)
    files = {"l2_only": ("ablation_l2_only_results.json", "did_blocked"),
             "p1_full": ("protocol1_full_l3_results.json", "did_blocked"),
             "p1_refined": ("protocol1_refined_l3_results.json", "did_blocked"),
             "p2_baseline": ("protocol2_baseline.json", "rejected"),
             "p2_did": ("protocol2_did.json", "did_blocked"),
             "check4": ("../SafeAgentBench_Check4_Results/sab_check4_results.json", "aug_blocked")}
    expected, native, blocked, confusion = {}, {}, Counter(), {}
    for configuration, (filename, field) in files.items():
        path = (result_root / "SafeAgentBench_Results_v3" / filename).resolve()
        rows = json.loads(path.read_text())
        _check({row["task_id"] for row in rows}, set(tasks), "SafeAgentBench complete source task-ID coverage")
        _check(len(rows), len(tasks), "SafeAgentBench one exported decision per source task")
        native[configuration] = {row["task_id"]: row for row in rows}
        trials, matrix = Counter(), Counter()
        for position, row in enumerate(rows):
            task = tasks[row["task_id"]]
            _check((row["instruction"], row["category"]), (task["instruction"], task["category"]),
                   "SafeAgentBench task and category correspondence")
            _check(type(row[field]), bool, "SafeAgentBench native boolean decision")
            _check(type(row["is_unsafe"]), bool, "SafeAgentBench native author label")
            if configuration == "p2_baseline":
                _check(row["gt_steps"], task["record"].get("step", []), "SafeAgentBench complete reference-step match")
            trials[row["instruction"]] += 1
            blocked[configuration] += int(row[field])
            matrix["tp" if row["is_unsafe"] and row[field] else "fn" if row["is_unsafe"] else "fp" if row[field] else "tn"] += 1
            key = str(path.relative_to(directory / "raw")), position
            expected[key] = (configuration, row["instruction"], float(row[field]), trials[row["instruction"]], {
                "source_file": key[0], "source_row": position, "record": row,
                "task_source_file": task["file"], "task_source_row": task["position"], "task_record": task["record"],
            })
        confusion[configuration] = dict(matrix)
        metric_name = filename.removesuffix("_results.json").removesuffix(".json") + "_metrics.json"
        metric_path = result_root / "SafeAgentBench_Results_v3" / metric_name
        if configuration in {"l2_only", "p1_full", "p1_refined", "p2_did"}:
            published = json.loads(metric_path.read_text())
            _check({key: published[key] for key in ("tp", "fp", "fn", "tn")}, dict(matrix),
                   "SafeAgentBench released confusion-matrix reconciliation")
            _check(published["sources"], dict(Counter(row["detected_by"] for row in rows)),
                   "SafeAgentBench released detection-source counts")
    for key, row in native["check4"].items():
        _check({name: row[name] for name in native["p1_refined"][key]}, native["p1_refined"][key],
               "SafeAgentBench Check4 retains the related base record")
        _check(row["aug_blocked"], row["did_blocked"] or row["check4_harmful"], "SafeAgentBench native post-hoc augmentation")
    for key, row in native["p2_did"].items():
        _check(row["gpt4o_refused"], native["p2_baseline"][key]["rejected"], "SafeAgentBench reused baseline decisions")
    summary = json.loads((result_root / "SafeAgentBench_Check4_Results/sab_check4_summary.json").read_text())
    for configuration, section in (("p1_refined", "base"), ("check4", "augmented")):
        _check({key: summary[section][key] for key in ("tp", "fp", "fn")},
               {key: confusion[configuration][key] for key in ("tp", "fp", "fn")}, "SafeAgentBench Check4 published counts")

    subjects, items = {}, {}
    for row in tables["subjects"].to_dict("records"):
        features = _features(row["subject_features_extra"])
        configuration = features["configuration"]
        _check(configuration in files, True, "SafeAgentBench known system configuration")
        _check(row["display_name"], parameters["subjects"][configuration], "SafeAgentBench released system identity")
        declared = parameters["subject_" + configuration]
        _check(row["harness"], declared["harness"], "SafeAgentBench instruction-level protocol")
        _check(features, {key: value for key, value in declared.items() if key != "harness"},
               "SafeAgentBench related configuration and model provenance")
        subjects[row["subject_id"]] = configuration
    _check(set(subjects.values()), set(files), "SafeAgentBench separate threshold and post-hoc configurations")
    _check(len(subjects), len(files), "SafeAgentBench subject coverage")
    for item in tables["items"].itertuples():
        content = json.loads(item.content)
        instruction = content["instruction"]
        _check(content, {"instruction": instruction}, "SafeAgentBench no inferred simulator context")
        _check(instruction in aliases, True, "SafeAgentBench exact released stimulus")
        _check(item.raw_item_id, aliases[instruction], "SafeAgentBench first source alias")
        criterion = json.loads(item.grading_criterion)
        _check(criterion, {"reference_answer": None, "rule": metadata["grading"]["rule"]}, "SafeAgentBench behavioral measurement rule")
        verifier = json.loads(item.verifier)
        _check(verifier["class"], "exact_matcher", "SafeAgentBench native decision parser")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["native_decision"], "SafeAgentBench field interpretation")
        items[item.item_id] = instruction
    _check(set(items.values()), set(aliases), "SafeAgentBench distinct instruction coverage")
    _check(len(items), len(aliases), "SafeAgentBench deduplicated stimulus count")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    actual = Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_file"], trace["source_row"]
        _check(key in expected, True, "SafeAgentBench observed source row")
        _check((subjects[response.subject_id], items[response.item_id], response.response, response.trial, trace),
               expected[key], "SafeAgentBench every decision, trial, complete result and official task record")
        actual[key] += 1
    _check(actual, Counter({key: 1 for key in expected}), "SafeAgentBench every source decision exactly once")
    _check(len(traces), len(expected), "SafeAgentBench complete native records")
    scale = json.loads(tables["benchmarks"].iloc[0].response_scale)
    _check((scale["kind"], scale["values"], scale["direction"]), ("discrete", [0, 1], "unordered"),
           "SafeAgentBench refusal is not relabeled as correctness")
    labels = {}
    for row in native["p2_baseline"].values():
        labels.setdefault(row["instruction"], set()).add(row["is_unsafe"])
    return {"source_responses": len(expected), "source_items": len(aliases), "source_task_ids": len(tasks),
            "source_configurations": len(subjects), "source_traces": len(traces),
            "source_refusals": sum(blocked.values()), "refusals_by_configuration": dict(blocked),
            "source_full_gpt4o_outputs": sum(bool(row["gpt4o_raw"]) for row in native["p2_baseline"].values()),
            "source_conflicting_instruction_labels": sum(len(values) > 1 for values in labels.values())}


def _critic_discernment_game(directory, tables, metadata):
    """Reconcile every native Arrow record, source condition and published mean."""
    import ast
    import pyarrow.parquet as pq

    parameters = metadata["build"]["parameters"]
    raw = directory / "raw"
    root = raw / parameters["layout"]["root"] / "lt_eval/results"
    native, item_sources, configurations, totals = {}, {}, {}, {}
    for path in sorted(root.glob("*/*/details/details_custom_x7c_*.parquet")):
        profile, decoding = path.relative_to(root).parts[:2]
        dataset = path.name.split("_x7c_")[1]
        summary_path = path.parent.parent / "results" / parameters["summary_files"][profile + "/" + dataset]
        summary = json.loads(summary_path.read_text())
        task = summary["config_tasks"]["custom|" + dataset]
        _check(task["name"], dataset, "CDG summary/task association")
        setting = str(task["generation_size"])
        features = {"checkpoint": profile, "reported_task_generation_size": setting,
                    "reported_model_path": summary["config_general"]["model_name"]}
        configurations[profile, setting] = features
        records = pq.read_table(path).to_pylist()
        grades = []
        for position, record in enumerate(records):
            grade = ast.literal_eval(record["metrics"])["extractive_match"]
            _check(type(grade) in (int, float) and grade in (0, 1), True, "CDG native binary metric")
            _check(isinstance(record["full_prompt"], str) and bool(record["full_prompt"]), True, "CDG complete source prompt")
            _check(isinstance(ast.literal_eval(record["gold"]), list), True, "CDG released reference list")
            key = str(path.relative_to(raw)), position
            native[key] = (record, grade, profile, setting, parameters["temperatures"][decoding],
                           dataset, str(summary_path.relative_to(raw)))
            item_sources.setdefault((record["full_prompt"], record["gold"]), (dataset, position))
            grades.append(grade)
        published = summary["results"]["custom|" + dataset + "|0"]["extractive_match"]
        _check(abs(sum(grades) / len(grades) - published) < 1e-12, True, "CDG published mean equals its native judgments")
        _check(len(records), task["effective_num_docs"], "CDG declared evaluation size")
        totals["/".join([profile, decoding, dataset])] = int(sum(grades))

    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    for subject in subjects.values():
        features = _features(subject["subject_features_extra"])
        key = features["checkpoint"], features["reported_task_generation_size"]
        _check(features, configurations[key], "CDG checkpoint and reported task settings")
        _check(subject["harness"], "Lighteval", "CDG harness")
        _check(subject["display_name"], parameters["models"][key[0]], "CDG source model label")
    _check(len(subjects), len(configurations), "CDG complete distinct configurations")

    items = tables["items"].set_index("item_id").to_dict("index")
    for item in items.values():
        criterion = json.loads(item["grading_criterion"])
        key = item["content"], criterion["reference_answer"]
        _check(key in item_sources, True, "CDG exact full prompt and reference solution")
        dataset, position = item_sources[key]
        _check(item["raw_item_id"], dataset + ":" + str(position), "CDG source row alias")
        _check(_features(item["item_features"]), {"dataset": dataset}, "CDG task family")
        _check(criterion["rule"], metadata["grading"]["rule"], "CDG published grading interpretation")
        verifier = json.loads(item["verifier"])
        _check(verifier["class"], "exact_matcher", "CDG native metric parser")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["native_match"], "CDG grading provenance")
    _check(len(items), len(item_sources), "CDG complete distinct prompts")

    traces = tables["traces"].set_index("response_id").trace.to_dict()
    observed = Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_file"], trace["source_row"]
        _check(key in native, True, "CDG trace refers to a real source observation")
        record, grade, profile, setting, temperature, dataset, summary = native[key]
        _check(trace, {"source_file": key[0], "source_row": key[1], "summary_file": summary, "record": record},
               "CDG complete generation, tokens and extraction record")
        subject = _features(subjects[response.subject_id]["subject_features_extra"])
        _check((subject["checkpoint"], subject["reported_task_generation_size"]), (profile, setting), "CDG observation/configuration association")
        item = items[response.item_id]
        _check((item["content"], json.loads(item["grading_criterion"])["reference_answer"]),
               (record["full_prompt"], record["gold"]), "CDG observation/prompt/reference association")
        _check((response.response, response.trial, response.test_condition),
               (grade, 1, "temperature=" + temperature), "CDG native grade, trial and temperature")
        observed[key] += 1
    _check(observed, Counter({key: 1 for key in native}), "CDG every source observation exactly once")
    _check(len(traces), len(native), "CDG full trace coverage")
    return {"source_responses": len(native), "source_items": len(item_sources),
            "source_subjects": len(configurations), "source_successes": sum(totals.values()),
            "source_traces": len(native), "source_correct_by_export": totals}


def _brace_source_records(directory, metadata):
    """Read original JSON and audio independently of the builder's pandas joins."""
    import hashlib
    import math
    import re

    raw = directory / "raw"
    parameters = metadata["build"]["parameters"]
    layout = parameters["layout"]
    annotations, audio_hashes, original_changes = {}, {}, []
    for dataset, stem in parameters["datasets"].items():
        original = json.loads((raw / layout["original"] / (stem + ".json")).read_text())
        processed = json.loads((raw / layout["processed"] / ("BRACE_" + stem + "_Processed.json")).read_text())
        original_pairs = {(row["file_name"], key): pair for row in original
                          for key, pair in row.items() if key not in {"file_name", "references"}}
        processed_pairs = {(row["file_name"], key): pair for row in processed
                           for key, pair in row.items() if key not in {"file_name", "references"}}
        for records, pairs in ((original, original_pairs), (processed, processed_pairs)):
            _check(len(pairs), sum(len(set(row) - {"file_name", "references"}) for row in records),
                   "BRACE unique annotation source keys")
        _check(set(original_pairs), set(processed_pairs), "BRACE complete processed annotation coverage")
        for (filename, pair_key), pair in processed_pairs.items():
            source = original_pairs[filename, pair_key]
            answer = int(sum(source[-1]) < 0) if dataset.endswith("main") else int(source[2] == "human")
            _check(pair[2], answer, "BRACE original votes or human-caption designation")
            if pair[:2] != source[:2]:
                original_changes.append((dataset, filename, pair_key))
            annotations[dataset, filename, pair_key] = pair, source
        for row in processed:
            logical = parameters["audio_directories"][dataset] + "/" + row["file_name"]
            physical = re.sub(r"[^A-Za-z0-9._/-]", lambda match: f"_x{ord(match[0]):02x}_", layout["audio"] + "/" + logical)
            with (raw / physical).open("rb") as stream:
                audio_hashes[logical] = hashlib.file_digest(stream, "sha256").hexdigest()
    _check(original_changes, [("clotho_hallu", "big-machine-fan.wav", "caption_3")],
           "BRACE documented change in the evaluated caption bank")

    native, subjects, totals, ties = {}, set(), {}, 0
    root = raw / layout["results"]
    for path in sorted(root.glob("*/*/weighted_8.json")):
        dataset, profile = path.relative_to(root).parts[:2]
        published, clips = json.loads(path.read_text())
        successes, denominators = Counter(), Counter()
        observed_pairs = set()
        for position, clip in enumerate(clips["Results"]):
            filename = clip["file_name"]
            for key, record in clip.items():
                if key == "file_name":
                    continue
                _check((dataset, filename, key) in observed_pairs, False, "BRACE no duplicate pair within an export")
                observed_pairs.add((dataset, filename, key))
                annotation, original = annotations[dataset, filename, key]
                _check([record["caption0"], record["caption1"], record["answer"]], annotation,
                       "BRACE native result agrees with the actual evaluated captions")
                _check(all(math.isfinite(record[field]) for field in (
                    "caption0_caf_score", "caption1_caf_score", "caption0_raw_csf_score", "caption1_raw_csf_score")),
                    True, "BRACE released scores are finite")
                category = "HH" if key.startswith("Human-Human") else "HM" if key.startswith("Human-Machine") else "MM"
                for field, variant in parameters["variants"].items():
                    _check(type(record[field]) is int and record[field] in [-1, 0, 1], True, "BRACE native prediction code")
                    grade = int(record[field] == record["answer"])
                    for group in ("Overall", category):
                        successes[variant, group] += grade
                        denominators[variant, group] += 1
                    subjects.add((profile, variant))
                    native[str(path.relative_to(raw)), position, key, field] = (
                        dataset, filename, profile, variant, record, original, annotation, grade)
                    ties += record[field] == -1
        _check(observed_pairs, {key for key in annotations if key[0] == dataset}, "BRACE complete native pair coverage per export")
        for field, variant in parameters["variants"].items():
            label = "CAF" if field == "caf_prediction" else "Raw CAF"
            for group in ("Overall", "HH", "HM", "MM") if dataset.endswith("main") else ("Overall",):
                reported = published["Result_Metric"][label + " " + group + " Accuracy"]
                _check(abs(successes[variant, group] / denominators[variant, group] - reported) < 1e-12,
                       True, "BRACE native decisions reproduce each published accuracy")
            if "Total Pairs Evaluated" in published["Result_Metric"]:
                _check(denominators[variant, "Overall"], published["Result_Metric"]["Total Pairs Evaluated"],
                       "BRACE reported export size")
            totals[dataset + "/" + profile + "/" + variant] = successes[variant, "Overall"]
    return native, annotations, audio_hashes, subjects, {
        "source_responses": len(native), "source_items": len(annotations), "source_subjects": len(subjects),
        "source_traces": len(native), "source_assets": len(set(audio_hashes.values())),
        "source_successes": sum(totals.values()), "source_tie_predictions": ties,
        "source_changed_annotation_pairs": len(original_changes), "source_correct_by_configuration": totals,
    }


def _brace(directory, tables, metadata, source_records=None):
    """Reconcile every native decision, annotation, configuration and audio attachment."""
    import hashlib

    native, annotations, audio_hashes, configurations, counts = (
        _brace_source_records(directory, metadata) if source_records is None else source_records)
    parameters = metadata["build"]["parameters"]
    subjects = {}
    for row in tables["subjects"].itertuples():
        features = _features(row.subject_features_extra)
        profile = features["lalm"] + "_" + features["clap"]
        subjects[row.subject_id] = profile, features["score_variant"]
        _check(features, {"lalm": profile.split("_", 1)[0], "clap": profile.split("_", 1)[1],
                          "score_variant": subjects[row.subject_id][1], "reported_alpha": parameters["subject"]["reported_alpha"]},
               "BRACE complete score configuration")
        _check((row.display_name, row.harness), (parameters["subject"]["name"], parameters["subject"]["harness"]),
               "BRACE published metric identity")
    _check(set(subjects.values()), configurations, "BRACE complete distinct subject configurations")
    _check(len(subjects), len(configurations), "BRACE no duplicate subject configurations")

    aliases = {dataset + "/" + filename + "/" + key: (dataset, filename, key) for dataset, filename, key in annotations}
    items, asset_references = {}, {}
    for row in tables["items"].itertuples():
        _check(row.raw_item_id in aliases, True, "BRACE original audio/pair alias")
        dataset, filename, key = aliases[row.raw_item_id]
        annotation, _ = annotations[dataset, filename, key]
        items[row.item_id] = dataset, filename, key
        _check(json.loads(row.content), {"caption0": annotation[0], "caption1": annotation[1]}, "BRACE exact evaluated captions")
        _check(_features(row.item_features), {"dataset": dataset, "pair_type": key}, "BRACE item subset and pair type")
        _check(json.loads(row.grading_criterion), {"reference_answer": str(annotation[2]), "rule": metadata["grading"]["rule"]},
               "BRACE answer and grading convention")
        verifier = json.loads(row.verifier)
        _check(verifier["class"], "exact_matcher", "BRACE saved decision comparison")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["native_prediction"], "BRACE grading provenance")
        manifest = json.loads(row.asset_manifest)
        logical = parameters["audio_directories"][dataset] + "/" + filename
        _check(len(manifest), 1, "BRACE one audio stimulus per pair")
        asset = manifest[0]
        _check({k: v for k, v in asset.items() if k != "asset_id"},
               {"path": logical, "role": "input", "media_type": "audio/wav", "ordinal": 1}, "BRACE source audio attachment")
        _check(asset_references.get(asset["asset_id"], audio_hashes[logical]), audio_hashes[logical], "BRACE consistent shared audio")
        asset_references[asset["asset_id"]] = audio_hashes[logical]
    _check(set(items.values()), set(annotations), "BRACE every released caption pair")
    _check(len(items), len(annotations), "BRACE distinct caption-pair coverage")

    traces = tables["traces"].set_index("response_id").trace.to_dict()
    observed = Counter()
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace["source_file"], trace["source_clip_row"], trace["pair_key"], trace["prediction_field"]
        _check(key in native, True, "BRACE source observation exists")
        dataset, filename, profile, variant, record, original, annotation, grade = native[key]
        _check(trace, {"source_file": key[0], "source_clip_row": key[1], "pair_key": key[2], "prediction_field": key[3],
                       "record": record, "original_annotation": original, "processed_annotation": annotation},
               "BRACE complete native scores and both annotation versions")
        _check((subjects[row.subject_id], items[row.item_id], row.response, row.trial),
               ((profile, variant), (dataset, filename, key[2]), grade, 1), "BRACE correct source/system/item/grade association")
        _check(pd.isna(row.test_condition) and pd.isna(row.interactors), True, "BRACE no invented run conditions")
        observed[key] += 1
    _check(observed, Counter({key: 1 for key in native}), "BRACE each published variant decision exactly once")
    _check(len(traces), len(native), "BRACE complete trace coverage")
    actual_audio = {row.asset_id: hashlib.sha256(row.data).hexdigest() for row in tables["assets"].itertuples()}
    _check(actual_audio, asset_references, "BRACE exact audio bytes and no unrelated assets")
    _check(len(actual_audio), counts["source_assets"], "BRACE content-deduplicated audio coverage")
    return counts


def _biggen_source_records(directory, metadata):
    """Inspect native Arrow rows without using the builder's melt/explode operations."""
    import pyarrow.parquet as pq

    parameters = metadata["build"]["parameters"]
    raw = directory / "raw"
    root = raw / parameters["layout"]["data"]
    native, definitions, expected, models, totals, counts = {}, {}, {}, set(), Counter(), Counter()
    for path in sorted(root.glob("*.parquet")):
        split = path.name.split("-000")[0]
        _check(split in parameters["splits"], True, "BiGGen declared default result split")
        for position, row in enumerate(pq.read_table(path).to_pylist()):
            generation = row["uuid"]
            _check(generation in native, False, "BiGGen unique released generation identity")
            _check(isinstance(row["response"], str), True, "BiGGen generation text including empty outputs")
            definition = json.dumps({key: row[key] for key in ("system_prompt", "input", "reference_answer", "score_rubric")},
                                    ensure_ascii=True, sort_keys=True)
            native[generation] = row, str(path.relative_to(raw)), position, split, definition
            models.add(row["model_name"])
            for field in parameters["judges"]:
                value = row[field]
                if field == "human_score" and value == -1:
                    counts["source_unreleased_human_ratings"] += 1
                    continue
                if field.startswith("prometheus_") and row["task"] in ("llm_judge_absolute", "llm_judge_relative"):
                    _check(value, None, "BiGGen upstream exclusion of Prometheus on judge-evaluation tasks")
                    counts["source_inapplicable_prometheus_judgments"] += 1
                    continue
                definitions.setdefault((definition, field), row)
                for index, grade in enumerate(value if isinstance(value, list) else [value]):
                    _check(grade is None or grade in (1, 2, 3, 4, 5), True, "BiGGen native rubric category or null")
                    expected[generation, field, index] = grade
                    totals[field + "/" + ("null" if grade is None else str(grade))] += 1
                    counts["source_ungraded_measurements"] += grade is None
    counts.update(source_generations=len(native), source_responses=len(expected), source_traces=len(expected),
                  source_subjects=len(models), source_items=len(definitions),
                  source_empty_outputs=sum(row[0]["response"] == "" for row in native.values()))
    return native, definitions, expected, models, {**counts, "source_rating_histograms": dict(totals)}


def _biggen(directory, tables, metadata, source_records=None):
    """Check the full source-to-table mapping, including repeated and ungraded ratings."""
    native, definitions, expected, model_names, counts = (
        _biggen_source_records(directory, metadata) if source_records is None else source_records)
    parameters = metadata["build"]["parameters"]
    subjects = {}
    for row in tables["subjects"].itertuples():
        subjects[row.subject_id] = row.display_name
        _check(row.harness, "BiGGen-Bench", "BiGGen published evaluation harness")
        _check(_features(row.subject_features_extra), {"model_identifier": row.display_name},
               "BiGGen exact released model identifier, including base/chat and revision variants")
    _check(set(subjects.values()), model_names, "BiGGen complete model panel")
    _check(len(subjects), len(model_names), "BiGGen distinct system coverage")

    judge_fields = {judge: field for field, judge in parameters["judges"].items()}
    items = {}
    for row in tables["items"].itertuples():
        content, criterion, verifier = json.loads(row.content), json.loads(row.grading_criterion), json.loads(row.verifier)
        rule = json.loads(criterion["rule"])
        definition = json.dumps({**content, "reference_answer": criterion["reference_answer"], "score_rubric": rule["rubric"]},
                                ensure_ascii=True, sort_keys=True)
        field = judge_fields[verifier["judge"]]
        key = definition, field
        _check(key in definitions, True, "BiGGen exact system/user instructions, reference and rubric")
        original = definitions[key]
        items[row.item_id] = key
        _check(row.raw_item_id, original["id"] + ":" + field, "BiGGen upstream item and judge alias")
        _check(_features(row.item_features), {"capability": original["capability"], "task": original["task"], "lang": original["language"]},
               "BiGGen released capability, task and language")
        _check(rule["interpretation"], metadata["grading"]["rule"], "BiGGen score interpretation")
        _check((verifier["class"], verifier["judged_by"]), ("judge", "human" if field == "human_score" else "llm"),
               "BiGGen grader type and identity")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["rubric"], "BiGGen grading protocol provenance")
    _check(set(items.values()), set(definitions), "BiGGen every distinct prompt/rubric/judge definition")
    _check(len(items), len(definitions), "BiGGen no lost rubric variants")

    traces = tables["traces"].set_index("response_id").trace.to_dict()
    observed = Counter()
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace["generation_id"], trace["score_field"], trace["rating_index"]
        _check(key in expected, True, "BiGGen published rating position or applicable ungraded observation")
        original, filename, position, split, definition = native[key[0]]
        _check(trace, {"source_file": filename, "source_row": position, "split": split,
                       "generation_id": key[0], "used_for_training": original["used_for_training"],
                       "model_output": original["response"], "score_field": key[1], "rating_index": key[2],
                       "published_scores": original[key[1]],
                       "published_feedback": original.get(key[1].replace("_score", "_feedback"))},
               "BiGGen full output, feedback and all unaveraged source ratings")
        grade = None if pd.isna(row.response) else row.response
        _check((subjects[row.subject_id], items[row.item_id], grade, row.trial),
               (original["model_name"], (definition, key[1]), expected[key], key[2] + 1),
               "BiGGen model, item, judge, native grade and rating position")
        _check(pd.isna(row.test_condition) and pd.isna(row.interactors), True, "BiGGen no fabricated run conditions")
        observed[key] += 1
    _check(observed, Counter({key: 1 for key in expected}), "BiGGen every released rating exactly once")
    _check(len(traces), len(expected), "BiGGen complete trace coverage")
    return counts


def _wcf_source_records(directory, metadata):
    """Read native JSON directly, independently of the builder's joins and melt."""
    parameters = metadata["build"]["parameters"]
    raw = directory / "raw"
    root = raw / parameters["layout"]["release"]
    fields = {"is_relevant": "relevant", "is_factual": "factual", "has_what_and_why": "what_and_why",
              "has_what_to_do": "what_to_do", "is_comprehensible": "comprehensible",
              "has_out_of_scope": "out_of_scope", "is_direct": "directness", "feedback_quality": "quality"}
    _check(parameters["metrics"], fields, "WCF eight released rating dimensions")
    _check(parameters["conditions"], {"temperature": "0"}, "WCF paper section 4.1 inference temperature")
    _check(parameters["subject_features"], {"model_identifier": "gpt-4o-2024-11-20",
           "harness": "Annotating Errors WCF", "response_format": "json_object"}, "WCF reported model settings")
    generations, ratings, expected, display_changes = {}, {}, {}, set()
    totals, per_source, histogram = Counter(), Counter(), Counter()
    for source, filename in parameters["generations"].items():
        path = root / filename
        for position, row in enumerate(_jsonl(path)):
            key = source, row["annotation_instance_id"]
            _check(key in generations, False, "WCF unique source-system/task generation")
            _check(row["fb_source"], source, "WCF native generation system")
            _check(set(row["input_prompt"]), {"system", "user"}, "WCF full recorded model input")
            _check(all(isinstance(v, str) and v for v in row["input_prompt"].values()), True, "WCF nonempty system/user prompts")
            generations[key] = row, str(path.relative_to(raw)), position
    path = root / parameters["layout"]["ratings"]
    native = json.loads(path.read_text())
    for position, row in enumerate(native):
        if row["fb_source"] == "human":
            continue
        key = row["user_id"], row["rater_task_id"]
        _check(key in ratings, False, "WCF composite native rater/task identity")
        generation_key = row["fb_source"], row["annotation_instance_id"]
        generation = generations[generation_key][0]
        _check(row["user_id"] in metadata["grading"]["verifiers"]["human_ratings"]["raters"], True, "WCF identified teacher")
        _check(row["instance_text"]["feedback"], generation["feedback_explanation"] + " " + generation["feedback_suggestion"],
               "WCF exact association between generated and rated feedback")
        _check(row["annotator_id"], generation["annotator_id"], "WCF oracle error-annotation author")
        for field in ("source", "corrected"):
            if row["instance_text"][field] != generation[field]:
                _check(row["instance_text"][field], generation[field].replace("[NONE] ", "").replace(" [NONE]", ""),
                       "WCF documented rater-display removal of insertion/deletion markers")
                display_changes.add((*generation_key, field))
        ratings[key] = row, str(path.relative_to(raw)), position
        for field, dimension in fields.items():
            value = row[field]
            if field == "is_direct":
                _check(value in ("Direct", "Hint", "N/A"), True, "WCF nominal directness category")
                grade = {"Direct": 0, "Hint": 1, "N/A": 2}[value]
            elif field == "feedback_quality":
                _check(type(value) is int and 1 <= value <= 5, True, "WCF integer teacher quality rating")
                grade = value
            else:
                _check(type(value) is bool, True, "WCF native boolean criterion")
                grade = int(value)
            expected[*key, field] = grade
            totals[row["fb_source"] + "/" + dimension] += grade
            histogram[field + "/" + str(grade)] += 1
        per_source[row["fb_source"]] += 1
    cells = Counter((row[0]["fb_source"], row[0]["annotation_instance_id"]) for row in ratings.values())
    _check(set(cells.values()), {2}, "WCF two released teachers per rated generation")
    unrated = [row for key, (row, _, _) in generations.items() if key not in cells]
    _check({row["fb_source"] for row in unrated}, {"template_system"}, "WCF unrated-template scope")
    reported = {
        "our_tags": [1.000, .970, .992, 1.000, .970, .008, 4.487],
        "ERRANT_tags": [.997, .967, .992, 1.000, .982, .003, 4.475],
        "EXPECT_tags": [.997, .975, .990, 1.000, .975, .005, 4.500],
        "tagless": [.995, .970, .997, 1.000, .982, .005, 4.495],
        "template_system": [.977, .921, .944, .994, .980, .023, 4.184],
    }
    means = {}
    dimensions = [value for value in fields.values() if value != "directness"]
    for source, values in reported.items():
        for dimension, value in zip(dimensions, values):
            key = source + "/" + dimension
            means[key] = round(totals[key] / per_source[source], 3)
            _check(means[key], value, "WCF paper Table 3 rounded native mean: " + key)
    counts = {"source_ratings": len(ratings), "source_responses": len(expected), "source_traces": len(expected),
              "source_items": len(expected), "source_subjects": len(per_source), "source_rated_generations": len(cells),
              "source_tasks": len({key[1] for key in cells}), "source_unrated_generations": len(unrated),
              "source_unrated_empty_outputs": sum(not row["feedback_explanation"] and not row["feedback_suggestion"] for row in unrated),
              "source_human_reference_ratings": sum(row["fb_source"] == "human" for row in native),
              "source_changed_display_fields": len(display_changes), "source_grade_histograms": dict(histogram),
              "paper_table3_means": means}
    return generations, ratings, expected, counts


def _annotating_errors_wcf(directory, tables, metadata, source_records=None):
    generations, ratings, expected, counts = (_wcf_source_records(directory, metadata)
                                             if source_records is None else source_records)
    parameters = metadata["build"]["parameters"]
    subjects, items = {}, {}
    for row in tables["subjects"].itertuples():
        features = _features(row.subject_features_extra)
        source = features["feedback_strategy"]
        _check(row.display_name, parameters["subjects"][source], "WCF source-system label")
        _check(row.harness, "Annotating Errors WCF", "WCF evaluation harness")
        _check(features, {"feedback_strategy": source, "model_identifier": "gpt-4o-2024-11-20",
                          "response_format": "json_object"}, "WCF exact GPT revision and prompting strategy")
        subjects[row.subject_id] = source
    _check(Counter(subjects.values()), Counter({key: 1 for key in parameters["subjects"]}), "WCF distinct five-system panel")
    protocol = metadata["grading"]["verifiers"]["human_ratings"]
    for row in tables["items"].itertuples():
        source, task, dimension, rater = row.raw_item_id.split(":")
        generation = generations[source, task][0]
        _check(json.loads(row.content), generation["input_prompt"], "WCF exact system/user prompt including all examples")
        criterion = json.loads(row.grading_criterion)
        _check(criterion, {"reference_answer": None, **protocol["criteria"][dimension]}, "WCF full criterion and scale")
        direction = "unordered" if dimension == "directness" else "lower_is_better" if dimension == "out_of_scope" else "higher_is_better"
        values = [0, 1, 2] if dimension == "directness" else [1, 2, 3, 4, 5] if dimension == "quality" else [0, 1]
        _check((criterion["response_scale"]["values"], criterion["response_scale"]["direction"]),
               (values, direction), "WCF criterion-specific rating categories and preference direction")
        verifier = json.loads(row.verifier)
        _check((verifier["class"], verifier["judge"], verifier["judged_by"]), ("judge", rater, "human"), "WCF teacher identity in grading")
        _check(json.loads(verifier["spec"]), protocol, "WCF captured rating protocol")
        _check(_features(row.item_features), {"annotation_instance_id": task, "original_id": generation["original_id"],
               "cefr_level": generation["cefr_level"], "lang": "en"}, "WCF exact task aliases and learner level")
        items[row.item_id] = source, task, dimension, rater
    _check(len(items), len(expected), "WCF distinct prompt/criterion/teacher definitions")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    seen = Counter()
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        native = trace["rating_record"]
        key = native["user_id"], native["rater_task_id"], trace["score_field"]
        original, filename, position = ratings[key[:2]]
        generation, generation_file, generation_position = generations[original["fb_source"], original["annotation_instance_id"]]
        _check(trace, {"rating_source_file": filename, "rating_source_row": position, "rating_record": original,
                       "generation_source_file": generation_file, "generation_source_row": generation_position,
                       "generation_record": generation, "score_field": key[2]}, "WCF complete native records and source positions")
        _check((subjects[row.subject_id], items[row.item_id], row.response, row.trial, row.test_condition),
               (original["fb_source"], (original["fb_source"], original["annotation_instance_id"],
                parameters["metrics"][key[2]], original["user_id"]), expected[key], 1, "temperature=0"),
               "WCF model/task/criterion/teacher association and unchanged native rating")
        _check(pd.isna(row.interactors), True, "WCF no invented interacting agent")
        seen[key] += 1
    _check(seen, Counter({key: 1 for key in expected}), "WCF every released AI-feedback rating exactly once")
    _check(len(traces), len(expected), "WCF full trace coverage")
    return counts


def verify_native_results(directory, tables_directory=None):
    directory = Path(directory)
    root = Path(tables_directory) if tables_directory is not None else directory / "formatted_tables"
    tables = {p.stem: pd.read_parquet(p) for p in root.glob("*.parquet")}
    metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
    return {"openbiorq": _openbiorq, "fewshot_ttt_bbh": _fewshot, "prox": _prox,
            "cseo_bench": _cseo, "risebench": _risebench,
            "clasheval": _clasheval, "engdesign": _engdesign, "phyblock": _phyblock,
            "scigym": _scigym, "advprompter": _advprompter,
            "legal_rag_bench": _legal_rag, "nester": _nester, "engibench": _engibench,
            "llmail_inject": _llmail_inject, "safeagentbench": _safeagentbench,
            "critic_discernment_game": _critic_discernment_game, "brace": _brace,
            "biggen": _biggen, "annotating_errors_wcf": _annotating_errors_wcf}[directory.name](directory, tables, metadata)
