"""Check complete native results independently of the pandas builders."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from measurement_db.scripts.curate_benchmarks.batch3_audits import (
    _actual, _check, _digest, _features,
)


def _agc_source_records(directory, metadata):
    """Read the native rows independently of the builder's DataFrame joins."""
    import csv
    import math
    import random
    from zipfile import ZipFile
    import pyarrow.parquet as pq

    raw = directory / "raw"
    paths = {name: raw / value for name, value in metadata["build"]["parameters"]["layout"].items()}
    native, lookup, prompts, panels = {}, {}, {}, {}
    for path in sorted((paths["release"] / "generations/prompts").glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            key = path.stem, row["instance_id"]
            _check(key not in prompts, True, "AGC unique source prompt key")
            prompts[key] = row["prompt"]
    for path in sorted((paths["release"] / "generations").glob("*/*/*.parquet")):
        for position, row in enumerate(pq.read_table(path).to_pylist()):
            key = str(path.relative_to(raw)), position
            measurement = row["benchmark"], row["model"], row["instance_id"]
            _check(measurement not in lookup, True, "AGC unique native model/component/item")
            if row["canonical_score"] is not None:
                _check(math.isfinite(row["canonical_score"]), True, "AGC finite native statistic")
            native[key], lookup[measurement] = row, key
    for row in pq.read_table(paths["release"] / "analysis/jrt_complete_ratings.parquet").to_pylist():
        source = lookup[row["benchmark"], row["model"], row["item_id"]]
        key = source, row["rater"]
        _check(key not in panels, True, "AGC unique original panel judgment")
        _check(math.isfinite(row["score"]), True, "AGC finite original panel judgment")
        panels[key] = row

    # The exported IRFL prompts omit their multimedia fields. Reconstruct the
    # source's exact text, seeded option order and bytes independently.
    restored, golds, images = {}, {}, {}
    with (paths["irfl"] / "idiom_detection_task.csv").open() as stream:
        bank = list(csv.DictReader(stream))
    with ZipFile(paths["irfl"] / "IRFL_images.zip") as archive:
        for index, row in enumerate(bank):
            task = f"idiom-detection-task_{index}"
            correct = json.loads(row["answer"])[0]
            order = [correct] + json.loads(row["distractors"])
            random.Random(f"idiom-detection-task:{index}:{row['phrase']}").shuffle(order)
            golds["irfl", task] = "ABCD"[order.index(correct)]
            definition = json.loads(row["definition"])[0]
            text = (f'Choose the image that best visualizes the meaning of the figurative expression: "{row["phrase"]}"\n\n'
                    f"Definition: {definition}\n\nSelect exactly one option and answer with a single letter: A, B, C, or D.\n\n")
            elements = [{"content_type": "text/plain", "text": text}]
            for letter, image in zip("ABCD", order, strict=True):
                filename = f"images/{image}.jpeg"
                elements.extend([{"content_type": "text/plain", "text": f"\n{letter})"},
                                 {"content_type": "image/jpeg", "location": filename}])
                images[filename] = archive.read(filename)
            elements.append({"content_type": "text/plain", "text": "\n\nAnswer:"})
            restored["irfl", task] = {"multimedia_elements": elements}
    with (paths["analobench"] / "AnaloBench-T1-Subset-S1.csv").open() as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            golds["analobench", f"id{index}"] = row["Label"]
    with ZipFile(paths["moh_x"]) as archive:
        import io
        with archive.open("data/MOH-X/MOH-X_formatted_svo_cleaned.csv") as handle:
            for index, row in enumerate(csv.DictReader(io.TextIOWrapper(handle))):
                golds["moh_x", f"id{index}"] = "Yes" if row["label"] == "1" else "No"
    _check((len(native), len(panels), len({row["model"] for row in native.values()}),
            len({row["benchmark"] for row in native.values()})),
           (267018, 182924, 83, 67), "AGC full released generation/panel coverage")
    return native, prompts, panels, restored, golds, images


def _agc_bench(directory, tables, metadata, source_records=None):
    """Check every output against the immutable generation and rater records."""
    import math
    import re
    from measurement_db.scripts.build_measurement_tables.response_scales import canonical_response_scale

    native, prompts, panels, restored, golds, images = (
        _agc_source_records(directory, metadata) if source_records is None else source_records)
    protocols = metadata["grading"]["verifiers"]
    subjects = {}
    for row in tables["subjects"].itertuples():
        original = _features(row.subject_features_extra)["model_identifier"]
        _check(row.harness, "AGC-Bench/HELM", "AGC subject harness")
        subjects[row.subject_id] = original
    _check(Counter(subjects.values()), Counter({row["model"]: 1 for row in native.values()}), "AGC exact source model labels")
    items = {row.item_id: row for row in tables["items"].itertuples()}
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    assets = tables["assets"].set_index("asset_id").to_dict("index")
    checked_items, seen, counts = set(), Counter(), Counter()
    invalid_bands = {"arastories": (1, 5), "future_ideas": (1, 5), "poetmt": (1, 5),
                     "rpgbench": (1, 5), "showerthoughts": (1, 6)}
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        source_key = trace["source_file"], trace["source_row"]
        original = native[source_key]
        _check(trace["generation"], original, "AGC complete native generation record without rounding or truncation")
        judge = trace["panel_rating"]
        rater = judge["rater"] if judge is not None else ""
        key = source_key, rater
        benchmark, task = original["benchmark"], original["instance_id"]
        if judge is not None:
            _check(judge, panels[key], "AGC original individual judge record")
            grade, status = judge["score"], "released_grade"
            if benchmark == "cpers" and not 1 <= grade <= 5:
                grade, status = None, "invalid_rating_preserved_in_trace"
            protocol = "panel_" + benchmark
            counts["source_panel_judgments"] += 1
        else:
            grade = original["canonical_score"]
            status = "not_released" if grade is None else "released_grade"
            if benchmark == "story_quality":
                grade, status = None, "non_grade_statistic"
            elif grade is not None and benchmark in invalid_bands:
                low, high = invalid_bands[benchmark]
                if not low <= grade <= high:
                    grade, status = None, "invalid_rating_preserved_in_trace"
            protocol = "canonical_" + benchmark
            counts["source_native_attempts"] += 1
        _check(trace["grade_status"], status, "AGC explicit missing/invalid/non-grade distinction")
        _check(None if pd.isna(row.response) else row.response, grade, "AGC unmodified native grade or explicitly unavailable grade")
        _check(subjects[row.subject_id], original["model"], "AGC response-model association")
        _check(row.trial, 1, "AGC one retained attempt per source model/task/grading protocol")
        _check(row.test_condition, "source_run=" + original["source_run_dir"], "AGC retained run provenance")
        _check(pd.isna(row.interactors), True, "AGC no invented interactors")
        counts["source_graded_observations"] += grade is not None
        counts["source_ungraded_observations"] += grade is None
        counts["source_invalid_ratings"] += status == "invalid_rating_preserved_in_trace"
        counts["source_non_grade_statistics"] += status == "non_grade_statistic"
        item = items[row.item_id]
        features = _features(item.item_features)
        _check(features["component_dataset"], benchmark, "AGC response-component association")
        _check(features["grading_channel"], protocol, "AGC native versus panel protocol")
        criterion = json.loads(item.grading_criterion)
        verifier = json.loads(item.verifier)
        _check(verifier.get("judge"), metadata["build"]["parameters"]["panel_raters"].get(rater), "AGC exact judge identity belongs to item")
        _check(json.loads(verifier["spec"]), protocols[protocol]["implementation"], "AGC matching grading implementation")
        _check(criterion["response_scale"], json.loads(canonical_response_scale(protocols[protocol]["response_scale"])), "AGC effective grading scale")
        if (benchmark, task) in golds:
            _check(criterion["reference_answer"], golds[benchmark, task], "AGC pinned task-bank reference")
        if benchmark == "irfl":
            _check(json.loads(item.content), restored[benchmark, task], "AGC complete reconstructed multimodal stimulus")
            if judge is None:
                option = re.search(r"\b([ABCD])\b", original["completion"] or "")
                predicted = option.group(1) if option else None
                _check(grade, float(predicted == golds[benchmark, task]), "AGC every IRFL grade agrees with shuffled image reference")
                counts["source_irfl_verified_attempts"] += 1
        else:
            _check(item.content, prompts[benchmark, task], "AGC complete recorded prompt")
        if benchmark == "analobench" and judge is None:
            match = re.search(r"(?:\:\s)?([A-D])(?:\.|\s|$)", original["completion"] or "", re.IGNORECASE)
            expected = .25 if match is None else float(match.group(1).upper() == golds[benchmark, task])
            _check(grade, expected, "AGC all AnaloBench grades including unparsed-option partial credit")
            counts["source_partial_credit_attempts"] += grade == .25
        if row.item_id not in checked_items:
            _check(criterion["rule"], protocols[protocol]["rule"], "AGC grading rule")
            if benchmark == "irfl":
                links = json.loads(item.asset_manifest)
                expected_paths = [part["location"] for part in restored[benchmark, task]["multimedia_elements"] if part["content_type"] == "image/jpeg"]
                _check([link["path"] for link in links], expected_paths, "AGC all image choices in their presented order")
                for link in links:
                    _check(assets[link["asset_id"]]["data"], images[link["path"]], "AGC original image bytes")
            checked_items.add(row.item_id)
        seen[key] += 1
    expected = Counter({(key, ""): 1 for key in native})
    expected.update({key: 1 for key in panels})
    _check(seen, expected, "AGC every native attempt and individual judgment exactly once")
    _check(len(traces), len(expected), "AGC complete trace associations")
    _check(checked_items, set(items), "AGC no unused task definitions")
    counts.update(source_responses=len(expected), source_traces=len(traces), source_subjects=len(subjects),
                  source_component_datasets=67, source_assets=len(assets), source_items=len(items))
    _check((counts["source_invalid_ratings"], counts["source_non_grade_statistics"]), (121, 4150), "AGC reviewed grading corrections")
    return dict(counts)


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


def _bertaqa_source_records(directory, metadata):
    """Match native API records with the independent question bank and scoring rule."""
    parameters = metadata["build"]["parameters"]
    raw = directory / "raw"
    banks, prompts, native, definitions = {}, {}, {}, {}
    counts, group_sizes, correct_groups = Counter(), Counter(), Counter()
    for language, filename in parameters["question_files"].items():
        rows = _jsonl(raw / parameters["layout"]["questions"] / filename)
        banks[language] = {row["id"]: row for row in rows}
        _check(len(banks[language]), len(rows), "BertaQA unique question-bank IDs")
        prompts[language] = {}
        for row in rows:
            question, answer = ("Galdera", "Erantzuna") if language == "eu" else ("Question", "Answer")
            text = f"{question}: {row['question']}\nA: {row['candidates'][0]}\nB: {row['candidates'][1]}\nC: {row['candidates'][2]}\n{answer}:"
            gold = "ABC"[row["answer"]]
            if text in prompts[language]:
                _check(prompts[language][text], gold, "BertaQA no conflicting references for identical questions")
            prompts[language][text] = gold
    api_ids, model_names = set(), set()
    files = sorted((raw / parameters["layout"]["results"]).glob("*/bertaqa_??_5-shot.jsonl"))
    for path in files:
        model, language = path.parent.name, path.stem.split("_")[1]
        provider = parameters["providers"][model]
        seen_ids = set()
        for position, row in enumerate(_jsonl(path)):
            _check(row["id"] in seen_ids, False, "BertaQA unique target per result file")
            seen_ids.add(row["id"])
            reference = banks[language][row["id"]]
            _check({key: row[key] for key in reference}, reference, "BertaQA question/options/reference and original annotations")
            messages = ([{"role": "system", "content": row["system"]}] if provider == "anthropic" else []) + row["messages"]
            _check([message["role"] for message in messages], ["system", *(["user", "assistant"] * 5), "user"],
                   "BertaQA system instruction, five demonstrations and target")
            _check(messages[0]["content"], "Respond always with a single letter: A, B or C.", "BertaQA saved system instruction")
            question, answer = ("Galdera", "Erantzuna") if language == "eu" else ("Question", "Answer")
            target = f"{question}: {row['question']}\nA: {row['candidates'][0]}\nB: {row['candidates'][1]}\nC: {row['candidates'][2]}\n{answer}:"
            _check(messages[-1]["content"], target, "BertaQA exact target prompt")
            for index in range(1, 11, 2):
                _check(prompts[language][messages[index]["content"]], messages[index + 1]["content"], "BertaQA native few-shot reference")
            output = row["response"]
            _check(output["model"], model, "BertaQA exact returned model identifier")
            _check(output["id"] in api_ids, False, "BertaQA unique recorded API response")
            api_ids.add(output["id"])
            text = output["content"][0]["text"] if provider == "anthropic" else output["choices"][0]["message"]["content"]
            gold = "ABC"[row["answer"]]
            _check(type(row["correct"]) is bool and row["correct"] == (text == gold), True, "BertaQA native exact-letter grade")
            definition = json.dumps({"messages": messages, "reference": gold}, ensure_ascii=False, sort_keys=True)
            definitions.setdefault(definition, (row, language))
            native[str(path.relative_to(raw)), position] = row, model, language, definition
            model_names.add(model)
            group = model + "/" + language + "/" + row["group"]
            group_sizes[group] += 1
            correct_groups[group] += row["correct"]
            counts["source_nonletter_outputs"] += text not in ("A", "B", "C")
            counts["source_successes"] += row["correct"]
            counts["source_anthropic_traces"] += provider == "anthropic"
        _check(seen_ids, set(banks[language]), "BertaQA complete language-specific task coverage")
    reported = {
        "gpt-3.5-turbo-0125": [55.08, 82.40, 47.25, 66.22],
        "gpt-4-0613": [69.88, 91.43, 62.94, 85.91],
        "gpt-4-0125-preview": [72.17, 91.68, 69.46, 89.21],
        "claude-3-haiku-20240307": [58.71, 84.16, 58.21, 79.85],
        "claude-3-sonnet-20240229": [58.33, 86.41, 56.13, 83.24],
        "claude-3-opus-20240229": [71.91, 91.85, 71.32, 90.89],
    }
    percentages = {}
    for model, expected in reported.items():
        groups = [(language, group) for language in ("en", "eu") for group in ("Euskal gaiak", "Gai orokorrak")]
        for (language, group), value in zip(groups, expected):
            key = model + "/" + language + "/" + group
            percentages[key] = round(100 * correct_groups[key] / group_sizes[key], 2)
            _check(percentages[key], value, "BertaQA paper Tables 2 and 4 accuracy: " + key)
    counts.update(source_responses=len(native), source_traces=len(native), source_items=len(definitions),
                  source_subjects=len(model_names), source_result_files=len(files),
                  source_language_tasks=sum(len(bank) for bank in banks.values()))
    return native, definitions, {**counts, "source_correct_by_group": dict(correct_groups), "paper_accuracy_percent": percentages}


def _bertaqa(directory, tables, metadata, source_records=None):
    native, definitions, counts = _bertaqa_source_records(directory, metadata) if source_records is None else source_records
    parameters = metadata["build"]["parameters"]
    subjects, items = {}, {}
    for row in tables["subjects"].itertuples():
        _check(row.harness, "BertaQA", "BertaQA evaluation harness")
        _check(_features(row.subject_features_extra), {"model_identifier": row.display_name,
               "api_provider": parameters["providers"][row.display_name]}, "BertaQA dated model identifier and API provider")
        subjects[row.subject_id] = row.display_name
    _check(Counter(subjects.values()), Counter({key: 1 for key in parameters["providers"]}), "BertaQA complete six-model panel")
    for row in tables["items"].itertuples():
        criterion = json.loads(row.grading_criterion)
        messages = json.loads(row.content)
        definition = json.dumps({"messages": messages, "reference": criterion["reference_answer"]}, ensure_ascii=False, sort_keys=True)
        original, language = definitions[definition]
        items[row.item_id] = definition
        _check(row.raw_item_id, f"{language}-{original['id']}", "BertaQA language and original question alias")
        _check(criterion, {"reference_answer": "ABC"[original["answer"]], "rule": metadata["grading"]["rule"]}, "BertaQA gold letter and strict scoring rule")
        _check(_features(row.item_features), {"lang": language, "category": original["category"], "group": original["group"],
               "difficulty": str(original["difficulty"]), "shot": "5"}, "BertaQA original question characteristics")
        verifier = json.loads(row.verifier)
        _check(verifier["class"], "exact_matcher", "BertaQA deterministic grading")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["released_accuracy"], "BertaQA upstream response selectors")
    _check(Counter(items.values()), Counter({key: 1 for key in definitions}), "BertaQA every distinct full prompt and reference")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    seen = Counter()
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace["source_file"], trace["source_row"]
        original, model, language, definition = native[key]
        settings = {"temperature": "0", "max_tokens": "1"} if model.startswith("claude-") else {"temperature": "0", "seed": "42"}
        _check(trace, {"source_file": key[0], "source_row": key[1], "record": original, "published_request_settings": settings},
               "BertaQA complete native API response, usage, cost, prompt and published request settings")
        _check((subjects[row.subject_id], items[row.item_id], row.response, row.trial, row.test_condition),
               (model, definition, int(original["correct"]), 1, "temperature=0"), "BertaQA native model/prompt/grade association")
        _check(pd.isna(row.interactors), True, "BertaQA no invented interaction participants")
        seen[key] += 1
    _check(seen, Counter({key: 1 for key in native}), "BertaQA every native result exactly once")
    _check(len(traces), len(native), "BertaQA full OpenAI and Anthropic trace coverage")
    return counts


def _afrimedqa_csv(path):
    """Read CSV independently, retaining the one bare-CR answer continuation."""
    import csv
    import sys

    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as stream:
        records = iter(csv.reader(stream))
        header = next(records)
        header = [name or f"Unnamed: {index}" for index, name in enumerate(header)]
        result = []
        for values in records:
            if len(values) != len(header):
                continuation = next(records)
                _check((len(header), len(values), len(continuation)), (14, 12, 3), "AfriMed-QA known CSV continuation shape")
                _check(values[0], "7276cbf1ad6d7295806602d5f4e6f3007543597eed7050cc437e00ae5384936a", "AfriMed-QA bare-CR source task")
                _check(continuation, [" E.  Ovulation can be confirmed by measurement of LH on day 14..", "E", "False"], "AfriMed-QA original answer continuation")
                values[-1] += "\r" + continuation[0]
                values.extend(continuation[1:])
            result.append(dict(zip(header, values, strict=True)))
    return result


def _afrimedqa_source_records(directory, metadata):
    """Reconcile native MCQ records, reference banks, duplicate files and nulls."""
    import re

    parameters = metadata["build"]["parameters"]
    raw = directory / "raw"
    root = raw / parameters["layout"]["release"]
    phase1_rows = _afrimedqa_csv(root / parameters["layout"]["phase1_reference"])
    phase1 = {row["sample_id"]: row for row in phase1_rows}
    questions = {" ".join(row["question"].split()): row for row in phase1_rows}
    _check((len(phase1), len(questions)), (3000, 3000), "AfriMed-QA unique Phase-1 references")
    phase2 = {row["sample_id"]: row for row in _afrimedqa_csv(root / parameters["layout"]["phase2_reference"])}
    experts = {key for key, row in phase2.items() if (row["question_type"], row["tier"], row["split"]) == ("mcq", "expert", "test")}
    medqa = _afrimedqa_csv(root / parameters["layout"]["medqa_reference"])
    foreign = {value for row in medqa for value in (row["sample_id"], row["question"])}
    _check(len(experts), 3910, "AfriMed-QA released expert question bank")
    for path, reason in parameters["excluded_results"].items():
        if reason.startswith("Byte-identical copy of "):
            other = reason.removeprefix("Byte-identical copy of ")
            _check((root / path).read_bytes(), (root / other).read_bytes(), "AfriMed-QA duplicate source bytes")
    ambiguous = [path for path, reason in parameters["excluded_results"].items() if not reason.startswith("Byte-identical copy of ")]
    fingerprints = []
    for path in ambiguous:
        fingerprints.append(sorted((r["sample_id"], r["model_prompt"], r["outputs"], r["preds"], r["correct"])
                                   for r in _afrimedqa_csv(root / path)))
    _check(len(fingerprints), 3, "AfriMed-QA reviewed ambiguous file group")
    _check(fingerprints[0] == fingerprints[1] == fingerprints[2], True, "AfriMed-QA cross-model copied prompts and outputs")

    native, definitions, trials = {}, {}, Counter()
    counts, files, models, tasks = Counter(), {}, set(), set()
    for path in sorted((root / "results").glob("*/*mcq*.csv")):
        relative = str(path.relative_to(root))
        if relative in parameters["excluded_results"]:
            continue
        rows = _afrimedqa_csv(path)
        recovered = "sample_id" not in rows[0]
        if not recovered and {row["sample_id"] for row in rows} <= foreign:
            counts["source_foreign_runs_excluded"] += 1
            continue
        if recovered:
            task_rows = []
            for row in rows:
                question = row["model_prompt"].split("###Question: ", 1)[1].split("\n###Options:", 1)[0]
                reference = questions[" ".join(question.split())]
                options = row["model_prompt"].split("\n###Options:\n", 1)[1].split("\n\n\n### Response:", 1)[0]
                parts = re.split(r"(?m)^([A-E])\. ", options)
                actual = {parts[index]: " ".join(parts[index + 1].split()) for index in range(1, len(parts), 2)}
                for letter in "ABCDE":
                    expected = " ".join(reference.get(letter, "").split())
                    value = actual.get(letter, "")
                    _check(value == expected or (not expected and value.upper() == "N/A"), True,
                           "AfriMed-QA all options in recovered output-only prompts")
                task_rows.append(reference)
            counts["source_recovered_ungraded"] += len(rows)
        else:
            task_rows = rows
        ids = {row["sample_id"] for row in task_rows}
        _check(len(ids), len(rows), "AfriMed-QA unique tasks in each native run")
        if ids == set(phase1):
            bank = "afrimedqa-v1"
        elif ids == experts:
            bank = "afrimedqa-v2"
        else:
            _check((len(ids), len(ids & experts), ids <= set(phase2)), (289, 160, True), "AfriMed-QA overlapping Phase-2 subset")
            bank = "afrimedqa-v2.5"
        successes = 0
        for position, (row, reference) in enumerate(zip(rows, task_rows, strict=True)):
            task, gold = reference["sample_id"], reference["answer"]
            if task in phase1:
                _check(gold, phase1[task]["answer"], "AfriMed-QA Phase-1 reference letter")
            else:
                option = phase2[task]["correct_answer"].split(",")[0].strip()
                _check(gold, "ABCDE"[int(option.removeprefix("option")) - 1], "AfriMed-QA Phase-2 reference letter")
            grade_text = row.get("correct", "")
            _check(grade_text in ("", "True", "False", "true", "false", "1", "0"), True, "AfriMed-QA finite native grade")
            grade = None if grade_text == "" else int(grade_text in ("True", "true", "1"))
            if grade is not None:
                _check(grade, int(row["answer"] == row["preds"]), "AfriMed-QA native exact-letter scoring")
            definition = row["model_prompt"], gold
            definitions.setdefault(definition, task)
            model = path.parent.name
            condition = "source=" + bank
            trial_key = model, definition, condition
            trials[trial_key] += 1
            key = str(path.relative_to(raw)), position
            trace = {"source_file": key[0], "source_row": position, "record": row}
            native[key] = model, definition, grade, condition, trials[trial_key], _digest(json.dumps(trace, ensure_ascii=False, sort_keys=True))
            models.add(model)
            tasks.add(task)
            successes += grade == 1
            counts["source_successes"] += grade == 1
            counts["source_ungraded"] += grade is None
            counts["source_bare_cr_answers"] += "\r" in row.get("outputs", "")
        files[relative] = {"responses": len(rows), "successes": successes}
    counts.update(source_responses=len(native), source_traces=len(native), source_items=len(definitions),
                  source_subjects=len(models), source_target_questions=len(tasks), source_result_files=len(files))
    return native, definitions, {**counts,
        "source_run_responses": {path: values["responses"] for path, values in files.items()},
        "source_run_successes": {path: values["successes"] for path, values in files.items()}}


def _afrimedqa(directory, tables, metadata, source_records=None):
    native, definitions, counts = _afrimedqa_source_records(directory, metadata) if source_records is None else source_records
    subjects, items = {}, {}
    for row in tables["subjects"].itertuples():
        _check(row.harness, "AfriMed-QA", "AfriMed-QA source harness")
        features = _features(row.subject_features_extra)
        _check(set(features), {"model_identifier"}, "AfriMed-QA original model label")
        subjects[row.subject_id] = features["model_identifier"]
    _check(Counter(subjects.values()), Counter({row[0]: 1 for row in native.values()}), "AfriMed-QA every model, including fine-tuned variants")
    for row in tables["items"].itertuples():
        criterion = json.loads(row.grading_criterion)
        definition = row.content, criterion["reference_answer"]
        _check(row.raw_item_id, definitions[definition], "AfriMed-QA original question alias")
        _check(criterion, {"reference_answer": definition[1], "rule": metadata["grading"]["rule"]}, "AfriMed-QA reference and grading rule")
        _check(_features(row.item_features), {"lang": "en"}, "AfriMed-QA item language")
        verifier = json.loads(row.verifier)
        _check(verifier["class"], "exact_matcher", "AfriMed-QA grading method")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["released_accuracy"], "AfriMed-QA upstream extraction and comparison")
        items[row.item_id] = definition
    _check(Counter(items.values()), Counter({key: 1 for key in definitions}), "AfriMed-QA complete full-prompt definitions")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    seen = Counter()
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace["source_file"], trace["source_row"]
        observed = (subjects[row.subject_id], items[row.item_id], None if pd.isna(row.response) else row.response,
                    row.test_condition, row.trial, _digest(json.dumps(trace, ensure_ascii=False, sort_keys=True)))
        _check(observed, native[key], "AfriMed-QA every model/prompt/reference/grade/trial and complete source record")
        _check(pd.isna(row.interactors), True, "AfriMed-QA no invented interactors")
        seen[key] += 1
    _check(seen, Counter({key: 1 for key in native}), "AfriMed-QA every native attempt exactly once")
    _check(len(tables["traces"]), len(native), "AfriMed-QA complete trace coverage")
    return counts


def _adaptivestep_source_records(directory, metadata):
    """Read native JSON lines and use the captured author's grading functions."""
    import ast
    import re
    import runpy
    from types import SimpleNamespace
    import pyarrow.parquet as pq

    raw = directory / "raw"
    paths = {key: raw / value for key, value in metadata["build"]["parameters"]["layout"].items()}
    utility = runpy.run_path(str(paths["math_util"]))
    tree = ast.parse(paths["math_util"].with_name("eval.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"remove_boxed", "process_results"}]
    _check({node.name for node in functions}, {"remove_boxed", "process_results"}, "AdaptiveStep captured grader functions")
    author = {"util": SimpleNamespace(**utility)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "captured_asprm_answer_checker", "exec"), author)
    references = {}
    for record in pq.read_table(paths["gsm_reference"]).to_pylist():
        reference = re.search(r"####\s*([+-]?\d*[\.,]?\d+)", record["answer"]).group(1)
        references["gsm8k", record["question"]] = reference
    with paths["math_reference"].open() as stream:
        for line in stream:
            record = json.loads(line)
            reference = author["remove_boxed"](utility["last_boxed_only_string"](record["solution"]))
            _check(reference is not None, True, "AdaptiveStep authoritative boxed reference")
            references["math500", record["problem"]] = reference

    native, definitions, counts, questions = {}, {}, Counter(), set()
    canonical_aliases, trial_counts = {}, Counter()
    arrays = ["pred", "math_random_list", "math_hard_list", "math_confidence_list"]
    for path in sorted(paths["math"].glob("*_testset_bo256.jsonl")):
        dataset = "gsm8k" if "GSM8k" in path.name else "math500"
        model = "MetaMath-Llama" if path.name.startswith("Llama31_") else "MetaMath-Mistral"
        with path.open() as stream:
            for position, line in enumerate(stream):
                record = json.loads(line)
                _check([len(record[key]) for key in arrays], [256] * 4, "AdaptiveStep aligned math samples")
                reference = references[dataset, record["question"]]
                original = (re.search(r"####\s*([+-]?\d*[\.,]?\d+)", record["gt_answer"]).group(1)
                            if dataset == "gsm8k" else author["remove_boxed"](utility["last_boxed_only_string"](record["gt_answer"])))
                content = record.get("input", record["question"])
                _check(record["question"] in content, True, "AdaptiveStep full recorded math prompt")
                origin = "recorded_input" if "input" in record else "released_question"
                definition = content, reference, dataset, origin, f"{dataset}_{record['idx']}"
                definitions[definition] = True
                questions.add((dataset, record["question"]))
                counts.update(source_pools=1, source_reference_corrections=int(original != reference))
                context = {key: value for key, value in record.items() if key not in arrays}
                for offset, completion in enumerate(record["pred"]):
                    if dataset == "gsm8k":
                        match = re.search(r"The answer is:\s*([+-]?\d*[\.,]?\d+)", completion)
                        grade = float(bool(match and match.group(1) == reference))
                        old_grade = float(bool(match and match.group(1) == original))
                    else:
                        grade = float(author["process_results"](completion, reference))
                        old_grade = float(author["process_results"](completion, original))
                    key = path.name, position, offset + 1
                    trace = {"source_file": path.name, "source_row": position, "trial": offset + 1,
                             "source_record": context, **{name: record[name][offset] for name in arrays},
                             "original_reference": original, "reference": reference, "original_response": old_grade}
                    trial_counts[model, definition] += 1
                    native[key] = model, definition, grade, trial_counts[model, definition], _digest(json.dumps(trace, sort_keys=True, ensure_ascii=False))
                    counts.update(source_responses=1, source_graded_responses=1, source_successes=int(grade),
                                  source_original_math_successes=int(old_grade), source_corrected_grades=int(grade != old_grade))
        counts.update(source_result_files=1)

    for path in sorted(paths["code"].glob("*_eval.jsonl")):
        dataset = "livecodebench" if "_lcb_" in path.name else "leetcode"
        orm_path = path.with_name(path.name.replace(".jsonl", "_orm.jsonl"))
        with path.open() as primary, orm_path.open() as secondary:
            for position, (line, orm_line) in enumerate(zip(primary, secondary, strict=True)):
                record, orm = json.loads(line), json.loads(orm_line)
                _check({key: value for key, value in record.items() if key != "code_confidence_list"},
                       {key: value for key, value in orm.items() if key != "code_confidence_list"},
                       "AdaptiveStep PRM/ORM files annotate identical candidates")
                candidates = record["code"] if dataset == "livecodebench" else record["pred"]
                _check([len(candidates), len(record["code_confidence_list"]), len(orm["code_confidence_list"])],
                       [64, 64, 64], "AdaptiveStep final code pool sizes")
                content = record["prompt_use"] if dataset == "livecodebench" else record["question"]
                question = record["question_content"] if dataset == "livecodebench" else record["question"]
                _check(question in content, True, "AdaptiveStep complete code task in recorded input")
                reference = None if dataset == "livecodebench" else record["answer"]
                origin = "recorded_input" if dataset == "livecodebench" else "released_question"
                alias = record["question_id"] if dataset == "livecodebench" else record["task_id"]
                identity = content, reference, dataset, origin
                canonical_aliases.setdefault(identity, f"{dataset}_{alias}")
                definition = *identity, canonical_aliases[identity]
                definitions[definition] = True
                questions.add((dataset, alias))
                context = {key: value for key, value in record.items() if key not in {
                    "pred", "code", "code_list", "code_confidence_list_pre", "code_confidence_list"}}
                filename, orm_filename = str(path.relative_to(raw)), str(orm_path.relative_to(raw))
                for offset, completion in enumerate(candidates):
                    key = filename, position, offset + 1
                    trace = {"source_file": filename, "orm_source_file": orm_filename, "source_row": position,
                             "trial": offset + 1, "source_record": context, "candidate": completion,
                             "code_confidence_list": record["code_confidence_list"][offset],
                             "orm_annotation": orm["code_confidence_list"][offset]}
                    trial_counts["LCD-DS", definition] += 1
                    native[key] = "LCD-DS", definition, None, trial_counts["LCD-DS", definition], _digest(json.dumps(trace, sort_keys=True, ensure_ascii=False))
                    counts.update(source_responses=1, source_ungraded_responses=1)
                counts.update(source_pools=1)
        counts.update(source_result_files=2)
    counts.update(source_unique_questions=len(questions), source_prompt_definitions=len(definitions), source_subjects=3)
    return native, definitions, dict(counts)


def _adaptivestep(directory, tables, metadata, source_records=None):
    native, definitions, counts = (_adaptivestep_source_records(directory, metadata)
                                   if source_records is None else source_records)
    _check(len(tables["responses"]), len(native), "AdaptiveStep complete candidate count")
    subjects = {}
    for row in tables["subjects"].itertuples():
        _check(row.harness, "ASPRM", "AdaptiveStep recorded subject harness")
        subjects[row.subject_id] = _features(row.subject_features_extra)["model_identifier"]
    _check(Counter(subjects.values()), Counter({"MetaMath-Llama": 1, "MetaMath-Mistral": 1, "LCD-DS": 1}),
           "AdaptiveStep policy models, not reward-model identities")
    items = {}
    for row in tables["items"].itertuples():
        features, criterion = _features(row.item_features), json.loads(row.grading_criterion)
        dataset = features["dataset"]
        _check(set(features), {"dataset", "prompt_origin"}, "AdaptiveStep item attributes")
        _check(criterion["rule"], metadata["grading"]["verifiers"][dataset]["rule"], "AdaptiveStep explicit grading rule")
        spec = json.loads(row.verifier)
        _check(spec["class"], "exact_matcher", "AdaptiveStep non-LLM verifier")
        _check(json.loads(spec["spec"]), metadata["grading"]["verifiers"][dataset], "AdaptiveStep grader provenance")
        items[row.item_id] = row.content, criterion["reference_answer"], dataset, features["prompt_origin"], row.raw_item_id
    _check(Counter(items.values()), Counter(definitions), "AdaptiveStep complete prompts, references and source aliases")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    seen = Counter()
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace["source_file"], trace["source_row"], trace["trial"]
        observed = (subjects[row.subject_id], items[row.item_id], None if pd.isna(row.response) else row.response, row.trial,
                    _digest(json.dumps(trace, sort_keys=True, ensure_ascii=False)))
        _check(observed, native[key], "AdaptiveStep every candidate/model/task/reference/grade and complete trace")
        _check(row.test_condition, "dataset=" + items[row.item_id][2], "AdaptiveStep source pool condition")
        _check(pd.isna(row.interactors), True, "AdaptiveStep no invented interaction context")
        seen[key] += 1
    _check(seen, Counter({key: 1 for key in native}), "AdaptiveStep every final candidate exactly once")
    _check(len(traces), len(native), "AdaptiveStep full trace coverage")
    return counts


def _algotune_html(text):
    """Read semantic HTML tokens with the standard parser, independently of bs4."""
    from html.parser import HTMLParser

    class Capture(HTMLParser):
        void = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

        def __init__(self):
            super().__init__()
            self.messages, self.best_files, self.filename = [], [], None
            self.kind, self.stack, self.tokens, self.text = None, [], [], []

        def handle_starttag(self, tag, attrs):
            classes = dict(attrs).get('class', '').split()
            if self.kind is None:
                if tag == 'div' and 'message' in classes:
                    self.kind, self.role = 'message', ' '.join(c for c in classes if c != 'message')
                elif tag == 'pre' and 'best-code' in classes:
                    self.kind = 'best_code'
                elif tag == 'div' and 'file-name' in classes:
                    self.kind = 'filename'
                else:
                    return
            self.tokens.append(('start', tag, sorted(attrs)))
            if tag not in self.void:
                self.stack.append(tag)

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)
            if tag not in self.void:
                self.handle_endtag(tag)

        def handle_endtag(self, tag):
            if self.kind is None or tag in self.void:
                return
            _check(bool(self.stack) and self.stack[-1] == tag, True, 'AlgoTune balanced captured message HTML')
            self.stack.pop()
            self.tokens.append(('end', tag))
            if not self.stack:
                if self.kind == 'message':
                    self.messages.append((self.role, _digest(json.dumps(self.tokens, ensure_ascii=False))))
                elif self.kind == 'filename':
                    self.filename = ''.join(self.text)
                else:
                    _check(bool(self.filename), True, 'AlgoTune final source filename')
                    self.best_files.append({'name': self.filename, 'content': ''.join(self.text)})
                self.kind, self.tokens, self.text = None, [], []

        def handle_data(self, data):
            if self.kind is not None:
                self.text.append(data)
                # Ignore layout indentation, retaining all whitespace in code.
                if data.strip() or 'pre' in self.stack or 'code' in self.stack:
                    self.tokens.append(('text', data))

    parsed = Capture()
    parsed.feed(text)
    parsed.close()
    _check(parsed.kind, None, 'AlgoTune complete captured HTML blocks')
    return parsed.messages, parsed.best_files


def _algotune_source_records(directory, metadata):
    import html
    import math
    import re
    paths = {name: directory / 'raw' / value for name, value in metadata['build']['parameters']['layout'].items()}
    summary = json.loads(paths['summary'].read_text())
    suffixes = metadata['build']['parameters']['model_page_suffix']
    native, definitions, counts = {}, {}, Counter()
    for task, models in summary.items():
        definitions[task] = ((paths['tasks']/task/'description.txt').read_text(),
                             (paths['tasks']/task/(task+'.py')).read_text())
        for model, result in models.items():
            _check(set(result), {'final_speedup'}, 'AlgoTune native result fields')
            speedup = result['final_speedup']
            if speedup == 'N/A':
                grade = 0.0
                counts['source_reported_failures'] += 1
            else:
                _check(math.isfinite(float(speedup)), True, 'AlgoTune finite reported speedup')
                grade = float(float(speedup) >= 1.0)
            page = paths['site'] / f'{task}_{suffixes[model]}.html'
            text = page.read_text()
            title = re.search(r'<title>AlgoTuner Log – (.*?) – (.*?)</title>', text)
            _check(title is not None, True, 'AlgoTune native page title')
            title_task, title_model = (html.unescape(value) for value in title.groups())
            _check((title_task, title_model.rsplit('/', 1)[-1]), (task, model.removesuffix(' (medium)')),
                   'AlgoTune page task/model identity independently confirms the filename mapping')
            messages, best_files = _algotune_html(text)
            native[model, task] = dict(grade=grade, speedup=speedup, messages=messages, best_files=best_files,
                                      source_file=str(page.relative_to(directory/'raw')))
            counts.update(source_responses=1, source_successes=int(grade), source_messages=len(messages),
                          source_traces=int(bool(messages)), source_final_files=len(best_files))
    counts.update(source_items=len(definitions), source_subjects=len({model for model, task in native}))
    return native, definitions, dict(counts)


def _algotune(directory, tables, metadata, source_records=None):
    native, definitions, counts = (_algotune_source_records(directory, metadata)
                                   if source_records is None else source_records)
    subjects = {}
    for row in tables['subjects'].itertuples():
        subjects[row.subject_id] = _features(row.subject_features_extra)['model_identifier']
        _check(row.harness, 'AlgoTuner', 'AlgoTune subject harness')
    _check(Counter(subjects.values()), Counter({model: 1 for model, task in native}), 'AlgoTune complete model labels')
    items = {}
    for row in tables['items'].itertuples():
        task = row.raw_item_id
        _check(row.content, definitions[task][0], 'AlgoTune full task instruction')
        _check(json.loads(row.grading_criterion)['rule'], metadata['grading']['rule'], 'AlgoTune explicit binary rule')
        verifier = json.loads(row.verifier)
        _check(verifier['class'], 'exact_matcher', 'AlgoTune recorded code-based grading')
        _check(json.loads(verifier['spec']), {**metadata['grading']['verifiers']['task'], 'task_code': definitions[task][1]},
               'AlgoTune full task-specific verifier')
        items[row.item_id] = task
    _check(Counter(items.values()), Counter({task: 1 for task in definitions}), 'AlgoTune complete task identities')
    traces = tables['traces'].set_index('response_id').trace.to_dict()
    seen, traced = Counter(), set()
    for row in tables['responses'].itertuples():
        key = subjects[row.subject_id], items[row.item_id]
        source = native[key]
        seen[key] += 1
        _check((row.response, row.trial), (source['grade'], 1), 'AlgoTune each native task/model outcome')
        if source['messages']:
            trace = json.loads(traces[row.response_id])
            _check(set(trace), {'source_file', 'speedup', 'messages', 'best_files'}, 'AlgoTune trace fields')
            _check((trace['source_file'], trace['speedup'], trace['best_files']),
                   (source['source_file'], source['speedup'], source['best_files']), 'AlgoTune native score, source and final files')
            messages = []
            for message in trace['messages']:
                _check(set(message), {'role', 'html'}, 'AlgoTune message fields')
                extracted, _ = _algotune_html(message['html'])
                _check(len(extracted), 1, 'AlgoTune one native block per message')
                _check(message['role'], extracted[0][0], 'AlgoTune message role')
                messages.extend(extracted)
            _check(messages, source['messages'], 'AlgoTune all conversation content and message order')
            traced.add(row.response_id)
        else:
            _check(row.response_id not in traces, True, 'AlgoTune absent conversation remains absent')
    _check(seen, Counter({key: 1 for key in native}), 'AlgoTune every native task/model result exactly once')
    _check(set(traces), traced, 'AlgoTune all trace associations')
    return counts



def _aider_source_records(directory, metadata):
    """Read native attempts and task definitions without the builder's joins."""
    import csv
    import hashlib
    import io
    from zipfile import ZipFile

    raw = directory / "raw"
    layout = metadata["build"]["parameters"]["layout"]
    task_root = raw / layout["tasks"] / "cpp/exercises/practice"
    records, definitions, trials = {}, {}, Counter()
    with ZipFile(raw / layout["archive"]) as archive:
        prefix = metadata["build"]["parameters"]["archive"]["root"] + "/"
        names = set(archive.namelist())
        with archive.open(prefix + "experiments_data/all_functional_tests.csv") as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
        published = {(row["build_id"], row["testcase"]): row for row in rows}
        _check(len(published), len(rows), "Aider unique published run/task rows")
        for member in sorted(name for name in names if name.endswith("/.aider.results.json")):
            relative = member.removeprefix(prefix)
            _, run, language, _, _, task, _ = relative.split("/")
            _check(language, "cpp", "Aider captured C++ cohort")
            native = json.loads(archive.read(member))
            _check(native["testcase"], task, "Aider task identity in filename and record")
            _check(native["testdir"], f"/benchmarks/{run}/cpp/exercises/practice/{task}", "Aider original run directory")
            row = published[run.split("--", 1)[1], task]
            _check((native["model"], native["edit_format"]), (row["model"], row["edit_format"]), "Aider native versus published model configuration")
            outcomes = native["tests_outcomes"]
            _check(1 <= len(outcomes) <= 2 and all(type(value) is bool for value in outcomes), True, "Aider native test-feedback cycles")
            grade = float(outcomes[-1])
            _check(grade, {"True": 1., "False": 0.}[row["pass2"]], "Aider native versus published verdict")

            stem = member.rsplit("/", 1)[0] + "/"
            history = archive.read(stem + ".aider.chat.history.md").decode("utf-8")
            prompt_lines = []
            for line in history.splitlines():
                if line.startswith("####"):
                    prompt_lines.append(line)
                elif prompt_lines:
                    break
            recorded_prompt = "\n".join(prompt_lines)
            boundary = "Only use standard libraries, don't suggest installing any packages."
            _check(boundary in recorded_prompt, True, "Aider complete first user prompt")
            suffix = recorded_prompt.split(boundary, 1)[1]
            header = history.split("\n####", 1)[0].splitlines()
            version = next(line.removeprefix("> Aider ").strip() for line in header if line.startswith("> Aider "))
            weak = next((line.removeprefix("> Weak model: ").strip() for line in header if line.startswith("> Weak model: ")), None)
            banner = next(line.split(": ", 1)[1].strip() for line in header if line.startswith(("> Model: ", "> Main model: ")))
            expected_banner = f"{native['model']} with {native['edit_format']} edit format"
            _check(banner == expected_banner or banner.startswith(expected_banner + ", "),
                   True, "Aider recorded main model and edit format")
            features = {"harness": "Aider", "harness_version": version, "harness_commit": native["commit_hash"],
                        "model_identifier": native["model"], "weak_model": weak,
                        "model_banner": banner,
                        "prompting_condition": row["experiment"], "edit_format": native["edit_format"],
                        "prompt_variant": hashlib.sha256(suffix.encode("utf-8")).hexdigest()}
            features = {key: value for key, value in features.items() if value is not None}
            for key in ["reasoning_effort", "thinking_tokens"]:
                if native.get(key) is not None:
                    features[key] = str(native[key])
            configuration = tuple(sorted(features.items()))

            config = json.loads(archive.read(stem + ".meta/config.json"))
            local = task_root / task
            _check(json.loads((local / ".meta/config.json").read_text()), config, "Aider task configuration matches the captured run")
            instructions = ""
            for name in ["introduction.md", "instructions.md", "instructions.append.md"]:
                filename = ".docs/" + name
                available = stem + filename in names
                _check((local / filename).exists(), available, "Aider optional instruction sections")
                if available:
                    text = archive.read(stem + filename).decode("utf-8")
                    _check((local / filename).read_text(), text, "Aider captured instruction text")
                    instructions += text
            tests = {name: archive.read(stem + name).decode("utf-8") for name in config["files"]["test"]}
            references = {name: archive.read(stem + name).decode("utf-8") for name in config["files"].get("example", [])}
            for name, text in {**tests, **references}.items():
                _check((local / name).read_text(), text, "Aider captured tests and reference implementation")
            cmake = archive.read(stem + "CMakeLists.txt").decode("utf-8")
            postbuild = archive.read(prefix + f"polyglot_artifacts/{run}/cpp/cpptest-postbuild.cmake").decode("utf-8")
            _check((local / "CMakeLists.txt").read_text(), cmake, "Aider captured CMake test configuration")
            _check((raw / layout["tasks"] / "cpp/cpptest-postbuild.cmake").read_text(), postbuild, "Aider captured post-build hook")
            initial = {name: (local / name).read_text() for name in config["files"]["solution"]}
            definition = {"content": {"instructions": instructions, "initial_files": initial},
                          "references": references, "test_files": tests, "cmake": cmake, "postbuild": postbuild}
            if task in definitions:
                _check(definitions[task], definition, "Aider consistent stimulus and grading across runs")
            else:
                definitions[task] = definition
            final_files = {name: archive.read(stem + name).decode("utf-8") for name in config["files"]["solution"]}
            trials[configuration, task] += 1
            records[relative] = {"native_record": native, "published_record": row, "history": history,
                                 "additional_instructions_md": suffix, "final_files": final_files,
                                 "features": features, "task": task, "grade": grade, "trial": trials[configuration, task]}
    _check(len(records), len(published), "Aider every published attempt has a native record")
    return records, definitions


def _aider(directory, tables, metadata, source_records=None):
    native, definitions = (_aider_source_records(directory, metadata) if source_records is None else source_records)
    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    items = tables["items"].set_index("item_id").to_dict("index")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    protocol = metadata["grading"]["verifiers"]["functional_tests"]
    _check(protocol["max_test_cycles"], 2, "Aider grading budget")
    _check(protocol["result_field"], "tests_outcomes[-1] (agrees with the published pass2 column)", "Aider grading field")
    seen, seen_items, seen_subjects = Counter(), set(), set()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_member"]
        original = native[key]
        expected_trace = {name: original[name] for name in [
            "native_record", "published_record", "history", "additional_instructions_md", "final_files"]}
        expected_trace.update(source_member=key, source_archive=metadata["build"]["parameters"]["layout"]["archive"])
        _check(trace, expected_trace, "Aider full native record, conversation and final files without truncation")
        _check(response.response, original["grade"], "Aider preserved native grade")
        _check(response.trial, original["trial"], "Aider distinct trials in source-directory order")
        _check(pd.isna(response.test_condition) and pd.isna(response.interactors), True, "Aider no invented run conditions")
        subject = subjects[response.subject_id]
        actual_features = _features(subject["subject_features_extra"])
        for field in ["harness", "harness_version", "reasoning_effort"]:
            if not pd.isna(subject[field]):
                actual_features[field] = subject[field]
        _check(actual_features, original["features"], "Aider complete recorded agent configuration and prompt variant")
        _check(subject["display_name"], original["native_record"]["model"], "Aider unmodified model label")

        item = items[response.item_id]
        definition = definitions[original["task"]]
        _check(item["raw_item_id"], "cpp/" + original["task"], "Aider response-to-exercise association")
        _check(json.loads(item["content"]), definition["content"], "Aider full instructions and initial source files")
        _check(_features(item["item_features"]), {"lang": "cpp"}, "Aider task language")
        criterion = json.loads(item["grading_criterion"])
        _check(criterion["rule"], metadata["grading"]["rule"], "Aider grading criterion")
        _check(json.loads(criterion["reference_answer"]), definition["references"], "Aider released reference implementations")
        specification = json.loads(json.loads(item["verifier"])["spec"])
        _check(specification, {**protocol, **{key: definition[key] for key in ["test_files", "cmake", "postbuild"]}},
               "Aider exact per-task tests, CMake build and post-build hook")
        seen[key] += 1
        seen_items.add(response.item_id)
        seen_subjects.add(response.subject_id)
    _check(seen, Counter({key: 1 for key in native}), "Aider complete source attempt census without duplication")
    _check(len(traces), len(native), "Aider complete response-trace associations")
    _check(seen_items, set(items), "Aider exact used task definitions")
    _check(seen_subjects, set(subjects), "Aider exact used model configurations")
    configurations = {tuple(sorted(row["features"].items())) for row in native.values()}
    _check(len(subjects), len(configurations), "Aider separate identities for recorded prompting variants")
    return {"source_responses": len(native), "source_successes": int(sum(row["grade"] for row in native.values())),
            "source_items": len(definitions), "source_subjects": len(configurations),
            "source_models": len({row["native_record"]["model"] for row in native.values()}),
            "source_conditions": len({row["published_record"]["experiment"] for row in native.values()}),
            "source_runs": len({row["published_record"]["build_id"] for row in native.values()}),
            "source_traces": len(native), "source_final_files": sum(len(row["final_files"]) for row in native.values()),
            "source_test_cycles": sum(len(row["native_record"]["tests_outcomes"]) for row in native.values())}


def _alpaca_source_records(directory, metadata):
    """Read native annotations and reference texts without the builder's table operations."""
    raw = directory / "raw"
    layout = metadata["build"]["parameters"]["layout"]
    release = raw / layout["release"]
    reference = json.loads((release / layout["reference"]).read_text())
    native = {}
    for path in sorted((release / "results").glob("*/weighted_alpaca_eval_gpt4_turbo/annotations.json")):
        for position, record in enumerate(json.loads(path.read_text())):
            native[str(path.relative_to(raw)), position] = record
    copies = 0
    for path in sorted((release / "results").glob(
            "*/weighted_alpaca_eval_gpt4_turbo/weighted_alpaca_eval_gpt4_turbo/annotations.json")):
        original = json.loads((path.parent.parent / "annotations.json").read_text())
        augmented = json.loads(path.read_text())
        _check([{key: value for key, value in row.items() if key != "glm_preference"}
                for row in augmented], original, "AlpacaEval nested export is the same original judgments")
        copies += 1
    _check((len(native), len(reference), copies), (177892, 805, 2), "AlpacaEval complete pinned native release")
    return native, reference, copies


def _alpacaeval(directory, tables, metadata, source_records=None):
    """Check every native comparison, its fixed opponent, grade and full judge record."""
    import math
    from urllib.parse import unquote

    native, reference, copies = (
        _alpaca_source_records(directory, metadata) if source_records is None else source_records)
    bank = {row["instruction"]: (position, row) for position, row in enumerate(reference)}
    _check(len(bank), len(reference), "AlpacaEval unique reference instructions")
    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    expected_models = {row["generator_2"] for row in native.values()}
    actual_models = []
    for subject in subjects.values():
        features = _features(subject["subject_features_extra"])
        _check(set(features), {"model_identifier"}, "AlpacaEval reported model identity")
        _check(subject["harness"], "AlpacaEval 2.0", "AlpacaEval source harness")
        actual_models.append(unquote(features["model_identifier"]))
    _check(Counter(actual_models), Counter({model: 1 for model in expected_models}),
           "AlpacaEval distinct reported generators without alias collapse")

    protocol = dict(metadata["grading"]["verifiers"]["weighted_preference"])
    layout = metadata["build"]["parameters"]["layout"]
    protocol["prompt_template"] = (directory / "raw" / layout["release"] / layout["judge_prompt"]).read_text()
    _check(protocol["judge_model"], "gpt-4-1106-preview", "AlpacaEval judge model")
    items = tables["items"].set_index("item_id").to_dict("index")
    for item in items.values():
        position, original = bank[item["content"]]
        _check(item["raw_item_id"], f"alpaca_eval:{position}", "AlpacaEval original reference-bank position")
        _check(_features(item["item_features"]), {"dataset": original["dataset"]}, "AlpacaEval task provenance")
        criterion = json.loads(item["grading_criterion"])
        _check(criterion.get("reference_answer"), None, "AlpacaEval reference is not a gold correctness answer")
        _check(json.loads(criterion["rule"]), {"rule": metadata["grading"]["rule"],
               "reference_model": "gpt4_1106_preview", "reference_output": original["output"]},
               "AlpacaEval complete fixed reference and grading rule")
        verifier = json.loads(item["verifier"])
        _check(verifier["class"], "judge", "AlpacaEval judgment-based verifier")
        _check(json.loads(verifier["spec"]), protocol, "AlpacaEval full judge prompt and configuration")

    traces = tables["traces"].set_index("response_id").trace.to_dict()
    seen, used_items, counts = Counter(), set(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_file"], trace["source_row"]
        original = native[key]
        _check(trace, {"source_file": key[0], "source_row": key[1], "native_record": original},
               "AlpacaEval complete original record without clipping or invented fields")
        _check(original["generator_1"], "gpt4_1106_preview", "AlpacaEval original opponent orientation")
        _check(original["annotator"], "weighted_alpaca_eval_gpt4_turbo", "AlpacaEval native annotator")
        _check(original.get("input") in (None, ""), True, "AlpacaEval no omitted extra task input")
        _check(unquote(_features(subjects[response.subject_id]["subject_features_extra"])["model_identifier"]),
               original["generator_2"], "AlpacaEval correct response-to-generator association")
        item = items[response.item_id]
        _check(item["content"], original["instruction"], "AlpacaEval correct response-to-instruction association")
        _check(bank[original["instruction"]][1]["output"], original["output_1"], "AlpacaEval unchanged fixed opponent output")
        preference = original["preference"]
        valid = isinstance(preference, (int, float)) and math.isfinite(preference) and 1 <= preference <= 2
        if valid:
            _check(response.response, float(preference) - 1.0, "AlpacaEval exact native soft preference")
            counts["source_graded"] += 1
        else:
            _check(preference is None or (preference == -1 and original.get("raw_completion") is None),
                   True, "AlpacaEval only reviewed unavailable judgments may lack a grade")
            _check(pd.isna(response.response), True, "AlpacaEval unavailable or invalid preference remains ungraded")
            counts["source_ungraded"] += 1
            counts["source_invalid_preferences"] += preference is not None
        _check(response.trial, 1, "AlpacaEval one primary annotation per generator and instruction")
        _check(pd.isna(response.test_condition), True, "AlpacaEval grading protocol belongs to the item")
        _check(response.interactors, "opponent=gpt4_1106_preview", "AlpacaEval fixed comparison partner")
        counts["source_identical_outputs"] += original["output_1"] == original["output_2"]
        seen[key] += 1
        used_items.add(response.item_id)
    _check(seen, Counter({key: 1 for key in native}), "AlpacaEval exact census without omitted or duplicated comparisons")
    _check(len(traces), len(native), "AlpacaEval complete trace associations")
    _check((len(items), len(used_items)), (len(bank), len(bank)), "AlpacaEval full instruction coverage")
    counts.update(source_responses=len(native), source_traces=len(native), source_subjects=len(subjects),
                  source_items=len(bank), source_derived_copies=copies)
    return dict(counts)


def _ai2d_source_records(directory, metadata):
    """Read native Excel cells and the official-checksum task TSV independently."""
    import csv
    import hashlib
    from openpyxl import load_workbook

    raw = directory / "raw"
    layout = metadata["build"]["parameters"]["layout"]
    release = raw / layout["release"]
    task_file = raw / layout["tasks"]
    _check(hashlib.md5(task_file.read_bytes()).hexdigest(), "0f593e0d1c7df9a3d69bf1f947e71975",
           "AI2D original harness task checksum, including every diagram")
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(10000000)
        with task_file.open(newline="") as stream:
            bank = {int(row["index"]): row for row in csv.DictReader(stream, delimiter="\t")}
    finally:
        csv.field_size_limit(previous_limit)

    def cells(path):
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook.active.iter_rows(values_only=True)
            names = next(rows)
            return [dict(zip(names, ("" if value is None else value for value in row), strict=True)) for row in rows]
        finally:
            workbook.close()

    native, models = {}, set()
    for path in sorted(release.glob("mmeval/*/*_AI2D_TEST.xlsx")):
        model = path.name.removesuffix("_AI2D_TEST.xlsx")
        models.add(model)
        seen = set()
        for position, row in enumerate(cells(path)):
            _check(row["index"] not in seen, True, "AI2D unique native question within primary model export")
            seen.add(row["index"])
            native[str(path.relative_to(raw)), position] = row
        _check(seen, set(bank), "AI2D all source questions in each primary model export")
    published = {row["index"]: row for row in cells(release / layout["published_grades"])}
    _check((len(native), len(models), len(bank), len(published)), (784352, 254, 3088, 3088),
           "AI2D complete pinned primary predictions, task bank, and native grades")
    return native, bank, published


def _ai2d_answer(text, choices):
    """Scalar form of the reviewed historical upstream matcher (VERBOSE disabled)."""
    if "Failed to obtain answer via API" in text:
        return None
    if any(marker in text for marker in ("Sorry, I can't help with images of people yet.",
            "I can't process this file.", "I'm sorry, but without the image provided", "Cannot determine the answer")):
        return "Z"
    tokens = text
    for character in ".()[],:;!*#{}":
        tokens = tokens.replace(character, " ")
    words = tokens.split()
    present = [letter for letter in choices if letter in words]
    if len(present) == 1:
        return present[0]
    if not present and "Z" in words:
        return "Z"
    present = [letter for letter, option in choices.items() if option.lower() in text.lower()]
    return present[0] if len(present) == 1 else "Z"


def _ai2d_test(directory, tables, metadata, source_records=None):
    """Check every original record, task/image association, and deterministic grade."""
    import base64
    import hashlib
    from urllib.parse import unquote

    native, bank, published = _ai2d_source_records(directory, metadata) if source_records is None else source_records
    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    models = {key: unquote(_features(row["subject_features_extra"])["model_identifier"])
              for key, row in subjects.items()}
    _check(Counter(models.values()), Counter({Path(key[0]).name.removesuffix("_AI2D_TEST.xlsx"): 1 for key in native}),
           "AI2D all reported models remain distinct")
    for row in subjects.values():
        _check(row["harness"], "VLMEvalKit", "AI2D recorded harness")
    items = tables["items"].set_index("item_id").to_dict("index")
    assets = tables["assets"].set_index("asset_id").to_dict("index")
    item_indices, used_assets = {}, set()
    for item_id, item in items.items():
        index = int(item["raw_item_id"].removeprefix("ai2d_test_"))
        source = bank[index]
        options = {letter: source[letter] for letter in "ABCD" if source[letter] not in ("", "None", "NA")}
        text = "Question: " + source["question"] + "\nOptions:\n"
        text += "".join(f"{letter}. {option}\n" for letter, option in options.items())
        text += "Please select the correct answer from the options above. \n"
        data = base64.b64decode(source["image"], validate=True)
        image_path = "images/" + hashlib.sha256(data).hexdigest() + ".jpg"
        _check(json.loads(item["content"]), {"multimedia_elements": [
            {"content_type": "image/jpeg", "location": image_path},
            {"content_type": "text/plain", "text": text}]}, "AI2D exact standard task text and image reference")
        links = json.loads(item["asset_manifest"])
        _check(len(links), 1, "AI2D one complete diagram per task")
        _check(links[0]["path"], image_path, "AI2D matching image path")
        _check(assets[links[0]["asset_id"]]["data"], data, "AI2D exact unmodified diagram bytes")
        used_assets.add(links[0]["asset_id"])
        _check(_features(item["item_features"]), {"category": source["category"],
               "abc_label": source["abcLabel"].lower(), "source_image_path": source["image_path"]},
               "AI2D original category and image condition")
        criterion = json.loads(item["grading_criterion"])
        _check(criterion["reference_answer"], source["answer"], "AI2D original correct option")
        _check(criterion["rule"], metadata["grading"]["rule"], "AI2D documented deterministic grading rule")
        _check(json.loads(json.loads(item["verifier"])["spec"]), metadata["grading"]["verifiers"]["exact_matching"],
               "AI2D fixed historical grading implementation")
        item_indices[item_id] = index
    _check(set(assets), used_assets, "AI2D no missing or orphan image assets")
    _check((len(items), len(used_assets)), (3088, 1201), "AI2D complete tasks and distinct diagrams")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    seen, counts = Counter(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_file"], trace["source_row"]
        row = native[key]
        model = Path(key[0]).name.removesuffix("_AI2D_TEST.xlsx")
        _check(models[response.subject_id], model, "AI2D response belongs to its original model")
        _check(item_indices[response.item_id], row["index"], "AI2D response belongs to its original question and diagram")
        source = bank[row["index"]]
        for column in ("question", "A", "B", "C", "D", "answer", "category", "abcLabel", "image_path"):
            original, reference = row[column], source[column]
            if column in "ABCD":
                original = "" if original in ("", "None", "NA") else original
                reference = "" if reference in ("", "None", "NA") else reference
            _check(str(original), reference, "AI2D unchanged native task field: " + column)
        choices = {letter: source[letter] for letter in "ABCD" if source[letter] not in ("", "None", "NA")}
        text = str(row["prediction"])
        unavailable = not text or "Failed to obtain answer via API" in text
        answer = None if unavailable else _ai2d_answer(text, choices)
        native_grade = published[row["index"]] if model == "GPT4o" else None
        status = "unavailable_output" if unavailable else "published_exact_matching" if native_grade else "derived_exact_matching"
        _check(trace, {"source_file": key[0], "source_row": key[1], "native_record": row,
               "grade_status": status, "extracted_answer": answer, "published_record": native_grade},
               "AI2D full native output and grading record without truncation")
        if unavailable:
            _check(pd.isna(response.response), True, "AI2D unavailable output remains ungraded")
        else:
            _check(response.response, float(answer == source["answer"]), "AI2D exact deterministic grade")
        if native_grade:
            _check(native_grade["prediction"], row["prediction"], "AI2D published grader evaluated the same output")
            _check(response.response, float(native_grade["hit"]), "AI2D unchanged published GPT-4o grade")
        _check(response.trial, 1, "AI2D one maintained primary record per model and task")
        _check(pd.isna(response.test_condition), True, "AI2D no invented inference setting")
        seen[key] += 1
        counts["source_ungraded"] += unavailable
        counts["source_published_grades"] += native_grade is not None
    _check(seen, Counter({key: 1 for key in native}), "AI2D complete source census without dropped or duplicate records")
    _check(len(traces), len(native), "AI2D complete trace coverage")
    counts.update(source_responses=len(native), source_traces=len(native), source_subjects=len(subjects),
                  source_items=len(items), source_assets=len(assets))
    return dict(counts)


def _alpha_sql_source_records(directory, metadata):
    """Read original JSON/archive records and independently execute the released SQL."""
    import concurrent.futures
    import hashlib
    import io
    import sqlite3
    import tempfile
    import time
    from zipfile import ZipFile

    raw = directory / "raw"
    layout = metadata["build"]["parameters"]["layout"]
    predictions = json.loads((raw / layout["predictions"]).read_text())
    configuration = yaml.safe_load((raw / layout["configuration"]).read_text())
    payloads, file_lists = {}, {}
    with ZipFile(raw / layout["tasks"]) as archive:
        questions = {str(row["question_id"]): row for row in json.loads(archive.read(layout["questions"]))}
        schemas = {row["db_id"]: row for row in json.loads(archive.read(layout["schemas"]))}
        with ZipFile(io.BytesIO(archive.read(layout["databases"]))) as databases:
            for db in {row["db_id"] for row in questions.values()}:
                prefix = f"dev_databases/{db}/"
                names = [name for name in databases.namelist()
                         if name.startswith(prefix) and name.endswith((".csv", ".sqlite"))]
                file_lists[db] = names
                for name in names:
                    payloads[name] = databases.read(name)
    _check(set(predictions), set(questions), "Alpha-SQL every released query matches one BIRD question")
    _check((len(questions), len(schemas)), (1534, 11), "Alpha-SQL complete original task/database census")
    _check(all(isinstance(value, str) and value for value in predictions.values()), True,
           "Alpha-SQL original predictions are complete SQL strings")
    timeout = float(metadata["grading"]["verifiers"]["execution"]["timeout_per_query_seconds"])

    with tempfile.TemporaryDirectory(prefix=".alpha-sql-audit-", dir=directory.parent) as temporary:
        temporary = Path(temporary)
        for db in file_lists:
            (temporary / (db + ".sqlite")).write_bytes(payloads[f"dev_databases/{db}/{db}.sqlite"])

        def execute(task):
            key, question = task
            connection = sqlite3.connect((temporary / (question["db_id"] + ".sqlite")).as_uri() + "?mode=ro", uri=True)
            connection.execute("PRAGMA query_only = ON")
            outcomes, rows = {}, {}
            try:
                # This scalar checker uses SQLite's progress callback, independently
                # of the builder's timer and query-table joins.
                for kind, query in (("prediction", predictions[key]), ("gold", question["SQL"])):
                    deadline = time.monotonic() + timeout
                    connection.set_progress_handler(lambda: time.monotonic() >= deadline, 1000)
                    try:
                        rows[kind] = set(connection.execute(query).fetchall())
                        outcomes[kind + "_status"], outcomes[kind + "_error"] = "ok", None
                    except sqlite3.Error as error:
                        rows[kind] = None
                        outcomes[kind + "_status"] = "timeout" if error.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT else "error"
                        outcomes[kind + "_error"] = str(error)
                    outcomes[kind + "_distinct_rows"] = None if rows[kind] is None else len(rows[kind])
            finally:
                connection.close()
            grade = None if rows["gold"] is None else float(rows["prediction"] is not None and rows["prediction"] == rows["gold"])
            outcomes["sqlite_version"] = sqlite3.sqlite_version
            return key, {"grade": grade, "execution": outcomes}

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            executions = dict(pool.map(execute, questions.items()))
    return predictions, questions, schemas, configuration, payloads, file_lists, executions


def _alpha_sql(directory, tables, metadata, source_records=None):
    """Check all source-to-table links, binary grades, SQL text, and database bytes."""
    import hashlib

    predictions, questions, schemas, configuration, payloads, file_lists, executions = (
        _alpha_sql_source_records(directory, metadata) if source_records is None else source_records)
    subjects = tables["subjects"].set_index("subject_id").to_dict("index")
    _check(len(subjects), 1, "Alpha-SQL one released system configuration")
    model = configuration["mcts_model_kwargs"]
    expected_features = {key: str(value) for key, value in model.items() if key not in ("model", "temperature")}
    expected_features.update({key: str(configuration[key]) for key in ("max_rollout_steps", "max_depth", "exploration_constant")})
    expected_features["model_identifier"] = model["model"]
    subject = next(iter(subjects.values()))
    _check(_features(subject["subject_features_extra"]), expected_features, "Alpha-SQL exact released inference and search settings")
    _check(subject["harness"], "Alpha-SQL", "Alpha-SQL scaffold is part of subject identity")
    _check(subject["harness_version"], "216e61fc86467591fa1dc5b7031ca536577b4d21", "Alpha-SQL captured harness revision")
    items = tables["items"].set_index("item_id").to_dict("index")
    assets = tables["assets"].set_index("asset_id").to_dict("index")
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    actual_hashes = {key: hashlib.sha256(row["data"]).hexdigest() for key, row in assets.items()}
    _check(set(actual_hashes.values()), set(source_hashes.values()), "Alpha-SQL exact complete database and description assets")
    item_keys, seen_items, used_assets = {}, Counter(), set()
    for item_id, item in items.items():
        key = item["raw_item_id"]
        question = questions[key]
        db = question["db_id"]
        _check(json.loads(item["content"]), {"question": question["question"], "evidence": question["evidence"],
            "database_id": db, "schema": schemas[db], "input_files": file_lists[db]},
            "Alpha-SQL full original question, evidence, schema and database inputs")
        _check(_features(item["item_features"]), {"database_id": db, "difficulty": question["difficulty"]},
               "Alpha-SQL original database and difficulty")
        _check(json.loads(item["grading_criterion"]), {"reference_answer": question["SQL"], "rule": metadata["grading"]["rule"]},
               "Alpha-SQL complete original reference query and grading rule")
        _check(json.loads(json.loads(item["verifier"])["spec"]), metadata["grading"]["verifiers"]["execution"],
               "Alpha-SQL explicit derived grading protocol")
        links = json.loads(item["asset_manifest"])
        _check([link["path"] for link in links], file_lists[db], "Alpha-SQL exact input-file association and order")
        for ordinal, link in enumerate(links, 1):
            name = link["path"]
            _check(actual_hashes[link["asset_id"]], source_hashes[name], "Alpha-SQL unmodified corresponding database/description bytes")
            _check((link["media_type"], link["role"], link["ordinal"]),
                   ("application/vnd.sqlite3" if name.endswith(".sqlite") else "text/csv", "input", ordinal),
                   "Alpha-SQL typed input attachment")
            used_assets.add(link["asset_id"])
        item_keys[item_id] = key
        seen_items[key] += 1
    _check(seen_items, Counter({key: 1 for key in questions}), "Alpha-SQL every complete task exactly once")
    _check(set(assets), used_assets, "Alpha-SQL no missing or orphaned assets")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    seen, counts = Counter(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = trace["source_key"]
        _check(trace, {"source_file": metadata["build"]["parameters"]["layout"]["predictions"],
            "source_key": key, "prediction": predictions[key], "native_question": questions[key],
            "configuration": configuration, "derived_execution": executions[key]["execution"]},
            "Alpha-SQL full original SQL, task record, configuration and independently checked execution")
        _check(item_keys[response.item_id], key, "Alpha-SQL matching original question, never positional zip")
        _check(response.subject_id in subjects, True, "Alpha-SQL correct evaluated system")
        expected = executions[key]["grade"]
        if expected is None:
            _check(pd.isna(response.response), True, "Alpha-SQL failed reference remains an ungraded attempt")
            counts["source_ungraded"] += 1
        else:
            _check(response.response, expected, "Alpha-SQL complete execution result-set comparison")
            counts["source_successes"] += expected == 1
        _check(response.trial, 1, "Alpha-SQL exactly one recorded final prediction per task")
        _check(response.test_condition, "temperature=" + str(model["temperature"]), "Alpha-SQL recorded sampling temperature")
        seen[key] += 1
    _check(seen, Counter({key: 1 for key in predictions}), "Alpha-SQL complete released attempts without duplication or omission")
    _check(len(traces), len(predictions), "Alpha-SQL complete trace coverage")
    counts.update(source_responses=len(predictions), source_items=len(questions), source_subjects=1,
                  source_traces=len(predictions), source_databases=len(schemas),
                  source_input_files=len(payloads), source_assets=len(set(source_hashes.values())))
    return dict(counts)


def _alignment_faking_source_records(directory, metadata):
    """Read scalar Arrow records and the complete published prompt columns independently."""
    import hashlib
    from html.parser import HTMLParser
    import pyarrow.parquet as pq

    class PromptTable(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.active = self.cell = False
            self.rows, self.row, self.parts = [], [], []

        def handle_starttag(self, tag, attrs):
            if tag == "div" and dict(attrs).get("class") == "system-prompt-table":
                self.active = True
            if self.active and tag == "tr":
                self.row = []
            if self.active and tag == "td":
                self.cell, self.parts = True, []

        def handle_data(self, data):
            if self.cell and data.strip():
                self.parts.append(data.strip())

        def handle_endtag(self, tag):
            if self.active and tag == "td":
                self.row.append("\n".join(self.parts))
                self.cell = False
            if self.active and tag == "tr":
                self.rows.append(self.row)
            if self.active and tag == "table":
                self.active = False

    raw = directory / "raw"
    layout = metadata["build"]["parameters"]["layout"]
    parser = PromptTable()
    parser.feed((raw / layout["article"]).read_text())
    _check([len(row) for row in parser.rows], [3, 3, 3, 3], "Alignment Faking full published prompt table")
    prompts = {name: "\n\n".join(row[column] for row in parser.rows[1:] if row[column])
               for name, column in (("No example no draft", 0), ("Singleturn", 1))}
    _check("=== START EXAMPLES ===" in prompts["Singleturn"], True, "Alignment Faking retains all few-shot examples")
    _check(all("<is_free_tier_user>" in text for text in prompts.values()), True,
           "Alignment Faking complete system prompts include the user-tier rule")
    native, configurations, contexts, counts = {}, {}, set(), Counter()
    files = sorted((raw / layout["transcripts"]).glob("*.parquet"))
    _check(len(files), 155, "Alignment Faking all original run exports")
    for path in files:
        for position, record in enumerate(pq.read_table(path).to_pylist()):
            key = str(path.relative_to(raw)), position
            grade = record["alignment_faking"]
            _check(grade is None or type(grade) is bool, True, "Alignment Faking nullable boolean source labels")
            _check(record["task_id"], "alignment_faking_eval", "Alignment Faking original evaluation scope")
            _check(record["response"].startswith("<canary>"), True, "Alignment Faking preserves native transcript protection")
            context = record["user_input"], record["is_free_tier_user"], record.get("system_prompt")
            configuration = path.stem, record["step"]
            features = {"model_identifier": path.stem, "training_step": str(record["step"]),
                        **{field: record[field] for field in ("initialised_from", "experiment_name", "intervention")}}
            _check(configurations.setdefault(configuration, features), features,
                   "Alignment Faking consistent released run/checkpoint configuration")
            payload = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            native[key] = (configuration, context, grade, hashlib.sha256(payload).hexdigest())
            contexts.add(context)
            counts["source_ungraded"] += grade is None
            counts["source_positive_judgments"] += grade is True
            counts["source_negative_judgments"] += grade is False
            counts["source_unknown_prompt_records"] += context[2] is None
    _check((len(native), len(configurations), len(contexts), counts["source_ungraded"]),
           (2138969, 4278, 1464, 45120), "Alignment Faking full original observation census")
    counts.update(source_responses=len(native), source_traces=len(native), source_runs=len(files),
                  source_subjects=len(configurations), source_items=len(contexts),
                  source_user_questions=len({context[0] for context in contexts}), source_published_prompt_variants=len(prompts))
    return native, configurations, contexts, prompts, dict(counts)


def _alignment_faking(directory, tables, metadata, source_records=None):
    """Check every source record, grade, run/checkpoint, input context and encoded trace."""
    import hashlib

    native, configurations, contexts, prompts, counts = (
        _alignment_faking_source_records(directory, metadata) if source_records is None else source_records)
    _check((len(tables["responses"]), len(tables["traces"])), (len(native), len(native)),
           "Alignment Faking no omitted or extra native observations")
    subjects, seen_subjects = {}, Counter()
    for row in tables["subjects"].itertuples():
        features = _features(row.subject_features_extra)
        key = features["model_identifier"], int(features["training_step"])
        _check(features, configurations[key], "Alignment Faking complete source system configuration")
        _check(row.harness, "Anthropic alignment-faking RL evaluation", "Alignment Faking source harness")
        _check(pd.isna(row.harness_version), True, "Alignment Faking unknown harness version stays unknown")
        subjects[row.subject_id] = key
        seen_subjects[key] += 1
    _check(seen_subjects, Counter({key: 1 for key in configurations}), "Alignment Faking distinct run/checkpoint identities")
    items, seen_contexts = {}, Counter()
    for row in tables["items"].itertuples():
        content = json.loads(row.content)
        key = content["user_input"], content["is_free_tier_user"], content["system_prompt_name"]
        _check(key in contexts, True, "Alignment Faking source input context exists")
        _check(content, {"user_input": key[0], "is_free_tier_user": key[1], "system_prompt_name": key[2],
                         "published_system_prompt": prompts.get(key[2])}, "Alignment Faking complete published stimulus")
        _check(json.loads(row.grading_criterion), {"reference_answer": None, "rule": metadata["grading"]["rule"]},
               "Alignment Faking behavioral grading rule")
        verifier = json.loads(row.verifier)
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["alignment_faking"],
               "Alignment Faking published classifier description")
        _check((verifier["judge"], verifier["judged_by"]),
               ("Sonnet 4-based alignment-faking classifier", "llm"), "Alignment Faking actual reported judge")
        items[row.item_id] = key
        seen_contexts[key] += 1
    _check(seen_contexts, Counter({key: 1 for key in contexts}), "Alignment Faking all available prompt/tier contexts")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    _check(set(traces), set(tables["responses"].response_id), "Alignment Faking complete one-to-one trace linkage")
    seen, trials = Counter(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        _check(set(trace), {"source_file", "source_row", "record"}, "Alignment Faking complete native trace envelope")
        key = trace["source_file"], trace["source_row"]
        configuration, context, grade, digest = native[key]
        payload = json.dumps(trace["record"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        _check(hashlib.sha256(payload).hexdigest(), digest, "Alignment Faking full original record and encoded transcript")
        _check((subjects[response.subject_id], items[response.item_id]), (configuration, context),
               "Alignment Faking correct source-to-system-and-item associations")
        _check(pd.isna(response.response) if grade is None else response.response == float(grade), True,
               "Alignment Faking native boolean judgments and explicit nulls")
        _check(pd.isna(response.test_condition), True, "Alignment Faking no invented response conditions")
        trials[response.subject_id, response.item_id] += 1
        _check(response.trial, trials[response.subject_id, response.item_id], "Alignment Faking repeated native attempts keep distinct trials")
        seen[key] += 1
    _check(seen, Counter({key: 1 for key in native}), "Alignment Faking exact source census without duplication")
    return counts


def _arcagi_source_records(directory, metadata):
    """Reconcile native JSON files, task history and scalar grid comparisons."""
    import hashlib
    import re
    from datetime import datetime, timezone

    raw = directory / "raw"
    parameters = metadata["build"]["parameters"]
    tasks, native, summaries, counts = {}, {}, {}, Counter()
    for snapshot, folder in parameters["tasks"].items():
        for path in sorted((raw / folder / "data/evaluation").glob("*.json")):
            puzzle = json.loads(path.read_text())
            _check({"train", "test"} <= puzzle.keys() and not (puzzle.keys() - {"train", "test", "name"}),
                   True, "ARC complete source task structure")
            _check(puzzle.get("name", path.stem), path.stem, "ARC optional native task alias")
            tasks[snapshot, path.stem] = str(path.relative_to(raw)), dict(train=puzzle["train"], test=puzzle["test"])
    for version, folder in parameters["results"].items():
        for path in sorted((raw / folder).glob("*/results.json")):
            for task_id, result in json.loads(path.read_text())["task_results"].items():
                summaries[version, path.parent.name, task_id] = dict(source_file=str(path.relative_to(raw)), record=result)
        for path in sorted((raw / folder).glob("*/*.json")):
            if not re.fullmatch("[0-9a-f]{8}", path.stem):
                continue
            payload = path.read_bytes()
            key = version, path.stem, hashlib.sha256(payload).hexdigest()
            entry = native.setdefault(key, dict(record=json.loads(payload), files=[], judgments=[], model=path.parent.name))
            _check(entry["record"], json.loads(payload), "ARC exact duplicate record agreement")
            entry["files"].append(str(path.relative_to(raw)))
            judgment = summaries.get((version, path.parent.name, path.stem))
            if judgment is not None:
                entry["judgments"].append(judgment)
            counts["source_result_files"] += 1

    configs, item_contexts = {}, {}
    for (version, task_id, digest), entry in native.items():
        configurations, temperatures, dates = {}, set(), []
        for pair in entry["record"]:
            for attempt in pair.values():
                if attempt is None:
                    continue
                recorded = attempt["metadata"]
                _check(recorded.get("task_id", task_id), task_id, "ARC recorded task identifier")
                request = dict(recorded["kwargs"])
                temperatures.add(request.pop("temperature", None))
                config = dict(model_identifier=entry["model"], api_model=recorded["model"],
                              provider=recorded["provider"], generation_parameters=request)
                configurations[json.dumps(config, sort_keys=True)] = config
                timestamp = datetime.fromisoformat(recorded["start_timestamp"])
                dates.append(timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp)
                counts["source_candidate_attempts"] += 1
        _check((len(configurations), len(temperatures)), (1, 1), "ARC one recorded configuration and temperature per task")
        configuration, features = next(iter(configurations.items()))
        configs[configuration] = features
        entry["configuration"], entry["temperature"] = configuration, next(iter(temperatures))
        snapshot = version
        if version == "v2" and min(dates) < datetime.fromisoformat(parameters["history"]["v2_cutoff"]):
            _check({date.date().isoformat() for date in dates}, {"2025-04-14"}, "ARC historical source matching is limited to the released April run")
            snapshot = "v2_20250414"
            counts["source_historical_task_associations"] += 1
        elif version == "v2":
            _check(min(dates).date().isoformat() >= "2025-07-22", True, "ARC later runs postdate all recorded task changes")
        task_file, puzzle = tasks[snapshot, task_id]
        entry["task_file"] = task_file
        entry["item"] = json.dumps([version, puzzle], sort_keys=True)
        item_contexts[entry["item"]] = version, puzzle
        covered, solved = set(), set()
        for position, pair in enumerate(entry["record"]):
            indices = {a["metadata"]["pair_index"] for a in pair.values()
                       if a is not None and a["metadata"].get("pair_index") is not None}
            _check(len(indices) <= 1, True, "ARC attempts agree on test-pair identity")
            index = next(iter(indices)) if indices else position
            _check(type(index) is int and 0 <= index < len(puzzle["test"]), True, "ARC recorded pair exists in the dated task")
            _check(index not in covered, True, "ARC no duplicated test-grid records")
            covered.add(index)
            for attempt in pair.values():
                if attempt is None:
                    continue
                matched = attempt["answer"] == puzzle["test"][index]["output"]
                if attempt.get("correct") is not None:
                    _check(attempt["correct"], matched, "ARC published candidate flag matches the exact grid")
                if matched:
                    solved.add(index)
        judgments = {row["record"]["score"] for row in entry["judgments"]}
        _check(len(judgments) <= 1, True, "ARC identical copies have the same published judgment")
        if judgments:
            entry["grade"] = next(iter(judgments))
            entry["origin"] = "published_task_judgment"
            _check(abs(entry["grade"] - len(solved) / len(puzzle["test"])) < 1e-12, True,
                   "ARC every published task judgment agrees with native grid matching")
        elif covered == set(range(len(puzzle["test"]))):
            entry["grade"] = len(solved) / len(puzzle["test"])
            entry["origin"] = "derived_exact_grid_match"
        else:
            entry["grade"], entry["origin"] = None, "ungraded_incomplete_export"
        counts["source_" + entry["origin"]] += 1
        counts["source_fractional_grades"] += entry["grade"] is not None and entry["grade"] not in (0, 1)
        counts["source_published_fractional_grades"] += bool(judgments) and entry["grade"] not in (0, 1)
        counts["source_duplicate_copies"] += len(entry["files"]) - 1
    _check((counts["source_result_files"], len(native), len(configs), len(item_contexts)),
           (35659, 35260, 80, 526), "ARC full release census and distinct task/configuration identities")
    counts.update(source_responses=len(native), source_subjects=len(configs), source_items=len(item_contexts), source_traces=len(native))
    return native, configs, item_contexts, dict(counts)


def _arcagi(directory, tables, metadata, source_records=None):
    """Check every released attempt, grade, source alias, task definition and setting."""
    import ast

    native, configurations, contexts, counts = (
        _arcagi_source_records(directory, metadata) if source_records is None else source_records)
    _check((len(tables["responses"]), len(tables["traces"])), (len(native), len(native)), "ARC complete unique observation coverage")
    _check(json.loads(tables["benchmarks"].iloc[0].response_scale),
           dict(kind="interval", min=0, max=1, direction="higher_is_better"), "ARC preserves its fractional scoring scale")
    subjects, seen_subjects = {}, Counter()
    for row in tables["subjects"].itertuples():
        features = _features(row.subject_features_extra)
        features["generation_parameters"] = ast.literal_eval(features["generation_parameters"])
        key = json.dumps(features, sort_keys=True)
        _check(key in configurations, True, "ARC exact native model/provider/request configuration")
        _check(row.harness, "ARC Prize benchmarking", "ARC source harness identity")
        _check(pd.isna(row.harness_version), True, "ARC unknown historical harness stays unknown")
        subjects[row.subject_id] = key
        seen_subjects[key] += 1
    _check(seen_subjects, Counter({key: 1 for key in configurations}), "ARC no collapsed model configurations")
    items, seen_items = {}, Counter()
    for row in tables["items"].itertuples():
        content = json.loads(row.content)
        _check(set(content), {"train", "test"}, "ARC full stimulus structure")
        _check(all(set(pair) == {"input"} for pair in content["test"]), True, "ARC test solutions never enter stimulus text")
        criterion = json.loads(row.grading_criterion)
        references = json.loads(criterion["reference_answer"])
        _check(criterion["rule"], metadata["grading"]["rule"], "ARC correct grading rule")
        _check(len(references), len(content["test"]), "ARC one reference grid per test input")
        version = _features(row.item_features)["split"]
        puzzle = dict(train=content["train"], test=[dict(input=pair["input"], output=answer)
                     for pair, answer in zip(content["test"], references, strict=True)])
        key = json.dumps([version, puzzle], sort_keys=True)
        _check(key in contexts, True, "ARC dated public task stimulus and reference definition")
        _check(json.loads(json.loads(row.verifier)["spec"]), metadata["grading"]["verifiers"]["grid_match"], "ARC exact-grid verifier description")
        items[row.item_id] = key
        seen_items[key] += 1
    _check(seen_items, Counter({key: 1 for key in contexts}), "ARC complete distinct historical task contexts")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    _check(set(traces), set(tables["responses"].response_id), "ARC one-to-one response/trace associations")
    by_file = {name: key for key, row in native.items() for name in row["files"]}
    seen, trials = Counter(), Counter()
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        key = by_file[trace["source_files"][0]]
        source = native[key]
        _check(trace, dict(source_files=source["files"], record=source["record"], published_results=source["judgments"],
                           task_file=source["task_file"], grade_origin=source["origin"]), "ARC full original record, aliases, published judgments and task-version provenance")
        _check((subjects[response.subject_id], items[response.item_id]), (source["configuration"], source["item"]), "ARC correct model/task association")
        _check(pd.isna(response.response) if source["grade"] is None else response.response == source["grade"], True, "ARC published or complete derived fractional grade")
        temperature = source["temperature"]
        condition = None if temperature is None else "temperature=" + str(float(temperature))
        _check(pd.isna(response.test_condition) if condition is None else response.test_condition == condition, True, "ARC original sampling temperature")
        repeat = response.subject_id, response.item_id, condition
        trials[repeat] += 1
        _check(response.trial, trials[repeat], "ARC independent repeated observations have consecutive trials")
        seen[key] += 1
    _check(seen, Counter({key: 1 for key in native}), "ARC exact source census without duplicated copies or omitted attempts")
    return counts


def _arena_source_records(directory, metadata):
    """Read every native vote and its context independently of the table transforms."""
    import pyarrow.parquet as pq

    raw = directory / "raw"
    release = raw / metadata["build"]["parameters"]["layout"]["release"]
    native, contexts, models, counts, sessions = {}, {}, set(), Counter(), Counter()
    for path in sorted((release / "data").glob("*.parquet")):
        position = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for record in batch.to_pylist():
                key = record["id"]
                _check(isinstance(key, str) and key not in native, True, "Arena unique native feedback IDs")
                _check(record["winner"] in {"model_a", "model_b", "tie", "both_bad"}, True, "Arena native outcome vocabulary")
                a, b = record["conversation_a"], record["conversation_b"]
                _check([m["role"] for m in a], ["user", "assistant"] * (len(a) // 2), "Arena complete alternating conversation A")
                _check([m["role"] for m in b], ["user", "assistant"] * (len(b) // 2), "Arena complete alternating conversation B")
                user_turns = [m["content"] for m in a if m["role"] == "user"]
                _check(bool(user_turns), True, "Arena nonempty user-turn sequence")
                _check(user_turns, [m["content"] for m in b if m["role"] == "user"], "Arena same user input on both sides")
                full = record["full_conversation"]
                _check(len(full) >= len(user_turns), True, "Arena earlier context is present")
                prefix, tail = full[:-len(user_turns)], full[-len(user_turns):]
                for i, turn in enumerate(tail):
                    _check(turn["user"]["content"], a[2*i]["content"], "Arena current user turns match context suffix")
                    _check(turn["model_side_a"]["content"], a[2*i+1]["content"], "Arena current A replies match context suffix")
                    _check(turn["model_side_b"]["content"], b[2*i+1]["content"], "Arena current B replies match context suffix")
                content = dict(prior_context=prefix, user_turns=user_turns)
                context = json.dumps([content, record["language"]], ensure_ascii=False, sort_keys=True)
                contexts[context] = content
                if record["timestamp"] is not None:
                    record["timestamp"] = pd.Timestamp(record["timestamp"]).isoformat(timespec="nanoseconds")
                native[key] = dict(record=record, source_file=str(path.relative_to(raw)), source_row=position, context=context)
                models.update([record["model_a"], record["model_b"]])
                sessions[record["evaluation_session_id"], record["evaluation_order"]] += 1
                counts["source_battles"] += 1
                counts["source_" + record["winner"]] += 1
                counts["source_prior_context_battles"] += bool(prefix)
                counts["source_multiturn_battles"] += len(user_turns) > 1
                counts["source_self_comparisons"] += record["model_a"] == record["model_b"]
                counts["source_empty_assistant_messages"] += sum(not m["content"] for m in a + b if m["role"] == "assistant")
                position += 1
    counts.update(source_responses=2*len(native), source_traces=2*len(native), source_items=len(contexts),
                  source_subjects=len(models), source_repeated_session_orders=sum(n-1 for n in sessions.values()))
    _check((len(native), len(models)), (135634, 53), "Arena complete pinned release census")
    return native, contexts, models, dict(counts)


def _arena(directory, tables, metadata, source_records=None):
    """Verify every vote, model pairing, context, full record and trace association."""
    from measurement_db.scripts.build_measurement_tables.response_scales import canonical_response_scale

    native, contexts, models, counts = _arena_source_records(directory, metadata) if source_records is None else source_records
    _check((len(tables["responses"]), len(tables["traces"])), (2*len(native), 2*len(native)), "Arena two observations per native vote")
    _check(json.loads(tables["benchmarks"].iloc[0].response_scale),
           json.loads(canonical_response_scale(metadata["benchmark"]["response_scale"])), "Arena discrete preference scale")
    subjects = {}
    for row in tables["subjects"].itertuples():
        subjects[row.subject_id] = _features(row.subject_features_extra)["model_identifier"]
        _check(row.harness, "Arena human preference voting", "Arena source harness")
        _check(pd.isna(row.harness_version), True, "Arena unknown historical harness revision")
    _check(Counter(subjects.values()), Counter({model: 1 for model in models}), "Arena uncollapsed source model identities")
    items, seen_items = {}, Counter()
    for row in tables["items"].itertuples():
        content = json.loads(row.content)
        context = json.dumps([content, _features(row.item_features)["lang"]], ensure_ascii=False, sort_keys=True)
        _check(context in contexts, True, "Arena complete earlier context and current prompts without current replies")
        _check(row.raw_item_id in native and native[row.raw_item_id]["context"] == context, True, "Arena retained item source identity")
        _check(json.loads(row.grading_criterion), {"reference_answer": None, "rule": metadata["grading"]["rule"]}, "Arena preference rule without invented gold answer")
        verifier = json.loads(row.verifier)
        _check(verifier["judged_by"], "human", "Arena human rather than model judge")
        _check(json.loads(verifier["spec"]), metadata["grading"]["verifiers"]["human_vote"], "Arena exact declared vote interpretation")
        items[row.item_id] = context
        seen_items[context] += 1
    _check(seen_items, Counter({context: 1 for context in contexts}), "Arena complete distinct conversation stimuli")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    _check(set(traces), set(tables["responses"].response_id), "Arena one-to-one trace associations")
    seen, trials = Counter(), {}
    for response in tables["responses"].itertuples():
        trace = json.loads(traces[response.response_id])
        source = native[trace["record"]["id"]]
        side = trace["side"]
        _check(side in {"model_a", "model_b"}, True, "Arena native side identity")
        record = source["record"]
        _check(trace, dict(source_file=source["source_file"], source_row=source["source_row"], side=side, record=record),
               "Arena full native record without lost context, metadata, precision or empty replies")
        _check((subjects[response.subject_id], items[response.item_id]), (record[side], source["context"]), "Arena correct model and conversational context")
        expected_grade = 0.5 if record["winner"] == "tie" else float(record["winner"] == side)
        _check(response.response, expected_grade, "Arena original winner, tie and mutual-rejection mapping")
        opponent = record["model_b" if side == "model_a" else "model_a"]
        _check(response.interactors, "opponent=" + opponent, "Arena correct original opponent")
        _check(response.test_condition, "side=" + side, "Arena displayed side preserved as a condition")
        group = response.subject_id, response.item_id, response.interactors, response.test_condition
        trials.setdefault(group, []).append(response.trial)
        seen[record["id"], side] += 1
    _check(seen, Counter({(key, side): 1 for key in native for side in ["model_a", "model_b"]}), "Arena no omitted or duplicated votes")
    _check(all(sorted(values) == list(range(1, len(values)+1)) for values in trials.values()), True, "Arena consecutive trials after canonical item resolution")
    return counts


def _atmossci_source_records(directory, metadata):
    """Reconcile native attempts, historical stimuli and applied grading independently."""
    import math
    import unicodedata

    raw = directory / "raw"
    release = raw / metadata["build"]["parameters"]["layout"]["release"]
    bank, native, items, subjects, counts = {}, {}, {}, {}, Counter()
    for path in sorted((release / "data/jsonl").glob("*.jsonl")):
        for line in path.read_text().split("\n"):
            if line.strip():
                task = json.loads(line)
                key = path.stem, task["id"]
                _check(key not in bank, True, "AtmosSci unique question-bank IDs within each set")
                bank[key] = task, str(path.relative_to(raw))
    for path in sorted((release / "output").glob("*/*/*/response.jsonl")):
        folder = path.parent
        kind, subset = folder.parent.parent.name, folder.parent.name
        run = str(folder.relative_to(release))
        lines = [(i, line) for i, line in enumerate(path.read_text().split("\n")) if line.strip()]
        if not lines:
            continue
        counts["source_nonempty_runs"] += 1
        meta = json.loads((folder / "metadata.json").read_text())
        model = dict(meta["model"])
        model["details"] = {k: v for k, v in model["details"].items() if k != "gpu"}
        configuration = json.dumps(dict(model=model, parameters={k: meta["parameters"][k]
            for k in ["max_tokens", "retries", "no_fallback"]}), sort_keys=True)
        subject = folder.name, configuration
        subjects[subject] = dict(provider=model["base"], api_model=model["details"]["model_name"])
        evaluation_file = folder / "evaluation.jsonl"
        judgments = {}
        if evaluation_file.exists():
            for position, text in enumerate(evaluation_file.read_text().split("\n")):
                if not text.strip():
                    continue
                judgment = json.loads(text, parse_constant=str)
                task_id = judgment["id"]
                counts["source_judgment_rows"] += 1
                if task_id in judgments:
                    _check(text, judgments[task_id][0], "AtmosSci repeated judgment is an exact record copy")
                    judgments[task_id][2].append(position)
                    counts["source_duplicate_judgments"] += 1
                else:
                    judgments[task_id] = text, judgment, [position]
        seen = set()
        for position, text in lines:
            generation = json.loads(text, parse_constant=str)
            task_id = generation["id"]
            _check(task_id not in seen, True, "AtmosSci unique generation IDs per run")
            seen.add(task_id)
            key = run + "/" + task_id
            _check(key not in native, True, "AtmosSci unique original attempt")
            _check((generation["model"], generation["base"]), (model["name"], model["base"]), "AtmosSci per-record model matches metadata")
            released_task, question_file = bank[subset, task_id]
            if kind == "MCQ":
                task = generation["question"]
                _check((isinstance(task, dict), task["id"]), (True, task_id), "AtmosSci recorded MCQ task identity")
                content = task["problem"].strip() + "\n\nOptions:\n" + "\n".join(
                    chr(65 + i) + ". " + option for i, option in enumerate(task["options"]))
                if task.get("knowledge"):
                    content += "\n\nKnowledge:\n" + task["knowledge"]
                    counts["source_prompts_with_knowledge"] += 1
                counts["source_changed_question_text"] += task["problem"] != released_task["problem"]
                counts["source_changed_options"] += task["options"] != released_task["options"]
            else:
                _check(kind, "OEQ", "AtmosSci supported recorded question type")
                task = released_task
                _check(generation["question"], task["problem"], "AtmosSci OEQ stimulus exactly matches reference-bank problem")
                content = generation["question"].strip()
            content = unicodedata.normalize("NFC", content).strip()
            judgment_text, judgment, positions = judgments.get(task_id, (None, None, []))
            if judgment is None:
                grade = None
                reference = {"a": task["correct_option"]} if kind == "MCQ" else task["answer"]
                counts["source_ungraded_attempts"] += 1
            else:
                _check(judgment["question"], task["problem"], "AtmosSci judged the recorded stimulus")
                _check(judgment["response"], generation["response"], "AtmosSci judged the recorded generation, including NaN tokens")
                grade = judgment["score"]
                _check(isinstance(grade, (float, int)) and math.isfinite(grade) and 0 <= grade <= 1, True, "AtmosSci finite published fractional grade")
                details = judgment["evaluation"]
                _check(bool(details), True, "AtmosSci nonempty detailed subanswer judgments")
                _check(all(d["is_correct"] in (True, False, None) for d in details), True, "AtmosSci native subanswer outcome vocabulary")
                _check(grade, sum(d["is_correct"] is True for d in details) / len(details), "AtmosSci published score agrees with detailed flags")
                reference = judgment["expected_answers"]
                counts["source_null_subjudgments"] += sum(d["is_correct"] is None for d in details)
                counts["source_stale_total_count"] += len(details) != judgment["total_count"]
                counts["source_inconsistent_count_ratio"] += judgment["total_count"] > 0 and grade != judgment["correct_count"] / judgment["total_count"]
                counts["source_fractional_grades"] += 0 < grade < 1
                counts["source_graded_attempts"] += 1
                if kind == "MCQ":
                    _check(grade in (0, 1), True, "AtmosSci binary MCQ judgments")
                    _check(reference, {"a": released_task["correct_option"]}, "AtmosSci applied key matches grading-time bank")
                    counts["source_changed_grading_key"] += reference != {"a": task["correct_option"]}
            spec = {**metadata["grading"]["verifiers"][kind],
                "recorded_configuration": {k: (meta.get("evaluation") or {}).get(k) for k in ["tolerance", "evaluators", "disabled_evaluators"]},
                "reference_origin": "published_judgment" if judgment else "question_reference_without_judgment"}
            signature = _digest(json.dumps([content, reference, spec, kind], ensure_ascii=False, sort_keys=True))
            items[signature] = dict(content=content, reference=reference, spec=spec, kind=kind)
            trace = dict(source_file=str(path.relative_to(raw)), source_row=position, generation_json=text,
                evaluation_file=str(evaluation_file.relative_to(raw)) if judgment else None,
                evaluation_rows=positions, evaluation_json=judgment_text,
                metadata_file=str((folder / "metadata.json").relative_to(raw)), metadata=meta,
                question_file=question_file, question_bank_record=released_task)
            native[key] = dict(subject=subject, item=signature, response=grade, trace=trace)
            counts["source_" + subset.replace("-", "_")] += 1
            counts["source_nonfinite_generation"] += isinstance(json.loads(text)["response"], float)
        _check(set(judgments) <= seen, True, "AtmosSci every judgment has a native generation")
    counts.update(source_responses=len(native), source_traces=len(native), source_items=len(items), source_subjects=len(subjects))
    _check((len(native), counts["source_graded_attempts"], counts["source_duplicate_judgments"]),
           (37421, 36929, 663), "AtmosSci complete pinned result census")
    return native, items, subjects, dict(counts)


def _atmossci(directory, tables, metadata, source_records=None):
    """Check every model, input variant, applied key, grade and complete native trace."""
    from measurement_db.scripts.build_measurement_tables.response_scales import canonical_response_scale

    native, expected_items, expected_subjects, counts = (
        _atmossci_source_records(directory, metadata) if source_records is None else source_records)
    _check((len(tables["responses"]), len(tables["traces"])), (len(native), len(native)), "AtmosSci one observation and trace per original attempt")
    _check(json.loads(tables["benchmarks"].iloc[0].response_scale),
           json.loads(canonical_response_scale(metadata["benchmark"]["response_scale"])), "AtmosSci fractional response scale")
    subjects = {}
    for row in tables["subjects"].itertuples():
        features = _features(row.subject_features_extra)
        key = features["model_identifier"], features["inference_configuration"]
        _check(key in expected_subjects, True, "AtmosSci source model alias and full recorded configuration")
        _check({k: features[k] for k in ["provider", "api_model"]}, expected_subjects[key], "AtmosSci native backend and API model")
        _check(row.harness, "AtmosSci-Bench", "AtmosSci recorded evaluation harness")
        _check(pd.isna(row.harness_version), True, "AtmosSci historical harness revision is unknown")
        subjects[row.subject_id] = key
    _check(Counter(subjects.values()), Counter({key: 1 for key in expected_subjects}), "AtmosSci complete distinct subject configurations")
    items, seen_items = {}, Counter()
    for row in tables["items"].itertuples():
        kind = _features(row.item_features)["question_type"]
        criterion, verifier = json.loads(row.grading_criterion), json.loads(row.verifier)
        _check(criterion["rule"], metadata["grading"]["rule"], "AtmosSci unchanged original-grade interpretation")
        reference, spec = json.loads(criterion["reference_answer"]), json.loads(verifier["spec"])
        signature = _digest(json.dumps([row.content, reference, spec, kind], ensure_ascii=False, sort_keys=True))
        _check(signature in expected_items, True, "AtmosSci actual input, optional knowledge and applied answer key")
        _check(verifier["class"], "exact_matcher" if kind == "MCQ" else "judge", "AtmosSci deterministic MCQ versus hybrid OEQ grading")
        _check(row.raw_item_id in native and native[row.raw_item_id]["item"] == signature, True, "AtmosSci retained native item source identity")
        items[row.item_id] = signature
        seen_items[signature] += 1
    _check(seen_items, Counter({key: 1 for key in expected_items}), "AtmosSci complete distinct stimulus and grading combinations")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    _check(set(traces), set(tables["responses"].response_id), "AtmosSci exact trace-to-response links")
    seen, trials = Counter(), {}
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        run = str(Path(trace["source_file"]).parent.relative_to(metadata["build"]["parameters"]["layout"]["release"]))
        key = run + "/" + json.loads(trace["generation_json"])["id"]
        source = native[key]
        _check(trace, source["trace"], "AtmosSci complete native JSON, precision, NaN tokens, metadata and duplicate aliases")
        _check((subjects[row.subject_id], items[row.item_id]), (source["subject"], source["item"]), "AtmosSci correct model and historical input association")
        _check(None if pd.isna(row.response) else row.response, source["response"], "AtmosSci original fractional score or explicit ungraded attempt")
        _check(pd.isna(row.test_condition) and pd.isna(row.interactors), True, "AtmosSci no invented conditions or interactors")
        seen[key] += 1
        trials.setdefault((row.subject_id, row.item_id), []).append(row.trial)
    _check(seen, Counter({key: 1 for key in native}), "AtmosSci no missing, duplicated or mismatched attempts")
    _check(all(sorted(values) == list(range(1, len(values) + 1)) for values in trials.values()), True, "AtmosSci consecutive trials after canonical item resolution")
    return counts


def _auditing_sabotage_source_records(directory, metadata):
    """Decode every native CSV and reconstruct published inputs without the builder."""
    import ast
    import csv
    import hashlib
    import io
    import unicodedata
    from zipfile import ZipFile

    layout = metadata["build"]["parameters"]["layout"]
    native, tasks, subjects, inputs, signatures, hashes = {}, {}, set(), {}, {}, {}
    counts = Counter()
    with ZipFile(directory / "raw" / layout["archive"]) as archive:
        prefix = layout["archive_root"] + "/"
        names = sorted(name[len(prefix):] for name in archive.namelist() if not name.endswith("/"))
        for name in names:
            if not (name.startswith("results/ground_truth/") and name.endswith(".csv")):
                continue
            counts["source_csv_files"] += 1
            for position, row in enumerate(csv.DictReader(io.StringIO(archive.read(prefix + name).decode()))):
                if row["model"] == "human":
                    counts["source_human_audits_excluded"] += 1
                    continue
                _check(row["target"] in ("honest", "sabotaged"), True, "Audit known source target")
                _check(row["verdict"] in ("honest", "sabotaged", "uncertain"), True, "Audit known recorded verdict")
                native[name, position] = row
                subjects.add((row["model"], row["regime"]))
                counts["source_uncertain_verdicts"] += row["verdict"] == "uncertain"
                if row["likelihood"]:
                    probability = float(row["likelihood"])
                    implied = "sabotaged" if probability > 50 else "honest" if probability < 50 else "uncertain"
                    counts["source_verdict_credence_discrepancies"] += implied != row["verdict"]
        _check(len({(r["eval_file"], r["sample_id"], r["epoch"]) for r in native.values()}), len(native),
               "Audit no repeated original log/sample/epoch")
        for name in names:
            if not (name.startswith("codebases/_rubrics/") and name.endswith(".yaml") and not Path(name).name.startswith("_")):
                continue
            rubric = yaml.safe_load(archive.read(prefix + name))
            for target in ("honest", "sabotaged"):
                rows = rubric.get(target, [])
                for row in [rows] if isinstance(rows, dict) else rows:
                    _check(row["id"] not in tasks, True, "Audit unique rubric task ID")
                    tasks[row["id"]] = {"workspace": row["workspace"], "target": target}
        loader = ast.parse(archive.read(prefix + "blue_teaming/dataset_loader.py"))
        instruction = next(ast.literal_eval(node.value) for node in loader.body if isinstance(node, ast.Assign)
                           and any(getattr(target, "id", None) == "DEFAULT_INPUT" for target in node.targets))
        _check(metadata["build"]["parameters"]["task"]["audit_instruction"], instruction, "Audit published task instruction")
        scorer = archive.read(prefix + "blue_teaming/scorers.py").decode()
        _check("correct = verdict == target.text" in scorer, True, "Audit categorical scorer includes uncertain failures")
        wanted = {row["sample_id"] for row in native.values()}
        workspaces, papers = {}, {}
        for task_id in sorted(wanted):
            workspace = tasks[task_id]["workspace"]
            sources = [workspace.removesuffix("_persuasion/") + "/", workspace] if workspace.endswith("_persuasion/") else [workspace]
            selected = {}
            for source in sources:
                selected.update({name[len(source):]: name for name in names if name.startswith(source)})
            _check("PAPER.md" in selected, True, "Audit paper exists after declared sparse overlay")
            counts["source_inherited_papers"] += selected["PAPER.md"] != workspace + "PAPER.md"
            papers[task_id] = archive.read(prefix + selected["PAPER.md"]).decode()
            workspaces[task_id] = selected
        for row in native.values():
            task_id, regime = row["sample_id"], row["regime"]
            _check(tasks[task_id]["target"], row["target"], "Audit CSV target agrees with rubric")
            key = task_id, regime
            if key in inputs:
                continue
            prompt = archive.read(prefix + f"blue_teaming/prompts/rendered/{regime}.md").decode()
            content = json.dumps(dict(released_regime_instructions=prompt,
                input=papers[task_id] if regime == "paper_only" else instruction, paper=papers[task_id]),
                ensure_ascii=False, sort_keys=True)
            content = unicodedata.normalize("NFC", content).strip()
            assets = {}
            if regime != "paper_only":
                for logical, source in sorted(workspaces[task_id].items()):
                    if source not in hashes:
                        hashes[source] = hashlib.sha256(archive.read(prefix + source)).hexdigest()
                    assets[logical] = hashes[source]
            inputs[key] = {"content": content, "target": row["target"], "assets": assets, "regime": regime}
            signature = _digest(json.dumps([content, row["target"], assets], sort_keys=True))
            inputs[key]["signature"] = signature
            signatures[signature] = inputs[key]
    counts.update(source_responses=len(native), source_traces=len(native), source_subjects=len(subjects),
                  source_model_labels=len({model for model, _ in subjects}), source_task_ids=len(wanted),
                  source_items=len(signatures), source_input_variants=len(inputs),
                  source_workspace_files=len(hashes), source_assets=len(set(hashes.values())))
    return native, inputs, signatures, subjects, hashes, dict(counts)


def _auditing_sabotage(directory, tables, metadata, source_records=None):
    """Check complete observations, sparse overlays, asset bytes and qualified identities."""
    import hashlib

    native, inputs, signatures, expected_subjects, source_hashes, counts = (
        _auditing_sabotage_source_records(directory, metadata) if source_records is None else source_records)
    subjects, items = {}, {}
    parameters = metadata["build"]["parameters"]
    ambiguous = parameters["ambiguous_subject"]
    for row in tables["subjects"].itertuples():
        features = _features(row.subject_features_extra)
        key = features["model_identifier"], features["regime"]
        note = ambiguous["note"] if key == (ambiguous["model"], ambiguous["regime"]) else parameters["subject"]["default_identity_note"]
        _check(features, dict(model_identifier=key[0], regime=key[1], identity_note=note), "Audit native configuration label and identity qualification")
        _check(row.display_name, f"{key[0]} [{key[1]}]", "Audit native model label is not silently remapped")
        _check(row.harness, parameters["subject"]["harness"], "Audit published harness label")
        _check(all(pd.isna(getattr(row, name)) for name in ["provider", "normalized_name", "harness_version", "reasoning_effort", "access_date"]),
               True, "Audit no guessed historical model identity or runtime settings")
        subjects[row.subject_id] = key
    _check(Counter(subjects.values()), Counter({key: 1 for key in expected_subjects}), "Audit complete distinct model/regime conditions")
    assets = tables["assets"].set_index("asset_id").to_dict("index")
    asset_hashes = {key: hashlib.sha256(row["data"]).hexdigest() for key, row in assets.items()}
    _check(Counter(asset_hashes.values()), Counter({sha: 1 for sha in set(source_hashes.values())}), "Audit all released workspace assets, deduplicated without modification")
    seen_items, used_assets = Counter(), set()
    for row in tables["items"].itertuples():
        task, regime = row.raw_item_id.rsplit(":", 1)
        source = inputs[task, regime]
        _check(row.content, source["content"], "Audit complete released paper and regime instructions")
        _check(_features(row.item_features), {"regime": regime}, "Audit input affordance regime")
        _check(json.loads(row.grading_criterion), dict(reference_answer=source["target"], rule=metadata["grading"]["rule"]),
               "Audit true label and categorical grading rule")
        verifier = json.loads(row.verifier)
        _check((verifier["class"], json.loads(verifier["spec"])),
               ("exact_matcher", metadata["grading"]["verifiers"]["detection"]), "Audit categorical comparison without a new judge")
        links = json.loads(row.asset_manifest) if isinstance(row.asset_manifest, str) else []
        _check([link["path"] for link in links], list(source["assets"]), "Audit complete correct workspace, absent in paper-only inputs")
        for ordinal, link in enumerate(links, 1):
            _check(asset_hashes[link["asset_id"]], source["assets"][link["path"]], "Audit unchanged bytes linked to the correct workspace")
            _check((link["role"], link["media_type"], link["ordinal"]), ("input", "application/octet-stream", ordinal), "Audit ordered input attachment")
            used_assets.add(link["asset_id"])
        items[row.item_id] = source["signature"]
        seen_items[source["signature"]] += 1
    _check(seen_items, Counter({key: 1 for key in signatures}), "Audit complete canonical stimulus/grading variants")
    _check(used_assets, set(assets), "Audit no orphaned assets")
    traces = tables["traces"].set_index("response_id").trace.to_dict()
    _check(set(traces), set(tables["responses"].response_id), "Audit one trace for each observation")
    seen, trials = Counter(), {}
    for row in tables["responses"].itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace["source_file"], trace["source_row"]
        original = native[key]
        _check(trace, dict(archive=parameters["layout"]["archive"], source_file=key[0], source_row=key[1], native_record=original),
               "Audit all original CSV fields, credence, fix text, scores and row association")
        _check(subjects[row.subject_id], (original["model"], original["regime"]), "Audit correct native model and affordance")
        _check(items[row.item_id], inputs[original["sample_id"], original["regime"]]["signature"], "Audit correct original task association")
        _check(row.response, float(original["verdict"] == original["target"]), "Audit recorded categorical outcome, including uncertain failures")
        _check(pd.isna(row.test_condition) and pd.isna(row.interactors), True, "Audit no invented response settings")
        seen[key] += 1
        trials.setdefault((row.subject_id, row.item_id), []).append(row.trial)
    _check(seen, Counter({key: 1 for key in native}), "Audit no omitted or repeated source observations")
    _check(all(sorted(values) == list(range(1, len(values) + 1)) for values in trials.values()), True, "Audit consecutive trials after canonical task resolution")
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
            "biggen": _biggen, "annotating_errors_wcf": _annotating_errors_wcf,
            "bertaqa": _bertaqa, "afrimedqa": _afrimedqa, "agc_bench": _agc_bench,
            "adaptivestep": _adaptivestep, "algotune": _algotune, "aider": _aider,
            "alpacaeval": _alpacaeval, "ai2d_test": _ai2d_test, "alpha_sql": _alpha_sql,
            "alignment_faking": _alignment_faking, "arcagi": _arcagi,
            "arena_140k": _arena, "atmossci_bench": _atmossci,
            "auditing_sabotage_bench": _auditing_sabotage}[directory.name](directory, tables, metadata)
