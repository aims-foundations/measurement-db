"""Independent, read-only counts from captured provider releases.

These checks do not import a builder or use its output to obtain source counts.
Each characterization states the scope of the corresponding source claim.
"""
import json
import csv
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import tomllib
from zipfile import ZipFile


def _json(path):
    return json.loads(path.read_text())


def source_observations(directory):
    directory = Path(directory)
    raw, slug = directory / "raw", directory.name
    if slug.startswith("medhelm_") or slug in {
        "helm_airbench", "helm_anthropic_redteam", "helm_bbq", "helm_bold", "helm_afr", "helm_cleva", "helm_thaiexam",
        "helm_harmbench", "helm_real_toxicity_prompts", "helm_simple_safety_tests", "helm_xstest",
    }:
        return verify_helm_upstream(directory)
    if slug == "mlrbench":
        return verify_mlrbench_upstream(directory)
    if slug == "cellverse":
        return verify_cellverse_upstream(directory)
    if slug == "dave":
        return verify_dave_upstream(directory)
    if slug == "software_world_models":
        bank = [row for path in (raw / "benchmark/data").glob("*.parquet")
                for row in pq.read_table(path).to_pylist()]
        files = list((raw / "generations/data").glob("*.parquet"))
        return {"source_item_records": len(bank), "source_model_exports": len(files),
                "source_responses": sum(pq.read_metadata(path).num_rows for path in files)}
    if slug == "putnam_axiom":
        rows = pq.read_table(raw / "grading.parquet").to_pylist()
        return {"source_solutions": len(rows),
                "source_problems": len({row["problem_id"] for row in rows}),
                "source_rubric_scores": sum(len(set(row["human_grade"]) - {"score_total"}) for row in rows)}
    if slug == "proofaug":
        import tarfile
        source = raw / "ProofAug/isabelle_src"
        with tarfile.open(source / "cum.tar.gz") as archive:
            solved = {json.loads(line)["problem_name"]
                      for line in archive.extractfile("output/cum_result.jsonl") if line.strip()}
        return {"source_items": sum(bool(line.strip()) for line in
                                     (source / "datasets/minif2f-test-curated.jsonl").read_text().splitlines()),
                "source_solved": len(solved)}
    if slug == "abstract_reason":
        files = list((raw / "abstract-reason-benchmark/gpt-4o-mini/result").rglob("output.json"))
        rows = [row for path in files for row in _json(path)]
        return {"source_result_files": len(files), "source_responses": len(rows),
                "source_successes": sum(bool(row["is_correct"]) for row in rows)}
    if slug == "agi_elo":
        bank = pq.read_table(raw / "mmlu_items.parquet").to_pylist()
        references = {row["id"]: "ABCD"[int(row["answer"])] for row in bank}
        files = sorted((raw / "mmlu_predictions").glob("*.pkl"))
        count = matched = 0
        for path in files:
            frame = pd.read_pickle(path)
            for task_id, label in zip(frame["Test Case"], frame["True Label"]):
                count += 1
                matched += references.get(task_id) == label
        return {"source_model_exports": len(files), "source_records": count,
                "source_matching_responses": matched, "source_reference_conflicts": count - matched}
    if slug == "fgbench":
        # The malformed 70B export is retained in raw but not imported.
        files = [p for folder in [raw / "results", raw / "gpt_request"]
                 for p in folder.glob("*.jsonl")
                 if p.name != "Llama-3.1-70B-Instruct.jsonl"
                 and (folder.name != "gpt_request" or p.name.startswith("benchmark_result_"))]
        return {
            "source_model_exports": len(files),
            "source_questions": sum(bool(line.strip()) for line in (raw / "test.jsonl").read_text().splitlines()),
            "source_responses": sum(sum(bool(line.strip()) for line in p.read_text().splitlines()) for p in files),
        }
    if slug == "svrpbench":
        records = [_json(path) for path in (raw / "results").glob("*.json")]
        return {
            "source_route_exports": len(records),
            "source_instances": len({(row["problem"], row["instance"]) for row in records}),
            "source_metric_observations": sum(len(row["metrics"]) for row in records),
        }
    if slug == "mathconstraint":
        names = {p.stem for p in raw.glob("*_instances/*.json")}
        count = successes = 0
        for path in raw.glob("results/*/*/*.json"):
            if path.parent.name not in {"easy", "hard", "hard2"} or path.stem not in {"no_tools", "tools"}:
                continue
            for row in _json(path)["results"]:
                if row.get("problem_name") not in names or row.get("correct") is None:
                    continue
                count += 1
                successes += bool(row["correct"])
        return {"source_responses": count, "source_successes": successes}
    if slug == "causal_axioms":
        count = successes = 0
        for path in sorted(raw.glob("length_*/*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt, gold, prediction = row.get("prompt"), str(row.get("completion", "")).strip(), row.get("prediction")
                if not isinstance(prompt, str) or not prompt.strip() or gold not in {"Yes", "No"} or prediction is None:
                    continue
                count += 1
                successes += str(prediction).strip().lower() == gold.lower()
        return {"source_responses": count, "source_successes": successes}
    if slug == "revolve":
        count = successes = 0
        for path in raw.glob("*_predictions.json"):
            for row in _json(path).values():
                if not isinstance(row, dict) or not isinstance(row.get("predictions"), list) or not isinstance(row.get("answer"), str):
                    continue
                gold = row["answer"].strip().upper()
                if gold not in {"A", "B", "C", "D"}:
                    continue
                for prediction in row["predictions"][:5]:
                    match = re.search(r"(?i)Answer\s*:\s*\$?([A-D])", prediction) if isinstance(prediction, str) else None
                    count += 1
                    successes += bool(match and match.group(1).upper() == gold)
        return {"source_responses": count, "source_successes": successes}
    if slug == "evouna":
        count = successes = ungraded = 0
        for name in ["NQ.json", "TQ.json"]:
            for row in _json(raw / name):
                if not (row.get("question") or "").strip():
                    continue
                for model in ["fid", "gpt35", "chatgpt", "gpt4", "newbing"]:
                    grade, answer = row.get("judge_" + model), row.get("answer_" + model)
                    missing = grade is None or grade == "nan"
                    if missing and not (isinstance(answer, str) and answer.strip()):
                        continue
                    count += 1
                    successes += grade is True
                    ungraded += missing
        return {"source_responses": count, "source_successes": successes, "source_ungraded": ungraded}
    if slug == "vl_rewardbench":
        bank = pd.read_parquet(raw / "items.parquet")
        references = {(row.id, row.query, tuple(row.response), tuple(row.human_ranking))
                      for row in bank.itertuples()}
        count = successes = ungraded = 0
        with ZipFile(raw / "inference_results.zip") as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".jsonl"):
                    continue
                for line in archive.read(name).splitlines():
                    row = json.loads(line)
                    if isinstance(row, dict) and "0" in row:
                        row = row["0"]
                    if not row or (row["id"], row["query"], tuple(row["response"]), tuple(row["ranking"])) not in references:
                        continue
                    count += 1
                    status = row["meta"]["flag_status"]
                    successes += status == "agree"
                    ungraded += status == "doesntMatch"
        return {"source_responses": count, "source_successes": successes,
                "source_ungraded": ungraded, "source_bank_rows": len(bank)}
    if slug == "donotanswer":
        with (raw / "data_en.csv").open(newline="") as handle:
            records = list(csv.DictReader(handle))
        fields = [name for name in records[0] if name.endswith(("_harmful", "_action"))]
        labels = [int(row[name]) for row in records for name in fields if row[name] != ""]
        return {"source_prompt_rows": len(records), "source_responses": len(labels),
                "source_label_sum": sum(labels)}
    if slug == "icpc2_code_selector":
        records = pd.read_csv(raw / "predictions_data.csv").to_dict("records")
        records = [row for row in records if isinstance(row["model"], str) and isinstance(row["query"], str)]
        return {
            "source_responses": len(records),
            "source_successes": sum(bool(row["true_positive"]) or bool(row["true_negative"]) for row in records),
            "source_items": len({(row["query"], row["top_k"]) for row in records}),
        }
    if slug == "ikp":
        root = raw / "ikp/data"
        probes = {row["id"]: row for row in _json(root / "probes/final_probe_set_v9.json")}
        count = successes = 0
        for path in sorted((root / "results").glob("*.json")):
            run = _json(path)
            if not isinstance(run, dict) or not run.get("model_name") or not isinstance(run.get("results"), list):
                continue
            for row in run["results"]:
                item = probes.get(row.get("probe_id"))
                if item is None or row.get("question") != item["question"]:
                    continue
                if not (row.get("model_response") or "").strip() or row.get("correct") is None:
                    continue
                count += 1
                successes += int(row["correct"])
        return {"source_responses": count, "source_successes": successes, "source_items": len(probes)}
    if slug in {'live_agent_risk', 'metaagent', 'swe_together', 'wikihow_agent', 'agentic_review_perturbation', 'scivisagentbench', 'confagents', 'swe_smith', 'os_harm'}:
        from measurement_db.scripts.curate_benchmarks.batch4_audits import verify_batch4
        return verify_batch4(directory)
    if slug in {'agentdojo', 'swebench_live', 'workbench_revisited', 'appworld', 'algotune'}:
        from measurement_db.scripts.curate_benchmarks.batch3_audits import verify_batch3
        return verify_batch3(directory)
    if slug in {'corebench', 'editbench', 'hcast', 'theagentcompany'}:
        from measurement_db.scripts.curate_benchmarks.batch2_audits import verify_batch2
        return verify_batch2(directory)
    if slug == "tau2_bench":
        count = successes = 0
        files = list((raw / "repo_results").glob("*.json"))
        files += [p for p in (raw / "s3_trajectories").glob("*/*.json")
                  if p.name != "submission.json"]
        for p in files:
            d = _json(p)
            domain = d.get("info", {}).get("environment_info", {}).get("domain_name")
            if domain not in {"airline", "retail", "telecom", "telecom-workflow"}:
                continue
            ids = {str(t["id"]) for t in d.get("tasks", [])}
            for run in d.get("simulations", []):
                grade = (run.get("reward_info") or {}).get("reward")
                if str(run.get("task_id")) in ids and grade in (0, 1):
                    count += 1
                    successes += int(grade)
        return {"source_responses": count, "source_successes": successes}
    if slug == "programbench":
        html = (raw / "extended.html").read_text()
        match = re.search(r"(?:var|const|let)\s+hm\s*=\s*", html)
        if not match:
            raise ValueError("captured ProgramBench matrix declaration is absent")
        matrix = json.JSONDecoder().raw_decode(html[match.end():])[0]
        return {"source_items": len(matrix["tasks"]), "source_model_rows": len(matrix["models"])}
    if slug == "swebench_java":
        return {"cached_item_descriptions": len(pd.read_parquet(raw / "instance_content.parquet"))}
    if slug == "frontieror":
        items = responses = 0
        for p in (raw / "details").glob("*.json"):
            d = _json(p)
            items += 1
            responses += len(d["per_model"])
        return {"source_items": items, "source_responses": responses}
    if slug == "matharena_platform":
        return {"captured_output_rows": sum(pq.read_metadata(p).num_rows for p in raw.glob("*.parquet"))}
    if slug == "swe_chat":
        return {"captured_sessions": pq.read_metadata(raw / "sessions.parquet").num_rows,
                "captured_conversation_rows": pq.read_metadata(raw / "conversations.parquet").num_rows}
    if slug == "osworld":
        verified = verify_osworld_upstream(directory)
        verified["official_task_definitions"] = len(list((raw / "task_definitions").glob("*/*.json")))
        raw = raw / "legacy_snapshot"
        return {**verified, "source_items": pq.read_metadata(raw / "items.parquet").num_rows,
                "source_responses": pq.read_metadata(raw / "responses.parquet").num_rows,
                "source_traces": pq.read_metadata(raw / "traces.parquet").num_rows}
    raise ValueError(f"No independent source audit for {slug}")


def verify_dave_upstream(directory, tables_directory=None):
    """Check every curated DAVE response, reference, trace and media association.

    The released result files omit clip IDs. Match the stated task and correct
    option to the bank independently of model answers; reject nonunique clips.
    Each imported run must cover the unambiguous bank exactly once. Some
    Ego4D exports shuffle rows, so positional agreement is recorded, not assumed.
    """
    from collections import Counter, defaultdict
    import hashlib

    directory = Path(directory)
    raw = directory / "raw"
    tables_directory = Path(tables_directory or directory / "formatted_tables")
    model_names = {
        "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
        "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "models/gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
        "gemini-2.0-flash-001": "gemini-2.0-flash-001",
        "pandagpt": "PandaGPT", "salmonn": "video-SALMONN", "videollama": "Video-LLaMA2",
    }
    banks, indices, media_hashes = {}, {}, {}
    for split in ["epic", "ego4d"]:
        banks[split] = _json(raw / "dataset" / f"{split}.json")
        index = defaultdict(list)
        for position, item in enumerate(banks[split]):
            question = item["choice_metadata"]["audio_visual_alignment"]
            choices = tuple(sorted(option.lower().strip() for option in question["choices"]))
            reference = question["choices"][question["ground_truth"]].lower().strip()
            index[item["type"], item["audio_class"].replace("_", " "), choices, reference].append(position)
        indices[split] = index
        with ZipFile(raw / "dataset" / f"{split}.zip") as archive:
            for positions in index.values():
                if len(positions) != 1:
                    continue
                item = banks[split][positions[0]]
                for field in ["video_with_overlayed_audio_path", "overlayed_audio_path"]:
                    member = item[field]
                    if (split, member) not in media_hashes:
                        media_hashes[split, member] = hashlib.sha256(archive.read(member)).hexdigest()

    expected = Counter()
    source_records = excluded = reordered = 0
    for path in sorted((raw / "results").glob("*.json")):
        split = "ego4d" if "_ego4d_" in path.name else "epic"
        for model, conditions in _json(path)["predictions"].items():
            if model not in model_names or "multimodal" not in conditions:
                continue
            matched_positions = []
            for position, row in enumerate(conditions.get("multimodal", [])):
                source_records += 1
                options = dict(re.findall(r"^\(([A-Z])\) (.+)$", row["prompt"], flags=re.MULTILINE))
                sound = re.search(r"when the (.*?) sound is heard", row["prompt"])[1]
                reference = row["ground_truth"][0]
                key = (row["question_type"], sound, tuple(sorted(value.lower().strip() for value in options.values())),
                       options[reference.strip("()")].lower().strip())
                matches = indices[split].get(key, [])
                if not matches:
                    raise ValueError("A released DAVE question/reference has no source match")
                if (row["response_dict"]["response_text"] in row["ground_truth"]) != row["is_correct"]:
                    raise ValueError("DAVE provider grade disagrees with its released evaluator")
                if len(matches) != 1:
                    excluded += 1
                    continue
                matched_positions.append(matches[0])
                reordered += matches[0] != position
                item = banks[split][matches[0]]
                media = tuple(media_hashes[split, item[field]] for field in
                              ["video_with_overlayed_audio_path", "overlayed_audio_path"])
                trace = json.dumps(row["response_dict"], sort_keys=True, ensure_ascii=False)
                expected[model_names[model], row["prompt"], reference, float(row["is_correct"]),
                         split + "/multimodal", trace, media] += 1
            expected_positions = {positions[0] for positions in indices[split].values() if len(positions) == 1}
            if set(matched_positions) != expected_positions or len(matched_positions) != len(expected_positions):
                raise ValueError("A DAVE model run does not cover each unambiguous stimulus exactly once")

    frames = {name: pd.read_parquet(tables_directory / f"{name}.parquet")
              for name in ["responses", "subjects", "items", "traces"]}
    items = frames["items"].set_index("item_id").to_dict("index")
    names = frames["subjects"].set_index("subject_id").display_name.to_dict()
    traces = frames["traces"].set_index("response_id").trace.to_dict()
    actual = Counter()
    for row in frames["responses"].itertuples():
        item = items[row.item_id]
        reference = json.loads(item["grading_criterion"])["reference_answer"]
        media = tuple(entry["asset_id"] for entry in json.loads(item["asset_manifest"]))
        trace = json.dumps(json.loads(traces[row.response_id]), sort_keys=True, ensure_ascii=False)
        actual[names[row.subject_id], item["content"], reference, row.response, row.test_condition, trace, media] += 1
    if actual != expected:
        raise ValueError(f"DAVE source reconciliation differs: {sum((expected-actual).values())} missing, "
                         f"{sum((actual-expected).values())} additional observations")
    return {"source_bank_items": sum(map(len, banks.values())), "source_core_records": source_records,
            "ambiguous_media_records": excluded, "source_responses": sum(expected.values()),
            "source_models": len(model_names), "source_assets": len(set(media_hashes.values())),
            "reordered_source_records": reordered,
            "source_successes": int(sum(key[3] * count for key, count in expected.items()))}


def verify_osworld_upstream(directory):
    """Reconcile captured official result.txt members against the old release.

    Historical trial ordinals are retained by the migration. This audit compares
    the complete multiset of (configured subject, source task, score), including
    repeated attempts, independently of how archive prefixes were numbered.
    """
    from collections import Counter

    raw = Path(directory) / 'raw'
    snapshot = raw / 'legacy_snapshot'
    subjects = pd.read_parquet(snapshot / 'subjects.parquet').set_index('subject_id').display_name.to_dict()
    items = pd.read_parquet(snapshot / 'items.parquet').set_index('item_id').raw_item_id.to_dict()
    expected = Counter((subjects[r.subject_id], items[r.item_id], float(r.response))
                       for r in pd.read_parquet(snapshot / 'responses.parquet').itertuples())
    actual = Counter()
    pattern = re.compile(r'(?:^|/)(chrome|gimp|libreoffice_calc|libreoffice_impress|libreoffice_writer|multi_apps|os|thunderbird|vlc|vs_code)/([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})(?:/|$)')
    archives = sorted((raw / 'upstream_scores').glob('*.json'))
    for path in archives:
        capture = _json(path)
        if capture['repository'] != 'xlangai/ubuntu_osworld_verified_trajs':
            raise ValueError('OSWorld score capture has an unexpected provider')
        if capture['revision'] != '5473c39e42a538a187a9b2c2b499db59d560fd8c':
            raise ValueError('Review a new OSWorld provider revision before updating the capture')
        for record in capture['scores']:
            match = pattern.search(record['member'])
            if match is None:
                raise ValueError('OSWorld score member has no source task identifier')
            actual[capture['model'], '/'.join(match.groups()), float(record['score'])] += 1
    if actual != expected:
        raise ValueError(f'OSWorld upstream reconciliation differs: '
                         f'{sum((expected-actual).values())} missing and '
                         f'{sum((actual-expected).values())} additional responses')
    return {'official_archives': len(archives), 'official_responses': sum(actual.values())}


def verify_helm_upstream(directory, tables_directory=None):
    """Reconcile every native HELM attempt against linked canonical tables.

    Deliberately use keyed native records rather than the builder's pandas
    transformation, checking prompts, grades, references, settings and traces.
    """
    import ast
    import gzip
    import unicodedata
    import yaml

    directory = Path(directory)
    tables_directory = Path(tables_directory or directory / "formatted_tables")
    metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
    configuration = metadata["build"]["parameters"]["helm"]
    expected = {}
    run_paths = sorted((directory / "raw/runs").rglob("run_spec.json*"))
    for path in run_paths:
        documents = {}
        opener = gzip.open if path.suffix == ".gz" else open
        suffix = ".json.gz" if path.suffix == ".gz" else ".json"
        for kind in ("run_spec", "instances", "per_instance_stats", "display_predictions", "display_requests"):
            with opener(path.with_name(kind + suffix), "rt") as handle:
                documents[kind] = json.load(handle)
        results = None
        if configuration.get("result_file"):
            with opener(path.with_name(configuration["result_file"] + (".gz" if path.suffix == ".gz" else "")), "rt") as handle:
                states = json.load(handle)["request_states"]
            results = {(str(row["instance"]["id"]), row["train_trial_index"]): row["result"] for row in states}
            if len(results) != len(states):
                raise ValueError("Duplicate native HELM results")
        spec = documents["run_spec"]
        if "metric" in configuration:
            metric = configuration["metric"]
            profile = metadata["grading"]["verifiers"]["published"]
        else:
            primary = spec["metric_specs"][0]
            names = primary.get("args", {}).get("names", [])
            metric = names[0] if names else metadata["build"]["parameters"]["primary_metrics"][primary["class_name"]]
            profile = metadata["grading"]["verifiers"][metric]
        bank = {str(record["id"]): record for record in documents["instances"]}
        if len(bank) != len(documents["instances"]):
            raise ValueError("Duplicate native HELM item identifiers")
        requests = {(str(record["instance_id"]), record["train_trial_index"]): record["request"]
                    for record in documents["display_requests"]}
        if len(requests) != len(documents["display_requests"]):
            raise ValueError("Duplicate native HELM request identifiers")
        grades = {}
        for record in documents["per_instance_stats"]:
            for stat in record["stats"]:
                if stat["name"]["name"] != metric:
                    continue
                key = str(record["instance_id"]), record["train_trial_index"]
                if key in grades:
                    raise ValueError("Duplicate selected native HELM grades")
                grades[key] = stat["mean"]
        prediction_keys = set()
        for record in documents["display_predictions"]:
            key = str(record["instance_id"]), record["train_trial_index"]
            identity = (spec["name"], *key)
            if identity in expected:
                raise ValueError("Duplicate native HELM attempt")
            prediction_keys.add(key)
            if metric in record["stats"] and grades.get(key) != record["stats"][metric]:
                raise ValueError("Native HELM grade files disagree")
            item = bank[key[0]]
            references = [ref["output"]["text"] for ref in item["references"] if "correct" in ref["tags"]]
            reference_kind = profile.get("reference_kind", "gold")
            if reference_kind == "gold":
                if not references:
                    raise ValueError("Expected a correct native reference")
                if profile.get("reference_encoding") != "json_array" and len(references) != 1:
                    raise ValueError("Multiple references need a declared array encoding")
            verifier_spec = profile
            if profile.get("class") == "judge":
                annotation = record["annotations"][profile["annotation_key"]]
                prompt = annotation.get("prompt_text") or ""
                rubric = None
                if prompt:
                    prefix, start, remainder = prompt.partition(profile["model_response_start"])
                    body, end, suffix_text = remainder.rpartition(profile["model_response_end"])
                    if not start or not end:
                        raise ValueError("Missing model-output boundary in native judge prompt")
                    if body.strip() != (record.get("predicted_text") or "").strip():
                        raise ValueError("Native judge prompt encloses a different model output")
                    rubric = prefix + start + "{{model_response}}" + end + suffix_text
                verifier_spec = {**profile, "rubric": rubric}
                if metric == "safety_score":
                    values = [value for field, value in annotation.items() if field.endswith("_score") and value is not None]
                    derived_grade = sum(values) / len(values) if values else None
                elif metric == "air_score":
                    derived_grade = annotation["score"]
                else:
                    raise ValueError("Review the independent check for this judge metric")
                if derived_grade != grades.get(key):
                    raise ValueError("Published HELM metric differs from its native judge ratings")
            features = None
            if reference_kind == "metadata":
                features = {"upstream_reference_metadata": item["references"]}
            elif profile.get("reference_kind") == "gold":
                features = {"reference_tags": sorted({tag for reference in item["references"] for tag in reference.get("tags", []) if tag != "correct"})}
            elif "sub_split" in item:
                features = {"sub_split": item["sub_split"]}
            if configuration.get("item_features"):
                features = {**(features or {}), "scenario": spec[configuration["item_features"]]}
            expected[identity] = dict(
                model=spec["adapter_spec"]["model"], adapter=spec["adapter_spec"],
                request=requests[key], prediction=record, grade=grades.get(key),
                reference=(references if profile.get("reference_encoding") == "json_array" else references[0] or None)
                          if reference_kind == "gold" else None,
                extra=item.get("extra_data") or {}, source_input=item["input"]["text"],
                verifier_spec=verifier_spec, features=features,
                metric=metric, profile=profile,
                result=results[key] if results is not None else None,
            )
        if set(grades) - prediction_keys:
            raise ValueError("Native HELM grades lack predictions")

    subjects = {row["subject_id"]: row for row in pq.read_table(tables_directory / "subjects.parquet").to_pylist()}
    items = {row["item_id"]: row for row in pq.read_table(tables_directory / "items.parquet").to_pylist()}
    traces = {row["response_id"]: row["trace"] for row in pq.read_table(tables_directory / "traces.parquet").to_pylist()}
    responses = pq.read_table(tables_directory / "responses.parquet").to_pylist()
    if len(traces) != len(responses):
        raise ValueError("Every HELM observation must have its complete published trace")
    seen = set()
    for row in responses:
        trace = json.loads(traces[row["response_id"]])
        prediction = trace["prediction"]
        identity = trace["source_run"], str(prediction["instance_id"]), prediction["train_trial_index"]
        if identity in seen or identity not in expected:
            raise ValueError(f"Unexpected or repeated HELM attempt: {identity}")
        seen.add(identity)
        native = expected[identity]
        profile = native["profile"]
        item, subject = items[row["item_id"]], subjects[row["subject_id"]]
        criterion = json.loads(item["grading_criterion"])
        rule = json.loads(criterion["rule"])
        settings_text = subject["subject_features_extra"].partition("request_settings=")[2]
        actual_settings = ast.literal_eval(settings_text)
        desired_settings = {key: value for key, value in native["request"].items() if key != "prompt"}
        checks = {
            "grade": row["response"] == native["grade"],
            "condition": row["test_condition"] == "scenario=" + configuration["scenario"],
            "model": subject["display_name"] == native["model"],
            "harness": subject["harness"] == "HELM",
            "release": f"helm_release={configuration['release']};" in subject["subject_features_extra"],
            "adapter": f"adapter_method={native['adapter']['method']};" in subject["subject_features_extra"],
            "training allowance": f"max_train_instances={native['adapter']['max_train_instances']};" in subject["subject_features_extra"],
            "generation settings": actual_settings == desired_settings,
            # Canonical item identity treats NFC-equivalent text and surrounding
            # whitespace alike. The request in the trace must still match exactly.
            "canonical prompt": unicodedata.normalize("NFC", item["content"]).strip()
                                == unicodedata.normalize("NFC", native["request"]["prompt"]).strip(),
            "reference": (json.loads(criterion["reference_answer"]) if profile.get("reference_encoding") == "json_array"
                          else criterion.get("reference_answer")) == native["reference"],
            "grading rule": rule == {"description": profile.get("criterion", metadata["grading"]["rule"]), "extra_data": native["extra"]},
            "item scale": criterion.get("response_scale") == profile.get("response_scale"),
            "verifier": json.loads(json.loads(item["verifier"])["spec"]) == native["verifier_spec"],
            "verifier class": json.loads(item["verifier"])["class"] == profile.get("class", "exact_matcher"),
            "full trace": trace == {"source_run": identity[0], "request": native["request"], "prediction": native["prediction"],
                                     **({"native_result": native["result"]} if configuration.get("result_file") else {})},
        }
        if native["features"] is not None:
            checks["item features"] = item["item_features"] == ";".join(
                f"{name}={value}" for name, value in sorted(native["features"].items()) if value is not None)
        if not all(checks.values()):
            raise ValueError(f"HELM {identity}: differing fields {[name for name, valid in checks.items() if not valid]}")
    if seen != set(expected):
        raise ValueError(f"Missing {len(set(expected) - seen)} native HELM attempts")
    grade_total = "source_successes" if directory.name.startswith("medhelm_") or configuration.get("metric") == "exact_match" else "source_grade_sum"
    if "metric" not in configuration:
        # Different task metrics have different units; keep their totals separate.
        metrics = sorted({value["metric"] for value in expected.values()})
        totals = {"source_grade_sums": {metric: sum(value["grade"] or 0 for value in expected.values() if value["metric"] == metric)
                                        for metric in metrics}}
    else:
        totals = {grade_total: sum(value["grade"] or 0 for value in expected.values())}
    return {"source_model_exports": len(run_paths), "source_responses": len(expected),
            **totals,
            "source_ungraded": sum(value["grade"] is None for value in expected.values())}


def verify_mlrbench_upstream(directory, tables_directory=None):
    """Reconcile every selected rubric grade with native tasks, artifacts and reviews."""
    import ast
    import tarfile

    directory = Path(directory)
    target = Path(tables_directory) if tables_directory is not None else directory / "formatted_tables"
    sources = {}
    with tarfile.open(next((directory / "raw").glob("mlrbench-*.tar.gz"))) as archive:
        for member in archive:
            path = member.name.partition("/")[2]
            if member.isfile() and path.startswith(("tasks/", "agent_results/ideas_and_proposals/",
                    "agent_reviews/idea_proposal_reviews_", "mlrbench/agent/", "mlrbench/evals/")):
                sources[path] = archive.extractfile(member).read().decode()

    prompts, rubrics = {}, {}
    for stage in ("idea", "proposal"):
        code = ast.parse(sources[f"mlrbench/agent/{stage}_generator.py"])
        function = next(node for node in code.body if isinstance(node, ast.FunctionDef) and node.name == "generate_" + stage)
        prefixes = [ast.literal_eval(node.value) for node in function.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets)]
        suffix = next(node.value for node in function.body if isinstance(node, ast.AugAssign))
        assert len(prefixes) == 1
        prompts[stage] = prefixes[0], suffix
        module = ast.parse(sources[f"mlrbench/evals/review_{stage}.py"])
        symbol = "RESEARCH_" + stage.upper() + "_RUBRIC"
        rubrics[stage] = next(ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets))

    expected, generations, tasks, models, reviewers = {}, set(), set(), set(), set()
    for path, text in sources.items():
        if not path.startswith("agent_reviews/idea_proposal_reviews_") or not path.endswith(".json"):
            continue
        _, panel, task, stage, filename = path.split("/")
        reviewer = panel.removeprefix("idea_proposal_reviews_")
        assert filename.startswith(stage + "_")
        model = filename.removesuffix(".json").removeprefix(stage + "_")
        native = json.loads(text)
        context = {"task": sources[f"tasks/{task}.md"]}
        if stage == "proposal":
            context.update(idea=sources[f"agent_results/ideas_and_proposals/{task}/idea.md"],
                           related_work=sources[f"agent_results/ideas_and_proposals/{task}/related_work.md"])
        prefix, suffix = prompts[stage]
        # Render the native f-string from its literal/variable segments without
        # importing or executing the provider's Python or the builder.
        content = prefix
        for part in suffix.values:
            if isinstance(part, ast.Constant):
                content += part.value
            else:
                assert isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
                assert part.conversion == -1 and part.format_spec is None
                content += context[part.value.id]
        generation = f"agent_results/ideas_and_proposals/{task}/{stage}/{stage}_{model}.md"
        assert generation in sources
        for metric, assessment in native.items():
            assert type(assessment["score"]) is int and assessment["score"] in range(1, 11)
            expected[path, metric] = (model, task, stage, reviewer, content, generation, assessment)
        generations.add(generation); tasks.add(task); models.add(model); reviewers.add(reviewer)

    frames = {p.stem: pd.read_parquet(p) for p in target.glob("*.parquet")}
    subjects = frames["subjects"].set_index("subject_id").to_dict("index")
    items = frames["items"].set_index("item_id").to_dict("index")
    traces = frames["traces"].set_index("response_id").trace.to_dict()
    seen = set()
    for response in frames["responses"].itertuples():
        subject, item = subjects[response.subject_id], items[response.item_id]
        trace = json.loads(traces[response.response_id])
        criterion = json.loads(json.loads(item["grading_criterion"])["rule"])
        metric = criterion["metric"]
        key = trace["source_review"], metric
        if key not in expected or key in seen:
            raise ValueError(f"Extra or repeated MLR-Bench observation: {key}")
        seen.add(key)
        model, task, stage, reviewer, content, generation, assessment = expected[key]
        verifier = json.loads(json.loads(item["verifier"])["spec"])
        checks = {
            "generator": subject["display_name"] == model and subject["harness"] == "MLR-Agent",
            "item": item["raw_item_id"] == f"{task}/{stage}/{reviewer}/{metric}",
            "input": item["content"] == content,
            "grade": response.response == assessment["score"],
            "rubric": criterion["rubric"] == rubrics[stage] and criterion["metric"] == metric,
            "reviewer": verifier["model"] == reviewer and verifier["grade_field"] == metric + ".score",
            "generation": trace["source_generation"] == generation and trace["generation"] == sources[generation],
            "assessment": trace["assessment"] == assessment,
        }
        if not all(checks.values()):
            raise ValueError(f"MLR-Bench {key}: differing {[name for name, matches in checks.items() if not matches]}")
    if seen != set(expected):
        raise ValueError(f"Missing {len(set(expected) - seen)} native MLR-Bench rubric grades")
    return {"source_rubric_grades": len(expected), "source_reviews": len({key[0] for key in expected}),
            "source_generated_artifacts": len(generations), "source_tasks": len(tasks),
            "source_generators": len(models), "source_reviewers": len(reviewers),
            "source_grade_sum": sum(value[-1]["score"] for value in expected.values())}


def verify_cellverse_upstream(directory, tables_directory=None):
    """Match each curated observation to the released output and independent task bank."""
    directory = Path(directory)
    target = Path(tables_directory) if tables_directory is not None else directory / "formatted_tables"
    bank = _json(directory / "raw/data/cta_scrna_full.json")
    outputs = _json(directory / "raw/results/ms_cta_response_deepseek_r1.json")
    native = {}
    for row in outputs:
        key = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        if key in native:
            raise ValueError("Repeated CellVerse source task")
        native[key] = row
    frames = {p.stem: pd.read_parquet(p) for p in target.glob("*.parquet")}
    subjects = frames["subjects"].set_index("subject_id").to_dict("index")
    items = frames["items"].set_index("item_id").to_dict("index")
    traces = frames["traces"].set_index("response_id").trace.to_dict()
    seen = set()
    successes = 0
    for response in frames["responses"].itertuples():
        item, subject = items[response.item_id], subjects[response.subject_id]
        prefix, number = item["raw_item_id"].split(":")
        index = int(number)
        if prefix != "cta_scrna_full" or index in seen or not 0 <= index < len(bank):
            raise ValueError("Unexpected or repeated CellVerse item")
        seen.add(index)
        messages = bank[index]["messages"]
        row = native[json.dumps(messages, ensure_ascii=False, sort_keys=True)]
        reference = messages[-1]["content"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert row["ground_truth"] == reference
        correct = float(row["prediction"] == reference)
        criterion = json.loads(item["grading_criterion"])
        checks = {
            "subject": subject["display_name"] == "DeepSeek-R1" and subject["harness"] == "CellVerse",
            "input": json.loads(item["content"]) == messages[:-1],
            "reference": criterion["reference_answer"] == reference,
            "grade": response.response == correct,
            "trace": traces[response.response_id] == row["model_response"],
        }
        if not all(checks.values()):
            raise ValueError(f"CellVerse row {index}: differing {[k for k, v in checks.items() if not v]}")
        successes += int(correct)
    if seen != set(range(len(bank))) or len(native) != len(bank):
        raise ValueError("CellVerse does not exactly cover the released 748-question subset")
    leaderboard = next(line for line in (directory / "raw/README.md").read_text().splitlines()
                       if line.startswith("|") and "**DeepSeek-R1" in line)
    reported = float(leaderboard.split("|")[7].strip())
    if abs(100 * successes / len(bank) - reported) > 0.005:
        raise ValueError("CellVerse exact matches disagree with the provider's rounded accuracy")
    return {"source_responses": len(outputs), "source_items": len(bank), "source_successes": successes,
            "source_traces": len(traces), "reported_accuracy_percent": reported}


SOURCE_AUDITS = dict.fromkeys(['cellverse', 'dave', 'abstract_reason', 'proofaug', 'putnam_axiom', 'software_world_models', 'agi_elo', 'agentdojo', 'agentic_review_perturbation', 'algotune', 'appworld', 'causal_axioms', 'confagents', 'corebench', 'donotanswer', 'editbench', 'evouna', 'fgbench', 'frontieror', 'hcast', 'icpc2_code_selector', 'ikp', 'live_agent_risk', 'matharena_platform', 'mathconstraint', 'metaagent', 'os_harm', 'osworld', 'programbench', 'revolve', 'scivisagentbench', 'svrpbench', 'swe_chat', 'swe_smith', 'swe_together', 'swebench_java', 'swebench_live', 'tau2_bench', 'theagentcompany', 'vl_rewardbench', 'wikihow_agent', 'workbench_revisited'], source_observations)

SOURCE_AUDITS.update(dict.fromkeys([
    "medhelm_ehr_sql",
    "medhelm_head_qa",
    "medhelm_med_mcqa",
    "medhelm_med_qa",
    "medhelm_medbullets",
    "medhelm_medcalc_bench",
    "medhelm_medec",
    "medhelm_medhallu",
    "medhelm_pubmed_qa",
    "medhelm_race_based_med",
], source_observations))

SOURCE_AUDITS.update(dict.fromkeys([
    "helm_airbench",
    "helm_anthropic_redteam",
    "helm_bbq",
    "helm_bold",
    "helm_harmbench",
    "helm_real_toxicity_prompts",
    "helm_simple_safety_tests",
    "helm_xstest",
], source_observations))

SOURCE_AUDITS.update(dict.fromkeys(["helm_afr", "helm_cleva", "helm_thaiexam"], source_observations))

SOURCE_AUDITS["mlrbench"] = source_observations

from measurement_db.scripts.curate_benchmarks.native_result_audits import verify_native_results
SOURCE_AUDITS.update(dict.fromkeys(["openbiorq", "fewshot_ttt_bbh", "prox", "cseo_bench", "risebench", "clasheval", "engdesign", "phyblock", "scigym", "advprompter"], verify_native_results))
SOURCE_AUDITS["legal_rag_bench"] = verify_native_results
SOURCE_AUDITS["nester"] = verify_native_results
SOURCE_AUDITS["engibench"] = verify_native_results
SOURCE_AUDITS["llmail_inject"] = verify_native_results
SOURCE_AUDITS["safeagentbench"] = verify_native_results
SOURCE_AUDITS["critic_discernment_game"] = verify_native_results
SOURCE_AUDITS["brace"] = verify_native_results
SOURCE_AUDITS["biggen"] = verify_native_results
SOURCE_AUDITS["annotating_errors_wcf"] = verify_native_results
SOURCE_AUDITS["bertaqa"] = verify_native_results
SOURCE_AUDITS["afrimedqa"] = verify_native_results
SOURCE_AUDITS["agc_bench"] = verify_native_results
SOURCE_AUDITS["adaptivestep"] = verify_native_results
SOURCE_AUDITS["algotune"] = verify_native_results
SOURCE_AUDITS["aider"] = verify_native_results
SOURCE_AUDITS["alpacaeval"] = verify_native_results
SOURCE_AUDITS["ai2d_test"] = verify_native_results
SOURCE_AUDITS["alpha_sql"] = verify_native_results
SOURCE_AUDITS["alignment_faking"] = verify_native_results
SOURCE_AUDITS["arcagi"] = verify_native_results
SOURCE_AUDITS["arena_140k"] = verify_native_results
SOURCE_AUDITS["atmossci_bench"] = verify_native_results
SOURCE_AUDITS["auditing_sabotage_bench"] = verify_native_results
SOURCE_AUDITS["autoresearchbench"] = verify_native_results
SOURCE_AUDITS["averimatec"] = verify_native_results
SOURCE_AUDITS["babilong"] = verify_native_results
SOURCE_AUDITS["bbq"] = verify_native_results
SOURCE_AUDITS["beavertails"] = verify_native_results

SOURCE_AUDITS["benger"] = verify_native_results

SOURCE_AUDITS["bedd_basalt"] = verify_native_results

SOURCE_AUDITS["bigfinancebench"] = verify_native_results

SOURCE_AUDITS["bountybench"] = verify_native_results

SOURCE_AUDITS["bird_sql"] = verify_native_results

SOURCE_AUDITS["braveguard"] = verify_native_results

SOURCE_AUDITS["bridging_gap"] = verify_native_results

SOURCE_AUDITS["care_enzymes"] = verify_native_results

SOURCE_AUDITS["ceobench"] = verify_native_results

SOURCE_AUDITS["chatgpt_drift"] = verify_native_results

SOURCE_AUDITS["chartmuseum"] = verify_native_results

SOURCE_AUDITS["ceval"] = verify_native_results
SOURCE_AUDITS["cmmlu"] = verify_native_results

SOURCE_AUDITS["chi_bench"] = verify_native_results

SOURCE_AUDITS["chipbench"] = verify_native_results

SOURCE_AUDITS["classroom_ai"] = verify_native_results

SOURCE_AUDITS["coffeebench"] = verify_native_results

SOURCE_AUDITS["complexbench"] = verify_native_results

SOURCE_AUDITS["csedb"] = verify_native_results

SOURCE_AUDITS["crow"] = verify_native_results

SOURCE_AUDITS["cruxeval"] = verify_native_results

SOURCE_AUDITS["das_med_hallucination"] = verify_native_results

SOURCE_AUDITS["cybench"] = verify_native_results

SOURCE_AUDITS["dataclawbench"] = verify_native_results
