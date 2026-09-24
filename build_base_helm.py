"""Shared HELM importers.

HelmTabularBuild imports pinned native requests, item definitions, predictions
and per-instance metrics through six table stages. Its source and grading
configuration live in each benchmark's metadata.yaml. Metric selection follows
the release declarations; original requests and complete outputs remain in traces.
"""

import gzip
import json
import pandas as pd
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_base import BenchmarkBuild, ExactMatcher, Judge  # noqa: E402
from scripts.build_measurement_tables.response_scales import validate_grade

class HelmTabularBuild(BenchmarkBuild):
    """Import native HELM generation exports through checked table joins.

    Each benchmark selects its upstream files and metric in metadata.yaml.
    Multi-task releases preserve their native scenario definitions and scales.
    """

    def download(self):
        names = [source.get("name") for source in self.source_manifest["upstream"]]
        return self.fetch_sources(*(["release"] if "release" in names else []), "runs", "grader")

    def build_tables(self):
        # 1. Load the five native tables for every run. Keep complete prediction
        # records so optional annotations and model outputs survive in traces.
        batches = {name: [] for name in (
            "run_spec", "instances", "per_instance_stats", "display_predictions", "display_requests")}
        settings = self.build_parameters["helm"]
        native_results = []
        declared = getattr(self, "source_files", ())
        paths = ([self.raw_dir / name for name in declared
                  if Path(name).parts[0] == "runs" and Path(name).name in {"run_spec.json", "run_spec.json.gz"}] if declared
                 else (self.raw_dir / "runs").rglob("run_spec.json*"))
        for path in sorted(paths):
            opener = gzip.open if path.suffix == ".gz" else open
            suffix = ".json.gz" if path.suffix == ".gz" else ".json"
            for name in batches:
                with opener(path.with_name(name + suffix), "rt") as handle:
                    records = json.load(handle)
                frame = pd.json_normalize(records, max_level=0)
                if name == "display_predictions":
                    frame["prediction_record"] = pd.Series(records, dtype=object)
                batches[name].append(frame.assign(run_key=path.parent.relative_to(self.raw_dir / "runs").as_posix()))
            if settings.get("result_file"):
                result_path = path.with_name(settings["result_file"] + (".gz" if path.suffix == ".gz" else ""))
                with opener(result_path, "rt") as handle:
                    state = json.load(handle)
                frame = pd.json_normalize(state["request_states"], max_level=0)
                frame["instance_id"] = frame.instance.map(lambda value: str(value["id"]))
                native_results.append(frame.assign(run_key=path.parent.relative_to(self.raw_dir / "runs").as_posix()))
        if not batches["run_spec"]:
            raise ValueError("No captured HELM runs; run download() first")
        runs, instances, stats, predictions, requests = (
            pd.concat(frames, ignore_index=True) for frames in batches.values())
        if "metric" not in settings:
            # The first declared metric is the task metric; later definitions
            # may describe auxiliary harms or efficiency measures.
            primary = pd.json_normalize(runs.metric_specs.str[0].tolist()).reindex(columns=["args.names", "class_name"])
            runs["metric"] = primary["args.names"].str[0].fillna(
                primary.class_name.map(self.build_parameters["primary_metrics"]))
            profiles = self.grading["verifiers"]
        else:
            runs["metric"] = settings["metric"]
            profiles = {settings["metric"]: self.grading["verifiers"]["published"]}
        if runs.metric.isna().any() or not set(runs.metric) <= profiles.keys():
            raise ValueError("Every HELM task metric needs an explicit grading description")
        keys = ["run_key", "instance_id", "train_trial_index"]
        instances = instances.rename(columns={"id": "instance_id"})
        for frame in (instances, stats, predictions, requests):
            frame["instance_id"] = frame.instance_id.astype(str)

        # 2. Explode metric lists and select the declared grade. A missing grade
        # remains null; it is neither a failure nor an absent model attempt.
        stats = stats.explode("stats", ignore_index=True)
        values = pd.json_normalize(stats.stats.tolist())
        stats = stats[keys].join(values).merge(runs[["run_key", "metric"]], on="run_key", validate="many_to_one")
        stats = stats.loc[stats["name.name"].eq(stats.metric)]
        stats = stats.rename(columns={"mean": "response"})[keys + ["response"]]
        predictions = predictions.merge(runs[["run_key", "metric"]], on="run_key", validate="many_to_one")
        attempts = predictions.merge(stats, on=keys, how="outer", validate="one_to_one", indicator=True)
        if attempts._merge.eq("right_only").any():
            raise ValueError("HELM grades without corresponding published predictions")
        attempts = attempts.drop(columns="_merge")
        display_grades = attempts.apply(lambda row: row.stats.get(row.metric), axis=1)
        display_has_metric = attempts.apply(lambda row: row.metric in row.stats, axis=1)
        agrees = attempts.response.eq(display_grades) | (attempts.response.isna() & display_grades.isna())
        if not agrees[display_has_metric].all():
            raise ValueError("HELM display and per-instance grades disagree")
        # Some display exports omit the selected metric entirely. Per-instance
        # statistics remain authoritative; their grades must fit the declared scale.
        for metric, rows in attempts.groupby("metric"):
            scale = profiles[metric].get("response_scale", self.INFO["response_scale"])
            for value in rows.response.dropna().unique():
                validate_grade(value, scale)

        # 3. Join each attempt to its own run's input and actual request, including
        # answer choices and instructions. A first-model item bank is insufficient.
        attempts = attempts.merge(requests[keys + ["request"]], on=keys, how="left", validate="one_to_one", indicator=True)
        if not attempts._merge.eq("both").all():
            raise ValueError("A HELM prediction lacks its actual request")
        attempts = attempts.drop(columns="_merge").merge(
            instances, on=["run_key", "instance_id"], how="left", validate="many_to_one", indicator=True)
        if not attempts._merge.eq("both").all():
            raise ValueError("A HELM prediction lacks its item definition")
        run_columns = ["run_key", "name", "adapter_spec"]
        if settings.get("item_features"):
            run_columns.append(settings["item_features"])
        attempts = attempts.drop(columns="_merge").merge(
            runs[run_columns], on="run_key", how="left", validate="many_to_one")
        if native_results:
            results = pd.concat(native_results, ignore_index=True).rename(columns={"result": "native_result"})
            attempts = attempts.merge(results[keys + ["native_result"]], on=keys, how="left", validate="one_to_one", indicator=True)
            if not attempts._merge.eq("both").all():
                raise ValueError("A HELM observation lacks its complete native result")
            attempts = attempts.drop(columns="_merge")
        attempts["response_key"] = attempts.index.astype(str)
        attempts["content"] = attempts.request.map(lambda value: value["prompt"])
        if not attempts.content.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("HELM requests require nonempty prompts")

        # 4. Subjects retain the actual generation settings. Distinct configurations
        # stay distinct even when the displayed model label is the same.
        attempts["request_settings"] = attempts.request.map(
            lambda value: json.dumps({key: item for key, item in value.items() if key != "prompt"}, sort_keys=True))
        configurations = attempts[["run_key", "request_settings", "adapter_spec"]].drop_duplicates(
            subset=["run_key", "request_settings"]).reset_index(drop=True)
        configurations["subject_key"] = configurations.index.astype(str)
        subjects = configurations.assign(
            raw_label=lambda frame: frame.adapter_spec.map(lambda value: value["model"]),
            features=lambda frame: frame.apply(lambda row: {
                "harness": "HELM", "helm_release": settings["release"],
                "adapter_method": row.adapter_spec["method"],
                "max_train_instances": row.adapter_spec["max_train_instances"],
                "request_settings": json.loads(row.request_settings),
            }, axis=1))[["subject_key", "raw_label", "features"]]
        attempts = attempts.merge(configurations[["run_key", "request_settings", "subject_key"]],
                                  on=["run_key", "request_settings"], how="left", validate="many_to_one")

        # 5. Preserve the reference and item-specific grading data. The declared
        # metric identifies what was measured; grades are not recomputed here.
        profile = next(iter(profiles.values()))
        if any(p.get(key) != profile.get(key) for p in profiles.values()
               for key in ("reference_kind", "reference_encoding", "class")):
            raise ValueError("HELM task profiles must share their reference representation and verifier class")
        if profile.get("reference_kind", "gold") == "gold":
            references = attempts[["response_key", "references"]].explode("references", ignore_index=True)
            reference_values = pd.json_normalize(references.references.tolist())
            references = references[["response_key"]].join(reference_values)
            references = references.loc[references.tags.map(lambda tags: "correct" in tags)]
            references = references.rename(columns={"output.text": "reference_answer"})
            if profile.get("reference_encoding") == "json_array":
                # Keep every accepted reference, including duplicates in translated
                # answer choices. The encoding is declared in the verifier spec.
                references = references.groupby("response_key", sort=False).reference_answer.agg(list).map(
                    lambda values: json.dumps(values, ensure_ascii=False)).reset_index()
            attempts = attempts.merge(references[["response_key", "reference_answer"]],
                                      on="response_key", how="left", validate="one_to_one", indicator=True)
            if not attempts._merge.eq("both").all():
                raise ValueError("Expected one correct-tagged reference per HELM item")
        else:
            # Safety exports use references for category labels, not gold answers.
            attempts["reference_answer"] = None
        verifiers = {name: ExactMatcher(spec=json.dumps(value, sort_keys=True)) for name, value in profiles.items()}
        attempts["verifier"] = attempts.metric.map(verifiers)
        if profile.get("class") == "judge":
            annotations = attempts.annotations.map(lambda value: value[profile["annotation_key"]])
            prompts = annotations.map(lambda value: value.get("prompt_text") or "")
            start_tag, end_tag = profile["model_response_start"], profile["model_response_end"]
            present = prompts.ne("")
            response_block = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
            enclosed = prompts.str.extract(response_block, flags=re.DOTALL, expand=False)
            if not (prompts[present].str.count(response_block, flags=re.DOTALL).eq(1)
                    & enclosed[present].str.strip().eq(attempts.predicted_text[present].fillna("").str.strip())).all():
                raise ValueError("Ambiguous model-response boundaries in a HELM judge prompt")
            # The response is an observation, not part of item identity. Preserve
            # the rubric with a placeholder; the complete original stays in traces.
            rubrics = prompts.str.replace(response_block,
                                          start_tag + "{{model_response}}" + end_tag,
                                          regex=True, flags=re.DOTALL)
            attempts["verifier"] = rubrics.map(lambda rubric: Judge(
                spec=json.dumps({**profile, "rubric": rubric or None}, sort_keys=True), judged_by="llm"))
        if "extra_data" not in attempts:
            attempts["extra_data"] = None
        items = attempts.assign(
            item_key=lambda frame: frame.response_key,
            raw_item_id=lambda frame: frame.instance_id,
            grading_criterion=lambda frame: frame.apply(lambda row: {
                "reference_answer": row.reference_answer or None,
                "rule": json.dumps({"description": profiles[row.metric].get("criterion", self.grading["rule"]),
                                    "extra_data": row.extra_data if isinstance(row.extra_data, dict) else {}}, sort_keys=True),
                **({"response_scale": profiles[row.metric]["response_scale"]}
                   if "response_scale" in profiles[row.metric] else {}),
            }, axis=1),
        )[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]]

        if profile.get("reference_kind") == "metadata":
            items["features"] = attempts.references.map(lambda references: {"upstream_reference_metadata": references})
        elif profile.get("reference_kind") == "gold":
            items["features"] = attempts.references.map(lambda references: {
                "reference_tags": sorted({tag for reference in references for tag in reference.get("tags", [])
                                          if tag != "correct"})})
        elif "sub_split" in attempts:
            items["features"] = attempts.sub_split.map(lambda value: {"sub_split": value})
        if settings.get("item_features"):
            if "features" not in items:
                items["features"] = [{}] * len(items)
            items["features"] = pd.DataFrame({"features": items.features, "scenario": attempts[settings["item_features"]]}).apply(
                lambda row: {**row.features, "scenario": row.scenario}, axis=1)

        # 6. Link observations to complete native traces. Number repeated attempts
        # only after canonical item identity is resolved by the shared writer.
        responses = attempts.assign(item_key=lambda frame: frame.response_key,
                                    test_condition="scenario=" + settings["scenario"])[
            ["response_key", "subject_key", "item_key", "response", "test_condition"]]
        traces = attempts.assign(trace=lambda frame: frame.apply(lambda row: json.dumps({
            "source_run": row["name"], "request": row.request,
            "prediction": row.prediction_record,
            **({"native_result": row.native_result} if native_results else {}),
        }, ensure_ascii=False, sort_keys=True), axis=1))[["response_key", "trace"]]
        return {"subjects": subjects, "items": items, "responses": responses, "traces": traces}
