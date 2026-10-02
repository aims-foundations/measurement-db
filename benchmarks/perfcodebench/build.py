#!/usr/bin/env python3
"""Tabulate PerfCodeBench's original candidate assessments and complete task contracts."""

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class PerfCodeBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        settings, paths = self.build_parameters, self.build_parameters["paths"]

        # 1. Load original archive members, task definitions and result JSON into tables.
        files = {}
        with tarfile.open(self.raw_dir / paths["archive"], "r|gz") as archive:
            for member in archive:
                if member.isfile():
                    files[member.name.removeprefix(paths["root"] + "/")] = archive.extractfile(member).read()
        sources = pd.DataFrame.from_dict(files, orient="index", columns=["data"]).rename_axis("source_member").reset_index()
        task_files = sources.loc[sources.source_member.str.fullmatch(paths["tasks"])].reset_index(drop=True)
        tasks = pd.json_normalize(task_files.data.map(json.loads), max_level=0)
        tasks["task_member"] = task_files.source_member
        tasks["task_record"] = task_files.data.map(json.loads)
        result_files = sources.loc[sources.source_member.str.fullmatch(paths["results"])].reset_index(drop=True)
        results = pd.json_normalize(result_files.data.map(json.loads), max_level=0)
        results["source_member"] = result_files.source_member
        results["source_record"] = result_files.data.map(json.loads)
        if tasks.task_id.duplicated().any() or not tasks.task_member.str.split("/").str[1].eq(tasks.task_id).all():
            raise ValueError("PerfCodeBench requires unique matching task IDs and archive locations")
        if not results.source_member.str.split("/").str[1].eq(results.task_id).all():
            raise ValueError("PerfCodeBench result filename and task ID disagree")

        # 2. Keep evaluated candidates and their recorded grading allowance; missing candidates stay absent.
        results["candidate_status"] = results.candidate.str["status"]
        observations = results.loc[~results.candidate_status.eq(settings["labels"]["missing"])].copy()
        observations["directory_model"] = observations.source_member.str.split("/candidate/").str[1].str.rsplit("/", n=1).str[0]
        if not observations.directory_model.eq(observations.model.str.replace("/", "__", regex=False)).all():
            raise ValueError("PerfCodeBench result model and candidate directory disagree")
        if observations.duplicated(["model", "task_id"]).any() or not observations.dry_run.eq(False).all():
            raise ValueError("PerfCodeBench requires one recorded, non-dry-run candidate per model/task")
        if not observations.candidate_status.isin([settings["labels"]["success"], *settings["failure_grades"]]).all():
            raise ValueError("Review an unknown PerfCodeBench execution status before assigning a grade")
        successful = observations.candidate_status.eq(settings["labels"]["success"])
        if not observations.loc[successful, "candidate"].map(lambda row: type(row.get("all_ok")) is bool).all():
            raise ValueError("A completed PerfCodeBench assessment requires an explicit Boolean verdict")
        observations["response"] = observations.candidate_status.map(settings["failure_grades"]).astype(float)
        observations.loc[successful, "response"] = observations.loc[successful, "candidate"].str["all_ok"].astype(float)
        observations["execution_repetitions"] = observations.candidate.str["runs"]
        conditions = observations[["task_id", "execution_repetitions", "benchmark_timeout_sec"]].drop_duplicates()
        if conditions.task_id.duplicated().any():
            raise ValueError("Separate changed grading allowances before combining PerfCodeBench observations")
        tasks = tasks.merge(conditions, on="task_id", how="inner", validate="one_to_one")
        if not observations.task_id.isin(tasks.task_id).all():
            raise ValueError("PerfCodeBench has an assessment without its original task definition")

        # 3. Join the complete baseline and first available interface, following the published prompt builder.
        tasks["solution_file"] = tasks.task_record.map(lambda row: row.get("solution_filename", settings["labels"]["default_solution"]))
        tasks["task_root"] = tasks.task_member.str.removesuffix("instance.json")
        tasks["baseline_member"] = tasks.task_root + "baseline/" + tasks.solution_file
        tasks["baseline_source"] = tasks.baseline_member.map(lambda name: files[name].decode())
        tasks["reference_member"] = tasks.task_root + "reference/" + tasks.solution_file
        tasks["reference_source"] = tasks.reference_member.map(lambda name: files[name].decode())
        interfaces = sources.loc[sources.source_member.str.fullmatch(paths["interfaces"])].copy()
        interfaces["task_id"] = interfaces.source_member.str.split("/").str[1]
        interfaces["priority"] = interfaces.source_member.str.rsplit("/", n=1).str[-1].map(settings["interface_priority"])
        interfaces["interface_source"] = interfaces.data.map(bytes.decode)
        interfaces = interfaces.sort_values(["task_id", "priority"]).drop_duplicates("task_id")
        tasks = tasks.merge(interfaces[["task_id", "interface_source"]], on="task_id", how="left", validate="one_to_one")
        tasks["interface_source"] = tasks.interface_source.fillna("")
        tasks["content"] = [settings["prompts"]["user"].format(solution_file=row.solution_file, task_id=row.task_id,
            title=row.title, goal=row.goal, metric=row.metric, correctness_rule=row.correctness_rule,
            includes=", ".join(row.task_record.get("allowed_external_includes", [])), baseline_source=row.baseline_source,
            interface_block=settings["prompts"]["interface"].format(interface_source=row.interface_source) if row.interface_source else "")
            for row in tasks.itertuples()]
        harness = sources.loc[sources.source_member.str.fullmatch(paths["harness"])].copy()
        harness["task_id"] = harness.source_member.str.split("/").str[1]
        harness["text"] = harness.data.map(bytes.decode)
        harness["source"] = harness[["source_member", "text"]].to_dict("records")
        tasks = tasks.join(harness.groupby("task_id", sort=False).source.agg(list).rename("harness_sources"), on="task_id")

        # 4. Define literal model identities and task-specific grading, keeping reference code out of the prompt.
        subjects = observations[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = settings["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = subjects.subject_key.map(lambda model: dict(recorded_model_label=model,
            harness=settings["labels"]["harness"], settings_status=settings["labels"]["settings_status"]))
        items = tasks.rename(columns={"task_id": "item_key"}).copy()
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(source_task_id=row.item_key, language=settings["languages"][Path(row.solution_file).suffix],
            prompt_basis=settings["labels"]["prompt_basis"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.reference_source,
            rule=json.dumps(dict(metric=self.grading["rule"], correctness_rule=row.correctness_rule,
                execution_repetitions=int(row.execution_repetitions), timeout_sec=int(row.benchmark_timeout_sec),
                build=row.task_record["build"], run=row.task_record.get("run"), harness_sources=row.harness_sources), ensure_ascii=False))
            for row in items.itertuples()]
        items["verifier"] = [ExactMatcher(spec=json.dumps(self.grading["verifiers"]["correct_and_runnable"], sort_keys=True))] * len(items)

        # 5. Link one assessment per generated candidate and preserve complete native output, errors and timings.
        observations = observations.merge(tasks[["task_id", "task_member"]], on="task_id", validate="many_to_one")
        observations["candidate_source"] = observations.candidate_path.map(lambda name: files[name].decode())
        if not observations.candidate_source.eq(observations.model_output_solution_source).all():
            raise ValueError("PerfCodeBench saved code differs from the source recorded in its assessment")
        responses = observations.rename(columns={"model": "subject_key", "task_id": "item_key", "source_member": "response_key"})
        responses["trial"] = 1
        responses["test_condition"] = [json.dumps(dict(metric=settings["labels"]["metric"],
            execution_repetitions=int(row.execution_repetitions), timeout_sec=int(row.benchmark_timeout_sec)), sort_keys=True)
            for row in responses.itertuples()]
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_archive=paths["archive"], source_member=row.response_key,
            source_record=row.source_record, task_member=row.task_member, candidate_source_member=row.candidate_path),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
                "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
                "responses": responses[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
                "traces": traces}


if __name__ == "__main__":
    PerfCodeBench(__file__).main_from_args()
