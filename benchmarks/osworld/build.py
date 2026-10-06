#!/usr/bin/env python3
"""Tabulate OSWorld's original task definitions, graded runs and native text logs."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class OSWorld(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("tasks", "harness", "results", "notice")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, paths = self.build_parameters, self.build_parameters["paths"]

        # 1. Load original task JSON and captured native ZIP members into tables.
        task_paths = sorted((self.raw_dir / paths["tasks"]).rglob("*.json"))
        task_records = [json.loads(path.read_text()) for path in task_paths]
        tasks = pd.json_normalize(task_records, max_level=0)
        tasks["raw_item_id"] = [str(path.relative_to(self.raw_dir / paths["tasks"])).removesuffix(".json") for path in task_paths]
        tasks["task_record"] = task_records
        if tasks.raw_item_id.duplicated().any() or not tasks.raw_item_id.str.rsplit("/", n=1).str[-1].eq(tasks.id).all():
            raise ValueError("OSWorld task IDs must match unique original definition paths")
        if not tasks.instruction.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("OSWorld requires the original nonempty task instruction")
        native_paths = sorted(path for path in (self.raw_dir / paths["results"]).rglob("*") if path.is_file())
        files = pd.DataFrame({"source_file": [str(path.relative_to(self.raw_dir / paths["results"])) for path in native_paths],
                              "text": [path.read_bytes().decode("utf-8") for path in native_paths]})
        # Restore the downloader's safe-filename escapes, such as a model ID's colon.
        files["source_file"] = files.source_file.str.replace(r"_x([0-9a-f]+)_", lambda match: chr(int(match[1], 16)), regex=True)
        files[["archive", "member"]] = files.source_file.str.split("/", n=1, expand=True)

        # 2. Pivot task files by their exact archive/run/task directory, never by listing order.
        records = files.join(files.member.str.extract(paths["attempt_member"]))
        records = records.loc[records.task_id.notna()].copy()
        records["raw_item_id"] = records.domain + "/" + records.task_id
        keys = ["archive", "run_prefix", "raw_item_id"]
        if records.duplicated(keys + ["file_kind"]).any():
            raise ValueError("OSWorld repeats a native file for the same run and task")
        attempts = records.pivot(index=keys, columns="file_kind", values="text").reset_index()
        for kind in parameters["trace_files"]:
            if kind not in attempts:
                attempts[kind] = None
        attempts = attempts.astype(object).where(attempts.notna(), None)
        attempted = attempts["result.txt"].notna() | attempts["traj.jsonl"].fillna("").ne("") | attempts["runtime.log"].fillna("").ne("")
        attempts = attempts.loc[attempted].copy()
        attempts["response"] = attempts["result.txt"].map(float, na_action="ignore")
        graded = attempts["result.txt"].notna()
        if not attempts.loc[graded, "response"].between(0, 1).all():
            raise ValueError("OSWorld grades must be finite original scores in [0, 1]")

        # 3. Keep published archive configurations separate; carry only released inference settings.
        settings = files.loc[files.member.str.endswith("args.json")].copy()
        settings["run_prefix"] = settings.member.str.removesuffix("args.json")
        settings["configuration"] = settings.text.map(json.loads).map(
            lambda record: {key: record[key] for key in parameters["settings"] if key in record})
        runs = attempts[["archive", "run_prefix"]].drop_duplicates().merge(
            settings[["archive", "run_prefix", "configuration"]], on=["archive", "run_prefix"], how="left", validate="one_to_one")
        runs["configuration"] = runs.configuration.map(lambda value: value if isinstance(value, dict) else {})
        runs["subject_key"] = runs.archive + "/" + runs.configuration.map(lambda value: json.dumps(value, sort_keys=True))
        subjects = runs.drop_duplicates("subject_key").copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.archive.str.removesuffix(".zip")
        subjects["features"] = [dict(harness=parameters["labels"]["harness"], source_archive=row.archive,
            recorded_configuration=json.dumps(row.configuration, sort_keys=True, allow_nan=False).replace(";", r"\u003b").replace("=", r"\u003d"),
            configuration_scope=parameters["labels"]["configuration_scope"])
            for row in subjects.itertuples()]

        # 4. Join complete task definitions and retain their setup and grading provenance.
        attempts = attempts.merge(runs[["archive", "run_prefix", "subject_key"]], on=["archive", "run_prefix"], validate="many_to_one")
        attempts = attempts.merge(tasks[["raw_item_id", "instruction", "snapshot", "task_record"]],
                                  on="raw_item_id", how="left", validate="many_to_one")
        if attempts.task_record.isna().any():
            raise ValueError("OSWorld has a native attempt without its original task definition")
        attempts["content"] = attempts["instruction.txt"].where(attempts["instruction.txt"].notna(), attempts.instruction)
        if not attempts.content.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("OSWorld requires a nonempty recorded or provider task instruction")
        # Run-specific wording takes precedence; changed stimuli remain separate items.
        items = attempts.drop_duplicates(["raw_item_id", "content"]).copy()
        items["item_key"] = range(len(items))
        items["features"] = [dict(domain=row.raw_item_id.split("/", 1)[0], snapshot=row.snapshot,
            definition_provenance=parameters["labels"]["definition_provenance"]) for row in items.itertuples()]
        items["grading_criterion"] = items.task_record.map(lambda value: {"rule": json.dumps(
            dict(interpretation=self.grading["rule"], task_definition=value), ensure_ascii=False, sort_keys=True)})
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["published_result"], sort_keys=True))
        attempts = attempts.merge(items[["raw_item_id", "content", "item_key"]],
                                  on=["raw_item_id", "content"], validate="many_to_one")

        # 5. Preserve missing grades and complete text, including malformed native JSONL or code.
        attempts["response_key"] = attempts.archive + "/" + attempts.run_prefix + attempts.raw_item_id
        attempts["trial"] = attempts.run_prefix.str.extract(paths["trial"], expand=False).fillna("1").astype(int)
        attempts["test_condition"] = [json.dumps(dict(source_archive=row.archive, run_prefix=row.run_prefix), sort_keys=True)
                                       for row in attempts.itertuples()]
        responses = attempts
        has_trace = attempts["traj.jsonl"].fillna("").ne("") | attempts["runtime.log"].fillna("").ne("")
        traces = attempts.loc[has_trace, ["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_archive=row["archive"], run_prefix=row["run_prefix"],
            task=row["raw_item_id"], native_files={kind: row[kind] for kind in parameters["trace_files"] if row[kind] is not None},
            media_capture=parameters["labels"]["media_capture"]), ensure_ascii=False, allow_nan=False)
            for row in attempts.loc[has_trace].to_dict("records")]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
                "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
                "responses": responses[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
                "traces": traces}


if __name__ == "__main__":
    OSWorld(__file__).main_from_args()
