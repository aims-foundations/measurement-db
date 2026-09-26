"""Tabulate published HealthAdminBench runs under their two released grading rules."""

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class HealthAdminBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "website")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]

        # 1. Read the released score tables and task JSONs without changing source cells.
        with tarfile.open(self.raw_dir / paths["archive"]) as archive:
            root = paths["root"] + "/"
            runs = pd.read_csv(archive.extractfile(root + paths["runs"]), dtype=str, keep_default_na=False)
            runs["source_record"] = runs.to_dict("records")
            runs["source_row"] = runs.index
            removed = pd.read_csv(archive.extractfile(root + paths["removed"]), dtype=str, keep_default_na=False)
            removed["source_record"] = removed.to_dict("records")
            files = [member for member in archive if member.isfile() and member.name.startswith(root + paths["tasks"]) and member.name.endswith(".json")]
            records = [json.load(archive.extractfile(member)) for member in files]
            tasks = pd.json_normalize(records, max_level=0)
            tasks["task_record"] = records
            tasks["task_path"] = [member.name.removeprefix(root + paths["tasks"]) for member in files]
        tasks["domain"] = tasks.task_path.str.split("/").str[0]
        tasks = tasks.rename(columns={"id": "task_id"})
        removed = removed.groupby("task_path", sort=False).agg(removed_checks=("source_record", list)).reset_index()
        tasks = tasks.merge(removed, on="task_path", how="left", validate="one_to_one")
        tasks["removed_checks"] = tasks.removed_checks.map(lambda value: value if isinstance(value, list) else [])

        # 2. Join the website's evaluation records by the complete run name.
        payload = json.loads((self.raw_dir / paths["website"]).read_text())
        website = pd.json_normalize(payload["data"], max_level=0).explode("results", ignore_index=True)
        website = website.rename(columns={"results": "website_record"})
        website["run_name"] = website.website_record.str["run_name"]
        if runs.run_name.duplicated().any() or website.run_name.duplicated().any() or set(runs.run_name) != set(website.run_name):
            raise ValueError("Source score and website records must have matching unique run names")
        runs = runs.merge(website[["run_name", "agent_name", "agent_provider", "website_record"]], on="run_name", how="left", validate="one_to_one")
        runs = runs.merge(tasks[["domain", "task_id", "task_path", "task_record", "removed_checks"]], on=["domain", "task_id"], how="left", validate="many_to_one")
        if runs.task_record.isna().any() or not runs.model.eq(runs.agent_name).all():
            raise ValueError("A run lacks a task definition or has conflicting model identity")

        # 3. Keep original and revised judgments separate; they grade the same run.
        judgments = runs.melt(id_vars=[column for column in runs if column not in self.build_parameters["protocols"]],
            value_vars=list(self.build_parameters["protocols"]), var_name="source_field", value_name="response")
        judgments["response"] = pd.to_numeric(judgments.response, errors="raise")
        if not judgments.response.isin([0, 1]).all():
            raise ValueError("Source task-pass judgments must be explicit binary values")
        judgments["protocol"] = judgments.source_field.map(self.build_parameters["protocols"])
        judgments["subject_key"] = judgments.model
        judgments["item_key"] = judgments.task_path + ":" + judgments.prompt_type + ":" + judgments.protocol
        judgments["response_key"] = judgments.run_name + ":" + judgments.source_field
        judgments["test_condition"] = "obs=" + judgments.input_type + ";prompt=" + judgments.prompt_type

        # 4. Build task inputs and grading definitions without exposing expected outcomes.
        items = judgments.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.domain + "/" + items.task_id
        items["selected_evals"] = [row.task_record["evals"] if row.protocol == "strict_all" else
            [value for index, value in enumerate(row.task_record["evals"]) if index not in {int(flag["eval_idx"]) for flag in row.removed_checks}]
            for row in items.itertuples()]
        items["content"] = [json.dumps(dict(multimedia_elements=[
            dict(content_type="text/plain", text=row.task_record["goal"]),
            dict(content_type="text/plain", text=self.build_parameters["presentation"]["start"].format(configuration=json.dumps(
                dict(website=row.task_record["website"], config=row.task_record["config"]), ensure_ascii=False)))]
            + ([dict(content_type="text/plain", text=self.build_parameters["presentation"]["guide"].format(
                steps="\n".join(row.task_record["metadata"]["step_by_step"])))] if row.prompt_type == "task_specific" else [])),
            ensure_ascii=False) for row in items.itertuples()]
        items["features"] = [dict(domain=row.domain, difficulty=row.task_record["difficulty"],
            input_scope=self.build_parameters["presentation"]["input_scope"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(rule=self.grading["rule"] + "\n" + json.dumps(dict(protocol=row.protocol, evals=row.selected_evals), ensure_ascii=False)) for row in items.itertuples()]
        items["verifier"] = [Judge(judged_by="llm", spec=json.dumps(dict(aggregation=self.grading["verifiers"][row.protocol],
            task_evaluator=self.grading["verifiers"]["task_evaluator"]), sort_keys=True))
            if any(value["type"] == "llm_judge" for value in row.selected_evals) else ExactMatcher(spec=json.dumps(dict(
                aggregation=self.grading["verifiers"][row.protocol], task_evaluator=self.grading["verifiers"]["task_evaluator"]), sort_keys=True))
            for row in items.itertuples()]

        # 5. Preserve model keys, complete evaluator strings and both source seed fields.
        subjects = runs[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key.map(self.build_parameters["model_labels"])
        if subjects.raw_label.isna().any():
            raise ValueError("An observed model lacks its declared source label")
        subjects["features"] = [dict(source_model=key, **self.build_parameters["model_features"]) for key in subjects.subject_key]
        traces = judgments[["response_key", "source_row", "source_field", "source_record", "website_record", "task_path", "task_record", "removed_checks"]].copy()
        traces["source_file"] = paths["runs"]
        traces["trace"] = [json.dumps(record, ensure_ascii=False, allow_nan=False) for record in traces.drop(columns="response_key").to_dict("records")]
        return {"subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": judgments[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces[["response_key", "trace"]]}


if __name__ == "__main__":
    HealthAdminBench(__file__).main()
