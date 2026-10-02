#!/usr/bin/env python3
"""Tabulate a dated SWE-Together export without merging revised score snapshots."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class SweTogether(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("website_metrics", "website_index", "website_tasks", "website_app", "task_bank", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        config = self.build_parameters
        paths = config["paths"]
        labels = config["labels"]

        # 1. Load task definitions and flatten the native task -> model -> trials table.
        tasks = pd.read_json(self.raw_dir / paths["tasks"], lines=True, dtype=False, convert_dates=False)
        tasks = tasks.astype(object).where(tasks.notna(), None)
        text = (self.raw_dir / paths["metrics"]).read_text()
        payload = json.JSONDecoder().raw_decode(text.split("window.METRICS = ", 1)[1])[0]
        cells = pd.DataFrame.from_dict({task: record["models"] for task, record in payload.items()}, orient="index")
        cells = cells.rename_axis("source_task").reset_index().melt(
            id_vars="source_task", var_name="subject_key", value_name="source_cell").dropna(subset=["source_cell"])
        cells["scores"] = cells.source_cell.map(lambda record: record["trials"])
        records = cells.explode("scores", ignore_index=True)
        records["trial"] = records.groupby(["source_task", "subject_key"], sort=False).cumcount() + 1
        records = records.loc[records.scores.notna()].copy()
        valid = records.scores.map(lambda score: type(score) in (int, float) and 0 <= score <= 1)
        if not valid.all():
            raise ValueError("SWE-Together requires finite published scores in [0, 1]")

        # 2. Join by task ID, with reviewed mappings for three shortened website IDs.
        records["item_key"] = records.source_task.replace(config["task_aliases"])
        tasks = tasks.rename(columns={"task_id": "item_key"})
        if set(records.item_key) != set(tasks.item_key):
            raise ValueError("SWE-Together task bank and scored task IDs differ")
        records = records.merge(tasks, on="item_key", how="left", validate="many_to_one", indicator=True)
        if not records["_merge"].eq("both").all() or tasks.instruction.isna().any():
            raise ValueError("A published score is missing its complete task definition")

        # 3. Keep literal model labels, complete instructions, context and grading criteria.
        subjects = records[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(harness="opencode", recorded_model_label=model,
            result_snapshot=labels["snapshot"], historical_inference_settings="not_recorded")
            for model in subjects.subject_key]
        items = tasks.copy()
        headers = items[["repo", "base_commit", "language"]].apply(
            lambda row: "\n".join(f"{label}: {value}" for label, value in
                zip(["Repository", "Base commit", "Language"], row) if value), axis=1)
        items["raw_item_id"] = items.item_key
        items["content"] = headers.where(headers.eq(""), headers + "\n\n") + items.instruction
        items["features"] = items[["repo", "repo_url", "base_commit", "language", "difficulty", "category",
                                    "tags", "docker_image", "allow_internet", "agent_timeout_sec"]].to_dict("records")
        criteria = items[["completeness_goals", "oracle_intents", "fail_to_pass", "pass_to_pass",
                          "test_manifest", "test_cmd", "log_parser"]].to_dict("records")
        items["grading_criterion"] = [dict(reference_answer=patch if patch else None,
            rule=json.dumps(dict(pass_rule=self.grading["rule"], task_grading=criterion), ensure_ascii=False, sort_keys=True))
            for patch, criterion in zip(items.reference_patch, criteria)]
        items["verifier"] = Judge(
            spec=json.dumps(self.grading["verifiers"]["published_judge"], sort_keys=True), judged_by="llm")

        # 4. Apply the documented threshold while preserving original replicate positions.
        records["response_key"] = records.source_task + "/" + records.subject_key + "/" + records.trial.astype(str)
        responses = records[["response_key", "subject_key", "item_key", "trial"]].copy()
        responses["response"] = records.scores.ge(self.grading["verifiers"]["published_judge"]["pass_threshold"]).astype(float)
        responses["test_condition"] = labels["condition"]

        # 5. These are complete published assessment records, not agent conversations.
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(
            record_type="published_metrics_cell", source_file=paths["metrics"], snapshot=labels["snapshot"],
            source_task=row.source_task, source_model=row.subject_key, source_trial=row.trial,
            source_score=row.scores, source_cell=row.source_cell, generated_output_available=False),
            ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    SweTogether(__file__).main_from_args()
