#!/usr/bin/env python3
"""Tabulate published FrontierOR task-level feasibility without rerunning solvers."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class FrontierOR(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("task_index", "website_index", "website_history", "website_metrics",
                                  "website_notice", "website_task_view", "results", "task_bank")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        config = self.build_parameters
        paths, labels = config["paths"], config["labels"]

        # 1. Load the task index, published metric records and full task artifacts.
        tasks = pd.read_json(self.raw_dir / paths["index"], dtype=False, convert_dates=False)
        website = pd.read_csv(self.raw_dir / paths["website_index"], dtype=str)
        if tasks.paper_id.duplicated().any() or set(tasks.paper_id) != set(website.paper_id):
            raise ValueError("FrontierOR task definitions and website coverage differ")
        details = pd.DataFrame([json.loads((self.raw_dir / paths["details"].format(paper_id=task)).read_text())
                                for task in tasks.paper_id])
        if not details.paper_id.equals(tasks.paper_id):
            raise ValueError("FrontierOR details do not match their source filenames")
        for field in ["description", "instance_schema", "solution_schema", "checker"]:
            tasks[field] = tasks.paper_id.map(lambda task: (self.raw_dir / paths[field].format(paper_id=task)).read_text())

        # 2. Flatten per-model assessments and join them by the original task key.
        cells = details[["paper_id", "per_model"]].explode("per_model", ignore_index=True)
        records = pd.json_normalize(cells.per_model.tolist(), max_level=0).assign(
            paper_id=cells.paper_id.to_numpy(), source_record=cells.per_model.to_numpy())
        if records.duplicated(["paper_id", "model"]).any() or not records.kind.isin(["model", "self_evolve"]).all():
            raise ValueError("FrontierOR requires unique published task/configuration assessments")
        records = records.merge(tasks[["paper_id"]], on="paper_id", how="left", validate="many_to_one", indicator=True)
        if not records["_merge"].eq("both").all():
            raise ValueError("A FrontierOR assessment has no task definition")
        valid = records.source_record.map(lambda row: row["feasibility"] is None or (
            type(row["feasibility"]) in (int, float) and row["feasibility"] in self.INFO["response_scale"]["values"]))
        if not valid.all():
            raise ValueError("FrontierOR requires the original five-instance feasibility fraction")

        # 3. Preserve literal model/configuration labels and separate evolution harnesses.
        subjects = records[["model", "kind"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["harness"] = subjects.subject_key.map(config["evolution_harnesses"])
        subjects.loc[subjects.kind.eq("model"), "harness"] = labels["one_shot_harness"]
        if subjects.subject_key.duplicated().any() or subjects.harness.isna().any():
            raise ValueError("An unrecognized FrontierOR subject configuration needs review")
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(harness=row.harness, recorded_model_label=row.subject_key,
            evaluation_kind=row.kind, result_snapshot=labels["snapshot"], historical_inference_settings="not_recorded")
            for row in subjects.itertuples()]

        # 4. Keep complete problem text and input/output schemas; describe the grader.
        items = tasks.rename(columns={"paper_id": "item_key"}).copy()
        items["raw_item_id"] = items.item_key
        items["content"] = (items.description + "\n\n# Instance Data Schema (instance_schema.json)\n\n" +
                            items.instance_schema + "\n\n# Solution Output Schema (solution_schema.json)\n\n" + items.solution_schema)
        items["features"] = [dict(task_artifact_revision=labels["task_revision"]) for _ in items.index]
        items["grading_criterion"] = [dict(rule=self.grading["rule"]) for _ in items.index]
        items["verifier"] = [Judge(spec=json.dumps(dict(
            protocol=self.grading["verifiers"]["published_feasibility"], reference_checker=checker), ensure_ascii=False))
            for checker in items.checker]

        # 5. Retain published aggregate grades and their full metric records, without rounding.
        records["response_key"] = records.paper_id + "/" + records.model
        responses = records.rename(columns={"paper_id": "item_key", "model": "subject_key", "feasibility": "response"})[
            ["response_key", "subject_key", "item_key", "response"]].copy()
        responses["trial"] = 1
        responses["test_condition"] = labels["condition"]
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(record_type="published_task_metrics", source_file=paths["details"].format(paper_id=row.paper_id),
            source_task=row.paper_id, source_record=row.source_record, snapshot=labels["snapshot"],
            generated_output_available=False, constituent_instance_results_available=False), ensure_ascii=False, allow_nan=False)
            for row in records.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
                "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
                "responses": responses, "traces": traces}


if __name__ == "__main__":
    FrontierOR(__file__).main_from_args()
