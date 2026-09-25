#!/usr/bin/env python3
"""Curate the released ComplexBench generations without inventing missing grades."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class ComplexBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read task definitions and generation exports directly into tables.
        tasks = pd.read_json(self.raw_dir / parameters["paths"]["tasks"])
        tasks["source_task"] = tasks.to_dict("records")
        files = sorted(self.raw_dir.glob(parameters["paths"]["generations"]))
        if not files:
            raise ValueError("ComplexBench has no generation exports")
        generations = pd.concat([
            pd.read_json(path, lines=True).assign(
                source_file=str(path.relative_to(self.raw_dir)),
                source_line=lambda frame: frame.index + 1)
            for path in files
        ], ignore_index=True)
        if tasks.main_id.duplicated().any() or generations.duplicated(["model", "main_id"]).any():
            raise ValueError("ComplexBench task or model/task identifiers are not unique")
        if tasks.instruction.isna().any() or generations.model.isna().any():
            raise ValueError("ComplexBench requires task text and a recorded model label")

        # 2. Join on native IDs and verify the actual instruction in every export.
        # Do not rely on file order, silently drop unmatched rows, or grade nulls.
        records = generations.merge(tasks, on="main_id", how="left",
            validate="many_to_one", indicator=True, suffixes=("", "_task"))
        if not records["_merge"].eq("both").all() or not records.instruction.eq(records.instruction_task).all():
            raise ValueError("ComplexBench generation instructions do not match the task bank")
        if not tasks.main_id.isin(records.main_id).all():
            raise ValueError("ComplexBench contains tasks without released generation records")

        # 3. Retain the complete rubric separately from the instruction presented
        # to the model. The hybrid grader is described, never run by this builder.
        subjects = records[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(harness=self.name, recorded_model_label=model,
            historical_inference_settings="not_recorded") for model in subjects.subject_key]
        items = tasks.rename(columns={"main_id": "item_key", "instruction": "content"}).copy()
        items["raw_item_id"] = items.item_key.astype(str)
        items["features"] = [dict(task_type=row.task_types, category=row.category) for row in items.itertuples()]
        items["grading_criterion"] = [{"rule": json.dumps(dict(
            protocol=self.grading["rule"], scoring_questions=questions),
            ensure_ascii=False, sort_keys=True)} for questions in items.scoring_questions]
        items["verifier"] = Judge(spec=json.dumps(self.grading["verifiers"]["official_pipeline"], sort_keys=True), judged_by="llm")

        # 4. Each row remains one model/instruction record. Historical judgments
        # were not released: an all-null float column is deliberate, not failure.
        records["response_key"] = records.source_file + ":" + records.source_line.astype(str)
        responses = records.rename(columns={"model": "subject_key", "main_id": "item_key"})[
            ["response_key", "subject_key", "item_key"]].copy()
        responses["response"] = pd.Series(None, index=responses.index, dtype="Float64")
        responses["trial"] = 1
        responses["test_condition"] = parameters["labels"]["condition"]

        # 5. Preserve every native field, full answer, rubric and missing output.
        native = records[["main_id", "model", "instruction", "generated"]].astype(object)
        records["source_record"] = native.where(native.notna(), None).to_dict("records")
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_line=row.source_line,
            source_record=row.source_record, source_task=row.source_task),
            ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    ComplexBench(__file__).main_from_args()
