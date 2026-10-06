#!/usr/bin/env python3
"""Tabulate recorded BEHAVIOR goal-interpretation answers and missing grades."""

import json
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class EmbodiedAgentInterface(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "outputs", "dataset_notice")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]
        labels = self.build_parameters["labels"]

        # 1. Load the native task dictionaries and archived answer lists as tables.
        prompts = pd.read_json(self.raw_dir / paths["prompts"], typ="series", dtype=False)
        prompts = prompts.rename_axis("identifier").reset_index(name="content")
        conditions = pd.read_json(self.raw_dir / paths["conditions"], orient="index", dtype=False)
        conditions = conditions.rename_axis("identifier").reset_index()
        with ZipFile(self.raw_dir / paths["archive"]) as archive:
            members = sorted(name for name in archive.namelist()
                             if fnmatchcase(name, paths["members"]))
            if not members or len(members) != len(set(members)):
                raise ValueError("EAI requires unique native model-output files")
            frames = []
            for member in members:
                with archive.open(member) as stream:
                    frame = pd.read_json(stream, dtype=False)
                if set(frame) != {"identifier", "llm_output"}:
                    raise ValueError("EAI answer export has unexpected fields; review its grading status")
                frame = frame.astype(object).where(frame.notna(), None)
                frames.append(frame.assign(
                    source_record=frame.to_dict("records"),
                    source_member=member,
                    source_row=frame.index + 1,
                    subject_key=Path(member).name.removesuffix("_outputs.json"),
                ))
        answers = pd.concat(frames, ignore_index=True)
        if answers.identifier.isna().any() or answers.duplicated(["subject_key", "identifier"]).any():
            raise ValueError("EAI requires one identified record per released model and task")

        # 2. Join by task identity, retaining every prompt and every recorded answer.
        tasks = prompts.merge(conditions, on="identifier", how="outer", validate="one_to_one", indicator=True)
        if not tasks["_merge"].eq("both").all() or tasks.content.isna().any() or tasks.goal_conditions.isna().any():
            raise ValueError("EAI prompts and grading references do not match")
        tasks = tasks.drop(columns="_merge")
        records = answers.merge(tasks, on="identifier", how="left", validate="many_to_one", indicator=True)
        if not records["_merge"].eq("both").all():
            raise ValueError("EAI answer has no matching task and grading reference")

        # 3. Keep literal model labels and full prompts; describe the intended grader.
        subjects = answers[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = labels["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(harness=self.name, recorded_model_label=model,
            historical_inference_settings=labels["settings_status"]) for model in subjects.subject_key]
        items = tasks.rename(columns={"identifier": "item_key"}).copy()
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(simulator=labels["simulator"], module=labels["module"])
                             for _ in items.index]
        items["grading_criterion"] = [dict(
            reference_answer=json.dumps(goals, ensure_ascii=False, sort_keys=True),
            rule=self.grading["rule"]) for goals in items.goal_conditions]
        items["verifier"] = Judge(
            spec=json.dumps(self.grading["verifiers"]["official_pipeline"], sort_keys=True))

        # 4. No historical grades are present. A recorded attempt remains an attempt.
        records["response_key"] = records.source_member + ":" + records.source_row.astype(str)
        responses = records.rename(columns={"identifier": "item_key"})[
            ["response_key", "subject_key", "item_key"]].copy()
        responses["response"] = pd.Series(None, index=responses.index, dtype="Float64")
        responses["trial"] = 1
        responses["test_condition"] = labels["condition"]

        # 5. Retain unmodified answers and their source positions, including invalid code.
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(
            source_archive=paths["archive"], source_member=row.source_member, source_row=row.source_row,
            source_record=row.source_record,
            source_conditions=dict(initial_conditions=row.initial_conditions, goal_conditions=row.goal_conditions),
            grade_status=labels["grade_status"]), ensure_ascii=False, allow_nan=False)
            for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    EmbodiedAgentInterface(__file__).main_from_args()
