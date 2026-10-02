#!/usr/bin/env python3
"""Curate released instruction decisions without executing agents or rejudging safety."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SafeAgentBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("tasks", "results", "provenance")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Load official tasks in the released export's order; keep complete definitions.
        frames = []
        for category, filename in parameters["task_files"].items():
            path = self.raw_dir / parameters["layout"]["tasks"] / filename
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            frame = pd.json_normalize(records, max_level=0)
            if category == "abstract":
                frame["instruction"] = frame.instruction.str[0]
            frames.append(frame.assign(task_record=records, task_category=category,
                                       task_source_row=frame.index,
                                       task_source_file=str(path.relative_to(self.raw_dir))))
        tasks = pd.concat(frames, ignore_index=True).rename(columns={"instruction": "task_instruction"})
        tasks["task_id"] = "t" + tasks.index.astype(str)

        # 2. Read each native export and translate only its boolean decision field.
        frames = []
        for configuration, filename in parameters["results"].items():
            path = self.raw_dir / parameters["layout"]["results"] / filename
            records = json.loads(path.read_text())
            frame = pd.json_normalize(records, max_level=0)
            decision = frame[parameters["decision_fields"][configuration]]
            if not decision.map(type).eq(bool).all():
                raise ValueError("A released decision is missing or not boolean")
            frames.append(frame.assign(native_record=records, subject_key=configuration,
                                       response=decision.astype(float), source_row=frame.index,
                                       source_file=str(path.relative_to(self.raw_dir))))
        observations = pd.concat(frames, ignore_index=True).merge(
            tasks[["task_id", "task_instruction", "task_category", "task_record", "task_source_file", "task_source_row"]],
            on="task_id", how="left", validate="many_to_one")
        if (observations.instruction.ne(observations.task_instruction).any()
                or observations.category.map(parameters["category_aliases"]).ne(observations.task_category).any()):
            raise ValueError("An exported task does not match the official task bank")

        # 3. Distinguish system configurations, including related threshold/post-hoc variants.
        subjects = pd.DataFrame(parameters["subjects"].items(), columns=["subject_key", "raw_label"])
        subjects["features"] = subjects.subject_key.map(lambda key: parameters["subject_" + key])

        # 4. Deduplicate exact instructions; keep refusal as an unordered behavior category.
        observations["item_key"] = observations.instruction.map(
            lambda text: json.dumps({"instruction": text}, ensure_ascii=True, sort_keys=True))
        items = observations.drop_duplicates("item_key")[["item_key", "task_id"]].rename(columns={"task_id": "raw_item_id"})
        items["content"] = items.item_key
        items["grading_criterion"] = [{"rule": self.grading["rule"]} for _ in items.index]
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["native_decision"], sort_keys=True))

        # 5. Preserve every recorded task attempt and both complete source records.
        observations["response_key"] = observations.index
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": row.source_row,
                                      "record": row.native_record, "task_source_file": row.task_source_file,
                                      "task_source_row": row.task_source_row, "task_record": row.task_record},
                                     ensure_ascii=False, allow_nan=False)
                           for row in observations.itertuples()]
        return {
            "subjects": subjects,
            "items": items,
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    SafeAgentBench(__file__).main_from_args()
