#!/usr/bin/env python3
"""Curate BigFinanceBench's recorded agent runs and independent judge assessments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class BigFinanceBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]
        grading = self.grading["verifiers"]
        release = self.raw_dir / paths["release"]

        # 1. Read the native tables, retaining complete records and their locations.
        tables = {}
        for name, pattern in self.build_parameters["tables"].items():
            frames = []
            for path in sorted(release.glob(pattern)):
                frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False, precise_float=True)
                frame = frame.astype(object).where(frame.notna(), None)
                frame["record"] = frame.to_dict("records")
                frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
            tables[name] = pd.concat(frames, ignore_index=True)
        tasks, runs, grades = tables["tasks"], tables["runs"], tables["grades"]
        keys = ["model", "question_id", "trial_idx"]
        if tasks.id.duplicated().any() or grades.duplicated(keys + ["judge"]).any():
            raise ValueError("Duplicate native question or judge assessment")

        # 2. Associate assessments with all matching traces; keep ambiguous retries.
        runs["content"] = [json.dumps(dict(system_prompt=row.system_prompt, question=row.question,
            tool_specs=row.tool_specs), ensure_ascii=False, sort_keys=True) for row in runs.itertuples()]
        runs["source_record"] = runs[["source_file", "source_row", "record"]].to_dict("records")
        if runs.groupby(keys).content.nunique().gt(1).any():
            raise ValueError("Retried trial has conflicting initial inputs")
        grouped = runs.groupby(keys, sort=False, as_index=False).agg(
            content=("content", "first"), run_records=("source_record", list))
        observations = grades.rename(columns={"record": "grade_record"}).merge(
            grouped, on=keys, how="left", validate="many_to_one")
        if observations.run_records.isna().any():
            raise ValueError("A published assessment has no released trace")
        observations["matching_run_indices"] = [[index for index, run in enumerate(records)
            if run["record"]["final_answer"] == answer]
            for records, answer in zip(observations.run_records, observations.final_answer)]
        if observations.matching_run_indices.str.len().eq(0).any():
            raise ValueError("No trace agrees with the final answer recorded by a judge")
        observations["trace_association"] = observations.matching_run_indices.str.len().eq(1).map(
            {True: "unique_final_answer_match", False: "ambiguous_final_answer_match"})
        observations = observations.merge(tasks[["id", "record"]].rename(
            columns={"id": "question_id", "record": "task_record"}), on="question_id", how="left", validate="many_to_one")
        if observations.task_record.isna().any() or not observations.reference_answer.eq(
                observations.task_record.str["reference_answer"]).all():
            raise ValueError("The item bank and published reference answers disagree")

        # 3. Preserve recorded model identities without guessing missing inference settings.
        subjects = runs.loc[runs.resolved_model.notna(), ["model", "resolved_model", "harness_version"]].drop_duplicates()
        if subjects.model.duplicated().any() or not set(runs.model).issubset(subjects.model):
            raise ValueError("A model's recorded harness configuration is absent or inconsistent")
        subjects["subject_key"] = subjects.model
        subjects["raw_label"] = subjects.model
        subjects["features"] = [dict(harness="BigFinanceBench", model_identifier=row.model,
            resolved_model=row.resolved_model, harness_version=row.harness_version) for row in subjects.itertuples()]

        # 4. Unpivot the two released measures, with distinct judge/metric protocols.
        if not observations.final_answer_correct.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("Final-answer correctness must be a recorded Boolean")
        if not observations.rubric_points_possible.gt(0).all():
            raise ValueError("Rubric fractions require a positive point denominator")
        observations["final_answer_correct"] = observations.final_answer_correct.astype(float)
        observations["rubric_fraction"] = observations.rubric_points_earned / observations.rubric_points_possible
        measures = list(grading["measures"])
        observations = observations.melt(id_vars=[column for column in observations.columns if column not in measures],
            value_vars=measures, var_name="metric", value_name="response")
        observations["item_key"] = (observations.question_id + ":" + observations.judge + ":" +
            observations.metric + ":" + observations.content)
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.question_id
        items["features"] = [dict(source_question_id=row.question_id, evaluation_only=row.task_record["evaluation_only"],
            do_not_train=row.task_record["do_not_train"], benchmark_canary=row.task_record["benchmark_canary"])
            for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.task_record["reference_answer"],
            rule=json.dumps(dict(metric=row.metric, rubric=row.task_record["rubric"],
                interpretation=grading["measures"][row.metric]["rule"]), ensure_ascii=False, sort_keys=True),
            response_scale=grading["measures"][row.metric]["scale"]) for row in items.itertuples()]
        items["verifier"] = [Judge(judge=row.judge, judged_by="llm",
            spec=json.dumps(grading["protocol"], ensure_ascii=False, sort_keys=True)) for row in items.itertuples()]

        # 5. Retain full outputs, grading explanations, source flags and retry evidence.
        observations["response_key"] = observations.source_file + ":" + observations.source_row.astype(str) + ":" + observations.metric
        observations["subject_key"] = observations.model
        observations["trial"] = observations.trial_idx.astype(int) + 1
        observations["test_condition"] = "judge=" + observations.judge + ";metric=" + observations.metric
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            metric=row.metric, grade_record=row.grade_record, task_record=row.task_record, run_records=row.run_records,
            matching_run_indices=row.matching_run_indices, trace_association=row.trace_association),
            ensure_ascii=False, allow_nan=False) for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BigFinanceBench(__file__).main_from_args()
