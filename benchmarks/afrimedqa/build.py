#!/usr/bin/env python3
"""Curate the released AfriMed-QA multiple-choice results with table operations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AfriMedQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["layout"]["release"]

        # 1. Load the question banks used to identify the released runs.
        reference = pd.read_csv(root / parameters["layout"]["phase1_reference"], dtype=str, na_filter=False)
        reference["question_key"] = reference.question.astype(object).str.replace(r"\s+", " ", regex=True).str.strip()
        phase2 = pd.read_csv(root / parameters["layout"]["phase2_reference"], dtype=str, na_filter=False)
        expert_ids = set(phase2.loc[phase2.question_type.eq("mcq") & phase2.tier.eq("expert") & phase2.split.eq("test"), "sample_id"])
        medqa = pd.read_csv(root / parameters["layout"]["medqa_reference"], dtype=str, na_filter=False)
        foreign_ids = set(medqa.sample_id) | set(medqa.question)

        # 2. Concatenate native records and recover references for output-only files.
        frames = []
        for path in sorted((root / "results").glob("*/*mcq*.csv")):
            if str(path.relative_to(root)) in parameters["excluded_results"]:
                continue
            # Upstream CSVs use LF rows; a bare CR inside one answer is data.
            frame = pd.read_csv(path, dtype=str, na_filter=False, lineterminator="\n")
            frame["native_record"] = frame.to_dict("records")
            if "sample_id" not in frame:
                frame["question_key"] = frame.model_prompt.str.extract(parameters["matching"]["question_pattern"], expand=False)
                frame["question_key"] = frame.question_key.astype(object).str.replace(r"\s+", " ", regex=True).str.strip()
                frame = frame.merge(reference[["question_key", "sample_id", "answer"]], on="question_key", how="left", validate="many_to_one")
            ids = set(frame.sample_id)
            if ids <= foreign_ids:
                continue
            if ids <= set(reference.sample_id):
                bank = "phase1"
            elif ids <= set(phase2.sample_id):
                bank = "expert_phase2" if ids == expert_ids else "subset_phase2"
            else:
                raise ValueError(f"Unmatched question bank in {path.name}")
            if frame.sample_id.duplicated().any() or frame[["model_prompt", "answer"]].isna().any().any():
                raise ValueError(f"Missing or duplicate question definitions in {path.name}")
            frames.append(frame.assign(subject_key=path.parent.name, source_file=str(path.relative_to(self.raw_dir)),
                                       source_row=frame.index, test_condition="source=" + parameters["banks"][bank]))
        observations = pd.concat(frames, ignore_index=True)
        observations["correct"] = observations.correct.fillna("")
        if not observations.correct.isin(["", *parameters["grade_values"]]).all():
            raise ValueError("Expected binary grades or an explicitly missing grade")

        # 3. Keep source model labels, including fine-tuned variants and undated APIs.
        subjects = observations[["subject_key"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [{**parameters["subject_features"], "model_identifier": label}
                                for label in subjects.subject_key]

        # 4. Preserve complete prompts and their matching reference letters.
        observations["item_key"] = observations.model_prompt + "\n" + observations.answer
        items = observations.drop_duplicates("item_key").copy()
        items["content"] = items.model_prompt
        items["raw_item_id"] = items.sample_id
        items["features"] = [dict(parameters["item_features"]) for _ in items.index]
        items["grading_criterion"] = items.answer.map(lambda answer: {"reference_answer": answer, "rule": self.grading["rule"]})
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["released_accuracy"], sort_keys=True))

        # 5. Keep every attempt, null grades and full source records without truncation.
        observations["response_key"] = observations.index
        observations["response"] = observations.correct.map(parameters["grade_values"]).astype(float)
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": row.source_row,
                                      "record": row.native_record}, ensure_ascii=False, allow_nan=False)
                           for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AfriMedQA(__file__).main_from_args()
