#!/usr/bin/env python3
"""Curate native ClashEval answers and the inputs for each answering condition."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ClashEval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the six released result tables, preserving file/row provenance.
        parameters = self.build_parameters
        native = pd.concat([
            pd.read_parquet(self.raw_dir / f"{model}.pqt").assign(
                subject_key=model, source_file=f"{model}.pqt", source_row=lambda frame: frame.index)
            for model in parameters["models"]
        ], ignore_index=True)
        native = native.astype(object).where(native.notna(), None)

        # 2. Each closed-book answer is reused across perturbations, not regenerated.
        # The author's prior-accuracy calculation likewise keeps one row per question.
        prior = native.drop_duplicates(["subject_key", "question"]).rename(columns={
            "prior_response": "output", "prior_correct": "response", "prior_logprobs": "logprobs",
        }).assign(condition="prior", context=None, mod_type=None)
        post = native.rename(columns={
            "post_response": "output", "post_correct": "response", "post_logprobs": "logprobs",
            "context_mod": "context",
        }).assign(condition="post")
        columns = ["subject_key", "question", "dataset", "answer_original", "condition", "context",
                   "mod_type", "output", "response", "logprobs", "source_file", "source_row"]
        observations = pd.concat([prior[columns], post[columns]], ignore_index=True)
        observations["response_key"] = observations.index

        # 3. Reconstruct the released system/user templates for each distinct input.
        identity = ["condition", "dataset", "question", "context", "answer_original", "mod_type"]
        items = observations.drop_duplicates(identity).copy()
        items["item_key"] = range(len(items))
        items["raw_item_id"] = items.condition + ":" + items.source_row.astype(str)
        templates = pd.concat([
            pd.Series(parameters[f"{condition}_prompts"], name="system").rename_axis("dataset").reset_index().assign(
                condition=condition, template=parameters["user_templates"][condition])
            for condition in ["prior", "post"]
        ], ignore_index=True)
        items = items.merge(templates, on=["condition", "dataset"], how="left", validate="many_to_one")
        if items.system.isna().any():
            raise ValueError("No released prompt template for an observed dataset/condition")
        items["user"] = [row["template"].format(question=row["question"], context=row["context"])
                         for row in items[["template", "question", "context"]].to_dict("records")]
        items["content"] = [json.dumps(row, ensure_ascii=False, allow_nan=False)
                            for row in items[["system", "user"]].to_dict("records")]
        items["features"] = items[["condition", "dataset", "mod_type"]].to_dict("records")

        # 4. Keep the true reference answer and the meaning of the native grade.
        items["grading_criterion"] = [
            {"reference_answer": answer, "rule": self.grading["rule"]}
            for answer in items.answer_original
        ]
        items["verifier"] = ExactMatcher(spec=json.dumps(
            self.grading["verifiers"]["published_correctness"], sort_keys=True))
        subjects = pd.DataFrame.from_dict(parameters["models"], orient="index", columns=["raw_label"])
        subjects = subjects.rename_axis("subject_key").reset_index()

        # 5. Join each observation to its input and retain unstripped output/log probabilities.
        responses = observations.merge(items[identity + ["item_key"]], on=identity,
                                       how="left", validate="many_to_one")
        traces = responses[["response_key"]].copy()
        traces["trace"] = [
            json.dumps(row, ensure_ascii=False, allow_nan=False)
            for row in responses[["output", "logprobs", "source_file", "source_row"]].to_dict("records")
        ]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ClashEval(__file__).main_from_args()
