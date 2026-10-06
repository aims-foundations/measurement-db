#!/usr/bin/env python3
"""Reshape EVOUNA's human judgments and corresponding model answers."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class EVOUNA(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("results")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read each released question table, preserving its source row number.
        frames = []
        for filename, dataset in self.build_parameters["datasets"].items():
            frame = pd.read_json(self.raw_dir / filename, dtype=False, convert_dates=False)
            frames.append(frame.assign(raw_item_id=dataset + "-" + frame.index.astype(str), dataset=dataset))
        questions = pd.concat(frames, ignore_index=True)
        questions["question"] = questions.question.fillna("").str.strip()
        questions = questions.loc[questions.question.ne("")].copy()
        questions["item_key"] = questions.raw_item_id
        questions["test_condition"] = (
            "dataset=" + questions.dataset + ";improper=" + questions.improper.astype(bool).astype(int).astype(str)
        )

        # 2. Give the question bank its reference answers and human grading protocol.
        judge_spec = json.dumps(self.grading["verifiers"]["human_annotation"], sort_keys=True)
        references = questions.golden_answer.map(lambda value: str(value) if value is not None else None)
        items = questions.assign(
            content=questions.question,
            grading_criterion=references.map(lambda answer: {"reference_answer": answer} if answer else {"rule": self.grading["rule"]}),
            verifier=Judge(spec=judge_spec, judged_by="human"),
        )

        # 3. Expand model columns into observed judgments and attach the matching answer.
        models = self.build_parameters["models"]
        labels = questions.melt(id_vars=["item_key", "test_condition"],
                                value_vars=["judge_" + model for model in models],
                                var_name="model", value_name="response")
        labels["subject_key"] = labels.model.str.removeprefix("judge_")
        # The source uses both null and the literal string "nan" for missing labels.
        labels["response"] = labels.response.replace("nan", None)
        if not labels.response.dropna().isin([False, True, 0, 1]).all():
            raise ValueError("EVOUNA judgments must be binary or absent")
        answers = questions.melt(id_vars="item_key", value_vars=["answer_" + model for model in models],
                                 var_name="model", value_name="trace")
        answers["subject_key"] = answers.model.str.removeprefix("answer_")
        responses = labels.merge(answers[["item_key", "subject_key", "trace"]],
                                on=["item_key", "subject_key"], how="left", validate="one_to_one")
        has_trace = responses.trace.map(lambda value: isinstance(value, str) and bool(value.strip()))
        responses = responses.loc[responses.response.notna() | has_trace].copy()
        responses = responses.assign(response_key=responses.index, response=responses.response.astype(float))
        subjects = pd.DataFrame({"subject_key": list(models), "raw_label": list(models.values())})
        has_trace = has_trace.loc[responses.index]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": responses.loc[has_trace, ["response_key", "trace"]],
        }


if __name__ == "__main__":
    EVOUNA(__file__).main_from_args()
