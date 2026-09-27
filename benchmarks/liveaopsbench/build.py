"""Tabulate the published LiveAoPSBench 2024 questions and native scores."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LiveAoPSBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]

        # 1. Read the native result table and both author question representations.
        released = json.loads((self.raw_dir / paths["results"]).read_text())
        attempts = pd.json_normalize(released["performances"], max_level=0)
        attempts["source_record"] = released["performances"]
        questions = pd.read_json(self.raw_dir / paths["questions"], lines=True, dtype=False, convert_dates=False)
        questions["question_record"] = questions.to_dict("records")
        original = pd.read_json(self.raw_dir / paths["source_questions"], lines=True, dtype=False, convert_dates=False)
        if not questions[["question", "solution"]].equals(original[["question", "solution"]]):
            raise ValueError("The author question exports disagree in content or order")
        if not questions.idx.eq(questions.index + 1).all():
            raise ValueError("The published task IDs are no longer one-based row positions")
        questions["source_question_record"] = original.to_dict("records")
        questions["question_id"] = questions.index.astype(str)

        # 2. Join zero-based result IDs to the matching published task positions.
        attempts = attempts.merge(questions, on="question_id", how="left", validate="many_to_one", indicator=True)
        if not attempts._merge.eq("both").all() or attempts.duplicated(["model", "question_id"]).any():
            raise ValueError("A result has an absent or duplicated model/task association")
        months = pd.to_datetime(attempts.date, unit="ms", utc=True).dt.strftime("%Y-%m")
        if not months.eq(pd.to_datetime(attempts.post_time, format="%Y-%m").dt.strftime("%Y-%m")).all():
            raise ValueError("Result and question publication months disagree")
        if not attempts["pass@1"].isin([0, 100]).all():
            raise ValueError("The released score is not binary pass@1")
        attempts["response"] = attempts["pass@1"].div(100)
        attempts["subject_key"] = attempts.model
        attempts["response_key"] = attempts.model + "#" + attempts.question_id
        attempts["test_condition"] = parameters["labels"]["condition"]

        # 3. Keep literal model labels and distinct question/reference definitions.
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=model)
                                for model in subjects.subject_key]
        keys = ["question", "answer"]
        items = questions.drop_duplicates(keys).copy()
        items["item_key"] = items.question_id
        items["raw_item_id"] = items.idx.astype(str)
        items["content"] = items.question
        items["grading_criterion"] = [dict(reference_answer=json.dumps(answer, ensure_ascii=False),
            rule=self.grading["rule"]) for answer in items.answer]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["reported"], sort_keys=True))
                             for _ in range(len(items))]
        attempts = attempts.merge(items[keys + ["item_key"]], on=keys, how="left", validate="many_to_one")

        # 4. Preserve each released result and both complete task source records.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_record=row.source_record,
            question_record=row.question_record, source_question_record=row.source_question_record),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LiveAoPSBench(__file__).main_from_args()
