"""Tabulate LiveBench's published question-level grading events."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LiveBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]

        # 1. Concatenate original exports, keeping every native record and location.
        frames = []
        for path in sorted(self.raw_dir.glob(paths["judgments"]), reverse=True):
            records = pq.read_table(path).to_pylist()
            frame = pd.DataFrame.from_records(records)
            frame["source_record"] = [dict(file=str(path.relative_to(self.raw_dir)), row=index, record=record)
                                      for index, record in enumerate(records)]
            if "task" not in frame and "grouping" in frame:
                frame = frame.rename(columns={"category": "task", "grouping": "category"})
            frames.append(frame)
        judgments = pd.concat(frames, ignore_index=True)
        event = ["question_id", "model", "turn", "tstamp"]
        if judgments[event + ["score"]].isna().any().any() or not judgments.score.between(0, 1).all():
            raise ValueError("A published event has a missing identity or an invalid grade")
        grouped = judgments.groupby(event, sort=False, dropna=False)
        if not grouped.score.nunique().eq(1).all():
            raise ValueError("Exports disagree about the score of the same grading event")
        provenance = grouped.source_record.agg(list).rename("source_records").reset_index()
        attempts = judgments.drop_duplicates(event).merge(provenance, on=event, validate="one_to_one")

        # 2. Select the latest author definition, preferring test to train at the same revision.
        frames = []
        question_paths = sorted(self.raw_dir.glob(paths["questions"]),
            key=lambda path: (path.parts[-3], path.name.startswith("test-")), reverse=True)
        for path in question_paths:
            records = pq.read_table(path).to_pylist()
            frame = pd.DataFrame.from_records(records)
            frame["question_source"] = [dict(file=str(path.relative_to(self.raw_dir)), row=index,
                record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                    default=str, allow_nan=False).encode()).hexdigest()) for index, record in enumerate(records)]
            if "task" not in frame and "grouping" in frame:
                frame = frame.rename(columns={"category": "task", "grouping": "category"})
            frames.append(frame[["question_id", "turns", "ground_truth", "question_source"]]
                          if "ground_truth" in frame else frame[["question_id", "turns", "question_source"]])
        questions = pd.concat(frames, ignore_index=True).drop_duplicates("question_id")
        attempts = attempts.merge(questions, on="question_id", how="inner", validate="many_to_one")
        items = questions[questions.question_id.isin(attempts.question_id)].copy()
        if not items.turns.map(len).eq(1).all() or not items.turns.str[0].str.strip().ne("").all():
            raise ValueError("Expected a nonempty single-turn question")

        # 3. Preserve literal model labels and the complete source grading definitions.
        subjects = attempts[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=model)
                                for model in subjects.subject_key]
        items["item_key"] = items["raw_item_id"] = items.question_id
        items["content"] = items.turns.str[0]
        items["grading_criterion"] = [dict(
            reference_answer=None if pd.isna(row.ground_truth) else json.dumps(row.ground_truth, ensure_ascii=False),
            rule=json.dumps(dict(rule=self.grading["rule"], question_source=row.question_source), sort_keys=True))
            for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["reported"], sort_keys=True))
                             for _ in range(len(items))]

        # 4. A timestamp identifies a grading event, not a new independent model generation.
        attempts["subject_key"] = attempts.model
        attempts["item_key"] = attempts.question_id
        attempts["response_key"] = [json.dumps(values) for values in attempts[event].itertuples(index=False, name=None)]
        attempts["response"] = attempts.score
        attempts["trial"] = 1
        attempts["test_condition"] = [json.dumps(dict(category=row.category, task=row.task,
            turn=row.turn, grading_timestamp=row.tstamp), sort_keys=True) for row in attempts.itertuples()]
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_records=row.source_records, question_source=row.question_source,
            answer_association=parameters["trace"]["answer_association"]), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LiveBench(__file__).main_from_args()
