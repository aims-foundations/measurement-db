#!/usr/bin/env python3
"""Curate Petavue's BIRD workbook without changing its recorded SQL verdicts."""

import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from xlsx2csv import Xlsx2csv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class BirdSQL(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]

        # 1. Read the Strict OOXML workbook as a table; retain unformatted cell values.
        buffer = StringIO()
        with Xlsx2csv(str(self.raw_dir / paths["results"]), skip_hidden_rows=False,
                ignore_formats=["float", "date", "time", "percentage"]) as workbook:
            workbook.convert(buffer, sheetid=int(parameters["workbook"]["sheet_id"]))
        buffer.seek(0)
        observations = pd.read_csv(buffer, dtype=str, keep_default_na=False)
        observations["record"] = observations.to_dict("records")
        observations["source_row"] = observations.index + 2  # One-based Excel row; row 1 contains headers.
        observations = observations.rename(columns=parameters["columns"])
        if observations.source_index.duplicated().any() or not observations.grade.isin(["0", "1"]).all():
            raise ValueError("Duplicated source row identifier or invalid published verdict")

        # 2. Join complete input-bank questions, schemas and hints using all three identifying fields.
        tasks = pd.read_csv(self.raw_dir / paths["tasks"], dtype=str, keep_default_na=False).rename(
            columns={"db_id": "database_id", "sql_query": "reference_answer", "question": "question_key"})
        tasks["item_key"] = tasks.index
        observations["question_key"] = observations.recorded_question.replace(parameters["question_aliases"])
        observations = observations.merge(tasks[["database_id", "question_key", "reference_answer", "item_key"]],
            on=["database_id", "question_key", "reference_answer"], how="left", validate="many_to_one")
        if observations.item_key.isna().any():
            raise ValueError("A published result has no unambiguous question/database/reference match")

        # 3. Separate model, serving environment and prompting conditions without guessing settings.
        configuration = ["model", "environment", "instruction_size", "shot_size"]
        subjects = observations[configuration].drop_duplicates().reset_index(drop=True)
        subjects["subject_key"] = subjects.index
        subjects["raw_label"] = subjects.model.map(parameters["models"])
        if subjects.raw_label.isna().any():
            raise ValueError("A published model has no documented model mapping")
        subjects["features"] = [dict(harness=parameters["harness"]["name"], source_model=row.model,
            serving_environment=row.environment, instruction_size=int(row.instruction_size),
            shot_size=int(row.shot_size), protocol_reference=parameters["harness"]["reference"])
            for row in subjects.itertuples()]
        observations = observations.merge(subjects[configuration + ["subject_key"]],
            on=configuration, how="left", validate="many_to_one")

        # 4. Keep input content separate from the reference SQL and recorded grading protocol.
        items = tasks.loc[tasks.item_key.isin(observations.item_key)].copy()
        items["raw_item_id"] = items.database_id + ":" + items.index_in_original
        items["content"] = [json.dumps(dict(question=row.question_key, database_id=row.database_id,
            schema=row.schema, evidence=row.evidence), ensure_ascii=False, sort_keys=True) for row in items.itertuples()]
        items["features"] = [dict(database_id=row.database_id, difficulty=row.difficulty,
            source_question_index=row.index_in_original) for row in items.itertuples()]
        items["grading_criterion"] = items.reference_answer.map(
            lambda answer: dict(reference_answer=answer, rule=self.grading["rule"]))
        verifier = json.dumps(self.grading["verifiers"]["recorded_execution"], sort_keys=True)
        items["verifier"] = [ExactMatcher(spec=verifier) for _ in range(len(items))]

        # 5. Preserve each complete spreadsheet record and its original observed outcome.
        observations["response_key"] = observations.source_row
        observations["response"] = observations.grade.astype(float)
        observations["trial"] = 1
        observations["test_condition"] = ("instruction_size=" + observations.instruction_size +
            ";environment=" + observations.environment + ";shot=" + observations.shot_size)
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=paths["results"], sheet=parameters["workbook"]["sheet_name"],
            source_row=row.source_row, record=row.record), ensure_ascii=False, allow_nan=False)
            for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BirdSQL(__file__).main_from_args()
