#!/usr/bin/env python3
"""Curate Alpha-SQL's released queries and derive read-only execution grades."""

import io
import json
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AlphaSQL(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        layout = self.build_parameters["layout"]
        grading = self.grading["verifiers"]["execution"]

        # 1. Join each released SQL prediction to its original BIRD question.
        with ZipFile(self.raw_dir / layout["tasks"]) as archive:
            questions = pd.read_json(io.BytesIO(archive.read(layout["questions"])))
            schemas = pd.read_json(io.BytesIO(archive.read(layout["schemas"])))
            database_archive = archive.read(layout["databases"])
        predictions = pd.Series(json.loads((self.raw_dir / layout["predictions"]).read_text()),
                                name="prediction").rename_axis("raw_item_id").reset_index()
        questions["raw_item_id"] = questions.question_id.astype(str)
        questions["native_record"] = questions.drop(columns="raw_item_id").to_dict("records")
        responses = predictions.merge(questions, on="raw_item_id", how="outer", validate="one_to_one",
                                      indicator=True)
        if not responses["_merge"].eq("both").all():
            raise ValueError("Every released prediction must match exactly one original question")
        responses = responses.sort_values("question_id").reset_index(drop=True)
        responses["response_key"] = responses.raw_item_id
        responses["item_key"] = responses.raw_item_id

        # 2. Retain the actual databases, column descriptions, and schema records.
        schemas["schema_record"] = schemas.to_dict("records")
        with ZipFile(io.BytesIO(database_archive)) as archive:
            files = pd.DataFrame({"path": [name for name in archive.namelist() if not name.endswith("/")]})
            files["db_id"] = files.path.str.extract(layout["database_pattern"], expand=False)
            files = files.loc[files.db_id.isin(questions.db_id) & files.path.str.endswith((".sqlite", ".csv"))].copy()
            files["data"] = files.path.map(archive.read)
        files["media_type"] = files.path.str.endswith(".sqlite").map(
            {True: "application/vnd.sqlite3", False: "text/csv"})
        files["attachment"] = [{"data": row.data, "path": row.path,
                                 "media_type": row.media_type, "role": "input"} for row in files.itertuples()]
        attachments = files.groupby("db_id").attachment.agg(list)
        databases = files.loc[files.path.str.endswith(".sqlite")].set_index("db_id")
        if not databases.index.is_unique:
            raise ValueError("Each database must have exactly one original SQLite file")
        responses = responses.merge(schemas[["db_id", "schema_record"]], on="db_id", how="left",
                                      validate="many_to_one")
        if responses.schema_record.isna().any() or not set(questions.db_id) <= set(databases.index):
            raise ValueError("A question is missing its original database or schema")

        # 3. Execute distinct queries read-only, then compare their complete row sets.
        # SQL execution is the one non-tabular step; no model or agent is run.
        queries = responses[["db_id", "SQL", "prediction"]].melt(
            id_vars="db_id", value_name="query").drop_duplicates(["db_id", "query"])
        results = []
        with tempfile.TemporaryDirectory(prefix=".alpha-sql-", dir=self.dir) as temporary:
            for row in databases.itertuples():
                (Path(temporary) / (row.Index + ".sqlite")).write_bytes(row.data)
            for row in queries.itertuples():
                database = Path(temporary) / (row.db_id + ".sqlite")
                connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
                allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                           sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
                connection.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
                timer = threading.Timer(float(grading["timeout_per_query_seconds"]), connection.interrupt)
                timer.start()
                try:
                    rows = set(connection.execute(row.query).fetchall())
                    status, error = "ok", None
                except sqlite3.Error as exception:
                    rows, error = None, str(exception)
                    status = "timeout" if exception.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT else "error"
                finally:
                    timer.cancel()
                    connection.close()
                results.append({"db_id": row.db_id, "query": row.query, "rows": rows,
                                "status": status, "error": error})
        results = pd.DataFrame(results)
        responses = responses.merge(results.rename(columns={"query": "SQL", "rows": "gold_rows",
            "status": "gold_status", "error": "gold_error"}), on=["db_id", "SQL"], validate="many_to_one")
        responses = responses.merge(results.rename(columns={"query": "prediction", "rows": "predicted_rows",
            "status": "prediction_status", "error": "prediction_error"}),
            on=["db_id", "prediction"], validate="many_to_one")
        responses["response"] = [None if row.gold_status != "ok" else
            float(row.prediction_status == "ok" and row.predicted_rows == row.gold_rows) for row in responses.itertuples()]

        # 4. Project one system configuration, complete items, and source-linked traces.
        configuration = yaml.safe_load((self.raw_dir / layout["configuration"]).read_text())
        model = configuration["mcts_model_kwargs"]
        subjects = pd.DataFrame([{"subject_key": model["model"], "raw_label": model["model"],
            "features": {**self.build_parameters["subject"],
                         **{key: str(value) for key, value in model.items() if key not in ("model", "temperature")},
                         **{key: str(configuration[key]) for key in ("max_rollout_steps", "max_depth", "exploration_constant")}}}])
        responses["subject_key"] = model["model"]
        responses["test_condition"] = "temperature=" + str(model["temperature"])
        items = responses[["item_key", "raw_item_id", "db_id", "question", "evidence", "SQL", "difficulty", "schema_record"]].copy()
        items["attachments"] = items.db_id.map(attachments)
        items["content"] = [json.dumps({"question": row.question, "evidence": row.evidence,
            "database_id": row.db_id, "schema": row.schema_record,
            "input_files": [entry["path"] for entry in row.attachments]}, ensure_ascii=False)
            for row in items.itertuples()]
        items["features"] = [{"database_id": row.db_id, "difficulty": row.difficulty} for row in items.itertuples()]
        items["grading_criterion"] = [{"reference_answer": sql, "rule": self.grading["rule"]} for sql in items.SQL]
        items["verifier"] = ExactMatcher(spec=json.dumps(grading, sort_keys=True))
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": layout["predictions"], "source_key": row.raw_item_id,
            "prediction": row.prediction, "native_question": row.native_record, "configuration": configuration,
            "derived_execution": {"sqlite_version": sqlite3.sqlite_version,
                "gold_status": row.gold_status, "gold_error": row.gold_error if pd.notna(row.gold_error) else None,
                "prediction_status": row.prediction_status,
                "prediction_error": row.prediction_error if pd.notna(row.prediction_error) else None,
                "gold_distinct_rows": None if row.gold_rows is None else len(row.gold_rows),
                "prediction_distinct_rows": None if row.predicted_rows is None else len(row.predicted_rows)}},
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "attachments", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AlphaSQL(__file__).main_from_args()
