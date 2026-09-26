"""Tabulate FIND's released conversations, references and native grading records."""

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class FindInterp(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("interpretations", "functions", "code")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Pivot native text files into one row per recorded interpreter/function attempt.
        with ZipFile(self.raw_dir / parameters["paths"]["interpretations"]) as archive:
            names = sorted(name for name in archive.namelist() if name.endswith((".json", ".txt", ".py")))
            files = pd.DataFrame({"path": names, "text": [archive.read(name).decode("utf-8") for name in names]})
        coordinates = files.path.str.extract(r"^(?P<attempt>results/[^/]+/[^/]+/f\d+)/(?P<file>.+)$")
        if coordinates.isna().any().any():
            raise ValueError("Unknown interpretation archive layout")
        files = files.join(coordinates)
        attempts = files.pivot(index="attempt", columns="file", values="text").reset_index()
        attempts["history"] = attempts["history.json"].map(json.loads)
        empty = attempts.history.eq("")
        if attempts.loc[empty].drop(columns=["attempt", "history"]).fillna("").drop(columns="history.json").ne("").any().any():
            raise ValueError("An empty history has other recorded output")
        attempts = attempts.loc[~empty].copy()
        coordinates = attempts.attempt.str.extract(r"^results/(?P<model>[^/]+)/(?P<condition>[^/]+)/(?P<function>f\d+)$")
        attempts = attempts.join(coordinates)
        attempts["category"] = attempts.condition.map(parameters["categories"])
        if attempts.category.isna().any() or not attempts.model.isin(parameters["models"]).all():
            raise ValueError("Unknown interpreter or function category")

        # 2. Retain every message supplied before the first assistant reply, including hints.
        messages = pd.json_normalize(attempts[["attempt", "history"]].to_dict("records"), record_path="history", meta="attempt")
        replies = messages.role.eq("assistant").groupby(messages.attempt).cumsum()
        initial = messages.loc[replies.eq(0)].copy()
        initial["message"] = initial[["role", "content"]].to_dict("records")
        prompts = initial.groupby("attempt").message.agg(list).rename("initial_messages")
        attempts = attempts.join(prompts, on="attempt", validate="one_to_one")
        attempts["content"] = [json.dumps(dict(initial_messages=value), ensure_ascii=False) for value in attempts.initial_messages]
        attempts["reference_key"] = attempts.category + "/" + attempts.function

        # 3. Join hidden reference code and state hashes as grading material, never as the prompt.
        with ZipFile(self.raw_dir / parameters["paths"]["functions"]) as archive:
            names = sorted(name for name in archive.namelist() if "/f" in name and not name.endswith("/"))
            payloads = [archive.read(name) for name in names]
        references = pd.DataFrame({"member": names, "payload": payloads})
        references["reference_key"] = references.member.str.extract(r"^find_dataset/([^/]+/f\d+)/", expand=False)
        references["sha256"] = [hashlib.sha256(value).hexdigest() for value in payloads]
        references["source"] = references[["member", "sha256"]].to_dict("records")
        states = references.groupby("reference_key").source.agg(list).rename("reference_files")
        code = references.loc[references.member.str.endswith("/function_code.py")].copy()
        code["reference_code"] = code.payload.str.decode("utf-8")
        attempts = attempts.merge(code[["reference_key", "reference_code"]], on="reference_key", how="left", validate="many_to_one")
        attempts = attempts.join(states, on="reference_key", validate="many_to_one")
        if attempts.reference_code.isna().any():
            raise ValueError("A recorded attempt has no reference function")

        # 4. Keep MSE, description annotations and unavailable grades distinct.
        grade_files = files.loc[files.file.isin(parameters["metrics"])].copy()
        grade_files["metric"] = grade_files.file.map(parameters["metrics"])
        grades = pd.json_normalize(grade_files.text.map(json.loads).tolist()).reindex(columns=["mse", "desc_score", "respones"])
        grade_files = grade_files.reset_index(drop=True).join(grades)
        grade_files["response"] = grade_files.mse.where(grade_files.metric.eq("mse"), grade_files.desc_score)
        grade_files["source_issue"] = None
        descriptions = grade_files.metric.eq("description")
        answers = grade_files.respones.str.extract(r"\[ANSWER\]:\s*([01])\b", expand=False)
        grade_files.loc[descriptions & answers.isna(), "source_issue"] = "judge_answer_unparseable"
        grade_files.loc[descriptions & answers.notna() & pd.to_numeric(answers).ne(grade_files.response), "source_issue"] = "grade_disagrees_with_written_answer"
        base = attempts[["attempt", "category"]].copy()
        base["metric"] = base.category.eq("numeric").map({True: "mse", False: "ungraded"})
        extra = grade_files.loc[~grade_files.metric.eq("mse"), ["attempt", "metric"]]
        measurements = pd.concat([base[["attempt", "metric"]], extra], ignore_index=True)
        measurements = measurements.merge(grade_files[["attempt", "metric", "response", "source_issue"]], on=["attempt", "metric"], how="left", validate="one_to_one")
        measurements = measurements.merge(attempts, on="attempt", validate="many_to_one")
        measurements["response_key"] = measurements.attempt + ":" + measurements.metric
        measurements["subject_key"] = measurements.model
        measurements["item_key"] = measurements.response_key

        # 5. Build canonical input tables and complete, source-linked traces.
        items = measurements[["item_key", "content", "reference_key", "reference_code", "reference_files", "condition", "metric"]].copy()
        items["raw_item_id"] = items.reference_key + ":" + items.metric
        items["features"] = items[["reference_key", "condition"]].to_dict("records")
        items["grading_criterion"] = [dict(rule=self.grading["verifiers"][row.metric]["rule"],
            response_scale=self.grading["verifiers"][row.metric]["response_scale"],
            reference_answer=json.dumps(dict(code=row.reference_code, archive=parameters["paths"]["functions"], files=row.reference_files), ensure_ascii=False))
            for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"][metric]["implementation"], sort_keys=True)) for metric in items.metric]
        subjects = attempts[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key.map(parameters["models"])
        subjects["features"] = [dict(**parameters["subject_features"], interpreter=model) for model in subjects.subject_key]
        native = files.groupby("attempt").apply(lambda group: dict(zip(group.file, group.text)), include_groups=False)
        traces = measurements[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(archive=parameters["paths"]["interpretations"], attempt=row.attempt,
            metric=row.metric, files=native[row.attempt], source_issues=[row.source_issue] if isinstance(row.source_issue, str) else []),
            ensure_ascii=False, allow_nan=False) for row in measurements.itertuples()]
        return {"subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": measurements[["response_key", "subject_key", "item_key", "response"]], "traces": traces}


if __name__ == "__main__":
    FindInterp(__file__).main_from_args()
