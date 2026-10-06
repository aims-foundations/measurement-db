"""Tabulate released Lambda FP Course assessments without guessing run identity."""

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LambdaFPCourse(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("artifact")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = parameters["paths"]["prefix"]
        with zipfile.ZipFile(self.raw_dir / parameters["paths"]["archive"]) as archive:
            # 1. Load the four supported assessment tables, retaining original rows.
            parts = []
            for kind, source_file in parameters["result_files"].items():
                table = pd.read_csv(archive.open(root + source_file), dtype=str, keep_default_na=False)
                table = table.rename(columns={"Unnamed: 5": ""})
                table["source_record"] = table.to_dict("records")
                table["source_file"], table["source_row"], table["subtask"] = source_file, range(len(table)), kind
                table = table.rename(columns={"Model":"model", "Attempt":"attempt", "performance":"Rating"})
                if kind == "codegen":
                    table["source_task"] = table.hw + "/" + table.question
                    table["attempt"] = ""
                elif kind == "explain":
                    table["source_task"] = "q" + table["index"].str.zfill(2)
                else:
                    table["source_task"] = table.Filename.str.replace("_", "_buggy_code_", n=1)
                parts.append(table)
            attempts = pd.concat(parts, ignore_index=True)
            attempts["item_key"] = attempts.subtask + "/" + attempts.source_task
            attempts["subject_key"] = attempts.model.map(parameters["model_aliases"])
            attempts["response"] = attempts.Rating.map(parameters["ordinal_values"]).astype(float)
            syntax = attempts.subtask.eq("repair_syntax")
            attempts.loc[syntax, "response"] = attempts.loc[syntax, "Fixed"].map({"True":1.0, "False":0.0})
            if attempts.subject_key.isna().any() or attempts.response.isna().any():
                raise ValueError("Unrecognized released model label or assessment category")

            # 2. Read source task definitions and join their explicit grading keys.
            parts = []
            for kind, source_file in parameters["task_files"].items():
                table = pd.read_csv(archive.open(root + source_file), dtype=str, keep_default_na=False)
                table["task_record"] = table.to_dict("records")
                table["task_metadata_file"] = source_file
                if kind == "codegen":
                    table["hw"] = table.Filename.str.extract(r"exercises/(hw\d+)/")
                    table["source_task"] = table.hw + "/q" + table.Question.str.extract(r"Question ([0-9.a-z]+):")[0]
                    table["task_file"] = "benchmarks/CodeGen/" + table.hw + ".txt"
                    # hw7/q1 has two function entries; retain both without choosing one.
                    table = table.groupby(["source_task", "task_file", "task_metadata_file"], sort=False).task_record.agg(list).reset_index()
                elif kind == "explain":
                    table["source_task"] = "q" + table["index"].str.zfill(2)
                    table["task_file"] = "benchmarks/Explain/" + table.source_task + ".txt"
                else:
                    table["source_task"] = table.Filename.str.removesuffix(".ml")
                    category = "Syntax" if kind == "repair_syntax" else "Type"
                    table["task_file"] = "benchmarks/Repair/" + category + " Error/" + table.Filename
                    if kind == "repair_syntax":
                        table = table.groupby(["source_task", "task_file", "task_metadata_file"], sort=False).task_record.agg(list).reset_index()
                table["item_key"] = kind + "/" + table.source_task
                table = table.loc[table.item_key.isin(attempts.item_key)].copy()
                table["content"] = [archive.read(root + name).decode("utf-8") for name in table.task_file]
                if kind == "repair_type":
                    table["content"] = (parameters["prompt_parts"]["type_prefix"] + table.Message
                        + parameters["prompt_parts"]["code_separator"] + table.content)
                elif kind == "repair_syntax":
                    table["content"] = parameters["prompt_parts"]["syntax_prefix"] + table.content
                table["subtask"] = kind
                table["reference_answer"] = table["solution(if applicable)"] if kind == "explain" else ""
                parts.append(table[["item_key", "source_task", "subtask", "content", "reference_answer", "task_file", "task_metadata_file", "task_record"]])
            items = pd.concat(parts, ignore_index=True)
            attempts = attempts.merge(items.drop(columns=["content", "subtask", "source_task"]),
                on="item_key", how="left", validate="many_to_one")
            if attempts.task_file.isna().any():
                raise ValueError("A released assessment has no original task definition")
            explain = attempts.subtask.eq("explain")
            if any(row.exam != row.task_record["question"] for row in attempts.loc[explain].itertuples()):
                raise ValueError("An Explain index disagrees with its original exam identifier")

            # 3. Associate saved outputs only when all three native keys are present.
            outputs = pd.DataFrame({"output_file": archive.namelist()})
            outputs["output_file"] = outputs.output_file.str.removeprefix(root)
            extracted = outputs.output_file.str.extract(
                r"^benchmarks/(?:ExplainOutput/q\d+|RepairOutput/Fixed (?:Type|Syntax) Error)/"
                r"(?P<model>[^/]+)_(?P<source_task>q\d+|hw\d+_buggy_code_\d+)_answer_(?P<attempt>\d+)\.(?:txt|ml)$")
            outputs = outputs.join(extracted).dropna()
            outputs["subject_key"] = outputs.model.map(parameters["model_aliases"])
            outputs["subtask"] = "explain"
            outputs.loc[outputs.output_file.str.contains("Fixed Type Error"), "subtask"] = "repair_type"
            outputs.loc[outputs.output_file.str.contains("Fixed Syntax Error"), "subtask"] = "repair_syntax"
            outputs = outputs.loc[outputs.subject_key.notna()].copy()
            attempts = attempts.merge(outputs.drop(columns="model"),
                on=["subject_key", "source_task", "subtask", "attempt"], how="left", validate="many_to_one")
            attempts["output"] = [None if pd.isna(name) else archive.read(root + name).decode("utf-8") for name in attempts.output_file]

        # 4. Preserve grading-specific item identities and source model labels.
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=name) for name in subjects.subject_key]
        items["raw_item_id"] = items.item_key
        items["features"] = [dict(subtask=row.subtask, source_task=row.source_task,
            input_scope=parameters["labels"]["input_scope"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=row.reference_answer or None,
            rule=self.grading["verifiers"][row.subtask]["rule"].format(source_task=row.source_task),
            response_scale=self.grading["verifiers"][row.subtask]["response_scale"]) for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"][kind], sort_keys=True)) for kind in items.subtask]
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_row.astype(str)
        attempts["test_condition"] = "subtask=" + attempts.subtask
        attempts["trial"] = attempts.groupby(["subject_key", "item_key"], sort=False).cumcount() + 1
        known = attempts.attempt.ne("")
        attempts.loc[known, "trial"] = pd.to_numeric(attempts.loc[known, "attempt"], errors="raise")

        # 5. Keep complete source records, task provenance and associated outputs.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            source_record=row.source_record, task_file=row.task_file, task_metadata_file=row.task_metadata_file,
            task_record=row.task_record, output_file=None if pd.isna(row.output_file) else row.output_file,
            output=None if pd.isna(row.output) else row.output, attempt_identity="not_recorded" if row.attempt == "" else "source_attempt"),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LambdaFPCourse(__file__).main_from_args()
