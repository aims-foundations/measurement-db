"""Tabulate LINGOLY's complete recorded prompts and native part-level answers."""

import ast
import io
import json
import re
import sys
import unicodedata as ud
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LingOly(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("upstream")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = parameters["paths"]
        password = parameters["archive_settings"]["password"].encode()

        # 1. Read the native question bank, predictions and scalar scorer.
        with zipfile.ZipFile(self.raw_dir / paths["archive"]) as archive:
            with zipfile.ZipFile(io.BytesIO(archive.read(paths["prefix"] + paths["questions"]))) as bank:
                sheets = pd.read_json(io.BytesIO(bank.read(
                    parameters["archive_settings"]["question_member"], pwd=password)),
                    lines=True, dtype=False, convert_dates=False)
            with zipfile.ZipFile(io.BytesIO(archive.read(paths["prefix"] + paths["responses"]))) as results:
                frames = {name: pd.read_json(io.BytesIO(results.read(name, pwd=password)), dtype=False, convert_dates=False)
                          for name in sorted(results.namelist()) if name.endswith(".json")}
            path = paths["prefix"] + paths["scorer"]
            functions = [node for node in ast.parse(archive.read(path)).body
                         if isinstance(node, ast.FunctionDef) and node.name in parameters["native_functions"].values()]
            if {node.name for node in functions} != set(parameters["native_functions"].values()):
                raise ValueError("Missing a declared native grading function")
            native = dict(ast=ast, re=re, ud=ud)
            exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), native)
        for duplicate, original in parameters["duplicate_results"].items():
            fields = ["questions", "overall_question_n", "model_answers"]
            if not frames[duplicate][fields].equals(frames[original][fields]):
                raise ValueError("The declared temporary export is no longer a duplicate")
            del frames[duplicate]
        if not all(name.endswith(("_lingoly.json", "_lingoly_nocontext.json")) for name in frames):
            raise ValueError("An unrecognized result file needs review")

        # 2. Flatten the bank to (sheet, question position, part) with exact references.
        sheets["questions"] = sheets.questions.map(json.loads)
        questions = sheets[["overall_question_n", "questions"]].explode("questions", ignore_index=True)
        questions["question_position"] = questions.groupby("overall_question_n", sort=False).cumcount()
        questions = questions.drop(columns="questions").join(pd.json_normalize(questions.questions, max_level=0))
        parts = questions.explode("subprompts", ignore_index=True)
        parts["reference_record"] = parts.subprompts
        parts = parts.drop(columns="subprompts").join(pd.json_normalize(parts.subprompts, max_level=0))
        bank_keys = ["overall_question_n", "question_position", "questionpart_n"]

        # 3. Expand parsed answers, then join the author's current grading reference.
        batches = []
        for name, frame in frames.items():
            suffix = "_lingoly_nocontext.json" if name.endswith("_nocontext.json") else "_lingoly.json"
            frame["source_record"] = frame.to_dict("records")
            batches.append(frame.assign(source_file=name, source_key=frame.index.astype(str),
                subject_key=name.removesuffix(suffix), condition="none" if name.endswith("_nocontext.json") else "full"))
        attempts = pd.concat(batches, ignore_index=True)
        attempts["question_position"] = attempts.groupby(["source_file", "overall_question_n"], sort=False).cumcount()
        attempts["questionpart_n"] = attempts.model_answers.map(list)
        attempts = attempts.explode("questionpart_n", ignore_index=True)
        attempts["model_answer"] = [answers[part] for answers, part in zip(attempts.model_answers, attempts.questionpart_n)]
        attempts = attempts.merge(parts[bank_keys + ["question_n", "answer", "reference_record"]],
                                  on=bank_keys, how="left", validate="many_to_one", indicator=True)
        if not attempts._merge.eq("both").all():
            raise ValueError("An answer cannot be associated with a question-bank part")
        predictions = attempts.model_answer.map(lambda value: ", ".join(value) if isinstance(value, list) else value)
        attempts["response"] = [float(native[parameters["native_functions"]["grading"]](
            prediction, reference, native[parameters["native_functions"]["comparison"]], None))
            for prediction, reference in zip(predictions, attempts.answer)]

        # 4. A stimulus is the complete observed prompt, with its selected part and condition.
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=model)
                                for model in subjects.subject_key]
        keys = ["questions", "questionpart_n", "answer", "condition"]
        items = attempts.drop_duplicates(keys).reset_index(drop=True)
        items["item_key"] = items.index.astype(str)
        items["raw_item_id"] = items.source_file + "#" + items.source_key + ":" + items.questionpart_n
        items["content"] = items.questions
        items["features"] = [dict(context=row.condition, target_part=json.dumps(row.questionpart_n, ensure_ascii=False))
                             for row in items.itertuples()]
        verifier = self.grading["verifiers"]["exact_match"]
        items["grading_criterion"] = [dict(reference_answer=json.dumps(row.answer, ensure_ascii=False),
            rule=self.grading["rule"].format(part=row.questionpart_n, condition=row.condition))
            for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(verifier, sort_keys=True)) for _ in range(len(items))]
        attempts = attempts.merge(items[keys + ["item_key"]], on=keys, how="left", validate="many_to_one")
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_key + ":" + attempts.questionpart_n
        attempts["test_condition"] = "context=" + attempts.condition

        # 5. Retain every original field, parsed answer and reference-bank annotation.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_key=row.source_key,
            target_part=row.questionpart_n, source_record=row.source_record, reference_record=row.reference_record),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LingOly(__file__).main_from_args()
