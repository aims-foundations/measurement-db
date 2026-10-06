"""Tabulate LexEval's released predictions with its native deterministic metrics."""

import ast
import importlib.metadata
import io
import json
import multiprocessing
import re
import string
import sys
import tempfile
import zipfile
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jieba
import pandas as pd
from rouge import Rouge

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LexEval(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("upstream")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        prefix = parameters["paths"]["prefix"]
        for package, version in parameters["packages"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Install the declared scorer dependency: {package}=={version}")

        # 1. Load the original scalar parsers and every complete prediction table.
        # Do not import generation code or initialize a model from the archive.
        with zipfile.ZipFile(self.raw_dir / parameters["paths"]["archive"]) as archive:
            path = prefix + parameters["paths"]["process_code"]
            tree = ast.parse(archive.read(path))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in parameters["native_functions"].values()]
            if {node.name for node in functions} != set(parameters["native_functions"].values()):
                raise ValueError("Missing a declared native scoring function")
            native = {"re": re, "string": string, "OrderedDict": OrderedDict}
            exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), native)
            parts = []
            files = sorted(name for name in archive.namelist()
                           if name.startswith(prefix + "model_output/") and name.endswith(".jsonl"))
            for name in files:
                relative = name.removeprefix(prefix)
                _, setting, model, filename = relative.split("/")
                frame = pd.read_json(io.BytesIO(archive.read(name)), lines=True, dtype=False, convert_dates=False)
                if set(frame) != {"input", "output", "answer"} or not frame.map(lambda value: isinstance(value, str)).all().all():
                    raise ValueError(f"Unexpected released prediction fields in {relative}")
                frame["source_record"] = frame.to_dict("records")
                parts.append(frame.assign(source_file=relative, source_key=frame.index.astype(str),
                    setting=setting, subject_key=model, task="_".join(filename.removesuffix(".jsonl").split("_")[-2:])))
        attempts = pd.concat(parts, ignore_index=True).rename(columns={"input": "content"})
        if not attempts.content.str.strip().ne("").all():
            raise ValueError("A released prediction has no usable input")
        attempts["metric"] = attempts.task.str.startswith("5_").map({False: "accuracy", True: "rouge_l"})

        # 2. Grade distinct output/reference pairs, then join scores back to every
        # attempt. Repetition changes observation counts, not deterministic grades.
        score_keys = ["metric", "output", "answer"]
        scores = attempts[score_keys].drop_duplicates().reset_index(drop=True)
        scores["response"] = 0.0
        scores["grade_status"] = "reconstructed_native_score"
        multiple_choice = scores.metric.eq("accuracy")
        parsed = scores.loc[multiple_choice, "output"].map(native[parameters["native_functions"]["extraction"]])
        scores.loc[multiple_choice, "response"] = parsed.eq(scores.loc[multiple_choice, "answer"]).astype(float)
        generation = scores.metric.eq("rouge_l")
        text = pd.concat([scores.loc[generation, "output"], scores.loc[generation, "answer"]]).drop_duplicates()
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".lexeval-tokenizer-", dir=self.tables_dir) as scratch:
            tokenizer = jieba.Tokenizer()
            tokenizer.tmp_dir = scratch
            tokens = {value: " ".join(tokenizer.cut(native[parameters["native_functions"]["normalization"]](value), cut_all=False))
                      for value in text}
            pairs = zip(scores.loc[generation, "output"].map(tokens), scores.loc[generation, "answer"].map(tokens))
            with ProcessPoolExecutor(max_workers=int(parameters["runtime"]["workers"]),
                                     mp_context=multiprocessing.get_context("spawn")) as workers:
                grades = list(workers.map(self.grade_generation, pairs, chunksize=32))
        scores.loc[generation, ["response", "grade_status"]] = pd.DataFrame(grades).to_numpy()
        attempts = attempts.merge(scores, on=score_keys, how="left", validate="many_to_one")

        # 3. Retain literal model labels without guessing historical settings.
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=model)
                                for model in subjects.subject_key]

        # 4. Preserve recorded input suffixes and distinguish task/setting/grade.
        keys = ["setting", "task", "content", "answer", "metric"]
        items = attempts[keys + ["source_file", "source_key"]].drop_duplicates(subset=keys).reset_index(drop=True)
        items["item_key"] = items.index.astype(str)
        items["raw_item_id"] = items.source_file + "#" + items.source_key
        items["features"] = [dict(task=row.task, recorded_prompt_setting=row.setting,
            input_scope=parameters["labels"]["input_scope"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=json.dumps(row.answer, ensure_ascii=False),
            rule=self.grading["verifiers"][row.metric]["rule"].format(task=row.task, setting=row.setting),
            response_scale=self.grading["verifiers"][row.metric]["response_scale"]) for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"][metric], sort_keys=True)) for metric in items.metric]
        attempts = attempts.merge(items[keys + ["item_key"]], on=keys, how="left", validate="many_to_one")
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_key
        attempts["test_condition"] = "task=" + attempts.task + ";prompt_setting=" + attempts.setting

        # 5. Keep lossless source records, including empty outputs and failures.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_key=row.source_key,
            source_record=row.source_record, grade_status=row.grade_status), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}

    @staticmethod
    def grade_generation(pair):
        """Isolate the native scalar metric and its explicit zero-on-error rule."""
        try:
            value = Rouge().get_scores([pair[0]], [pair[1]], avg=True)["rouge-l"]["f"]
            return value, "reconstructed_native_score"
        except Exception as error:
            return 0.0, "native_rouge_fallback_" + type(error).__name__


if __name__ == "__main__":
    LexEval(__file__).main_from_args()
