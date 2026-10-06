"""Tabulate every released LawBench prediction using its recorded input."""

import importlib
import importlib.metadata
import json
import multiprocessing
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class LawBench(BenchmarkBuild):
    def _check_python_version(self):
        required = self.build_parameters["runtime"]["python_version"]
        current = ".".join(map(str, sys.version_info[:3]))
        if current != required:
            raise ValueError(f"LawBench requires Python {required}; found {current}. "
                "The native information-extraction scorer depends on Python's float summation.")

    def download(self):
        self._check_python_version()
        return self.fetch_sources("upstream")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        self._check_python_version()
        parameters = self.build_parameters
        prefix = parameters["paths"]["prefix"]
        archive_path = self.raw_dir / parameters["paths"]["archive"]

        # 1. Load unchanged, reviewed native metrics directly from the pinned ZIP.
        # Workers use a fixed hash seed for stable floating-point summation order.
        for package, version in parameters["packages"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Install the declared scorer dependency: {package}=={version}")
        grader_path = str(archive_path) + "/" + prefix + parameters["paths"]["grader_directory"]
        sys.path.insert(0, grader_path)
        previous_environment = {name: os.environ.get(name) for name in ["PYTHONHASHSEED", "TMPDIR"]}
        parts = []
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".native-grading-", dir=self.tables_dir) as scratch:
                os.environ.update(PYTHONHASHSEED=parameters["runtime"]["python_hash_seed"], TMPDIR=scratch)
                with zipfile.ZipFile(archive_path) as archive, ProcessPoolExecutor(
                    max_workers=int(parameters["runtime"]["workers"]),
                    mp_context=multiprocessing.get_context("spawn")) as workers:

                    # 2. Read complete records, then apply the native scalar score.
                    files = sorted(name for name in archive.namelist()
                        if name.startswith(prefix + "predictions/") and name.endswith(".json"))
                    for number, name in enumerate(files, 1):
                        relative = name.removeprefix(prefix)
                        _, setting, model, filename = relative.split("/")
                        task = Path(filename).stem
                        frame = pd.DataFrame.from_dict(json.loads(archive.read(name)), orient="index")
                        if set(frame.columns) != {"origin_prompt", "prediction", "refr"}:
                            raise ValueError(f"Unexpected prediction fields in {relative}")
                        frame["source_record"] = frame.to_dict("records")
                        frame["source_key"] = frame.index.astype(str)
                        frame["source_file"], frame["setting"], frame["subject_key"], frame["task"] = relative, setting, model, task
                        frame["content"] = frame.origin_prompt
                        messages = frame.origin_prompt.map(lambda value: isinstance(value, list))
                        for value in frame.loc[messages, "origin_prompt"]:
                            if len(value) != 1 or value[0].get("role") != "HUMAN" or set(value[0]) != {"role", "prompt"}:
                                raise ValueError("Unexpected saved message structure")
                        frame.loc[messages, "content"] = frame.loc[messages, "origin_prompt"].map(lambda value: value[0]["prompt"])
                        frame["grade_status"] = "reconstructed_native_score"
                        frame["response"] = pd.Series(float("nan"), index=frame.index, dtype="float64")
                        if task in parameters["corpus_tasks"]:
                            frame["grade_status"] = parameters["corpus_tasks"][task]
                        else:
                            pattern = parameters["excluded_references"].get(task)
                            if pattern is not None:
                                frame.loc[frame.refr.str.contains(pattern, regex=True), "grade_status"] = "upstream_excludes_reference"
                            eligible = frame.grade_status.eq("reconstructed_native_score")
                            module, function = parameters["native_functions"][task].split(".")
                            grader = getattr(importlib.import_module("evaluation_functions." + module), function)
                            results = workers.map(grader, ([row] for row in frame.loc[eligible, "source_record"]), chunksize=64)
                            frame.loc[eligible, "response"] = [result["score"] for result in results]
                        parts.append(frame)
                        if number % 100 == 0 or number == len(files):
                            print(f"Read and graded {number}/{len(files)} released files", flush=True)
        finally:
            sys.path.remove(grader_path)
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        attempts = pd.concat(parts, ignore_index=True)

        # 3. Retain source model labels without guessing historical configurations.
        subjects = attempts[["subject_key"]].drop_duplicates().copy()
        subjects["raw_label"] = parameters["labels"]["subject_prefix"] + subjects.subject_key
        subjects["features"] = [dict(**parameters["subject_features"], source_model_label=name) for name in subjects.subject_key]

        # 4. Distinct recorded prompts, references and grading conditions are items.
        keys = ["setting", "task", "content", "refr"]
        items = attempts[keys + ["source_file", "source_key"]].drop_duplicates(subset=keys).reset_index(drop=True)
        items["item_key"] = items.index.astype(str)
        items["raw_item_id"] = items.source_file + "#" + items.source_key
        items["features"] = [dict(task=row.task, recorded_prompt_setting=row.setting,
            input_scope=parameters["labels"]["input_scope"]) for row in items.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=json.dumps(row.refr, ensure_ascii=False),
            rule=self.grading["verifiers"][row.task]["rule"].format(task=row.task, setting=row.setting),
            response_scale=self.grading["verifiers"][row.task]["response_scale"]) for row in items.itertuples()]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"][task], sort_keys=True)) for task in items.task]
        attempts = attempts.merge(items[keys + ["item_key"]], on=keys, how="left", validate="many_to_one")
        attempts["response_key"] = attempts.source_file + "#" + attempts.source_key
        attempts["test_condition"] = "task=" + attempts.task + ";prompt_setting=" + attempts.setting

        # 5. Preserve every original prompt, prediction, reference and source key.
        # Shared registration assigns occurrence trials after resolving item IDs.
        traces = attempts[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=row.source_file, source_key=row.source_key,
            source_record=row.source_record, grade_status=row.grade_status), ensure_ascii=False, allow_nan=False)
            for row in attempts.itertuples()]
        return {"subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": attempts[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces}


if __name__ == "__main__":
    LawBench(__file__).main_from_args()
