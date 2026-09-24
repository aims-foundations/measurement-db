#!/usr/bin/env python3
"""Curate the published Aider C++ study's native attempts and conversations."""

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Aider(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        paths = {name: self.raw_dir / value for name, value in parameters["layout"].items()}

        # 1. Read the published table and native logs directly from the archive.
        with ZipFile(paths["archive"]) as archive:
            prefix = parameters["archive"]["root"] + "/"
            published = pd.read_csv(archive.open(prefix + "experiments_data/all_functional_tests.csv"),
                                    dtype=str, keep_default_na=False)
            published["published_record"] = published.to_dict("records")
            published = published[["build_id", "testcase", "experiment", "model", "edit_format", "pass2",
                                   "published_record"]].rename(columns={"model": "reported_model",
                                                                         "edit_format": "reported_format"})
            members = sorted(name for name in archive.namelist() if name.endswith("/.aider.results.json"))
            records = [json.loads(archive.read(name)) for name in members]
            native = pd.json_normalize(records, max_level=0).assign(native_record=records,
                        source_member=[name.removeprefix(prefix) for name in members])
            directories = [name.rsplit("/", 1)[0] + "/" for name in members]
            native["history"] = [archive.read(directory + ".aider.chat.history.md").decode("utf-8")
                                 for directory in directories]
            configs = [json.loads(archive.read(directory + ".meta/config.json")) for directory in directories]
            native["final_files"] = [{name: archive.read(directory + name).decode("utf-8")
                                      for name in config["files"]["solution"]}
                                     for directory, config in zip(directories, configs)]

        # 2. Reconcile every published result with its native model and verdict.
        native["run"] = native.source_member.str.split("/").str[1]
        native["build_id"] = native.run.str.split("--", n=1).str[1]
        observations = native.merge(published, on=["build_id", "testcase"], how="outer",
                                    validate="one_to_one", indicator=True)
        if not observations._merge.eq("both").all():
            raise ValueError("Aider published rows and native result files do not match")
        valid = observations.tests_outcomes.map(
            lambda values: isinstance(values, list) and 1 <= len(values) <= 2
            and all(isinstance(value, bool) for value in values))
        if not valid.all():
            raise ValueError("Aider results require one or two explicit boolean test outcomes")
        observations["response"] = observations.tests_outcomes.str[-1].astype(float)
        if (not observations.model.eq(observations.reported_model).all()
                or not observations.edit_format.eq(observations.reported_format).all()
                or not observations.response.eq(observations.pass2.map({"True": 1.0, "False": 0.0})).all()):
            raise ValueError("Aider published grades or configurations disagree with native results")

        # 3. Keep prompting variants and recorded agent settings as distinct subjects.
        first_prompt = observations.history.str.extract(r"(?m)(^####[^\n]*(?:\n####[^\n]*)*)", expand=False)
        observations["additional_instructions_md"] = first_prompt.str.split(
            parameters["parsing"]["instruction_boundary"], n=1, regex=False).str[1]
        if observations.additional_instructions_md.isna().any():
            raise ValueError("Aider logs are missing the recorded initial instruction boundary")
        observations["prompt_variant"] = observations.additional_instructions_md.map(
            lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
        for column, label in [("harness_version", "Aider"), ("weak_model", "Weak model:")]:
            observations[column] = observations.history.str.extract(r"(?m)^> " + label + r" ([^\n]+)", expand=False).str.strip()
        observations["model_banner"] = observations.history.str.extract(
            r"(?m)^> (?:Main model|Model): ([^\n]+)", expand=False).str.strip()
        subject_columns = ["model", "experiment", "prompt_variant", "edit_format", "weak_model",
                           "model_banner", "harness_version", "commit_hash", "reasoning_effort", "thinking_tokens"]
        subjects = observations[subject_columns].drop_duplicates().reset_index(drop=True)
        subjects["subject_key"] = subjects.index
        subjects["raw_label"] = subjects.model
        subjects["features"] = [{"harness": "Aider", "model_identifier": row.model,
                                  "prompting_condition": row.experiment, "prompt_variant": row.prompt_variant,
                                  "edit_format": row.edit_format, "weak_model": row.weak_model,
                                  "model_banner": row.model_banner,
                                  "harness_version": row.harness_version, "harness_commit": row.commit_hash,
                                  "reasoning_effort": row.reasoning_effort, "thinking_tokens": row.thinking_tokens}
                                 for row in subjects.astype(object).where(subjects.notna(), None).itertuples()]
        observations = observations.merge(subjects[subject_columns + ["subject_key"]], on=subject_columns,
                                           how="left", validate="many_to_one")
        observations = observations.sort_values("source_member", kind="stable").reset_index(drop=True)

        # 4. Load complete instructions, initial files, reference code and tests.
        definitions = []
        for task in sorted(observations.testcase.unique()):
            directory = paths["tasks"] / "cpp/exercises/practice" / task
            config = json.loads((directory / ".meta/config.json").read_text())
            instructions = "".join((directory / ".docs" / name).read_text()
                                    for name in ["introduction.md", "instructions.md", "instructions.append.md"]
                                    if (directory / ".docs" / name).exists())
            initial_files = {name: (directory / name).read_text() for name in config["files"]["solution"]}
            references = {name: (directory / name).read_text() for name in config["files"].get("example", [])}
            tests = {name: (directory / name).read_text() for name in config["files"]["test"]}
            definitions.append({"item_key": task, "raw_item_id": "cpp/" + task,
                "content": json.dumps({"instructions": instructions, "initial_files": initial_files}, ensure_ascii=False),
                "grading_criterion": {"rule": self.grading["rule"], "reference_answer": json.dumps(references, ensure_ascii=False)},
                "verifier": ExactMatcher(spec=json.dumps({**self.grading["verifiers"]["functional_tests"],
                    "test_files": tests, "cmake": (directory / "CMakeLists.txt").read_text(),
                    "postbuild": (paths["tasks"] / "cpp/cpptest-postbuild.cmake").read_text()}, ensure_ascii=False)),
                "features": {"lang": "cpp"}})
        items = pd.DataFrame(definitions)

        # 5. Project linked tables; each benchmark run remains one observation.
        observations = observations.rename(columns={"testcase": "item_key", "source_member": "response_key"})
        trace_records = observations[["response_key", "native_record", "published_record", "history",
                                       "additional_instructions_md", "final_files"]].rename(
                                           columns={"response_key": "source_member"}).to_dict("records")
        traces = pd.DataFrame({"response_key": observations.response_key,
            "trace": [json.dumps({"source_archive": parameters["layout"]["archive"], **record},
                                 ensure_ascii=False, allow_nan=False) for record in trace_records]})
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items,
            "responses": observations[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    Aider(__file__).main_from_args()
