"""Tabulate the released MMedBench rationale examples without inventing grades."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MMedBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters["layout"]

        # 1. Load the recorded requests, outputs, adapted cases and original items.
        attempts = pd.read_csv(self.raw_dir / layout["results"], dtype=str, keep_default_na=False)
        attempts["source_record"] = attempts.to_dict("records")
        attempts["source_row"] = range(len(attempts))
        adapted, original = [], []
        with ZipFile(self.raw_dir / layout["original_bank"]) as archive:
            for task_id, language in parameters["languages"].items():
                paths = list((self.raw_dir / layout["adapted_bank"]).glob(task_id + "_*.jsonl"))
                if len(paths) != 1:
                    raise ValueError("Expected one original adapted task file per language")
                # Despite its extension, this upstream file is one JSON object.
                task = json.loads(paths[0].read_text())
                frame = pd.json_normalize(task["Instances"], max_level=0)
                frame = frame.rename(columns={"input": "task_input", "output": "reference"})
                adapted.append(frame.assign(task_id=task_id, language=language,
                    adapted_row=range(len(frame)), adapted_file=str(paths[0].relative_to(self.raw_dir))))
                name = layout["original_test_prefix"] + language + ".jsonl"
                records = [json.loads(line) for line in archive.read(name).splitlines()]
                frame = pd.json_normalize(records, max_level=0)
                frame["original_record"] = records
                original.append(frame.assign(language=language, original_file=name,
                    original_row=range(len(frame))))
        adapted = pd.concat(adapted, ignore_index=True)
        original = pd.concat(original, ignore_index=True).rename(columns={"rationale": "reference"})

        # 2. Match the exact recorded query/reference to both source task banks.
        wrapper = parameters["chat_wrapper"]
        if not (attempts.input.str.startswith(wrapper["prefix"]).all()
                and attempts.input.str.endswith(wrapper["suffix"]).all()):
            raise ValueError("Unknown recorded chat format")
        parts = attempts.input.str.slice(len(wrapper["prefix"]), -len(wrapper["suffix"])).str.split(
            wrapper["separator"], regex=False, expand=True)
        if parts.shape[1] != 2 or parts.isna().any().any():
            raise ValueError("Expected exactly one recorded system/user boundary")
        attempts["system"], attempts["query"] = parts[0], parts[1]
        adapted["query"] = ("Input:\n" + adapted.task_input + "\nOutput:\n").str.strip()
        attempts = attempts.merge(adapted, on=["task_id", "query"], how="left", validate="many_to_one")
        if attempts.reference.isna().any() or not attempts.GT.eq(attempts.reference).all():
            raise ValueError("Recorded request/reference does not match the adapted task")
        attempts = attempts.merge(original[["language", "reference", "original_file", "original_row", "original_record"]],
            on=["language", "reference"], how="left", validate="many_to_one")
        if attempts.original_record.isna().any():
            raise ValueError("Missing original MMedBench reference")
        if not all(row.original_record["question"] in row.task_input
                   and all(value in row.task_input for value in row.original_record["options"].values())
                   for row in attempts.itertuples()):
            raise ValueError("Original question/options do not match the adapted input")
        attempts["target_in_demonstrations"] = [
            ("Input:\n" + row.task_input + "\n\nOutput:\n" + row.reference + "\n\n\n") in row.system
            for row in attempts.itertuples()]

        # 3. Preserve the full pre-tokenization request and its rationale protocol.
        subjects = pd.DataFrame([dict(subject_key=parameters["labels"]["model_label"],
            raw_label=parameters["labels"]["model_label"], features=parameters["subject_features"])])
        items = attempts.copy()
        items["item_key"] = items.source_row
        items["raw_item_id"] = items.original_file + "#" + items.original_row.astype(str)
        items["content"] = items.input
        items["features"] = [dict(**parameters["item_features"], task_id=row.task_id,
            language=row.language, target_in_demonstrations=bool(row.target_in_demonstrations))
            for row in items.itertuples()]
        protocol = self.grading["verifiers"]["rationale_similarity"]
        items["grading_criterion"] = [dict(reference_answer=row.reference, rule=protocol["rule"])
            for row in items.itertuples()]
        items["verifier"] = ExactMatcher(spec=json.dumps(protocol, sort_keys=True))

        # 4. Keep every original generation; the CSV contains no published grades.
        responses = attempts.assign(response_key=attempts.source_row, item_key=attempts.source_row,
            subject_key=parameters["labels"]["model_label"], response=None)
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(source_file=layout["results"], source_row=row.source_row,
            source_record=row.source_record, adapted_file=row.adapted_file, adapted_row=int(row.adapted_row),
            adapted_record=dict(input=row.task_input, output=row.reference), original_file=row.original_file,
            original_row=int(row.original_row), original_record=row.original_record,
            target_in_demonstrations=bool(row.target_in_demonstrations), grade_status=parameters["labels"]["grade_status"]),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {"subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]], "traces": traces}


if __name__ == "__main__":
    MMedBench(__file__).main_from_args()
