"""Tabulate IneqMath's original attempts while preserving unavailable grades."""

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class IneqMath(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters, paths = self.build_parameters, self.build_parameters["paths"]

        # 1. Load the eight original exports as a table of saved records.
        exports = pd.Series(parameters["settings"], name="setting").rename_axis("source_file").reset_index()
        with tarfile.open(self.raw_dir / paths["archive"]) as archive:
            exports["source_record"] = [json.load(archive.extractfile(paths["root"] + "/" + name)) for name in exports.source_file]
        records = exports.explode("source_record", ignore_index=True)
        records["source_row"] = records.groupby("source_file", sort=False).cumcount()
        records = records.join(pd.json_normalize(records.source_record, max_level=0))

        # 2. Consolidate the exact duplicated export, retaining both source paths.
        records["primary_file"] = records.source_file.replace(parameters["duplicate_exports"])
        records["record_json"] = records.source_record.map(lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))
        groups = records.groupby(["primary_file", "source_row"], sort=False)
        if groups.record_json.nunique().ne(1).any():
            raise ValueError("Declared duplicate exports contain conflicting records")
        locations = groups.source_file.agg(list).rename("source_files").reset_index()
        records = records.drop_duplicates(["primary_file", "source_row"]).merge(
            locations, on=["primary_file", "source_row"], validate="one_to_one")
        if records[["prompt", "response", "data_split", "type", "data_id"]].isna().any().any() or records.prompt.eq("").any():
            raise ValueError("A native attempt is missing its recorded input, output or task identity")
        records["item_key"] = records.primary_file + "/" + records.source_row.astype(str)
        records["raw_item_id"] = records.data_split + ":" + records.data_id.astype(str)

        # 3. Retain actual prompts and references; missing judgments stay missing.
        items = records[["item_key", "raw_item_id", "prompt"]].rename(columns={"prompt": "content"})
        items["features"] = [dict(split=row.data_split, problem_type=row.type, prompt_condition=row.setting,
            **parameters["observation"]) for row in records.itertuples()]
        items["grading_criterion"] = [dict(reference_answer=answer or None, rule=self.grading["rule"]) for answer in records.answer]
        items["verifier"] = Judge(spec=json.dumps(self.grading["verifiers"]["final_answer"], sort_keys=True), judged_by="llm")
        model = parameters["subject"]["model_identifier"]
        subjects = pd.DataFrame([dict(subject_key=model, raw_label=model, features=parameters["subject"])])
        responses = records[["item_key"]].assign(response_key=records.item_key, subject_key=model, response=None,
            test_condition="split=" + records.data_split + ";setting=" + records.setting)

        # 4. Preserve complete source records and duplicate-export provenance.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(primary_file=row.primary_file, source_row=int(row.source_row),
            source_files=row.source_files, source_record=row.source_record,
            grade_status=parameters["observation"]["grade_status"]), ensure_ascii=False, allow_nan=False)
            for row in records.itertuples()]
        return {"subjects": subjects, "items": items, "responses": responses, "traces": traces}


if __name__ == "__main__":
    IneqMath(__file__).main_from_args()
