#!/usr/bin/env python3
"""Curate recorded medical responses and their released hallucination verdicts."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class DASMedicalHallucination(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "legacy_protocol")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]
        bundle = self.raw_dir / paths["bundle"]
        manifest = json.loads((bundle / "manifest.json").read_text())

        # 1. Read the manifest-declared generations and judgments as tables.
        frames = {"generations": [], "judgments": []}
        for entry in manifest["files"]:
            group = next((name for name in frames if entry["path"].startswith(paths[name])), None)
            if group is None:
                continue
            data = json.loads((bundle / entry["path"]).read_text())
            rows = data["results"] if group == "generations" else data
            table = pd.json_normalize(rows, max_level=0)
            table["native"] = rows
            table["source_file"] = entry["path"]
            table["source_row"] = range(len(table))
            table["attempt_key"] = [json.dumps({key: row.get(key) for key in
                ["prompt", "response", "metadata", "error"]}, sort_keys=True, ensure_ascii=False,
                allow_nan=False) for row in rows]
            frames[group].append(table)
        if not all(frames.values()):
            raise ValueError("DAS requires both released generation and detector files")
        generations = pd.concat(frames["generations"], ignore_index=True)
        judgments = pd.concat(frames["judgments"], ignore_index=True)

        # 2. Join exact native attempts, retaining generations without a grade.
        # Repeated prompts are distinct attempts; file-row positions alone do
        # not identify which response the detector actually evaluated.
        records = generations.merge(judgments[["attempt_key", "native", "source_file", "source_row", "merged_codes"]],
            on="attempt_key", how="outer", validate="one_to_one", indicator=True,
            suffixes=("_generation", "_judgment"))
        if records._merge.eq("right_only").any():
            raise ValueError("A DAS judgment has no exact matching recorded generation")
        records = records.sort_values(["source_file_generation", "source_row_generation"], kind="stable").reset_index(drop=True)
        settings = pd.json_normalize(records.metadata)
        records["subject_key"] = settings.model
        records["item_key"] = records.prompt.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
        records["response_key"] = records.source_file_generation + "/" + records.source_row_generation.astype(int).astype(str)
        records["test_condition"] = records.metadata.map(lambda values: json.dumps({key: value
            for key, value in values.items() if key not in ["model", "timestamp", "response_time"]},
            sort_keys=True, allow_nan=False))

        # 3. Preserve exact presented prompts, including variants absent from
        # the separately released seed bank. The source alias is the model ID
        # actually recorded by the run; unavailable checkpoints stay unknown.
        items = records[["item_key", "prompt"]].drop_duplicates().rename(columns={"prompt": "content"})
        items["raw_item_id"] = items.item_key
        items["features"] = items.item_key.map(lambda digest: {"prompt_sha256": digest})
        items["grading_criterion"] = [{"rule": self.grading["rule"]} for _ in range(len(items))]
        items["verifier"] = [Judge(spec=json.dumps(self.grading["verifiers"]["detector"], sort_keys=True),
            judged_by="llm") for _ in range(len(items))]
        subjects = records[["subject_key"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = subjects.subject_key.map(lambda model: {"harness": self.name, "reported_model": model})

        # 4. Translate the native categories without imputing missing grades.
        codes = records.merged_codes.map(lambda value: "0" if isinstance(value, list) and not value else value).explode()
        category_values = self.grading["verifiers"]["detector"]["category_values"]
        if not codes.dropna().isin(category_values).all():
            raise ValueError("DAS contains an unknown detector category")
        records["response"] = codes.map(category_values).groupby(level=0).max()
        records["trial"] = records.groupby(["subject_key", "item_key", "test_condition"], sort=False).cumcount() + 1
        responses = records[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]]

        # 5. Retain every complete native generation and its judgment record.
        traces = records[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(generation_file=row.source_file_generation,
            generation_row=int(row.source_row_generation), generation=row.native_generation,
            judgment_file=row.source_file_judgment if isinstance(row.source_file_judgment, str) else None,
            judgment_row=int(row.source_row_judgment) if pd.notna(row.source_row_judgment) else None,
            judgment=row.native_judgment if isinstance(row.native_judgment, dict) else None,
            trace_scope="released_generation_and_detector_record"), ensure_ascii=False, allow_nan=False)
            for row in records.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses,
            "traces": traces,
        }


if __name__ == "__main__":
    DASMedicalHallucination(__file__).main_from_args()
