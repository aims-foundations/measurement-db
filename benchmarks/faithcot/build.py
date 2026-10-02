"""Preserve released reasoning inputs, complete traces, and two distinct grades."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class FaithCoT(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Load original records directly into a table, retaining their native text.
        with ZipFile(self.raw_dir / parameters["paths"]["records"]) as archive:
            names = sorted(name for name in archive.namelist() if "/response_" in name and name.endswith(".json"))
            texts = [archive.read(name).decode("utf-8") for name in names]
        records = pd.json_normalize([json.loads(text) for text in texts], max_level=0)
        records["source_file"], records["native_json"] = names, texts
        coordinates = records.source_file.str.extract(r"^faithcot/(?P<task>[^/]+)/(?P<model>[^/]+)/response_(?P<index>\d+)\.json$")
        records = records.join(coordinates)
        if coordinates.isna().any().any() or not records.model.isin(parameters["models"]).all():
            raise ValueError("Unknown native record path or model")
        if not records.task.isin(parameters["domains"]).all() or not records.unfaithfulness.dropna().isin([0, 1]).all():
            raise ValueError("Unknown task or nonbinary human annotation")
        sample = pd.json_normalize(records.sample_0, max_level=0)

        # 2. Retain unavailable grades and expose contradictory combined labels.
        records["correct"] = sample.parsed_final_answer.astype("string").str.strip().eq(records.label.str.strip()).astype("Float64")
        records["faithful"] = 1 - records.unfaithfulness.astype("Float64")
        expected_type = (records.correct.fillna(0).astype(int) * 2 + 2 - records.faithful.fillna(0).astype(int))
        type_conflict = records.faithful_type.notna() & records.faithful_type.ne(expected_type)
        records["source_issues"] = [
            (["parsed_answer_unavailable"] if missing_answer else []) +
            (["human_annotation_unavailable"] if missing_label else []) +
            (["combined_type_inconsistent_or_undefined"] if conflict else [])
            for missing_answer, missing_label, conflict in zip(records.correct.isna(), records.faithful.isna(), type_conflict)]
        stimuli = records[["cot_prompt", "question", "options", "final_answer_str", "prefix"]].to_dict("records")
        records["content"] = [json.dumps(row, ensure_ascii=False, allow_nan=False) for row in stimuli]
        measurements = records.melt(id_vars=["source_file", "native_json", "task", "model", "index", "label", "content", "source_issues"],
                                    value_vars=["correct", "faithful"], var_name="metric", value_name="response")
        measurements["item_key"] = measurements.source_file + ":" + measurements.metric
        measurements["response_key"] = measurements.item_key
        measurements["subject_key"] = measurements.model

        # 3. Keep grading protocols separate while sharing the original input fields.
        items = measurements[["item_key", "content", "task", "index", "label", "metric"]].copy()
        items["raw_item_id"] = items.task + ":" + items["index"] + ":" + items.metric
        items["features"] = items[["task"]].to_dict("records")
        items["grading_criterion"] = [dict(rule=self.grading["verifiers"][metric]["rule"],
            **({"reference_answer": label} if metric == "correct" else {})) for metric, label in zip(items.metric, items.label)]
        items["verifier"] = [
            ExactMatcher(spec=json.dumps(self.grading["verifiers"][metric], sort_keys=True)) if metric == "correct" else
            Judge(judged_by="human", spec=json.dumps(self.grading["verifiers"][metric], sort_keys=True))
            for metric in items.metric]
        subjects = records[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects["raw_label"] = subjects.subject_key.map(parameters["models"])
        subjects["features"] = [dict(harness="FaithCoT-Bench", reported_model=model) for model in subjects.subject_key]

        # 4. Link both grades to the same complete source trajectory and annotations.
        measurements["trace"] = [json.dumps(dict(source_file=path, metric=metric, source_record_json=text,
            source_issues=issues), ensure_ascii=False, allow_nan=False)
            for path, metric, text, issues in zip(measurements.source_file, measurements.metric, measurements.native_json, measurements.source_issues)]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": measurements[["response_key", "subject_key", "item_key", "response"]],
            "traces": measurements[["response_key", "trace"]],
        }


if __name__ == "__main__":
    FaithCoT(__file__).main_from_args()
