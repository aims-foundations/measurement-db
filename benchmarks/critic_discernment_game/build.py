#!/usr/bin/env python3
"""Curate the released Lighteval records without executing or regrading models."""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class CriticDiscernmentGame(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["layout"]["root"]

        # 1. Read the native detail tables and their matching configuration summaries.
        frames, configurations = [], []
        for path in sorted((root / "lt_eval/results").glob("*/*/details/details_custom_x7c_*.parquet")):
            profile, decoding = path.relative_to(root / "lt_eval/results").parts[:2]
            dataset = path.name.split("_x7c_")[1]  # The downloader encodes upstream pipe characters.
            summary_path = path.parent.parent / "results" / parameters["summary_files"][profile + "/" + dataset]
            summary = json.loads(summary_path.read_text())
            task = summary["config_tasks"]["custom|" + dataset]
            subject_key = "/".join([profile, decoding, dataset])
            frame = pd.read_parquet(path)
            frames.append(frame.assign(
                native_record=frame.to_dict("records"), source_row=frame.index,
                source_file=str(path.relative_to(self.raw_dir)),
                summary_file=str(summary_path.relative_to(self.raw_dir)),
                subject_key=subject_key, dataset=dataset,
                test_condition="temperature=" + parameters["temperatures"][decoding]))
            configurations.append({
                "subject_key": subject_key, "raw_label": parameters["models"][profile],
                "features": {"harness": "Lighteval", "checkpoint": profile,
                             "reported_task_generation_size": str(task["generation_size"]),
                             "reported_model_path": summary["config_general"]["model_name"]},
            })
        observations = pd.concat(frames, ignore_index=True)
        subjects = pd.DataFrame(configurations)

        # 2. Parse only the saved metric; preserve all other native fields in traces.
        observations["response"] = observations.metrics.map(ast.literal_eval).str["extractive_match"]
        if not observations.response.isin([0.0, 1.0]).all():
            raise ValueError("A released extractive_match value is missing or not binary")

        # 3. Deduplicate exact prompts and references, keeping the original whitespace.
        observations["item_key"] = [json.dumps({"prompt": row.full_prompt, "reference": row.gold}, ensure_ascii=True)
                                    for row in observations.itertuples()]
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.dataset + ":" + items.source_row.astype(str)
        items["content"] = items.full_prompt
        items["features"] = items.dataset.map(lambda name: {"dataset": name})
        items["grading_criterion"] = items.gold.map(
            lambda gold: {"reference_answer": gold, "rule": self.grading["rule"]})
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["native_match"], sort_keys=True))

        # 4. Retain every observation and its complete native record, including token data.
        observations["response_key"] = observations.index
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": row.source_row,
                                      "summary_file": row.summary_file, "record": row.native_record},
                                     ensure_ascii=False, allow_nan=False)
                           for row in observations.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    CriticDiscernmentGame(__file__).main_from_args()
