#!/usr/bin/env python3
"""Curate AISafetyLab's released AdvPrompter inputs, replies and safety judgments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class AdvPrompter(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the two result exports and retain their original row positions.
        parameters = self.build_parameters
        native = pd.concat([
            pd.read_json(self.raw_dir / path, dtype=False, convert_dates=False).assign(
                subject_key=defense, source_file=path, source_row=lambda frame: frame.index)
            for defense, path in parameters["result_files"].items()
        ], ignore_index=True)
        native["response_key"] = native.index

        # 2. A defended model is a distinct subject; missing configuration stays unknown.
        subjects = native[["subject_key"]].drop_duplicates().assign(
            raw_label=parameters["target"]["model"])
        subjects["features"] = subjects.rename(columns={"subject_key": "defense"})[["defense"]].to_dict("records")

        # 3. The target sees final_query; the judge sees the original query and reply.
        identity = ["query", "final_query"]
        items = native.drop_duplicates(identity).copy()
        items["item_key"] = range(len(items))
        items["raw_item_id"] = parameters["target"]["item_prefix"] + items.source_row.astype(str)
        items["content"] = items.final_query
        items["grading_criterion"] = [
            {"rule": json.dumps({"rule": self.grading["rule"], "judge_query": query}, ensure_ascii=False)}
            for query in items["query"]
        ]
        items["verifier"] = Judge(judged_by="llm", spec=json.dumps(
            self.grading["verifiers"]["llama_guard"], sort_keys=True))

        # 4. Preserve the released grades without thresholding or rejudging them.
        responses = native.merge(items[identity + ["item_key"]], on=identity,
                                 how="left", validate="many_to_one")
        responses = responses.rename(columns={"final_score": "response"}).assign(
            interactors=parameters["target"]["interactors"])

        # 5. Keep every original field, including the attacker target prefix and full reply.
        traces = native[["response_key"]].copy()
        fields = ["query", "target", "final_query", "final_response", "final_score", "source_file", "source_row"]
        traces["trace"] = [json.dumps(row, ensure_ascii=False, allow_nan=False)
                           for row in native[fields].to_dict("records")]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "interactors"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AdvPrompter(__file__).main_from_args()
