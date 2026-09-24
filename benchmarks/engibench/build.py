#!/usr/bin/env python3
"""Curate saved EngiOpt notebook outcomes without executing their code."""

import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class EngiBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("examples")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["layout"]["root"]

        # 1. Load native notebook cells as a table, retaining complete saved outputs.
        notebooks, frames = {}, []
        for filename in parameters["examples"]:
            notebooks[filename] = json.loads((root / filename).read_text())
            cells = pd.json_normalize(notebooks[filename]["cells"], max_level=0)
            frames.append(cells.assign(subject_key=filename, source_cell=cells.index))
        cells = pd.concat(frames, ignore_index=True)
        cells["code"] = cells.source.str.join("")

        # 2. Read literal design conditions; explode each printed grade array by position.
        conditions = cells[["subject_key"]].join(cells.code.str.extract(
            parameters["patterns"]["conditions"]).rename(columns={0: "conditions"})).dropna()
        conditions["conditions"] = conditions.conditions.map(ast.literal_eval)
        conditions = conditions.explode("conditions", ignore_index=True)
        conditions["design_index"] = conditions.groupby("subject_key").cumcount()
        outputs = cells[["subject_key", "source_cell", "outputs"]].explode("outputs").dropna(subset=["outputs"])
        outputs = outputs.reset_index(drop=True)
        outputs["text"] = pd.json_normalize(outputs.outputs).reindex(columns=["text"]).text.str.join("")
        grades = outputs.text.str.extractall(parameters["patterns"]["grades"]).rename(columns={0: "heading", 1: "response"})
        grades = grades.droplevel("match").join(outputs[["subject_key", "source_cell"]])
        grades["response"] = grades.response.map(json.loads)
        grades = grades.explode("response", ignore_index=True)
        grades["criterion"] = grades.heading.map(parameters["metrics"])
        grades["design_index"] = grades.groupby(["subject_key", "criterion"]).cumcount()
        responses = grades.merge(conditions, on=["subject_key", "design_index"], validate="many_to_one")

        # 3. The same conditions share an item within each numerical grading protocol.
        responses["content"] = [json.dumps({"problem": parameters["problem"], "conditions": value}, sort_keys=True)
                                for value in responses.conditions]
        items = responses.drop_duplicates(["content", "criterion"]).copy()
        items["item_key"] = range(len(items))
        items["raw_item_id"] = "beams2d:example:" + items.design_index.astype(str) + ":" + items.criterion
        items["grading_criterion"] = [{"rule": json.dumps({"criterion": kind, "rule": parameters["rules"][kind]})}
                                     for kind in items.criterion]
        items["verifier"] = [ExactMatcher(spec=json.dumps({**self.grading["verifiers"]["native_gap"], "criterion": kind}))
                             for kind in items.criterion]
        subjects = pd.Series(parameters["examples"], name="raw_label").rename_axis("subject_key").reset_index()
        subjects["features"] = [{"harness": "EngiOpt Beams2D " + parameters["problem"]["version"],
                                 "source_run": filename} for filename in subjects.subject_key]

        # 4. Link native grade positions and preserve each full notebook as the trace.
        responses = responses.merge(items[["content", "criterion", "item_key"]],
                                    on=["content", "criterion"], validate="many_to_one")
        responses["response_key"] = responses.index
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.subject_key, "source_cell": row.source_cell,
                                      "design_index": row.design_index, "notebook": notebooks[row.subject_key]},
                                     ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    EngiBench(__file__).main_from_args()
