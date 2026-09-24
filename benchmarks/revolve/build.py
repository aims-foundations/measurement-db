#!/usr/bin/env python3
"""Expand REVOLVE's released prediction trajectories into five optimization stages."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class Revolve(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("predictions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read each question-indexed trajectory table with its model and dataset.
        models = self.build_parameters["models"]
        datasets = self.build_parameters["datasets"]
        stages = list(self.build_parameters["stages"].values())
        frames = []
        for filename, model in models.items():
            frame = pd.read_json(self.raw_dir / filename, orient="index", dtype=False, convert_axes=False)
            frames.append(frame.assign(content=frame.index, subject_key=model, dataset=datasets[filename]))
        data = pd.concat(frames, ignore_index=True)
        data["gold"] = data.answer.str.strip().str.upper()
        data = data.loc[data.gold.isin(["A", "B", "C", "D"]) & data.predictions.map(lambda value: isinstance(value, list) and len(value) > 0)].copy()

        # 2. Register each question/reference definition with a stable source alias.
        items = data[["content", "gold"]].drop_duplicates().reset_index(drop=True)
        spec = json.dumps(self.grading["verifiers"]["answer_letter"], sort_keys=True)
        items = items.assign(
            item_key=items.index,
            raw_item_id=items.content.map(lambda prompt: "question_" + hashlib.sha256(prompt.encode()).hexdigest()[:24]),
            grading_criterion=items.gold.map(lambda answer: {"reference_answer": answer, "rule": self.grading["rule"]}),
            verifier=ExactMatcher(spec=spec),
        )

        # 3. Expand the recorded stages and grade the first matching answer letter.
        data = data.assign(trajectory_key=data.index, predictions=data.predictions.str[:len(stages)])
        responses = data.explode("predictions", ignore_index=True).rename(columns={"predictions": "trace"})
        stage_index = responses.groupby("trajectory_key", sort=False).cumcount()
        stage_names = stage_index.map(dict(enumerate(stages)))
        responses["test_condition"] = "dataset=" + responses.dataset + ";stage=" + stage_names
        letter = responses.trace.str.extract(self.build_parameters["parsing"]["answer_letter"], expand=False).str.upper()
        responses["response"] = letter.eq(responses.gold).fillna(False).astype(float)
        responses = responses.merge(items[["content", "gold", "item_key"]], on=["content", "gold"],
                                    how="left", sort=False, validate="many_to_one")
        responses["response_key"] = responses.index
        subjects = pd.DataFrame({"raw_label": list(dict.fromkeys(models.values()))})
        subjects["subject_key"] = subjects.raw_label
        subjects["features"] = subjects.raw_label.str.rsplit("+", n=1).str[-1].map(lambda method: {"harness": method})
        has_trace = responses.trace.map(lambda value: isinstance(value, str))
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": responses.loc[has_trace, ["response_key", "trace"]],
        }


if __name__ == "__main__":
    Revolve(__file__).main_from_args()
