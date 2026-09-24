#!/usr/bin/env python3
"""Curate released human ratings of written corrective feedback with table operations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class AnnotatingErrorsWCF(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["layout"]["release"]

        # 1. Load native ratings and generation records, retaining their source positions.
        path = root / parameters["layout"]["ratings"]
        ratings = pd.read_json(path, dtype=False, convert_dates=False)
        ratings["rating_record"] = ratings.to_dict("records")
        ratings["rating_source_row"] = ratings.index
        ratings["rating_source_file"] = str(path.relative_to(self.raw_dir))
        ratings = ratings.loc[ratings.fb_source.isin(parameters["subjects"])].copy()
        if ratings.duplicated(["user_id", "rater_task_id"]).any():
            raise ValueError("A teacher's rating occurs more than once in the release")
        frames = []
        for source, filename in parameters["generations"].items():
            path = root / filename
            frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
            if not frame.fb_source.eq(source).all():
                raise ValueError(f"Generation source does not match its declared system: {filename}")
            frame["generation_record"] = frame.to_dict("records")
            frames.append(frame.assign(generation_source_file=str(path.relative_to(self.raw_dir)),
                                       generation_source_row=frame.index))
        generations = pd.concat(frames, ignore_index=True)

        # 2. Join each teacher's rating to the exact saved generation and full input prompt.
        keys = ["fb_source", "annotation_instance_id"]
        rated = ratings.merge(generations[keys + ["input_prompt", "original_id", "cefr_level",
                                                  "generation_record", "generation_source_file", "generation_source_row"]],
                              on=keys, how="left", validate="many_to_one", indicator=True)
        if not rated._merge.eq("both").all():
            raise ValueError("A human rating has no corresponding released generation")
        rated["trial"] = rated.groupby(keys + ["user_id"]).cumcount() + 1
        rated["test_condition"] = "temperature=" + parameters["conditions"]["temperature"]
        rated["subject_key"] = rated.fb_source
        subjects = rated[["subject_key"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key.map(parameters["subjects"])
        subjects["features"] = [{**parameters["subject_features"], "feedback_strategy": source}
                                for source in subjects.subject_key]

        # 3. Unpivot the eight criteria without averaging teachers or reversing any scale.
        grades = rated.melt(id_vars=["rating_source_row"], value_vars=list(parameters["metrics"]),
                            var_name="score_field", value_name="response")
        observations = grades.merge(rated, on="rating_source_row", validate="many_to_one")
        observations["metric"] = observations.score_field.map(parameters["metrics"])
        direct = observations.metric.eq("directness")
        if not observations.loc[direct, "response"].dropna().isin(parameters["directness"]).all():
            raise ValueError("Unknown feedback-directness category")
        observations.loc[direct, "response"] = observations.loc[direct, "response"].map(parameters["directness"])
        observations["response"] = pd.to_numeric(observations.response).astype(float)

        # 4. Each prompt, criterion and identified teacher defines a measurement item.
        observations["item_key"] = (observations.fb_source + ":" + observations.annotation_instance_id
                                    + ":" + observations.metric + ":" + observations.user_id)
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.item_key
        items["content"] = items.input_prompt.map(lambda prompt: json.dumps(prompt, ensure_ascii=False, sort_keys=True))
        items["features"] = [{"annotation_instance_id": row.annotation_instance_id, "original_id": row.original_id,
                               "cefr_level": row.cefr_level, "language": "en"}
                              for row in items.itertuples()]
        protocol = self.grading["verifiers"]["human_ratings"]
        items["grading_criterion"] = items.metric.map(lambda metric: protocol["criteria"][metric])
        items["verifier"] = [Judge(judge=rater, judged_by="human", spec=json.dumps(protocol, sort_keys=True))
                             for rater in items.user_id]

        # 5. Preserve full native records: repeated ratings assess the same saved generation.
        observations["response_key"] = observations.index
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"rating_source_file": row.rating_source_file,
                                      "rating_source_row": row.rating_source_row, "rating_record": row.rating_record,
                                      "generation_source_file": row.generation_source_file,
                                      "generation_source_row": row.generation_source_row,
                                      "generation_record": row.generation_record, "score_field": row.score_field},
                                     ensure_ascii=False, allow_nan=False)
                           for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    AnnotatingErrorsWCF(__file__).main_from_args()
