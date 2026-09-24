#!/usr/bin/env python3
"""Curate native BiGGen generations and rubric judgments with table operations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class BiggenBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "protocol")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["layout"]["data"]

        # 1. Read only the four declared result splits, keeping native row positions.
        frames = []
        for split, pattern in parameters["splits"].items():
            for path in sorted(root.glob(pattern)):
                frame = pd.read_parquet(path).reset_index(drop=True).astype(object)
                frame = frame.where(frame.notna(), None)
                frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)),
                                           source_row=frame.index, split=split))
        generations = pd.concat(frames, ignore_index=True).rename(columns={"response": "model_output"})
        if generations.uuid.duplicated().any():
            raise ValueError("A generation appears more than once in the selected release")
        generations["subject_key"] = generations.model_name
        subjects = generations[["subject_key"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [{**parameters["subject"], "model_identifier": name} for name in subjects.subject_key]

        # 2. Unpivot the judge columns; their identities belong to the grading protocol.
        grades = generations.melt(id_vars=["uuid"], value_vars=list(parameters["judges"]),
                                  var_name="score_field", value_name="published_scores")
        observations = grades.merge(generations, on="uuid", how="left", validate="many_to_one")
        human = observations.score_field.eq("human_score")
        missing_human = observations.loc[human, "published_scores"].eq(int(parameters["missing"]["human_sentinel"]))
        observations = observations.drop(index=missing_human[missing_human].index)
        skipped_tasks = parameters["missing"]["excluded_prometheus_tasks"].split(",")
        skipped = observations.score_field.str.startswith("prometheus_") & observations.task.isin(skipped_tasks)
        if observations.loc[skipped, "published_scores"].notna().any():
            raise ValueError("A formerly excluded Prometheus task now has scores; review the protocol")
        observations = observations.loc[~skipped].copy()

        # 3. Preserve every rating position; scalar and wholly missing grades use one row.
        # Arrow list columns arrive as arrays. This conversion only restores native JSON types.
        observations["published_scores"] = observations.published_scores.map(
            lambda value: value.tolist() if hasattr(value, "tolist") else value)
        observations["published_scores"] = observations.published_scores.map(
            lambda value: [None if pd.isna(score) else score for score in value] if isinstance(value, list) else value)
        observations["published_scores"] = observations.published_scores.astype(object).where(
            observations.published_scores.notna(), None)
        observations["response"] = observations.published_scores
        observations = observations.explode("response", ignore_index=True)
        observations["rating_index"] = observations.groupby(["uuid", "score_field"], sort=False).cumcount()
        if not observations.response.dropna().isin([1, 2, 3, 4, 5]).all():
            raise ValueError("A published rating falls outside the item rubric's 1–5 scale")
        observations["trial"] = observations.rating_index + 1

        # 4. Retain full system/user instructions and every distinct reference/rubric.
        generations["definition"] = [json.dumps({"system_prompt": row.system_prompt, "input": row.input,
                                                  "reference_answer": row.reference_answer, "score_rubric": row.score_rubric},
                                                 ensure_ascii=True, sort_keys=True)
                                     for row in generations.itertuples()]
        observations = observations.merge(generations[["uuid", "definition"]], on="uuid", validate="many_to_one")
        observations["item_key"] = observations.definition + ":" + observations.score_field
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.id + ":" + items.score_field
        items["content"] = [json.dumps({"system_prompt": row.system_prompt, "input": row.input}, ensure_ascii=True)
                            for row in items.itertuples()]
        items["features"] = [{"capability": row.capability, "task": row.task, "language": row.language}
                             for row in items.itertuples()]
        items["grading_criterion"] = [{"reference_answer": row.reference_answer,
                                        "rule": json.dumps({"rubric": row.score_rubric, "interpretation": self.grading["rule"]},
                                                           ensure_ascii=True, sort_keys=True)}
                                       for row in items.itertuples()]
        items["verifier"] = items.score_field.map(lambda field: Judge(
            judge=parameters["judges"][field], judged_by="human" if field == "human_score" else "llm",
            spec=json.dumps(self.grading["verifiers"]["rubric"], sort_keys=True)))

        # 5. Keep full generations, saved feedback and the unaveraged source score list.
        observations["response_key"] = observations.index
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row["source_file"], "source_row": row["source_row"],
                                      "split": row["split"], "generation_id": row["uuid"],
                                      "used_for_training": row["used_for_training"], "model_output": row["model_output"],
                                      "score_field": row["score_field"], "rating_index": row["rating_index"],
                                      "published_scores": row["published_scores"],
                                      "published_feedback": row.get(row["score_field"].replace("_score", "_feedback"))},
                                     ensure_ascii=False, allow_nan=False)
                           for row in observations.to_dict("records")]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "trial"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BiggenBench(__file__).main_from_args()
