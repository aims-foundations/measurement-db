#!/usr/bin/env python3
"""Curate the complete released BertaQA API conversations with table operations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class BertaQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("results", "questions")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        root = self.raw_dir / parameters["layout"]["results"]

        # 1. Load each result file and retain its native record before adding columns.
        frames = []
        for path in sorted(root.glob("*/bertaqa_??_5-shot.jsonl")):
            frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False, precise_float=True)
            frame["native_record"] = frame.to_dict("records")
            model, language = path.parent.name, path.stem.split("_")[1]
            provider = parameters["providers"][model]
            frame["conversation"] = frame.messages
            if provider == "anthropic":
                frame["conversation"] = [[{"role": "system", "content": system}, *messages]
                                         for system, messages in zip(frame.system, frame.messages)]
            frame["test_condition"] = "temperature=" + parameters[provider + "_request"]["temperature"]
            frames.append(frame.assign(subject_key=model, language=language, api_provider=provider,
                                       source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
        observations = pd.concat(frames, ignore_index=True)
        if not observations.correct.map(lambda value: type(value) is bool).all():
            raise ValueError("Expected native boolean correctness grades")
        if observations.duplicated(["source_file", "id"]).any():
            raise ValueError("A source result file repeats a question ID")

        # 2. Keep exact model revisions and the published prompting/scoring protocol.
        subjects = observations[["subject_key", "api_provider"]].drop_duplicates()
        subjects["raw_label"] = subjects.subject_key
        subjects["features"] = [{**parameters["subject_features"], "model_identifier": row.subject_key,
                                  "api_provider": row.api_provider} for row in subjects.itertuples()]
        observations["gold"] = observations.answer.astype(str).map(parameters["answer_letters"])
        if observations.gold.isna().any():
            raise ValueError("A reference answer is outside the three-option question format")

        # 3. Full conversations distinguish different few-shot samples for one question.
        observations["content"] = observations.conversation.map(
            lambda messages: json.dumps(messages, ensure_ascii=False, sort_keys=True))
        observations["item_key"] = observations.content + "\n" + observations.gold
        items = observations.drop_duplicates("item_key").copy()
        items["raw_item_id"] = items.language + "-" + items.id.astype(str)
        items["features"] = [{"language": row.language, "category": row.category, "group": row.group,
                               "difficulty": row.difficulty, "shot": parameters["prompt"]["shots"]}
                              for row in items.itertuples()]
        items["grading_criterion"] = items.gold.map(lambda gold: {"reference_answer": gold, "rule": self.grading["rule"]})
        items["verifier"] = ExactMatcher(spec=json.dumps(self.grading["verifiers"]["released_accuracy"], sort_keys=True))

        # 4. Preserve every grade and complete API record, including non-letter failures.
        observations["response_key"] = observations.index
        observations["response"] = observations.correct.astype(float)
        traces = observations[["response_key"]].copy()
        traces["trace"] = [json.dumps({"source_file": row.source_file, "source_row": row.source_row,
                                      "record": row.native_record,
                                      "published_request_settings": parameters[row.api_provider + "_request"]},
                                     ensure_ascii=False, allow_nan=False)
                           for row in observations.itertuples()]
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": observations[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    BertaQA(__file__).main_from_args()
