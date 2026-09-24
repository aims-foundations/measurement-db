#!/usr/bin/env python3
"""Flatten Abstract-Reason's recorded chat prompts and preserve its correctness flags."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AbstractReason(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("results")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Concatenate native result arrays, retaining the task and source row.
        source = self.raw_dir / "abstract-reason-benchmark" / self.build_parameters["subject"]["model"] / "result"
        frames = []
        for path in sorted(source.rglob("output.json")):
            frame = pd.read_json(path, dtype=False, convert_dates=False)
            relative = path.relative_to(source)
            frames.append(frame.assign(source_row=frame.index, level=relative.parts[0], task=relative.parts[-2]))
        observations = pd.concat(frames, ignore_index=True).assign(response_key=lambda frame: frame.index)
        if not observations.is_correct.isin([False, True, 0, 1]).all():
            raise ValueError("Abstract-Reason contains an unknown correctness flag")

        # 2. Explode chat messages and concatenate them in the original order.
        messages = observations[["response_key", "prompt"]].explode("prompt", ignore_index=True)
        fields = pd.json_normalize(messages.prompt.tolist()).reindex(columns=["role", "content"]).fillna("")
        messages["text"] = fields.role + ": " + fields.content
        prompts = messages.groupby("response_key", sort=False).text.agg("\n".join).rename("content")
        observations = observations.join(prompts, on="response_key")

        # 3. Keep the reference and the applicable numeric or exact-match grading protocol.
        numeric = observations.level.str.contains(r"l0|l3", regex=True)
        observations["grading_mode"] = numeric.map({True: "numeric", False: "exact"})
        specs = self.grading["verifiers"]
        items = observations.assign(
            item_key=observations.response_key,
            raw_item_id=observations.level + "/" + observations.task + ":" + observations.source_row.astype(str),
            grading_criterion=[{"reference_answer": answer, "rule": specs[mode]["rule"]}
                               for answer, mode in zip(observations.answer, observations.grading_mode)],
            verifier=observations.grading_mode.map(lambda mode: ExactMatcher(spec=json.dumps(specs[mode], sort_keys=True))),
        )

        # 4. Link the original correctness flags and complete, unmodified model outputs.
        subjects = pd.DataFrame({"subject_key": [0], "raw_label": [self.build_parameters["subject"]["model"]]})
        responses = observations.assign(subject_key=0, item_key=observations.response_key,
                                         response=observations.is_correct.astype(float),
                                         test_condition="level=" + observations.level + ";task=" + observations.task)
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": responses[["response_key", "raw_output"]].rename(columns={"raw_output": "trace"}),
        }


if __name__ == "__main__":
    AbstractReason(__file__).main_from_args()
