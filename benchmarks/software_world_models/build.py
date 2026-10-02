#!/usr/bin/env python3
"""Join Software World Models' captured prompts to its per-model outcome predictions."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class SoftwareWorldModels(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("items", "generations")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load the native item bank and concatenate the model-generation tables.
        bank = pd.concat([pd.read_parquet(path) for path in
                          sorted((self.raw_dir / "benchmark/data").glob("*.parquet"))], ignore_index=True)
        observations = pd.concat([pd.read_parquet(path) for path in
                                  sorted((self.raw_dir / "generations/data").glob("*.parquet"))], ignore_index=True)
        predictions = pd.json_normalize(observations.prediction.fillna("{}").map(json.loads))
        references = pd.json_normalize(observations.ground_truth.map(json.loads))
        observations = observations.assign(predicted_outcome=predictions["outcome.outcome"],
                                           recorded_reference=references["outcome.outcome"])

        # 2. Match sample identity and verify that both releases contain the same reference.
        observations = observations.merge(bank[["sample_id", "ground_truth_outcome"]],
                                           on="sample_id", how="left", validate="many_to_one", indicator=True)
        if not observations._merge.eq("both").all() or not observations.recorded_reference.eq(observations.ground_truth_outcome).all():
            raise ValueError("A Software World Models observation has no matching item/reference")

        # 3. Preserve complete prompts and explicit three-category correctness grading.
        spec = json.dumps(self.grading["verifiers"]["outcome"], sort_keys=True)
        items = bank.assign(
            item_key=bank.sample_id, raw_item_id=bank.sample_id,
            content=(bank.system_prompt.fillna("") + "\n\n" + bank.user_prompt.fillna("")).str.strip(),
            grading_criterion=bank.ground_truth_outcome.map(lambda value: {
                "reference_answer": value, "rule": self.grading["rule"],
            }),
            verifier=ExactMatcher(spec=spec),
        )
        subjects = observations[["model"]].drop_duplicates().rename(columns={"model": "subject_key"})
        subjects = subjects.assign(raw_label=subjects.subject_key,
                                   features=[self.build_parameters["subject_features"]] * len(subjects))

        # 4. Keep every recorded attempt, including repeated canonical prompts and parse failures.
        responses = observations.assign(
            response_key=observations.index, subject_key=observations.model, item_key=observations.sample_id,
            response=observations.predicted_outcome.eq(observations.ground_truth_outcome).fillna(False).astype(float),
            test_condition=self.build_parameters["evaluation"]["condition"],
        )
        traces = responses[["response_key", "raw_response"]].rename(columns={"raw_response": "trace"})
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces.loc[traces.trace.map(lambda value: isinstance(value, str))],
        }


if __name__ == "__main__":
    SoftwareWorldModels(__file__).main_from_args()
