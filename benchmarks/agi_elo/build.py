#!/usr/bin/env python3
"""Join AGI-Elo's released MMLU predictions to its question and answer bank."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class AGIEloBuild(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("*")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load the item bank and the checksum-pinned, native prediction tables.
        bank = pd.read_parquet(self.raw_dir / "mmlu_items.parquet")
        bank["gold"] = bank.answer.astype(int).map(dict(enumerate("ABCD")))
        frames = []
        for path in sorted((self.raw_dir / "mmlu_predictions").glob("*.pkl")):
            frame = pd.read_pickle(path).reset_index(drop=True)
            frames.append(frame.rename(columns={
                "Test Case": "id", "True Label": "gold", path.stem: "trace",
            }).assign(subject_key=path.stem))
        attempts = pd.concat(frames, ignore_index=True)

        # 2. Require both task identity and reference-answer agreement. Historical
        # answer conflicts remain in raw rather than receiving a changed grade.
        observations = attempts.merge(bank.drop(columns="answer"), on=["id", "gold"],
                                      how="inner", sort=False, validate="many_to_one")

        # 3. Preserve each complete question and ordered choices in the item text.
        items = observations[["id", "question", "subject", "gold"]].drop_duplicates()
        items = items.merge(bank[["id", "choices"]], on="id", validate="one_to_one")
        options = pd.DataFrame(items.choices.tolist(), index=items.index, columns=list("ABCD"))
        content = items.question
        for label in options.columns:
            content = content + "\n" + label + ". " + options[label]
        spec = json.dumps(self.grading["verifiers"]["multiple_choice"], sort_keys=True)
        items = items.assign(
            item_key=items.id, raw_item_id=items.id, content=content,
            features=items.subject.map(lambda value: {"mmlu_subject": value}),
            grading_criterion=items.gold.map(lambda value: {
                "reference_answer": value, "rule": self.grading["rule"],
            }),
            verifier=ExactMatcher(spec=spec),
        )

        # 4. Apply the author's answer extraction; retain the complete model output.
        patterns = self.build_parameters["answer_parser"]
        answer = (observations.trace.astype("string")
                  .str.replace(patterns["reasoning_block"], "", regex=True)
                  .str.extract(patterns["choice"], expand=False))
        responses = observations.assign(
            response_key=observations.index, item_key=observations.id,
            response=answer.eq(observations.gold).fillna(False).astype(float),
        )
        subjects = observations[["subject_key"]].drop_duplicates()
        subjects = subjects.assign(
            raw_label=subjects.subject_key,
            features=subjects.subject_key.map(lambda value: {
                "harness": "AGI-Elo MMLU", "reported_model": value,
            }),
        )
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "features", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": responses[["response_key", "trace"]],
        }


if __name__ == "__main__":
    AGIEloBuild(__file__).main_from_args()
