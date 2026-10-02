"""Flatten the released Few-Shot TTT experiments and preserve every prediction."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class FewshotTTTBBH(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Flatten each result file's task -> examples list into a table.
        models = self.build_parameters["models"]
        root = self.raw_dir / self.build_parameters["layout"]["results"]
        paths = sorted(path for method in models for path in (root / method).glob("*.json"))
        observations = pd.concat([
            pd.json_normalize(json.loads(path.read_text()), record_path="examples", meta="task").assign(
                subject_key=path.parent.name, source_file=str(path.relative_to(root)),
                source_row=lambda frame: frame.index,
            ) for path in paths
        ], ignore_index=True)
        if observations[["question", "true_answer", "prediction"]].isna().any().any():
            raise ValueError("A Few-Shot TTT example lacks its question, reference or prediction")

        # 2. Keep the inference/training method as part of subject identity.
        subjects = pd.Series(models, name="raw_label").rename_axis("subject_key").reset_index()
        subjects["features"] = subjects.subject_key.map(lambda method: {
            **self.build_parameters["subject_features"], "method": method,
        })
        items = observations[["task", "question", "true_answer"]].drop_duplicates().reset_index(drop=True)
        items = items.assign(
            item_key=items.index, raw_item_id=None, content=items.question,
            grading_criterion=items.true_answer.map(lambda answer: {"reference_answer": str(answer), "rule": self.grading["rule"]}),
            verifier=ExactMatcher(spec=json.dumps(self.grading["verifiers"]["exact_match"], sort_keys=True)),
            features=items.task.map(lambda task: {"task": task}),
        )

        # 3. Apply the upstream string comparison and link every attempt to its item.
        responses = observations.merge(
            items[["task", "question", "true_answer", "item_key"]],
            on=["task", "question", "true_answer"], how="left", validate="many_to_one",
        ).assign(
            response_key=observations.source_file + "::" + observations.source_row.astype(str),
            response=observations.prediction.astype(str).str.strip().str.lower().eq(
                observations.true_answer.astype(str).str.strip().str.lower()).astype(float),
        )
        traces = responses[["response_key", "prediction"]].rename(columns={"prediction": "trace"})
        traces["trace"] = traces.trace.astype(str)
        # The shared builder numbers repeated subject-item attempts after canonical item deduplication.
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response"]],
            "traces": traces,
        }


if __name__ == "__main__":
    FewshotTTTBBH(__file__).main_from_args()
