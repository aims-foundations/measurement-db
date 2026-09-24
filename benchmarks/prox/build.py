"""Normalize the ProX release's two checkpoints and nine math evaluation tasks."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ProX(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the released JSONL files directly into tables, retaining their task/checkpoint keys.
        models = self.build_parameters["models"]
        paths = sorted(path for checkpoint in models for path in (self.raw_dir / checkpoint).glob("*.jsonl"))
        runs = pd.concat([
            pd.read_json(path, lines=True, dtype=False, convert_dates=False, precise_float=True).assign(
                subject_key=path.parent.name, task=path.stem, source_file=str(path.relative_to(self.raw_dir)),
                source_row=lambda frame: frame.index,
            ) for path in paths
        ], ignore_index=True)
        if runs[["question", "gt"]].isna().any().any() or not runs.question.str.strip().ne("").all():
            raise ValueError("A ProX observation lacks its question or reference")
        # Parallel lists describe separate completions; explode them together to preserve alignment.
        responses = runs.explode(["code", "pred", "score"], ignore_index=True)
        if not responses.score.map(lambda value: isinstance(value, bool)).all():
            raise TypeError("ProX scores must be the released boolean grading verdicts")

        # 2. Keep the checkpoint and evaluation configuration in the subject table.
        subjects = pd.Series(models, name="raw_label").rename_axis("subject_key").reset_index()
        subjects["features"] = subjects.subject_key.map(lambda checkpoint: {
            **self.build_parameters["subject_features"], "training_tokens": checkpoint,
        })
        items = runs[["task", "idx", "question", "gt"]].drop_duplicates().copy()
        items = items.assign(
            item_key=items.task + ":" + items.idx.astype(str), raw_item_id=items.task + ":" + items.idx.astype(str),
            content=items.question.str.strip(),
            grading_criterion=items["gt"].map(lambda answer: {
                "reference_answer": str(answer) if str(answer).strip() else None, "rule": self.grading["rule"],
            }),
            verifier=ExactMatcher(spec=json.dumps(self.grading["verifiers"]["math_equal"], sort_keys=True)),
            features=items.task.map(lambda task: {"task": task}),
        )
        if items.item_key.duplicated().any():
            raise ValueError("ProX checkpoints disagree on a task's question or reference")

        # 3. Retain every native verdict and complete model output, including repeated questions.
        responses = responses.assign(
            response_key=responses.index, item_key=responses.task + ":" + responses.idx.astype(str),
            response=responses.score.astype(float), test_condition=self.build_parameters["conditions"]["test_condition"],
        )
        traces = responses[["response_key", "code"]].rename(columns={"code": "trace"})
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier", "features"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces,
        }


if __name__ == "__main__":
    ProX(__file__).main_from_args()
