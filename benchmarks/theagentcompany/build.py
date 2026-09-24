"""Join TheAgentCompany's native checkpoint scores, task definitions and full trajectories."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class TheAgentCompany(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("runs", "tasks")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read the released score records for the named agent configurations.
        layout = self.build_parameters["layout"]
        runs = self.raw_dir / layout["runs"]
        subjects = pd.DataFrame({
            "raw_label": self.build_parameters["models"],
            "harness": self.build_parameters["harnesses"],
        }).rename_axis("subject_key").reset_index()
        subjects["features"] = subjects[["harness"]].to_dict("records")
        paths = sorted(path for run in subjects.subject_key for path in (runs / run / "results").glob("eval_*.json"))
        scores = pd.json_normalize([json.loads(path.read_text()) for path in paths])
        responses = pd.DataFrame({
            "subject_key": [path.parent.parent.name for path in paths],
            "item_key": [path.stem.removeprefix("eval_").removesuffix("-image") for path in paths],
            "response_key": [str(path.relative_to(runs)) for path in paths],
            "earned": pd.to_numeric(scores["final_score.result"]),
            "available": pd.to_numeric(scores["final_score.total"]),
        })
        if (not np.isfinite(responses[["earned", "available"]]).all().all()
                or not responses.available.gt(0).all()
                or not responses.earned.between(0, responses.available).all()):
            raise ValueError("TheAgentCompany checkpoint credits must be finite and satisfy 0 <= earned <= available > 0")
        if responses.duplicated(["subject_key", "item_key"]).any():
            raise ValueError("Duplicate TheAgentCompany task score within a run")
        responses = responses.assign(response=responses.earned / responses.available, trial=1)

        # 2. Load native trajectories verbatim; state_*.json files are auxiliary snapshots.
        paths = sorted(path for run in subjects.subject_key for path in (runs / run / "trajectories").glob("traj_*"))
        text = []
        for path in paths:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    text.append(handle.read())
            else:
                text.append(path.read_text())
        traces = pd.DataFrame({
            "subject_key": [path.parent.parent.name for path in paths],
            "item_key": [path.name for path in paths], "trace": text,
            "source_file": [str(path.relative_to(runs)) for path in paths],
        })
        traces["item_key"] = traces.item_key.str.removeprefix("traj_").str.replace(
            r"\.(json(?:\.gz)?|txt)$", "", regex=True,
        ).str.removesuffix("-image")
        # One released trajectory has identical compressed and uncompressed copies.
        traces = traces.drop_duplicates(["subject_key", "item_key", "trace"])
        if traces.duplicated(["subject_key", "item_key"]).any():
            raise ValueError("Conflicting trajectory copies for one TheAgentCompany attempt")
        # A released trajectory without a score is an ungraded attempt, not a failure.
        responses = responses.merge(
            traces, on=["subject_key", "item_key"], how="outer", validate="one_to_one",
        )
        responses = responses.assign(response_key=responses.response_key.fillna(responses.source_file), trial=1)

        # 3. Attach the complete task prompt and its captured grading implementation.
        items = responses[["item_key"]].drop_duplicates().copy()
        task_root = self.raw_dir / layout["tasks"]
        items = items.assign(
            raw_item_id=items.item_key,
            content=items.item_key.map(lambda task: (task_root / task / "task.md").read_text().strip()),
            grading_criterion=[{"rule": self.grading["rule"]} for _ in items.index],
            verifier=items.item_key.map(lambda task: ExactMatcher(spec=json.dumps({
                **self.grading["verifiers"]["checkpoint"],
                "grading_code": (task_root / task / "evaluator.py").read_text(),
            }, sort_keys=True))),
        )
        return {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial"]],
            "traces": responses.loc[responses.trace.notna(), ["response_key", "trace"]],
        }


if __name__ == "__main__":
    TheAgentCompany(__file__).main_from_args()
