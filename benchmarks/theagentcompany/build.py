"""Join TheAgentCompany's native checkpoint scores, task definitions and full trajectories."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


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
        evaluations = [json.loads(path.read_text()) for path in paths]
        scores = pd.json_normalize(evaluations)
        responses = pd.DataFrame({
            "subject_key": [path.parent.parent.name for path in paths],
            "item_key": [path.stem.removeprefix("eval_").removesuffix("-image") for path in paths],
            "response_key": [str(path.relative_to(runs)) for path in paths],
            "earned": pd.to_numeric(scores["final_score.result"]),
            "available": pd.to_numeric(scores["final_score.total"]),
            "evaluation": evaluations,
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
            testlet_id="1.0.0/" + items.item_key,
            content=items.item_key.map(lambda task: (task_root / task / "task.md").read_text().strip()),
            grading_criterion=[{"rule": self.grading["rule"]} for _ in items.index],
            verifier=items.item_key.map(lambda task: ExactMatcher(spec=json.dumps({
                **self.grading["verifiers"]["checkpoint"],
                "grading_code": (task_root / task / "evaluator.py").read_text(),
            }, sort_keys=True))),
        )
        # 4. Keep each released checkpoint, including its native partial-credit scale.
        components, outcomes = {}, []
        for row in responses.itertuples():
            if not isinstance(row.evaluation, dict):
                parent = items.loc[items.item_key.eq(row.item_key)].iloc[0]
                component = "ungraded_task_outcome"
                components[row.item_key, component] = dict(parent_item_key=row.item_key,
                    component_id=component, grading_criterion=parent.grading_criterion, verifier=parent.verifier)
                outcomes.append(dict(parent_response_key=row.response_key, component_id=component, response=None))
                continue
            checkpoints = row.evaluation.get("checkpoints")
            if not isinstance(checkpoints, list) or not checkpoints:
                raise ValueError("A scored task has no released checkpoint results")
            for index, checkpoint in enumerate(checkpoints, 1):
                total, earned = checkpoint["total"], checkpoint["result"]
                if (type(total) is not int or type(earned) not in (int, bool)
                        or not 0 <= earned <= total):
                    raise ValueError("Checkpoint credits must be nonnegative integer counts")
                if total == 0:
                    continue  # An empty rubric contributes no scored item.
                component = f"checkpoint:{index}:out-of:{total}"
                key = row.item_key, component
                components[key] = dict(parent_item_key=row.item_key, component_id=component,
                    grading_criterion={"rule": json.dumps(dict(rule=self.grading["verifiers"]["checkpoint"]["checkpoint_rule"],
                        checkpoint_index=index, available_credit=total), sort_keys=True)},
                    verifier=ExactMatcher(spec=json.dumps(dict(
                        **self.grading["verifiers"]["checkpoint"], checkpoint_index=index,
                        available_credit=total,
                        grading_code=(task_root / row.item_key / "evaluator.py").read_text()), sort_keys=True)))
                outcomes.append(dict(parent_response_key=row.response_key,
                    component_id=component, response=earned / total))
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(evaluation=row.evaluation if isinstance(row.evaluation, dict) else None,
            trajectory=row.trace if isinstance(row.trace, str) else None), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        tables = {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "testlet_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial"]],
            "traces": traces,
        }
        return component_tables(tables, pd.DataFrame(components.values()), pd.DataFrame(outcomes))


if __name__ == "__main__":
    TheAgentCompany(__file__).main_from_args()
