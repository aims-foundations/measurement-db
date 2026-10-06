#!/usr/bin/env python3
"""Join MathConstraint's frozen instances to released per-model solver verdicts."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MathConstraintBuild(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("benchmark")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Read frozen problem definitions and reconcile identical bank overlaps.
        paths = sorted(self.raw_dir.glob("*_instances/*.json"))
        problems = pd.json_normalize([json.loads(path.read_text()) for path in paths], max_level=0)
        problems["name"] = problems.name.fillna(pd.Series([path.stem for path in paths]))
        problems = problems.loc[problems.prompt.map(lambda value: isinstance(value, str) and bool(value))].copy()
        problems["gold"] = problems.solution.map(lambda solution: json.dumps(solution, sort_keys=True) if solution is not None else "SATISFIABLE")
        problems.loc[~problems.satisfiable.astype(bool), "gold"] = "UNSATISFIABLE"
        definitions = problems[["name", "prompt", "gold"]].drop_duplicates()
        if definitions.name.duplicated().any():
            raise ValueError("A MathConstraint problem name has conflicting definitions")

        # 2. Flatten each model/split/condition result file into an observation table.
        frames = []
        for path in sorted((self.raw_dir / "results").glob("*/*/*.json")):
            if path.parent.name not in self.build_parameters["splits"] or path.stem not in self.build_parameters["conditions"]:
                continue
            run = json.loads(path.read_text())
            frame = pd.json_normalize(run["results"], max_level=0)
            condition = "split=" + path.parent.name + ";tools=" + self.build_parameters["conditions"][path.stem]
            frames.append(frame.assign(subject_key=run.get("model") or path.parent.parent.name, test_condition=condition))
        observations = pd.concat(frames, ignore_index=True).dropna(subset="correct")
        if not observations.correct.isin([False, True, 0, 1]).all():
            raise ValueError("MathConstraint correct fields must be binary")
        observations = observations.merge(definitions, left_on="problem_name", right_on="name",
                                           how="inner", sort=False, validate="many_to_one")

        # 3. Construct items and link recorded verdicts without rerunning the solver.
        items = observations[["name", "prompt", "gold"]].drop_duplicates().reset_index(drop=True)
        spec = json.dumps(self.grading["verifiers"]["solver"], sort_keys=True)
        items = items.assign(
            item_key=items.index, raw_item_id=items.name, content=items.prompt,
            grading_criterion=items.gold.map(lambda answer: {"reference_answer": answer, "rule": self.grading["rule"]}),
            verifier=ExactMatcher(spec=spec),
        )
        responses = observations.merge(items[["name", "item_key"]], on="name", how="left", sort=False, validate="many_to_one")
        # The release has explicit chain-of-thought text for some models and an answer for others.
        reasoning = responses.cot_trace.fillna("")
        responses = responses.assign(response_key=responses.index, response=responses.correct.astype(float),
                                     trace=responses.cot_trace.where(reasoning.astype(bool), responses.raw_response))
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        has_trace = responses.trace.map(lambda value: isinstance(value, str))
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": responses.loc[has_trace, ["response_key", "trace"]],
        }


if __name__ == "__main__":
    MathConstraintBuild(__file__).main_from_args()
