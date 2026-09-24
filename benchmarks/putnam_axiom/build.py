#!/usr/bin/env python3
"""Join Putnam-AXIOM's human rubric grades to its full canonical problem statements."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class PutnamAxiom(BenchmarkBuild):

    def download(self):
        return self.fetch_sources("items", "grades")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load the problem bank and expand the released human-grade struct.
        bank = pd.read_parquet(self.raw_dir / "full_eval.parquet")
        runs = pd.read_parquet(self.raw_dir / "grading.parquet")
        dimensions = self.build_parameters["rubric_dimensions"]
        scores = pd.json_normalize(runs.human_grade).reindex(columns=dimensions).set_axis(runs.index)
        runs = runs.drop(columns="human_grade").join(scores)

        # 2. Recover full statements only when source excerpts and reference solutions agree.
        runs = runs.merge(bank[["id", "problem", "solution"]], left_on="problem_id", right_on="id",
                          how="left", sort=False, validate="many_to_one", indicator=True)
        if not runs._merge.eq("both").all() or not runs.ground_truth_solution.eq(runs.solution).all():
            raise ValueError("Putnam-AXIOM grades and canonical reference solutions disagree")
        if not all(full.startswith(excerpt.removesuffix("...")) for full, excerpt in zip(runs.problem, runs.problem_statement)):
            raise ValueError("A released problem excerpt does not match the canonical statement")

        # 3. Melt the five rubric fields; each criterion is a distinct grading protocol.
        responses = runs.melt(id_vars=["problem_id", "student_model", "model_output", "problem", "solution"],
                               value_vars=list(dimensions), var_name="rubric", value_name="response")
        responses["rubric"] = responses.rubric.map(dimensions)
        items = responses[["problem_id", "problem", "solution", "rubric"]].drop_duplicates().reset_index(drop=True)
        specs = self.grading["verifiers"]
        items = items.assign(
            item_key=items.index, raw_item_id=items.problem_id + ":" + items.rubric, content=items.problem,
            grading_criterion=[{"reference_answer": answer, "rule": specs[rubric]["rule"]}
                               for answer, rubric in zip(items.solution, items.rubric)],
            verifier=items.rubric.map(lambda rubric: Judge(spec=json.dumps(specs[rubric], sort_keys=True), judged_by="human")),
        )

        # 4. Link each score and its complete solution; omit the redundant score_total sum.
        responses = responses.merge(items[["problem_id", "rubric", "item_key"]],
                                     on=["problem_id", "rubric"], how="left", sort=False, validate="many_to_one")
        responses = responses.assign(response_key=responses.index, subject_key=responses.student_model,
                                     test_condition="rubric=" + responses.rubric)
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        traces = responses[["response_key", "model_output"]].rename(columns={"model_output": "trace"})
        return {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition"]],
            "traces": traces.loc[traces.trace.map(lambda value: isinstance(value, str) and bool(value.strip()))],
        }


if __name__ == "__main__":
    PutnamAxiom(__file__).main_from_args()
