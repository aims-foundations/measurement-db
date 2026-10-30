#!/usr/bin/env python3
"""Join MathConstraint's frozen instances to released per-model solver verdicts."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


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
            frame['source_record'] = run['results']
            frame['source_row'] = frame.index
            frame['source_file'] = str(path.relative_to(self.raw_dir))
            frame['run_metadata'] = [{key: value for key, value in run.items() if key != 'results'}] * len(frame)
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
        # Some native instance names repeat exactly the same question. Group
        # their shared stimulus without changing or discarding the run labels.
        items['testlet_id'] = items.prompt.str.normalize('NFC').str.strip().map(lambda text: hashlib.sha256(text.encode()).hexdigest())
        responses = observations.merge(items[["name", "item_key", "testlet_id"]], on="name", how="left", sort=False, validate="many_to_one")
        responses['response_key'] = responses.source_file + ':' + responses.source_row.astype(str)
        responses['response'] = responses.correct.astype(float)
        responses['trial'] = responses.groupby(['subject_key', 'testlet_id', 'test_condition'], sort=False).cumcount() + 1
        responses['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            source_record=row.source_record, run_metadata=row.run_metadata), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        subjects = responses[["subject_key"]].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        tables = {
            "subjects": subjects,
            "items": items[["item_key", "raw_item_id", "testlet_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "test_condition", "trial"]],
            "traces": responses[["response_key", "trace"]],
        }
        # The native solver records two judgments. UNSAT has no witness to
        # check: its vacuous solution_correct=True is not a second success.
        outcomes = []
        for row in responses.itertuples():
            for check in ['satisfiability_correct'] + (['solution_correct'] if row.is_satisfiable else []):
                grade = getattr(row, check)
                if type(grade) is not bool:
                    raise ValueError('A MathConstraint component is not an explicit native Boolean')
                outcomes.append(dict(parent_response_key=row.response_key, component_id=check, response=int(grade)))
        definitions = []
        for row in items.itertuples():
            for check in ['satisfiability_correct'] + (['solution_correct'] if row.gold != 'UNSATISFIABLE' else []):
                component = self.grading['verifiers']['components'][check]
                reference = row.gold if check == 'solution_correct' else ('UNSATISFIABLE' if row.gold == 'UNSATISFIABLE' else 'SATISFIABLE')
                definitions.append(dict(parent_item_key=row.item_key, component_id=check,
                    grading_criterion=dict(reference_answer=reference, rule=component['rule']),
                    verifier=ExactMatcher(spec=json.dumps(dict(self.grading['verifiers']['solver'],
                        selected_field=check, component_rule=component['rule']), sort_keys=True))))
        return component_tables(tables, pd.DataFrame(definitions), pd.DataFrame(outcomes))


if __name__ == "__main__":
    MathConstraintBuild(__file__).main_from_args()
