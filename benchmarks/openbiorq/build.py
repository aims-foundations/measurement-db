"""Join OpenBioRQ's questions, rubrics, predictions and checklist verdicts."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge
from measurement_db.scripts.curate_benchmarks.testlet_tables import component_tables


class OpenBioRQ(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        # 1. Load the question bank and join each question's frozen grading rubric.
        layout = self.build_parameters["layout"]
        questions = pd.read_json(self.raw_dir / layout["tasks"], lines=True, dtype=False, convert_dates=False, precise_float=True)
        rubrics = pd.read_json(self.raw_dir / layout["rubrics"], lines=True, dtype=False, convert_dates=False, precise_float=True)
        items = questions.merge(rubrics, on="task_id", how="outer", validate="one_to_one", indicator=True)
        if not items._merge.eq("both").all() or not items.self_contained_question.eq(items.question).all():
            raise ValueError("OpenBioRQ questions and grading rubrics do not match")
        items = items.assign(
            item_key=items.task_id, raw_item_id=items.task_id, testlet_id=items.task_id,
            content=items.self_contained_question,
            grading_criterion=items.apply(lambda row: {
                "rule": json.dumps({"aggregation": self.grading["rule"], "criteria": row.criteria}, sort_keys=True),
                "reference_answer": json.dumps(row.gold_answer, ensure_ascii=False, sort_keys=True),
            }, axis=1),
            verifier=Judge(spec=json.dumps(self.grading["verifiers"]["checklist"], sort_keys=True), judged_by="llm"),
        )

        # 2. Distinguish tool-enabled and no-tool configurations of the same base model.
        subjects = pd.Series(self.build_parameters["models"], name="raw_label").rename_axis("subject_key").reset_index()
        subjects["features"] = subjects.subject_key.map(lambda model: {
            **self.build_parameters["subject_features"], "tool_mode": self.build_parameters["tool_access"][model],
        })
        paths = [self.raw_dir / layout["results"] / model for model in subjects.subject_key]
        predictions = pd.concat([
            pd.read_json(path / "predictions.jsonl", lines=True, dtype=False, convert_dates=False, precise_float=True).assign(
                subject_key=path.name, source_record=(path / "predictions.jsonl").read_text().splitlines(),
            ) for path in paths
        ], ignore_index=True)
        verdicts = pd.concat([
            pd.read_json(path / "checklist.jsonl", lines=True, dtype=False, convert_dates=False, precise_float=True).assign(subject_key=path.name)
            for path in paths
        ], ignore_index=True)

        # 3. Keep all recorded attempts; missing checklist scores remain null.
        responses = predictions.merge(
            verdicts, on=["subject_key", "task_id"], how="outer", validate="one_to_one", indicator=True,
        )
        if not responses._merge.eq("both").all() or not responses.task_id.isin(items.item_key).all():
            raise ValueError("An OpenBioRQ prediction, verdict or question has no counterpart")
        scores = pd.to_numeric(responses.checklist_score)
        observed = scores.dropna()
        if not np.isfinite(observed).all() or not observed.between(0, 1).all():
            raise ValueError("OpenBioRQ checklist scores must lie in [0, 1]")
        responses = responses.assign(
            item_key=responses.task_id, response_key=responses.subject_key + "::" + responses.task_id,
            response=scores.ge(self.grading["verifiers"]["checklist"]["success_threshold"]).astype(float).where(scores.notna()),
            trial=1, test_condition=self.build_parameters["conditions"]["test_condition"],
        )

        # 4. Retain each complete native prediction record, including messages and tool responses.
        traces = responses[["response_key"]].copy()
        traces["trace"] = [json.dumps(dict(prediction=json.loads(row.source_record),
            checklist_score=None if pd.isna(row.checklist_score) else row.checklist_score,
            verdicts=row.verdicts if isinstance(row.verdicts, list) else []),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]

        # 5. Preserve individual checklist judgments instead of thresholding their average.
        protocol = self.grading["verifiers"]["checklist"]
        definitions = []
        for row in items.itertuples():
            for criterion in row.criteria:
                definitions.append(dict(parent_item_key=row.item_key, component_id=str(criterion["id"]),
                    grading_criterion=dict(reference_answer=json.dumps(row.gold_answer, ensure_ascii=False, sort_keys=True),
                        rule=json.dumps(dict(rule=protocol["criterion_rule"], criterion=criterion), ensure_ascii=False, sort_keys=True)),
                    verifier=Judge(judged_by="llm", spec=json.dumps(dict(protocol,
                        selected_criterion_id=criterion["id"]), sort_keys=True))))
        outcomes = []
        criteria_by_task = items.set_index('item_key').criteria.to_dict()
        for row in responses.itertuples():
            verdicts = {verdict['id']: protocol['verdict_values'][verdict['v']]
                for verdict in row.verdicts} if isinstance(row.verdicts, list) else {}
            if not set(verdicts) <= {criterion['id'] for criterion in criteria_by_task[row.item_key]}:
                raise ValueError('A released verdict has no matching rubric criterion')
            for criterion in criteria_by_task[row.item_key]:
                outcomes.append(dict(parent_response_key=row.response_key, component_id=str(criterion['id']),
                    response=verdicts.get(criterion['id'])))
        tables = {
            "subjects": subjects[["subject_key", "raw_label", "features"]],
            "items": items[["item_key", "raw_item_id", "testlet_id", "content", "grading_criterion", "verifier"]],
            "responses": responses[["response_key", "subject_key", "item_key", "response", "trial", "test_condition"]],
            "traces": traces,
        }
        return component_tables(tables, pd.DataFrame(definitions), pd.DataFrame(outcomes))


if __name__ == "__main__":
    OpenBioRQ(__file__).main_from_args()
