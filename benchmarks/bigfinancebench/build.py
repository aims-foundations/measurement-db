#!/usr/bin/env python3
"""Curate BigFinanceBench's recorded agent runs and independent judge assessments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class BigFinanceBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources("release", "harness")

    def build_tables(self) -> dict[str, pd.DataFrame]:
        paths = self.build_parameters["paths"]
        grading = self.grading["verifiers"]
        release = self.raw_dir / paths["release"]

        # 1. Read the native tables, retaining complete records and their locations.
        tables = {}
        for name, pattern in self.build_parameters["tables"].items():
            frames = []
            for path in sorted(release.glob(pattern)):
                frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False, precise_float=True)
                frame = frame.astype(object).where(frame.notna(), None)
                frame["record"] = frame.to_dict("records")
                frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)), source_row=frame.index))
            tables[name] = pd.concat(frames, ignore_index=True)
        tasks, runs, grades = tables["tasks"], tables["runs"], tables["grades"]
        keys = ["model", "question_id", "trial_idx"]
        if tasks.id.duplicated().any() or grades.duplicated(keys + ["judge"]).any():
            raise ValueError("Duplicate native question or judge assessment")

        # 2. Associate assessments with all matching traces; keep ambiguous retries.
        runs["content"] = [json.dumps(dict(system_prompt=row.system_prompt, question=row.question,
            tool_specs=row.tool_specs), ensure_ascii=False, sort_keys=True) for row in runs.itertuples()]
        runs["source_record"] = runs[["source_file", "source_row", "record"]].to_dict("records")
        if runs.groupby(keys).content.nunique().gt(1).any():
            raise ValueError("Retried trial has conflicting initial inputs")
        grouped = runs.groupby(keys, sort=False, as_index=False).agg(
            content=("content", "first"), run_records=("source_record", list))
        observations = grades.rename(columns={"record": "grade_record"}).merge(
            grouped, on=keys, how="left", validate="many_to_one")
        if observations.run_records.isna().any():
            raise ValueError("A published assessment has no released trace")
        observations["matching_run_indices"] = [[index for index, run in enumerate(records)
            if run["record"]["final_answer"] == answer]
            for records, answer in zip(observations.run_records, observations.final_answer)]
        if observations.matching_run_indices.str.len().eq(0).any():
            raise ValueError("No trace agrees with the final answer recorded by a judge")
        observations["trace_association"] = observations.matching_run_indices.str.len().eq(1).map(
            {True: "unique_final_answer_match", False: "ambiguous_final_answer_match"})
        observations = observations.merge(tasks[["id", "record"]].rename(
            columns={"id": "question_id", "record": "task_record"}), on="question_id", how="left", validate="many_to_one")
        if observations.task_record.isna().any() or not observations.reference_answer.eq(
                observations.task_record.str["reference_answer"]).all():
            raise ValueError("The item bank and published reference answers disagree")

        # 3. Preserve recorded model identities without guessing missing inference settings.
        subjects = runs.loc[runs.resolved_model.notna(), ["model", "resolved_model", "harness_version"]].drop_duplicates()
        if subjects.model.duplicated().any() or not set(runs.model).issubset(subjects.model):
            raise ValueError("A model's recorded harness configuration is absent or inconsistent")
        subjects["subject_key"] = subjects.model
        subjects["raw_label"] = subjects.model
        subjects["features"] = [dict(harness="BigFinanceBench", model_identifier=row.model,
            resolved_model=row.resolved_model, harness_version=row.harness_version) for row in subjects.itertuples()]

        # 4. One attempt is one recorded agent execution, shared by its judges.
        # Three assessments match multiple retries; keep those associations unknown.
        executions = {}
        original_tasks = tasks.set_index('id').record.to_dict()
        for row in runs.itertuples():
            key = row.source_file + ':' + str(row.source_row)
            executions[key] = dict(attempt_key=key, raw_attempt_id=key, subject_key=row.model,
                evidence=dict(source_file=row.source_file, source_row=row.source_row,
                    record=row.record, task_record=original_tasks[row.question_id],
                    assessments=[], ambiguous_assessments=[]))
        definitions, responses, traces = {}, [], []
        for row in observations.itertuples():
            assessment = dict(source_file=row.source_file, source_row=row.source_row, record=row.grade_record)
            matching = [row.run_records[index] for index in row.matching_run_indices]
            attempt_keys = [run['source_file'] + ':' + str(run['source_row']) for run in matching]
            unique = len(attempt_keys) == 1
            for key in attempt_keys:
                executions[key]['evidence']['assessments' if unique else 'ambiguous_assessments'].append(assessment)
            checks = [('final_answer_correct', row.grade_record['final_answer_correct'],
                dict(metric='final_answer_correct', rule=grading['measures']['final_answer_correct']['rule']))]
            for index, line in enumerate(row.grade_record['rubric_lines']):
                if type(line['earned']) is not bool:
                    raise ValueError('A rubric line lacks an explicit Boolean verdict')
                checks.append(('rubric:' + str(index), line['earned'],
                    dict(metric='rubric_line', criterion=line['text'], points=line['points'],
                        rule='Recorded earned flag for this rubric line; points are its original aggregation weight')))
            points = sum(line['points'] for line in row.grade_record['rubric_lines'])
            earned = sum(line['points'] for line in row.grade_record['rubric_lines'] if line['earned'])
            if (points, earned) != (row.rubric_points_possible, row.rubric_points_earned):
                raise ValueError('Rubric components do not reproduce the published point totals')
            for name, grade, rule in checks:
                if type(grade) is not bool:
                    raise ValueError('A native assessment lacks its Boolean verdict')
                criterion = dict(reference_answer=row.task_record['reference_answer'],
                    rule=json.dumps(rule, ensure_ascii=False, sort_keys=True))
                item = json.dumps([row.question_id, row.judge, row.content, name, criterion], ensure_ascii=False, sort_keys=True)
                definitions[item] = dict(item_key=item, raw_item_id=row.question_id + '::' + name,
                    testlet_id=row.question_id, content=row.content,
                    features=dict(source_question_id=row.question_id, evaluation_only=row.task_record['evaluation_only'],
                        do_not_train=row.task_record['do_not_train'], benchmark_canary=row.task_record['benchmark_canary']),
                    grading_criterion=criterion, verifier=Judge(judge=row.judge, judged_by='llm',
                        spec=json.dumps(grading['protocol'], ensure_ascii=False, sort_keys=True)))
                key = row.source_file + ':' + str(row.source_row) + ':' + name
                responses.append(dict(response_key=key, subject_key=row.model, item_key=item,
                    response=float(grade), trial=int(row.trial_idx) + 1,
                    test_condition='judge=' + row.judge, attempt_key=attempt_keys[0] if unique else None))
                if not unique:
                    traces.append(dict(response_key=key, trace=json.dumps(dict(assessment=assessment,
                        matching_run_records=matching, trace_association='ambiguous_final_answer_match'),
                        ensure_ascii=False, allow_nan=False)))
        attempts = []
        for execution in executions.values():
            evidence = execution.pop('evidence')
            attempts.append(dict(**execution, trace=json.dumps(evidence, ensure_ascii=False, allow_nan=False)))
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=pd.DataFrame(definitions.values()), responses=pd.DataFrame(responses),
            attempts=pd.DataFrame(attempts), traces=pd.DataFrame(traces, columns=['response_key', 'trace']))



if __name__ == "__main__":
    BigFinanceBench(__file__).main_from_args()
