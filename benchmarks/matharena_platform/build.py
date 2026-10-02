"""Tabulate complete MathArena research-mathematics attempts and native grades."""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class MathArenaPlatform(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate complete original rows, keeping source coordinates and native values.
        frames = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['outputs'])):
            native = pq.read_table(path).to_pylist()
            frame = pd.json_normalize(native, max_level=0)
            frames.append(frame.assign(native_record=native, competition=path.parents[1].name,
                source_file=str(path.relative_to(self.raw_dir)), source_row=range(len(frame))))
        records = pd.concat(frames, ignore_index=True)
        records['problem_idx'] = records.problem_idx.astype(str)
        records['protocol'] = records.competition.map(parameters['versions'])
        if records.protocol.isna().any():
            raise ValueError('A released version has no declared grading protocol')
        if records.duplicated(['competition', 'model_config', 'problem_idx', 'idx_answer']).any():
            raise ValueError('Duplicate native model/configuration/problem/attempt key')

        # 2. Join the released task bank; it supplies references, never model predictions.
        banks = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['tasks'])):
            bank = pd.read_parquet(path, dtype_backend='pyarrow').astype(object)
            banks.append(bank.assign(competition=path.parents[1].name))
        bank = pd.concat(banks, ignore_index=True).astype(object).where(lambda table: table.notna(), None)
        bank['problem_idx'] = bank.problem_idx.astype(str)
        bank = bank.rename(columns={'problem': 'bank_problem', 'sample_solution': 'reference_solution', 'points': 'bank_points'})
        records = records.merge(bank[['competition', 'problem_idx', 'bank_problem', 'reference_solution', 'bank_points']],
            on=['competition', 'problem_idx'], how='left', validate='many_to_one')
        if not records.problem.eq(records.bank_problem).all():
            raise ValueError('A recorded request no longer matches its pinned task definition')
        has_points = records.max_points_judge_1.notna()
        if not records.loc[has_points, 'max_points_judge_1'].eq(records.loc[has_points, 'bank_points']).all():
            raise ValueError('A released judge uses a different maximum from the pinned task rubric')

        # 3. Keep recorded requests intact; mark missing-message fallbacks explicitly.
        has_request = records.user_message.map(lambda value: isinstance(value, str) and bool(value))
        records['content'] = records.user_message
        records['input_scope'] = parameters['labels']['recorded_input']
        records.loc[~has_request, 'content'] = [json.dumps(dict(problem=row.problem, formal_statement=row.formal_statement), ensure_ascii=False)
            for row in records.loc[~has_request].itertuples()]
        records.loc[~has_request, 'input_scope'] = parameters['labels']['fallback_input']
        records['reference'] = None
        final_answer = records.protocol.isin(['arxivmath', 'arxivmath_answer_judge'])
        records.loc[final_answer, 'reference'] = records.loc[final_answer, 'gold_answer']
        proof = records.protocol.eq('usamo')
        records.loc[proof, 'reference'] = records.loc[proof, 'reference_solution']
        records['reference'] = records.reference.map(lambda value: str(value) if pd.notna(value) and str(value).strip() else None)

        # 4. Distinguish published model configurations and full task/grading definitions.
        subjects = records[['model_name', 'model_config']].drop_duplicates().copy()
        subjects['subject_key'] = range(len(subjects))
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.model_name
        subjects['features'] = [dict(**parameters['subject_features'], source_model_label=row.model_name,
            source_model_config=row.model_config) for row in subjects.itertuples()]
        records = records.merge(subjects[['model_name', 'model_config', 'subject_key']],
            on=['model_name', 'model_config'], how='left', validate='many_to_one')
        records['rubric'] = records.native_record.map(lambda row: json.dumps([
            {field: part[field] for field in parameters['rubric_fields'] if field in part}
            for part in json.loads(row.get('grading_details_judge_1') or '[]')], sort_keys=True, ensure_ascii=False))
        definitions = ['competition', 'problem_idx', 'content', 'input_scope', 'reference', 'protocol', 'rubric']
        records['item_key'] = records.groupby(definitions, sort=False, dropna=False).ngroup()
        items = records.drop_duplicates('item_key').copy()
        items['raw_item_id'] = items.competition + '::' + items.problem_idx
        items['features'] = [dict(competition=row.competition, problem_idx=row.problem_idx, input_scope=row.input_scope)
            for row in items.itertuples()]
        specifications = self.grading['verifiers']
        items['scale'] = items.protocol.map({name: spec['response_scale'] for name, spec in specifications.items()})
        point_scores = items.protocol.isin(['brokenarxiv', 'usamo'])
        maximum = items.loc[point_scores, 'bank_points']
        if not (maximum.gt(0) & maximum.eq(maximum.astype(int))).all():
            raise ValueError('The native normalized rubric must declare a positive integer maximum')
        items.loc[point_scores, 'scale'] = pd.Series([dict(kind='discrete',
            values=[points / limit for points in range(int(limit) + 1)], direction='higher_is_better')
            for limit in maximum], index=maximum.index)
        items['grading_criterion'] = [dict(
            **({'reference_answer': row.reference} if pd.notna(row.reference) else {}),
            rule=json.dumps(dict(native_rule=specifications[row.protocol]['rule'], rubric=json.loads(row.rubric),
                **({'formal_statement': row.formal_statement} if row.protocol == 'arxivlean' else {})), ensure_ascii=False, sort_keys=True),
            response_scale=row.scale) for row in items.itertuples()]
        items['verifier'] = [Judge(judged_by='llm', spec=json.dumps(specifications[row.protocol], sort_keys=True))
            if specifications[row.protocol]['verifier_class'] == 'judge'
            else ExactMatcher(spec=json.dumps(specifications[row.protocol], sort_keys=True)) for row in items.itertuples()]

        # 5. Preserve native grades and entire source records, including failed/empty attempts.
        records['response'] = pd.to_numeric(records.correct, errors='raise')
        records['response_key'] = records.source_file + '#' + records.source_row.astype(str)
        traces = records[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            record=row.native_record), ensure_ascii=False, allow_nan=False) for row in records.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': records[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    MathArenaPlatform(__file__).main_from_args()
