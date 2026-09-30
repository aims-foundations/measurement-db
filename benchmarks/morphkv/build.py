#!/usr/bin/env python3
"""Tabulate MorphKV's released LongGenBench outputs and constraint judgments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MorphKV(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('release')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read original tables and join each generation to its complete task definition.
        bank = pd.read_json(self.raw_dir / parameters['paths']['bank']).rename_axis('bank_row').reset_index()
        frames = []
        for path in sorted((self.raw_dir / parameters['paths']['results']).glob('*.json')):
            frame = pd.read_json(path)
            frame['native_record'] = frame.to_dict('records')
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)),
                source_row=range(len(frame)), subject_key=path.stem))
        records = pd.concat(frames, ignore_index=True)
        records['record_key'] = records.source_file + '/' + records.source_row.astype(str)
        records = records.merge(bank, left_on='input', right_on='prompt', how='left', validate='many_to_one', suffixes=('', '_bank'))
        if records.bank_row.isna().any() or set(records.subject_key) != set(parameters['runs']):
            raise ValueError('Review missing task definitions or unexpected source configurations')
        for field in ['type', 'number', 'checks_once', 'checks_range', 'checks_periodic']:
            if not records[field].eq(records[field + '_bank']).all():
                raise ValueError('A released task differs from its original bank definition')

        # 2. Expand constraints and retain those with an output block, as in the evaluator.
        checks = records.melt(id_vars='record_key', value_vars=['checks_once', 'checks_range', 'checks_periodic'],
            var_name='kind', value_name='checks')
        checks['kind'] = checks.kind.str.removeprefix('checks_')
        checks = checks.assign(entry=checks.checks.map(dict.items).map(list)).explode('entry').dropna(subset=['entry']).reset_index(drop=True)
        checks[['block_id', 'requirement']] = pd.DataFrame(checks.entry.tolist(), index=checks.index)
        blocks = records[['record_key', 'type', 'output_blocks']].explode('output_blocks').reset_index(drop=True)
        block_tables = []
        for kind, pattern in parameters['block_patterns'].items():
            group = blocks.loc[blocks.type.eq(kind)].copy()
            group['block_id'] = group.output_blocks.str.extract(pattern, expand=False)
            block_tables.append(group.dropna(subset=['block_id']))
        blocks = pd.concat(block_tables).drop_duplicates(['record_key', 'block_id'])
        checks = checks.merge(blocks[['record_key', 'block_id']], on=['record_key', 'block_id'], validate='many_to_one')

        # 3. Join released judgments; missing judgments stay null rather than failures.
        verdicts = records.melt(id_vars='record_key', value_vars=['results_once', 'results_range', 'results_periodic'],
            var_name='kind', value_name='verdicts').dropna(subset=['verdicts'])
        verdicts['kind'] = verdicts.kind.str.removeprefix('results_')
        verdicts = verdicts.assign(entry=verdicts.verdicts.map(dict.items).map(list)).explode('entry').dropna(subset=['entry']).reset_index(drop=True)
        verdicts[['block_id', 'verdict']] = pd.DataFrame(verdicts.entry.tolist(), index=verdicts.index)
        observations = checks.merge(verdicts[['record_key', 'kind', 'block_id', 'verdict']],
            on=['record_key', 'kind', 'block_id'], how='left', validate='one_to_one')
        if not verdicts.verdict.isin(['yes', 'no']).all() or observations.verdict.notna().sum() != len(verdicts):
            raise ValueError('Unknown verdict or a judgment without a matching generated-block constraint')
        observations = observations.merge(records, on='record_key', how='left', validate='many_to_one')
        observations['response'] = observations.verdict.map({'yes': 1., 'no': 0.})

        # 4. Each item has a full generation prompt and one explicit grading constraint.
        observations['criterion'] = [json.dumps(dict(rule=self.grading['rule'], kind=row.kind,
            block_type=row.type, block_id=row.block_id, requirement=row.requirement), sort_keys=True)
            for row in observations.itertuples(index=False)]
        items = observations.drop_duplicates(['input', 'criterion']).copy()
        items['item_key'] = range(len(items))
        items['raw_item_id'] = 'bank_' + items.bank_row.astype(str) + ':' + items.kind + ':' + items.block_id
        items['content'] = items.input
        items['features'] = [dict(source_benchmark=parameters['labels']['benchmark'], task_type=row.type) for row in items.itertuples(index=False)]
        items['grading_criterion'] = [dict(rule=rule) for rule in items.criterion]
        items['verifier'] = Judge(spec=json.dumps(self.grading['verifiers']['published_judge'], sort_keys=True), judged_by='llm')
        observations = observations.merge(items[['input', 'criterion', 'item_key']], on=['input', 'criterion'], how='left', validate='many_to_one')

        # 5. Keep literal run identities and complete source records for every observation.
        subjects = records[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(harness=parameters['labels']['harness'], source_run=key) for key in subjects.subject_key]
        observations['response_key'] = observations.record_key + '/' + observations.kind + '/' + observations.block_id
        observations['test_condition'] = 'source_file=' + observations.source_file
        traces = observations[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row, bank_row=row.bank_row,
            kind=row.kind, block_id=row.block_id, native_record=row.native_record), ensure_ascii=False)
            for row in observations.itertuples(index=False)]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=observations[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    MorphKV(__file__).main_from_args()
