#!/usr/bin/env python3
"""Curate ERBench's released binary observations and their matching original logs."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ERBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('release')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Concatenate native CSV tables, retaining every field and row position.
        paths = sorted((self.raw_dir / parameters['paths']['results']).glob('*/crafted_df/*.csv'))
        frames = []
        for path in paths:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False).rename(columns={'Unnamed: 0': ''})
            frame['csv_record'] = frame.to_dict('records')
            frames.append(frame.assign(source_file=str(path.relative_to(self.raw_dir)),
                source_row=range(len(frame)), subject_key=path.parent.parent.name, domain=path.stem))
        records = pd.concat(frames, ignore_index=True).rename(columns={'question': 'csv_question'})
        if set(records.subject_key) != set(parameters['models']):
            raise ValueError('Review a missing or unexpected source model folder')
        if not records.model_answer.isin(['yes', 'no', 'unsure']).all() or not records.gold_answer.isin(['yes', 'no']).all():
            raise ValueError('Missing or unknown answers cannot be interpreted as failures')

        # 2. Select the verified source condition and load complete log entries.
        is_cot = (records.subject_key + '/' + records.domain).isin(parameters['cot_results'])
        records['prompting'] = is_cot.astype(str).map(parameters['prompting'])
        records['log_file'] = parameters['paths']['results'] + '/' + records.subject_key + '/' + records.domain + is_cot.map({False: '', True: '_cot'}) + '.log'
        logs = pd.DataFrame({'log_file': records.log_file.drop_duplicates().tolist()})
        logs['text'] = [(self.raw_dir / path).read_text() for path in logs.log_file]
        entries = logs.text.str.extractall(parameters['patterns']['entry']).reset_index(level=1, drop=True)
        entries = entries.join(logs[['log_file']]).reset_index(drop=True)
        entries['source_row'] = entries.groupby('log_file', sort=False).cumcount()
        entries = entries.join(entries.log_entry.str.extract(parameters['patterns']['question']))
        entries = entries.join(entries.log_entry.str.extract(parameters['patterns']['gold']))
        if len(entries) != len(records) or entries[['content', 'log_gold']].isna().any().any():
            raise ValueError('Every released CSV observation requires one complete original log entry')
        records = records.merge(entries, on=['log_file', 'source_row'], how='left', validate='one_to_one', suffixes=('', '_log'))
        if not (records.entity_idx.eq(records.entity_idx_log) & records.question_idx.eq(records.question_idx_log)
                & records.gold_answer.eq(records.log_gold) & records.csv_question.eq(' ' + records.content.str.split(':').str[0])).all():
            raise ValueError('The released CSV and original log disagree on the task or gold answer')

        # 3. Preserve full questions and grading; identical content/criteria share an item.
        items = records.drop_duplicates(['content', 'gold_answer']).copy()
        items['item_key'] = range(len(items))
        items['raw_item_id'] = items.domain + ':' + items.entity_idx + ':' + items.question_idx
        items['features'] = [dict(domain=row.domain) for row in items.itertuples(index=False)]
        items['grading_criterion'] = [dict(reference_answer=row.gold_answer, rule=self.grading['rule'])
                                     for row in items.itertuples(index=False)]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['answer'], sort_keys=True))
        records = records.merge(items[['content', 'gold_answer', 'item_key']],
            on=['content', 'gold_answer'], how='left', validate='many_to_one')

        # 4. Keep literal source identities and conditions, then translate native grades.
        subjects = records[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(harness='ERBench', source_model=row.subject_key) for row in subjects.itertuples(index=False)]
        records['response_key'] = records.source_file + '/' + records.source_row.astype(str)
        records['response'] = records.model_answer.eq(records.gold_answer).astype(float)
        records['test_condition'] = [json.dumps(dict(task='binary', domain=row.domain, prompting=row.prompting), sort_keys=True)
                                     for row in records.itertuples(index=False)]

        # 5. Keep complete original log text and parsed records, with their source links.
        traces = records[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            csv_record=row.csv_record, log_file=row.log_file, log_entry=row.log_entry), ensure_ascii=False)
            for row in records.itertuples(index=False)]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=records[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    ERBench(__file__).main_from_args()
