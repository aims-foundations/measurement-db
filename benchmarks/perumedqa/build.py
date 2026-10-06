#!/usr/bin/env python3
"""Curate PeruMedQA's original question bank and published model outputs."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class PeruMedQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('release')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read native tables and verify their preserved question-bank indices.
        bank = pd.read_csv(self.raw_dir / parameters['paths']['bank'], keep_default_na=False)
        frames = []
        for path in sorted((self.raw_dir / parameters['paths']['results']).rglob('*.parquet')):
            frame = pd.read_parquet(path)
            if not frame[bank.columns].reset_index(drop=True).equals(bank.loc[frame.index].reset_index(drop=True)):
                raise ValueError('A native result does not match its original question-bank row')
            frame['native_record'] = frame.to_dict('records')
            frames.append(frame.assign(source_path=str(path.relative_to(self.raw_dir)),
                source_row=range(len(frame)), source_dataset_row=frame.index))
        records = pd.concat(frames, ignore_index=True).rename(columns={'model_basename': 'subject_key', 'question': 'content'})
        if records[['content', 'answer_llm', 'correct_answer', 'subject_key']].isna().any().any():
            raise ValueError('Missing native fields require review; empty decoded strings remain valid attempts')
        if not records.correct_answer.isin(list('ABCDE')).all() or set(records.subject_key) != set(parameters['model_identifiers']):
            raise ValueError('Review missing or unexpected source models or reference answers')

        # 2. Keep complete formatted questions and their original grading protocol.
        items = records.drop_duplicates(['content', 'correct_answer']).copy()
        items['item_key'] = range(len(items))
        items['raw_item_id'] = 'row_' + items.source_dataset_row.astype(str)
        items['features'] = [dict(language=parameters['labels']['language']) for _ in range(len(items))]
        items['grading_criterion'] = [dict(reference_answer=row.correct_answer, rule=self.grading['rule'])
                                     for row in items.itertuples(index=False)]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['answer'], sort_keys=True))
        records = records.merge(items[['content', 'correct_answer', 'item_key']],
            on=['content', 'correct_answer'], how='left', validate='many_to_one')

        # 3. Retain source identities and identify the fine-tuned model's year pools.
        subjects = records[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(harness='PeruMedQA', source_model=row.subject_key,
            declared_model=parameters['model_identifiers'][row.subject_key],
            inference_sources=parameters['inference_sources'][row.subject_key]) for row in subjects.itertuples(index=False)]
        records['test_condition'] = [json.dumps(dict(exam=row.source_file, year=row.year, source_folder=row.source_folder,
            fine_tuning_partition=(parameters['labels']['held_out'] if str(row.year) == parameters['labels']['held_out_year']
                else parameters['labels']['development']) if row.subject_key == parameters['labels']['fine_tuned_model'] else None), sort_keys=True)
            for row in records.itertuples(index=False)]

        # 4. Follow the authors' first-match parser; unparseable attempts stay ungraded.
        predicted = records.answer_llm.str.extract(parameters['patterns']['answer'], expand=False)
        records['response'] = predicted.eq(records.correct_answer).astype(float).where(predicted.notna())
        records['response_key'] = records.source_path + '/' + records.source_row.astype(str)

        # 5. Preserve every original field, full output, and both source row positions.
        traces = records[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_path, source_row=row.source_row,
            source_dataset_row=row.source_dataset_row, native_record=row.native_record), ensure_ascii=False)
            for row in records.itertuples(index=False)]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=records[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    PeruMedQA(__file__).main_from_args()
