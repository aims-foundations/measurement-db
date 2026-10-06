#!/usr/bin/env python3
"""Tabulate released MORQA evaluator ratings with their complete source inputs."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class MORQA(BenchmarkBuild):
    def download(self):
        return self.fetch_sources(*(source['name'] for source in self.source_manifest['upstream']))

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read released result and annotation tables, retaining complete native records.
        result_tables, rating_tables = [], []
        for collection, path in parameters['result_files'].items():
            lang, dataset, split = collection.split('/')
            rating_path = parameters['rating_files'][collection]
            results = pd.read_json(self.raw_dir / path)
            results['native_record'] = json.loads((self.raw_dir / path).read_text())
            results['query_id'] = results.post_id.astype(str)
            results['record_key'] = path + '/' + results.index.astype(str)
            results = results.assign(source_file=path, source_row=range(len(results)),
                lang=lang, dataset=dataset, split=split)
            result_tables.append(results)
            ratings = pd.read_json(self.raw_dir / rating_path)
            ratings['native_annotation'] = json.loads((self.raw_dir / rating_path).read_text())
            ratings['query_id'] = ratings[parameters['id_fields'][dataset]].astype(str)
            rating_tables.append(ratings.assign(annotation_file=rating_path, annotation_row=range(len(ratings)),
                lang=lang, dataset=dataset, split=split))
        records = pd.concat(result_tables, ignore_index=True)
        annotations = pd.concat(rating_tables, ignore_index=True)

        # 2. Join the original question text by dataset, split, language and source ID.
        question_tables = []
        for collection, path in parameters['question_files'].items():
            dataset, split = collection.split('/')
            questions = pd.read_json(self.raw_dir / path)
            questions['query_id'] = questions[parameters['id_fields'][dataset]].astype(str)
            if split != 'all':
                questions['split'] = split
            for lang in ['en', 'zh']:
                fields = ['query_title_' + lang, 'query_content_' + lang]
                if not set(fields).issubset(questions.columns):
                    continue
                table = questions[['query_id', 'split', *fields]].rename(columns=dict(zip(fields, ['query_title','query_content'])))
                question_tables.append(table.assign(dataset=dataset, lang=lang, question_file=path))
        questions = pd.concat(question_tables, ignore_index=True)
        keys = ['dataset', 'split', 'lang', 'query_id']
        records = records.merge(questions, on=keys, how='left', validate='many_to_one')
        if records[['query_title', 'query_content', 'question_file']].isna().any().any():
            raise ValueError('A released evaluator input has no complete matching question')

        # 3. Associate each candidate with every original human metric and rater, without averaging.
        annotations = annotations.rename(columns={'author_id': 'author_id_candidate'})
        annotation_keys = keys + ['author_id_candidate']
        if annotations.duplicated(annotation_keys + ['metric', 'author_metric']).any():
            raise ValueError('Repeated human annotation identity requires review')
        joined = records[['record_key', 'candidate', *annotation_keys]].merge(annotations,
            on=annotation_keys, how='left', validate='one_to_many')
        if joined.annotation_file.isna().any() or not joined.candidate.eq(joined.system_input).all():
            raise ValueError('Human annotations do not match the exact released candidate text')
        joined['annotation'] = [dict(source_file=row.annotation_file, source_row=row.annotation_row,
            native_record=row.native_annotation) for row in joined.itertuples(index=False)]
        records = records.merge(joined.groupby('record_key', sort=False).annotation.agg(list).rename('human_annotations'),
            left_on='record_key', right_index=True, how='left', validate='one_to_one')

        # 4. Form content-and-grading items with full query, candidate and released references.
        items = records.copy()
        items['item_key'] = items.record_key
        items['raw_item_id'] = items.lang + ':' + items.dataset + ':' + items.split + ':' + items.query_id + ':' + items.author_id_candidate
        items['content'] = [json.dumps(dict(instruction=parameters['instructions'][row.lang], query_title=row.query_title,
            query_content=row.query_content, candidate=row.candidate, references=row.responses), ensure_ascii=False)
            for row in items.itertuples(index=False)]
        items['features'] = [dict(lang=row.lang, dataset=row.dataset, split=row.split, query_id=row.query_id,
            candidate_author=row.author_id_candidate, question_file=row.question_file) for row in items.itertuples(index=False)]
        items['grading_criterion'] = [dict(rule=self.grading['rule'], reference_answer=json.dumps(values, ensure_ascii=False))
            for values in items.human_annotations]
        items['verifier'] = Judge(spec=json.dumps(self.grading['verifiers']['published'], sort_keys=True), judged_by='llm')

        # 5. Melt evaluator columns; preserve exact values and complete linked source records.
        observations = records.melt(id_vars=['record_key', 'source_file', 'source_row', 'native_record'],
            value_vars=[key for key in parameters['evaluators'] if key in records], var_name='subject_key', value_name='response')
        observations = observations.loc[[row.subject_key in row.native_record for row in observations.itertuples(index=False)]].copy()
        observations['response'] = pd.to_numeric(observations.response, errors='raise')
        observations['item_key'] = observations.record_key
        observations['response_key'] = observations.record_key + '/' + observations.subject_key
        observations['test_condition'] = 'source_file=' + observations.source_file
        subjects = observations[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(harness=parameters['labels']['harness'], source_evaluator=key) for key in subjects.subject_key]
        traces = observations[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=row.source_row,
            evaluator=row.subject_key, native_record=row.native_record), ensure_ascii=False) for row in observations.itertuples(index=False)]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=observations[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    MORQA(__file__).main_from_args()
