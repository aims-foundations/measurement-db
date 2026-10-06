#!/usr/bin/env python3
"""Curate the original Real-POCQi questions, answers, and physician preferences."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class RealPOCQi(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Read the three native tables, retaining original rows and their positions.
        tables = {}
        for name in ['questions', 'answers', 'ratings']:
            table = pd.read_parquet(self.raw_dir / layout['release'] / layout[name])
            table['native_' + name] = table.astype(object).where(table.notna(), None).to_dict('records')
            table[name + '_row'] = table.index
            tables[name] = table
        questions, answers, ratings = (tables[name] for name in ['questions', 'answers', 'ratings'])
        if questions.question_id.duplicated().any() or answers.duplicated(['question_id', 'provider_key']).any():
            raise ValueError('Question and generated-answer keys must be unique')
        if not ratings.choice.dropna().isin(parameters['slot_a_scores']).all():
            raise ValueError('Unknown native preference label; do not replace it with a tie')
        if not ratings.axis.isin(parameters['axes']).all() or ratings.slot_a_provider.eq(ratings.slot_b_provider).any():
            raise ValueError('Unknown rating dimension or a self-comparison')

        # 2. Associate each observed query with its distinct human grading dimension.
        items = ratings[['question_id', 'axis']].drop_duplicates().merge(questions,
            on='question_id', how='left', validate='many_to_one').reset_index(drop=True)
        if items.question_text.isna().any():
            raise ValueError('A rated question has no released input text')
        items['item_key'] = items.index
        items['raw_item_id'] = items.question_id + ':' + items.axis
        items['content'] = items.question_text
        items['features'] = [dict(question_id=row.question_id, axis=row.axis, specialty=row.specialty) for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=self.grading['rule'] + ' Rating dimension: ' + parameters['axes'][axis] + '.')
            for axis in items.axis]
        items['verifier'] = [Judge(spec=json.dumps(dict(**self.grading['verifiers']['physician'], axis=axis),
            sort_keys=True), judged_by='human') for axis in items.axis]

        # 3. Melt each physician rating into its two coupled subject perspectives.
        ratings['a_credit'] = ratings.choice.map({key: float(value) for key, value in parameters['slot_a_scores'].items()})
        responses = ratings.melt(id_vars=ratings.columns.difference(['slot_a_provider', 'slot_b_provider'], sort=False),
            value_vars=['slot_a_provider', 'slot_b_provider'],
            var_name='slot', value_name='subject_key')
        responses = responses.merge(ratings[['ratings_row', 'slot_a_provider', 'slot_b_provider']],
            on='ratings_row', how='left', validate='many_to_one')
        responses['opponent'] = responses.slot_b_provider.where(responses.slot.eq('slot_a_provider'), responses.slot_a_provider)
        responses['response'] = responses.a_credit.where(responses.slot.eq('slot_a_provider'), 1 - responses.a_credit)
        responses['response_key'] = responses.index
        responses = responses.merge(items[['question_id', 'axis', 'item_key']],
            on=['question_id', 'axis'], how='left', validate='many_to_one')

        # 4. Join both full answers by their source keys, with no stripping or clipping.
        responses = responses.merge(answers[['question_id', 'provider_key', 'native_answers', 'answers_row']],
            left_on=['question_id', 'subject_key'], right_on=['question_id', 'provider_key'], how='left', validate='many_to_one')
        opponents = answers[['question_id', 'provider_key', 'native_answers', 'answers_row']].rename(
            columns=dict(provider_key='opponent', native_answers='opponent_answer', answers_row='opponent_answer_row'))
        responses = responses.merge(opponents, on=['question_id', 'opponent'], how='left', validate='many_to_one')
        if responses.answers_row.isna().any() or responses.opponent_answer_row.isna().any():
            raise ValueError('A rated system or opponent has no released answer')
        subjects = responses[['subject_key']].drop_duplicates()
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.subject_key
        subjects['features'] = [dict(source_provider=key, harness=parameters['labels']['harness']) for key in subjects.subject_key]
        responses['test_condition'] = [json.dumps(dict(source_row=int(row.ratings_row), axis=row.axis,
            render_mode=row.render_mode, slot=row.slot), sort_keys=True) for row in responses.itertuples()]
        responses['interactors'] = 'opponent=' + responses.opponent

        # 5. Preserve the original five-level votes and complete answer associations.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['release'] + '/' + layout['ratings'],
            source_row=int(row.ratings_row), slot=row.slot, native_rating=row.native_ratings,
            answer_row=int(row.answers_row), native_answer=row.native_answers,
            opponent_answer_row=int(row.opponent_answer_row), opponent_answer=row.opponent_answer),
            ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition', 'interactors']], traces=traces)


if __name__ == '__main__':
    RealPOCQi(__file__).main_from_args()
