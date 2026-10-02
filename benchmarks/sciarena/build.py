"""Tabulate SciArena's recorded human preferences and released literature context."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class SciArena(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        layout = self.build_parameters['layout']

        # 1. Read the original votes and join their independently released input papers.
        records = json.loads((self.raw_dir / layout['votes']).read_bytes(),
            parse_constant=lambda value: {'nonfinite_float': value.lower()})
        battles = pd.json_normalize(records, max_level=0)
        battles['native_record'] = records
        battles['source_row'] = battles.index
        context = pd.json_normalize(json.loads((self.raw_dir / layout['paperbank']).read_bytes(),
            parse_constant=lambda value: {'nonfinite_float': value.lower()}), max_level=0)
        context = context.rename(columns={'question type': 'question_type'})
        if battles.id.isna().any() or context.id.isna().any() or set(battles.id) != set(context.id):
            raise ValueError('The two original releases must contain the same non-null vote IDs')
        shared = [column for column in records[0] if column != 'id']
        original = battles.set_index('id')[shared].sort_index()
        enriched = context.set_index('id')[shared].sort_index()
        if not original.equals(enriched):
            raise ValueError('The paper-bank release disagrees with the original recorded votes')
        battles = battles.merge(context[['id', 'paper_bank']], on='id', validate='one_to_one')

        # 2. Preserve the question and retrieved papers, keeping current replies out of inputs.
        items = battles[['id', 'question', 'paper_bank', 'question_type', 'subject']].copy()
        items['content'] = [json.dumps(dict(question=row.question, paper_bank=row.paper_bank),
            ensure_ascii=False, allow_nan=False) for row in items.itertuples()]
        items['features'] = [dict(question_type=row.question_type, discipline=row.subject,
            input_scope=self.build_parameters['labels']['input_scope']) for row in items.itertuples()]
        items = items.assign(item_key=items.id, raw_item_id=items.id,
            grading_criterion=[dict(rule=self.grading['rule'])] * len(items),
            verifier=Judge(spec=json.dumps(self.grading['verifiers']['human_vote'], sort_keys=True), judged_by='human'))

        # 3. Unpivot each recorded vote into its two model-side preference observations.
        responses = battles.melt(id_vars=['id', 'vote'], value_vars=['modelA', 'modelB'],
            var_name='side', value_name='subject_key')
        scores = pd.DataFrame(self.grading['verifiers']['human_vote']['scores']).rename_axis('vote').reset_index().melt(
            id_vars='vote', var_name='side', value_name='response')
        responses = responses.merge(scores, on=['vote', 'side'], how='left', validate='many_to_one')
        if responses.subject_key.isna().any() or responses.response.isna().any():
            raise ValueError('A native model label or vote has no declared interpretation')
        opponents = responses[['id', 'side', 'subject_key']].rename(columns={'subject_key': 'opponent'})
        opponents['side'] = opponents.side.map({'modelA': 'modelB', 'modelB': 'modelA'})
        responses = responses.merge(opponents, on=['id', 'side'], validate='one_to_one')
        responses = responses.assign(response_key=responses.id + '/' + responses.side,
            item_key=responses.id, test_condition='side=' + responses.side, interactors='opponent=' + responses.opponent)

        # 4. Retain literal model labels and complete original replies, citations and vote categories.
        subjects = responses[['subject_key']].drop_duplicates().assign(raw_label=lambda frame: frame.subject_key)
        subjects['features'] = subjects.subject_key.map(lambda model: dict(model_identifier=model, **self.build_parameters['subject']))
        traces = responses[['response_key', 'id', 'side']].merge(
            battles[['id', 'source_row', 'native_record']], on='id', validate='many_to_one')
        traces['trace'] = [json.dumps(dict(source_file=layout['votes'], source_row=row.source_row,
            paperbank_source=layout['paperbank'], side=row.side, record=row.native_record),
            ensure_ascii=False, allow_nan=False) for row in traces.itertuples()]
        return {'subjects': subjects[['subject_key', 'raw_label', 'features']],
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition', 'interactors']],
            'traces': traces[['response_key', 'trace']]}


if __name__ == '__main__':
    SciArena(__file__).main_from_args()
