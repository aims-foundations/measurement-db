#!/usr/bin/env python3
"""Tabulate PRISM's model outputs, original human ratings and selected conversations."""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class PRISM(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout, labels = parameters['layout'], parameters['labels']

        # 1. Read the original tables; preserve native output records and source flags.
        utterances = pd.read_json(self.raw_dir / layout['utterances'], lines=True, convert_dates=False, dtype=False)
        utterances['native_record'] = utterances.to_dict('records')
        conversations = pd.read_json(self.raw_dir / layout['conversations'], lines=True, convert_dates=False, dtype=False)
        models = pd.read_json(self.raw_dir / layout['models'], lines=True, convert_dates=False, dtype=False)
        metadata = pd.json_normalize([json.loads(line) for line in
            (self.raw_dir / layout['metadata']).read_text().splitlines()], max_level=0)
        metadata = metadata.astype(object).where(metadata.notna(), None)
        metadata['native_metadata'] = metadata.to_dict('records')
        utterances = utterances.merge(metadata.loc[metadata.column_id.eq('model_response'), ['utterance_id', 'native_metadata']],
            on='utterance_id', how='left', validate='one_to_one')
        if utterances.native_metadata.isna().any():
            raise ValueError('Every original utterance needs its corresponding source metadata record')

        # 2. Build each turn's history from the selected branch, not the alternatives.
        expanded = conversations[['conversation_id', 'conversation_history']].explode('conversation_history', ignore_index=True)
        messages = pd.json_normalize(expanded.conversation_history.tolist(), max_level=0)
        messages['conversation_id'] = expanded.conversation_id
        users = messages.loc[messages.role.eq('user'), ['conversation_id', 'turn', 'content']].rename(columns={'content': 'user_prompt'})
        selected = messages.loc[messages.role.eq('model') & messages.if_chosen.eq(True), ['conversation_id', 'turn', 'content']].drop_duplicates()
        turns = users.merge(selected, on=['conversation_id', 'turn'], how='left', validate='one_to_one').sort_values(['conversation_id', 'turn'])
        if turns.content.isna().any():
            raise ValueError('A conversation turn has no unambiguous selected response')
        turns['exchange'] = [[dict(role='user', content=row.user_prompt), dict(role='assistant', content=row.content)] for row in turns.itertuples()]
        turns['history'] = turns.groupby('conversation_id').exchange.transform(lambda rows: rows.cumsum())
        turns['input_content'] = turns.history.map(lambda history: json.dumps(history[:-1], ensure_ascii=False))
        utterances = utterances.merge(turns[['conversation_id', 'turn', 'user_prompt', 'input_content']],
            on=['conversation_id', 'turn', 'user_prompt'], how='left', validate='many_to_one')
        if utterances.input_content.isna().any():
            raise ValueError('An utterance is missing its original user prompt and conversation history')

        # 3. Subjects are generating models; humans remain anonymous response raters.
        subjects = utterances[['model_name', 'model_provider']].drop_duplicates().merge(models,
            left_on='model_name', right_on='long_name', how='left', validate='one_to_one', suffixes=('', '_configuration'))
        if subjects.long_name.isna().any():
            raise ValueError('A generating model is missing its published configuration')
        subjects['subject_key'] = subjects.model_name
        subjects['raw_label'] = subjects.model_name
        subjects['features'] = [dict(harness=labels['harness'], source_model_name=row.model_name,
            source_model_provider=row.model_provider, documented_header=row.header,
            documented_generation_settings=json.dumps(row.selected_params, sort_keys=True)) for row in subjects.itertuples()]

        # 4. Unpivot the two measures without renormalizing scores or inventing choices.
        if not utterances.score.between(1, 100).all() or not utterances.score.mod(1).eq(0).all():
            raise ValueError('PRISM ratings must be finite integers from 1 through 100')
        if not utterances.if_chosen.map(lambda value: isinstance(value, bool)).all():
            raise ValueError('PRISM chosen flags must be original booleans')
        responses = utterances.melt(id_vars=[column for column in utterances if column not in self.grading['verifiers']],
            value_vars=list(self.grading['verifiers']), var_name='dimension', value_name='response')
        responses['response'] = responses.response.astype(float)
        responses['response_key'] = responses.utterance_id + ':' + responses.dimension
        responses['subject_key'] = responses.model_name
        responses['item_key'] = responses.response_key
        responses['test_condition'] = labels['test_condition_prefix'] + responses.utterance_id + ';measure=' + responses.dimension
        responses['interactors'] = responses.user_id.map(lambda user: json.dumps(dict(human_rater=user)))

        # 5. Keep measurement-specific grading and complete original outcome provenance.
        items = responses.rename(columns={'input_content': 'content'}).copy()
        items['raw_item_id'] = items.conversation_id + ':turn_' + items.turn.astype(str) + ':' + items.dimension
        items['features'] = [dict(dimension=row.dimension, input_scope=labels['input_scope']) for row in items.itertuples()]
        items['grading_criterion'] = items.dimension.map(lambda dimension: dict(rule=self.grading['verifiers'][dimension]['rule'],
            response_scale=self.grading['verifiers'][dimension]['response_scale']))
        items['verifier'] = items.dimension.map(lambda dimension: Judge(judge=labels['judge'], judged_by='human',
            spec=json.dumps(self.grading['verifiers'][dimension], sort_keys=True)))
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=layout['utterances'], dimension=row.dimension,
            utterance=row.native_record, metadata=row.native_metadata), ensure_ascii=False, allow_nan=False) for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition', 'interactors']], traces=traces)


if __name__ == '__main__':
    PRISM(__file__).main_from_args()
