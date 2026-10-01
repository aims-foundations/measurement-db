"""Tabulate model-attributed OASST1 replies and their original human preference ranks."""

import gzip
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class OASST(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        source_file = parameters['layout']['messages']

        # 1. Load complete native records; select attributed model replies, not rank tiers.
        with gzip.open(self.raw_dir / source_file, 'rt') as stream:
            records = pd.Series([json.loads(line) for line in stream], dtype=object)
        messages = pd.json_normalize(records, max_level=0)
        messages['native_record'] = records
        messages['source_row'] = messages.index
        if messages.message_id.duplicated().any():
            raise ValueError('The source contains duplicate message identities')
        responses = messages.loc[messages.role.eq('assistant') & messages.synthetic.eq(True)
            & messages.model_name.notna() & messages.model_name.ne('')].copy()
        responses['response_key'] = responses.message_id
        responses['subject_key'] = responses.model_name
        responses['response'] = responses['rank']

        # 2. Join each reply to its complete released input; this cohort uses root prompts.
        prompts = messages[['message_id', 'parent_id', 'role', 'text', 'native_record']].rename(columns={
            'message_id': 'prompt_id', 'parent_id': 'ancestor_id', 'role': 'prompt_role',
            'text': 'content', 'native_record': 'native_prompt'})
        responses = responses.merge(prompts, left_on='parent_id', right_on='prompt_id',
            how='left', validate='many_to_one')
        if responses.prompt_id.isna().any() or responses.ancestor_id.notna().any() or not responses.prompt_role.eq('prompter').all():
            raise ValueError('A reply lacks the complete root-prompt context required by this release')

        # 3. Retain literal generation configurations and one copy of each distinct prompt.
        subjects = responses[['subject_key']].drop_duplicates().copy()
        subjects['raw_label'] = subjects.subject_key.str.split(',', n=1).str[0]
        subjects['features'] = [dict(harness=parameters['labels']['harness'],
            source_model_configuration=json.dumps([part.split('=', 1) for part in name.split(',')]),
            configuration_scope=parameters['labels']['configuration_scope'])
            for name in subjects.subject_key]
        items = responses[['prompt_id', 'content']].drop_duplicates('content').copy()
        items['item_key'] = items.prompt_id
        items['raw_item_id'] = items.prompt_id
        items['features'] = [dict(input_scope=parameters['labels']['input_scope'])] * len(items)
        items['grading_criterion'] = [dict(rule=self.grading['rule'])] * len(items)
        items['verifier'] = [Judge(judged_by='human', spec=json.dumps(
            self.grading['verifiers']['preference'], sort_keys=True))] * len(items)
        responses = responses.merge(items[['content', 'item_key']], on='content', validate='many_to_one')

        # 4. Preserve the comparison group and all released candidate identities.
        peers = messages.loc[messages.parent_id.isin(responses.parent_id), ['parent_id', 'message_id', 'model_name']]
        peers = peers.sort_values('message_id')
        peers = peers.astype(object).where(peers.notna(), None)
        peers['candidate'] = peers[['message_id', 'model_name']].to_dict('records')
        groups = peers.groupby('parent_id', sort=False).candidate.agg(list).rename('candidates').reset_index()
        responses = responses.merge(groups, on='parent_id', validate='many_to_one')
        responses['test_condition'] = 'comparison_group=' + responses.parent_id
        responses['interactors'] = responses.candidates.map(lambda values: json.dumps(dict(reply_candidates=values), allow_nan=False))

        # 5. Link every graded or ungraded attempt to its full original input and output.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=source_file, source_row=int(row.source_row),
            native_record=row.native_record, native_prompt=row.native_prompt), allow_nan=False)
            for row in responses.itertuples()]
        return {'subjects': subjects,
            'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': responses[['response_key', 'subject_key', 'item_key', 'response', 'test_condition', 'interactors']],
            'traces': traces}


if __name__ == '__main__':
    OASST(__file__).main_from_args()
