#!/usr/bin/env python3
"""Tabulate the released Psychosis-bench conversations and their original ratings."""

import json
from pathlib import Path
import sys
import tarfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class PsychosisBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters

        # 1. Read the published log, scenario bank and grading implementation.
        with tarfile.open(self.raw_dir / parameters['layout']['archive']) as archive:
            source = {'/'.join(Path(member.name).parts[1:]): archive.extractfile(member).read().decode()
                      for member in archive if member.isfile()}
        cases = pd.json_normalize(json.loads(source[parameters['layout']['cases']])['cases'], max_level=0)
        experiments = pd.Series(source[parameters['layout']['log']]).str.extractall(
            r'(?ms)^## Experiment (?P<experiment>\d+): (?P<case_name>.*?) × (?P<model>[^\n]+)\n'
            r'(?P<body>.*?)(?=^## Experiment \d+: |\Z)').reset_index(drop=True)
        experiments['configuration'] = experiments.body.str.extract(r'(?ms)^## Metadata\n(.*?)(?:\n\n---\s*)?\Z')
        experiments['judge'] = experiments.configuration.str.extract(r'(?m)^- \*\*judge_model:\*\* (.+)$')
        experiments = experiments.merge(cases, left_on='case_name', right_on='name', validate='many_to_one')

        # 2. Expand the turns and check their association with the original prompts.
        turns = experiments.body.str.extractall(
            r'(?ms)^### Turn (?P<turn>\d+)\n(?P<body>.*?)(?=^### Turn \d+\n|^## Metadata\n|\Z)'
        ).reset_index().rename(columns={'level_0': 'experiment_row'})
        text = turns.body.str.extract(
            r'(?s)^\s*\*\*👤 User:\*\*\n(?P<user>.*?)\n\n\*\*🤖 Assistant:\*\*\n'
            r'(?P<assistant>.*?)(?:\n\n\*\*📊 Scores:\*\*\n(?P<score_text>.*?))?\n\n---\s*$')
        if text[['user', 'assistant']].isna().any().any():
            raise ValueError('A published conversation turn could not be parsed in full')
        turns = turns.drop(columns=['body', 'match']).join(text).merge(
            experiments.drop(columns='body'), left_on='experiment_row', right_index=True, validate='many_to_one')
        turns['turn'] = turns.turn.astype(int)
        turns['user'] = turns.user.str.strip()
        turns['assistant'] = turns.assistant.str.strip()
        if not (turns.user == [row.prompts[row.turn - 1] for row in turns.itertuples()]).all():
            raise ValueError('Published user messages differ from their scenario prompts')
        turns = turns.sort_values(['experiment_row', 'turn']).reset_index(drop=True)

        # 3. Accumulate each conversation, stopping before its current assistant answer.
        turns['exchange'] = [[dict(role='user', content=row.user), dict(role='assistant', content=row.assistant)]
                             for row in turns.itertuples()]
        turns['history'] = turns.groupby('experiment_row').exchange.transform(lambda rows: rows.cumsum())
        turns['content'] = turns.history.map(lambda messages: json.dumps(messages[:-1], ensure_ascii=False))
        turns['turn_key'] = turns.index
        subjects = turns.drop_duplicates('model')[['model']].rename(columns={'model': 'subject_key'})
        subjects['raw_label'] = subjects.subject_key
        subjects['features'] = [dict(harness=parameters['labels']['harness'],
            inference_api=parameters['labels']['inference_api'], model_alias=model) for model in subjects.subject_key]

        # 4. Join each applicable rubric to its released grade; absent ratings stay null.
        ratings = turns.score_text.fillna('').str.extractall(
            r'(?m)^- (?P<dimension>DCS|HES|SIS) \([^)]+\): (?P<response>-?\d+)$'
        ).reset_index().rename(columns={'level_0': 'turn_key'})
        ratings['response'] = ratings.response.astype(float)
        dimensions = pd.DataFrame.from_dict(self.grading['verifiers'], orient='index').rename_axis('dimension').reset_index()
        measured = turns.merge(dimensions, how='cross').query('turn >= start_turn').merge(
            ratings[['turn_key', 'dimension', 'response']], on=['turn_key', 'dimension'], how='left', validate='one_to_one')
        measured['response_key'] = measured.index
        measured['item_key'] = measured.response_key
        measured['subject_key'] = measured.model
        measured['interactors'] = json.dumps(dict(user_sim=parameters['labels']['user_sim']))

        # 5. Project canonical tables with complete context, grading scope and target text.
        items = measured.copy()
        items['raw_item_id'] = items.experiment + ':turn_' + items.turn.astype(str) + ':' + items.dimension
        items['features'] = items[['id', 'theme', 'condition', 'harm_type', 'turn', 'dimension']].to_dict('records')
        items['grading_criterion'] = [dict(rule=row.rule, response_scale=row.response_scale) for row in items.itertuples()]
        items['verifier'] = [Judge(spec=json.dumps(dict(**self.grading['verifiers'][row.dimension],
            harm_context=row.harm_type if row.dimension == 'HES' else None,
            implementation=source[parameters['layout']['scorer']]), ensure_ascii=False),
            judge=row.judge, judged_by='llm' if row.dimension != 'SIS' else None) for row in items.itertuples()]
        traces = measured[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(experiment=row.experiment, case_id=row.id, turn=row.turn,
            dimension=row.dimension, assistant=row.assistant, configuration=row.configuration,
            released_rating=None if pd.isna(row.response) else row.response), ensure_ascii=False)
            for row in measured.itertuples()]
        return dict(subjects=subjects, items=items[['item_key', 'raw_item_id', 'content', 'features',
            'grading_criterion', 'verifier']],
            responses=measured[['response_key', 'subject_key', 'item_key', 'response', 'interactors']], traces=traces)


if __name__ == '__main__':
    PsychosisBench(__file__).main_from_args()
