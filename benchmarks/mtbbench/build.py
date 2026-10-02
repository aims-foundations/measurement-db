#!/usr/bin/env python3
"""Curate MTBBench's released case questions, patient files, and agent judgments."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class MTBBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Expand the ordered case banks; retain only information already available.
        frames = []
        for cohort, filename in parameters['questions'].items():
            bank = pd.read_json(self.raw_dir / layout['tasks'] / filename, typ='series', convert_axes=False)
            cases = bank.rename('events').rename_axis('case_id').reset_index()
            events = cases.explode('events', ignore_index=True)
            events['event_index'] = events.groupby('case_id', sort=False).cumcount()
            events['case_events'] = events.events.map(lambda event: [{key: value for key, value in event.items() if key != 'answer'}])
            events['case_events'] = events.groupby('case_id', sort=False).case_events.transform(lambda series: series.cumsum())
            events = events.join(pd.json_normalize(events.events, max_level=0)).assign(cohort=cohort)
            events = events.loc[events.question.notna()].copy()
            events['question_index'] = events.groupby('case_id', sort=False).cumcount()
            frames.append(events)
        items = pd.concat(frames, ignore_index=True).assign(item_key=lambda table: table.index)
        items['raw_item_id'] = items.cohort + ':' + items.case_id + ':' + items.question_index.astype(str)
        items['content'] = [json.dumps(dict(cohort=row.cohort, case_id=row.case_id, case_events=row.case_events),
            ensure_ascii=False, allow_nan=False) for row in items.itertuples()]

        # 2. Link exact source assets without exposing files from later case stages.
        items['asset_paths'] = items.case_events.map(lambda events: list(dict.fromkeys(
            path for event in events for path in event.get('file_paths', []))))
        files = items[['asset_paths']].explode('asset_paths').dropna().drop_duplicates('asset_paths').set_index('asset_paths')
        files['data'] = [ (self.raw_dir / layout['tasks'] / path).read_bytes() for path in files.index ]
        files['media_type'] = [parameters['media_types'][Path(path).suffix] for path in files.index]
        assets = files.to_dict('index')
        items['attachments'] = [[dict(path=path, role='source', **assets[path]) for path in paths] for paths in items.asset_paths]

        # 3. Concatenate native question records and associate the full case session.
        frames = []
        for path in sorted((self.raw_dir / layout['release']).glob(layout['logs'])):
            records = json.loads(path.read_text())
            sessions = [row['conversation'] for row in records if 'conversation' in row]
            if len(sessions) != 1:
                raise ValueError('Each released case must have exactly one conversation')
            table = pd.json_normalize(records, max_level=0)
            table['native_record'] = records
            table['source_row'] = table.index
            table = table.loc[table.question.notna()].copy()
            table['question_index'] = range(len(table))
            table['conversation'] = [sessions[0]] * len(table)
            table['system_prompt'] = json.dumps([message for message in sessions[0] if message['role'] == 'system'],
                ensure_ascii=False, sort_keys=True)
            frames.append(table.assign(source_file=str(path.relative_to(self.raw_dir)), model=path.parent.name,
                cohort=path.parents[1].name.removeprefix('agent_logs_'), case_id=path.name.split('_chatlog_')[0]))
        responses = pd.concat(frames, ignore_index=True).assign(response_key=lambda table: table.index)
        responses = responses.merge(items[['cohort', 'case_id', 'question_index', 'question', 'answer', 'item_key']],
            on=['cohort', 'case_id', 'question_index', 'question', 'answer'], how='left', validate='many_to_one')
        if responses.item_key.isna().any():
            raise ValueError('A recorded question or answer differs from the frozen case bank')
        if not responses.correct.dropna().map(lambda value: type(value) is bool).all():
            raise ValueError('Published grades must be boolean or unavailable')
        responses['response'] = responses.correct.map({True: 1., False: 0.})
        responses['test_condition'] = 'source_file=' + responses.source_file

        # 4. Distinguish the recorded model and system protocol without guessing settings.
        subjects = responses[['model', 'system_prompt']].drop_duplicates().reset_index(drop=True)
        subjects['subject_key'] = subjects.index
        subjects['raw_label'] = parameters['labels']['subject_prefix'] + subjects.model
        subjects['features'] = [dict(source_model=row.model, system_prompt=row.system_prompt,
            harness=parameters['labels']['harness']) for row in subjects.itertuples()]
        responses = responses.merge(subjects[['model', 'system_prompt', 'subject_key']],
            on=['model', 'system_prompt'], how='left', validate='many_to_one')
        items['features'] = [dict(cohort=row.cohort, case_id=row.case_id, question_index=str(row.question_index),
            event_index=str(row.event_index)) for row in items.itertuples()]
        items['grading_criterion'] = [dict(reference_answer=answer, rule=self.grading['rule']) for answer in items.answer]
        items['verifier'] = ExactMatcher(spec=json.dumps(self.grading['verifiers']['published'], sort_keys=True))

        # 5. Preserve complete native traces, including the upstream-mutated final session.
        traces = responses[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, source_row=int(row.source_row),
            native_record=row.native_record, conversation=row.conversation), ensure_ascii=False, allow_nan=False)
            for row in responses.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']], items=items[['item_key', 'raw_item_id', 'content', 'features',
            'attachments', 'grading_criterion', 'verifier']], responses=responses[['response_key', 'subject_key',
            'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    MTBBench(__file__).main_from_args()
