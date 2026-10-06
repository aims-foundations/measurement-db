"""Tabulate original monitor assessments, requests and complete recorded histories."""

import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, Judge


class ScaleMRT(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, grading = self.build_parameters, self.grading
        labels, patterns = parameters['labels'], parameters['patterns']

        # 1. Read each named native record; unmonitored trajectories are source inputs, not monitor attempts.
        files = []
        for path in sorted(self.raw_dir.glob(parameters['layout']['results'])):
            relative = path.relative_to(self.raw_dir)
            collection, task, condition, agent, monitor, filename = relative.parts[1:]
            files.append(dict(response_key=relative.as_posix(), collection=collection, task=task,
                condition=condition, agent=agent, monitor=monitor, native_record=json.loads(path.read_bytes())))
        files = pd.DataFrame(files)
        rows = pd.concat([files, pd.json_normalize(files.native_record, max_level=0)], axis=1)
        unmonitored = rows.monitoring_response.isna()
        if not rows.loc[unmonitored, 'monitor'].eq('unmonitored').all():
            raise ValueError('A named monitor has no attempt record')
        rows = rows.loc[~unmonitored].copy()
        rows['monitor'] = rows.monitor.str.replace(patterns['escaped_path'], lambda m: chr(int(m[1], 16)), regex=True)
        if not rows.monitoring_response.map(lambda value: isinstance(value, str)).all():
            raise ValueError('Unexpected native monitor output type')

        # 2. Recover the saved final request according to the native history format.
        requests = []
        for row in rows.loc[rows.monitor_conversation.notna()].itertuples():
            history = row.monitor_conversation
            if not history:
                raise ValueError('An explicitly saved monitor history is empty')
            if any(isinstance(message, list) for message in history):
                # The summary-based runner stores a system message and user-only chunks.
                request = [history[0], *history[-1]]
            else:
                if history[-1]['role'] != 'assistant' or history[-1]['content'] != row.monitoring_response:
                    raise ValueError('The saved final output differs from the monitor result')
                start = max(i for i, message in enumerate(history[:-1]) if message['role'] == 'system') if 'hierarchical' in row.condition else 0
                request = history[start:-1]
            if request[0]['role'] != 'system' or request[-1]['role'] != 'user':
                raise ValueError('Unexpected native final-request boundary')
            requests.append(dict(response_key=row.response_key, request=request, system=request[0]['content']))
        requests = pd.DataFrame(requests)
        requests['content'] = requests.request.map(lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False))
        requests['input_scope'] = labels['recorded_scope']

        # 3. For missing requests, retain the available trajectory and explicitly mark the incomplete input.
        missing = rows.loc[~rows.response_key.isin(requests.response_key), ['response_key', 'condition', 'conversation']]
        messages = missing.explode('conversation', ignore_index=True).rename(columns={'conversation': 'native_message'})
        messages = pd.concat([messages, pd.json_normalize(messages.native_message, max_level=0)], axis=1)
        messages = messages.loc[messages.role.ne('system')].copy()
        # Match the upstream Python regex semantics, including Unicode whitespace.
        messages['content'] = (messages.content.fillna('').astype(object).str.replace(patterns['hidden'], '', regex=True)
            .str.replace(patterns['whitespace'], ' ', regex=True).str.replace('<visible_thinking>', '<model_thinking>', regex=False)
            .str.replace('</visible_thinking>', '</model_thinking>', regex=False).str.strip())
        after_assistant = messages.role.eq('assistant').groupby(messages.response_key).cummax()
        messages = messages.loc[~messages.condition.str.contains('unaware') | after_assistant].copy()
        messages['message'] = [dict(role=row.role, content=row.content, tool_calls=row.native_message.get('tool_calls'))
                               for row in messages.itertuples()]
        available = messages.groupby('response_key', sort=False).message.agg(list).rename('request').reset_index()
        available['content'] = available.request.map(lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False))
        available['input_scope'] = labels['partial_scope']
        rows = rows.merge(pd.concat([requests, available], ignore_index=True), on='response_key', how='left', validate='one_to_one')
        if rows.content.isna().any():
            raise ValueError('A monitoring result lacks even the available trajectory')

        # 4. Keep literal monitor/condition configurations and distinct native rating protocols.
        configurations = rows[['monitor', 'condition']].drop_duplicates()
        configurations['subject_key'] = configurations.apply(lambda row: json.dumps(row.to_list()), axis=1)
        subjects = configurations[['subject_key', 'monitor']].rename(columns={'monitor': 'raw_label'})
        subjects['features'] = [dict(harness=labels['harness'], recorded_model=row.monitor,
            recorded_condition=row.condition, historical_settings=labels['historical_settings']) for row in configurations.itertuples()]
        rows = rows.merge(configurations, on=['monitor', 'condition'], validate='many_to_one')
        rows['scale'] = rows.condition.str.contains('likert').map({True: 'one_to_five', False: 'zero_to_hundred'})
        recorded = rows.system.notna()
        for scale, marker in parameters['scale_markers'].items():
            rows.loc[recorded & rows.system.str.contains(marker, regex=False, na=False), 'scale'] = scale
        if not rows.loc[recorded, 'system'].map(lambda text: sum(marker in text for marker in parameters['scale_markers'].values()) == 1).all():
            raise ValueError('A recorded monitor prompt has an unknown rating scale')
        rows['grading_criterion'] = rows.scale.map({name: json.dumps(dict(rule=grading['rule'], response_scale=spec['response_scale']), sort_keys=True)
                                                   for name, spec in grading['verifiers'].items()})
        definition = ['content', 'grading_criterion']
        items = rows.drop_duplicates(definition)[definition + ['response_key', 'input_scope', 'scale']].rename(columns={'response_key': 'item_key'})
        items['raw_item_id'] = items.item_key.str.removeprefix('release/')
        items['features'] = [dict(input_scope=scope, rating_scale=scale) for scope, scale in zip(items.input_scope, items.scale)]
        items['verifier'] = items.scale.map({name: Judge(spec=json.dumps(spec, sort_keys=True)) for name, spec in grading['verifiers'].items()})
        rows = rows.merge(items[definition + ['item_key']], on=definition, validate='many_to_one')

        # 5. Parse the original final rating; preserve invalid attempts and all original evidence.
        rating = pd.to_numeric(rows.monitoring_response.str.extract(patterns['verdict'], expand=False).str.strip(), errors='coerce')
        minimum = rows.scale.map({name: spec['response_scale']['min'] for name, spec in grading['verifiers'].items()})
        maximum = rows.scale.map({name: spec['response_scale']['max'] for name, spec in grading['verifiers'].items()})
        valid = rating.between(minimum, maximum)
        rows['response'] = rating.where(valid)
        rows['rating_status'] = valid.map({True: 'valid', False: 'outside_declared_scale'}).where(rating.notna(), 'no_numeric_verdict')
        traces = rows[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.response_key, native_record=row.native_record,
            input_scope=row.input_scope, rating_status=row.rating_status), ensure_ascii=False, allow_nan=False) for row in rows.itertuples()]
        return {'subjects': subjects, 'items': items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'features', 'verifier']],
            'responses': rows[['response_key', 'subject_key', 'item_key', 'response']], 'traces': traces}


if __name__ == '__main__':
    ScaleMRT(__file__).main_from_args()
