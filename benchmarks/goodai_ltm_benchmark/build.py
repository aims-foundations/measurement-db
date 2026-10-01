#!/usr/bin/env python3
"""Tabulate GoodAI's original memory tests, conversation records and published grades."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class GoodAILTM(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Load native result records and decode their published path keys.
        with ZipFile(self.raw_dir / layout['archive']) as archive:
            members = {name.removeprefix(layout['prefix']): name for name in archive.namelist() if not name.endswith('/')}
            files = pd.Series(sorted(members), name='source_file')
            keys = files.str.extract(parameters['patterns']['result'])
            keys['source_file'] = files
            keys = keys.dropna().reset_index(drop=True)
            records = [json.loads(archive.read(members[name])) for name in keys.source_file]
            attempts = pd.concat([keys, pd.json_normalize(records, max_level=0)], axis=1)
            attempts['record'] = records
            attempts['response_key'] = attempts.source_file
            attempts['definition_file'] = (layout['tests'] + attempts.release_group + '/definitions/' +
                attempts.task_type + '/' + attempts.example + '.def.json')

            # 2. Join original definitions and scheduling settings, retaining all fields.
            definition_files = sorted(attempts.definition_file.unique())
            definitions = pd.DataFrame(dict(definition_file=definition_files,
                definition=[json.loads(archive.read(members[name])) for name in definition_files]))
            groups = sorted(attempts.release_group.unique())
            configurations = pd.DataFrame(dict(release_group=groups, configuration=[yaml.safe_load(
                archive.read(members[layout['tests'] + group + '/definitions/config.yml'])) for group in groups]))
            attempts = attempts.merge(definitions, on='definition_file', validate='many_to_one').merge(
                configurations, on='release_group', validate='many_to_one')
            resource_paths = [*parameters['task_programs'].values(), *parameters['shared_resources'].values()]
            resources = {name: archive.read(members[name]) for name in resource_paths}

        # 3. Keep exact source sessions and their declared model/context configurations.
        subjects = attempts[['session']].drop_duplicates().copy()
        parts = subjects.session.str.split(' - ', n=3, expand=True).reindex(columns=range(4))
        subjects['wrapper'] = parts[0]
        subjects['raw_label'] = parts[1].where(parts[1].notna() & ~parts[1].str.isdigit().fillna(False), parts[0])
        subjects['context_window'] = parts[2].where(parts[2].notna(), parts[1].where(parts[1].str.isdigit().fillna(False)))
        subjects['agent_configuration'] = parts[3]
        subjects['subject_key'] = subjects.session
        subjects['features'] = [dict(harness=row.wrapper, source_session=row.session,
            context_window=None if pd.isna(row.context_window) else row.context_window,
            agent_configuration=None if pd.isna(row.agent_configuration) else row.agent_configuration)
            for row in subjects.itertuples()]
        attempts['subject_key'] = attempts.session

        # 4. Describe the task program, keeping evaluated answers out of item inputs.
        # Restaurant is adaptive: only its initial instructions/menu precede grading.
        messages = attempts[['response_key', 'task_log']].explode('task_log', ignore_index=True)
        parsed = messages.task_log.str.extract(parameters['patterns']['message'])
        messages = pd.concat([messages[['response_key']], parsed], axis=1)
        initial = messages.loc[messages.role.eq('Test')].groupby('response_key', sort=False).message.agg(list)
        attempts['initial_messages'] = attempts.response_key.map(initial)
        content, criteria, verifiers, attachments = [], [], [], []
        for row in attempts.itertuples():
            definition = {key: value for key, value in row.definition.items() if key not in parameters['grading_fields']}
            program = parameters['task_programs'][row.task_type]
            stimulus = dict(task_type=row.task_type, release_group=row.release_group, definition=definition,
                configuration=row.configuration, program_reference=program,
                program_reference_revision=parameters['labels']['program_revision'])
            paths = [program, parameters['shared_resources']['interface'], parameters['shared_resources']['scheduler']]
            if row.task_type == parameters['labels']['dynamic_task']:
                intro, menu = row.initial_messages[:2]
                if parameters['labels']['menu_marker'] not in menu:
                    raise ValueError('The dynamic task does not begin with its recorded menu')
                stimulus.update(initial_instruction=intro, initial_menu_message=menu,
                    task_random_key=row.task_type + ' - ' + row.example,
                    dynamic_scope=parameters['labels']['dynamic_scope'])
                paths.append(parameters['shared_resources']['menu'])
            elif not definition['script'] or definition['script'][0] != row.initial_messages[0]:
                raise ValueError('The released task definition does not match its recorded initial instruction')
            content.append(json.dumps(stimulus, sort_keys=True, ensure_ascii=False, allow_nan=False))
            attachments.append([dict(path=name, data=resources[name],
                media_type='application/json' if name.endswith('.json') else 'text/x-python',
                role='task_program_reference') for name in paths])
            protocol = dict(self.grading['verifiers'][row.task_type])
            revised = any(key.startswith('auto') for key in row.record)
            protocol['revision_status'] = parameters['labels']['revised' if revised else 'unmarked']
            criterion = dict(rule=protocol['rule'], response_scale=dict(kind='interval', min=0,
                max=row.max_score, direction='higher_is_better'))
            if row.task_type == parameters['labels']['dynamic_task']:
                criterion['rule'] = json.dumps(dict(description=protocol['rule'], published_rubric=row.expected_responses),
                    ensure_ascii=False, sort_keys=True, allow_nan=False)
            else:
                criterion['reference_answer'] = json.dumps(row.expected_responses, ensure_ascii=False, allow_nan=False)
            criteria.append(criterion)
            spec = json.dumps(protocol, sort_keys=True, ensure_ascii=False)
            verifiers.append(ExactMatcher(spec=spec) if protocol['kind'] == 'deterministic' and not revised else Judge(spec=spec))
        items = attempts.assign(item_key=attempts.response_key, raw_item_id=attempts.definition_file,
            content=content, grading_criterion=criteria, verifier=verifiers, attachments=attachments)
        items['features'] = [dict(task=row.task_type, subset=row.release_group) for row in items.itertuples()]

        # 5. Preserve native score units, source occasions and complete original traces.
        attempts['item_key'] = attempts.response_key
        attempts['response'] = attempts.score
        attempts['trial'] = attempts.repetition.astype(int) + 1
        attempts['test_condition'] = attempts.source_file
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, record=row.record),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'grading_criterion', 'verifier', 'features', 'attachments']],
            responses=attempts[['response_key', 'subject_key', 'item_key', 'response', 'trial', 'test_condition']],
            traces=attempts[['response_key', 'trace']])


if __name__ == '__main__':
    GoodAILTM(__file__).main_from_args()
