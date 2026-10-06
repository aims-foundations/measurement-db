#!/usr/bin/env python3
"""Tabulate InterCode's released episodes, task resources and native rewards."""

import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class InterCode(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self) -> dict[str, pd.DataFrame]:
        parameters = self.build_parameters
        layout = parameters['layout']

        # 1. Flatten both original result layouts without losing episode fields.
        with ZipFile(self.raw_dir / layout['archive']) as archive:
            members = {name.removeprefix(layout['prefix']): name for name in archive.namelist() if not name.endswith('/')}
            frames = []
            for name in sorted(members):
                if not name.startswith(layout['results']) or not name.endswith('.json') or '/human/' in name:
                    continue
                data = json.loads(archive.read(members[name]))
                records = data.get('logs', data)
                frame = pd.DataFrame.from_dict(records, orient='index').rename_axis('source_record').reset_index()
                frame['record'] = frame.source_record.map(records)
                frame['source_file'] = name
                frame['run_metadata'] = [data.get('meta')] * len(frame)
                frames.append(frame)
            attempts = pd.concat(frames, ignore_index=True)
            summaries = pd.json_normalize(attempts.summary).reindex(columns=['max_reward', 'turns_max'])
            attempts['response'] = summaries.max_reward
            attempts['max_turns'] = summaries.turns_max.map(lambda value: None if pd.isna(value) else int(value)).astype(object)
            attempts['bank'] = attempts.dataset.map(parameters['task_banks'])
            if attempts.bank.isna().any():
                raise ValueError('An InterCode result has no documented task-bank mapping')

            # 2. Join task-bank positions, keeping the recorded question authoritative.
            definitions = []
            for name in sorted(attempts.bank.unique()):
                table = pd.json_normalize(json.loads(archive.read(members[name])), max_level=0)
                table['definition'] = table.to_dict('records')
                table = table.rename_axis('task_position').reset_index().assign(bank=name)
                definitions.append(table[['bank', 'task_position', 'definition']])
            attempts = attempts.merge(pd.concat(definitions, ignore_index=True), left_on=['bank', 'task_id'],
                right_on=['bank', 'task_position'], how='left', validate='many_to_one')
            attempts['response_key'] = attempts.source_file + ':' + attempts.source_record
            if attempts.definition.isna().any() or not attempts.source_record.eq(attempts.task_id.astype(str)).all():
                raise ValueError('An InterCode task position does not match its original episode')
            matches = attempts['query'].eq(attempts.definition.map(lambda task: task['query']))
            variants = parameters['question_variants']
            for row in attempts.loc[~matches].itertuples():
                if (variants.get(row.response_key) != row.query or
                        parameters['definition_variants'].get(row.response_key) != row.definition['query']):
                    raise ValueError('An undocumented InterCode question differs from its task bank')

            # Released input resources are archived bytes, never executed here.
            resources = {}
            for bank, task in attempts[['bank', 'task_id']].drop_duplicates().itertuples(index=False):
                pattern = parameters['resources'][bank].format(task=task)
                paths = [name for name in sorted(members) if re.fullmatch(pattern, name)
                    and '/solution/' not in name and not Path(name).name.startswith('.')]
                resources[bank, task] = [dict(path=name, data=archive.read(members[name]),
                    media_type=parameters['media_types'].get(Path(name).suffix, 'application/octet-stream'),
                    role='input') for name in paths]

        # 3. Separate model, prompting strategy and recorded interaction budgets.
        filenames = attempts.source_file.map(lambda name: Path(name).name)
        attempts['model'] = filenames.str.extract('(' + '|'.join(map(re.escape, parameters['models'])) + ')', expand=False)
        attempts['model'] = attempts.model.fillna(attempts.source_file.map(lambda name: Path(name).parent.name))
        if not attempts.model.isin(parameters['models']).all():
            raise ValueError('An InterCode model label has no documented source mapping')
        attempts['strategy'] = filenames.str.extract('(plan_solve_refine|plan_solve|react)', expand=False).fillna('try_again')
        attempts.loc[attempts.source_file.isin(parameters['initial_ctf_files']), 'strategy'] = 'initial_ctf'
        attempts['refine_turns'] = attempts.run_metadata.map(lambda meta: meta.get('refine_turns') if meta else None)
        attempts['configuration'] = [dict(environment=row.environment, strategy=row.strategy,
            max_turns=None if pd.isna(row.max_turns) else int(row.max_turns),
            refine_turns=None if pd.isna(row.refine_turns) else int(row.refine_turns)) for row in attempts.itertuples()]
        attempts['subject_key'] = attempts.model + ':' + attempts.configuration.map(lambda value: json.dumps(value, sort_keys=True))
        subjects = attempts.drop_duplicates('subject_key').copy()
        subjects['raw_label'] = subjects.model
        subjects['features'] = [dict(harness=parameters['labels']['harness'], **row.configuration) for row in subjects.itertuples()]

        # 4. Build distinct stimuli and grading protocols on their native scales.
        attempts['protocol'] = attempts.environment
        attempts.loc[attempts.source_file.isin(parameters['initial_ctf_files']), 'protocol'] = 'ic_ctf_initial'
        attempts['schema_provided'] = filenames.str.contains('handicap', regex=False)
        attempts['item_key'] = (attempts.bank + ':' + attempts.task_id.astype(str) + ':' + attempts['query'] +
            ':' + attempts.protocol + ':' + attempts.schema_provided.astype(str))
        items = attempts.drop_duplicates('item_key').copy()
        content, criteria = [], []
        for row in items.itertuples():
            task = row.definition
            protocol = self.grading['verifiers'][row.protocol]
            stimulus = dict(question=row.query, environment=row.environment, task_bank=row.bank,
                task_position=int(row.task_id), resource_revision=parameters['labels']['resource_revision'])
            if row.environment == 'ic_sql':
                stimulus['database'] = task['db']
                if row.schema_provided:
                    stimulus['provided_schema'] = task['db_tables']
            content.append(json.dumps(stimulus, sort_keys=True, ensure_ascii=False, allow_nan=False))
            criterion = dict(rule=protocol['rule'], response_scale=protocol['response_scale'])
            # CTF logs do not pin the historical flags; do not substitute today's flags.
            if row.environment != 'ic_ctf':
                criterion['reference_answer'] = task['gold'] if isinstance(task['gold'], str) else json.dumps(task['gold'], ensure_ascii=False)
            if row.environment == 'ic_python':
                criterion['rule'] = json.dumps(dict(description=protocol['rule'], tests=task['tests'],
                    test_setup_code=task['test_setup_code']), ensure_ascii=False, sort_keys=True)
            criteria.append(criterion)
        items['content'], items['grading_criterion'] = content, criteria
        items['raw_item_id'] = items.bank + ':' + items.task_id.astype(str)
        items['features'] = [dict(environment=row.environment, task_bank=row.bank) for row in items.itertuples()]
        items['attachments'] = [resources[row.bank, row.task_id] for row in items.itertuples()]
        items['verifier'] = items.protocol.map(lambda name: ExactMatcher(spec=json.dumps(self.grading['verifiers'][name], sort_keys=True)))

        # 5. Retain every published occasion and its complete, unmodified episode.
        attempts['test_condition'] = attempts.response_key
        attempts['trace'] = [json.dumps(dict(source_file=row.source_file, source_record=row.source_record,
            run_metadata=row.run_metadata, record=row.record), ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'attachments', 'grading_criterion', 'verifier']],
            responses=attempts[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']],
            traces=attempts[['response_key', 'trace']])


if __name__ == '__main__':
    InterCode(__file__).main_from_args()
