"""Tabulate complete original tau2 simulations, task definitions and run settings."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher, Judge


class Tau2Bench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_tables(self):
        parameters, labels = self.build_parameters, self.build_parameters['labels']

        # 1. Read each complete original run and its separate submission description.
        paths = sorted(self.raw_dir.glob(parameters['layout']['repository_runs']))
        paths += sorted(path for path in self.raw_dir.glob(parameters['layout']['submitted_runs'])
            if path.name != 'submission.json')
        records = [json.loads(path.read_text()) for path in paths]
        runs = pd.json_normalize(records, max_level=0)
        runs['source_file'] = [str(path.relative_to(self.raw_dir)) for path in paths]
        runs['submission'] = [json.loads(path.with_name('submission.json').read_text())
            if path.with_name('submission.json').exists() else None for path in paths]
        runs['source_submission'] = pd.Series([path.parent.name if path.parent.parent.name == 's3_trajectories'
            else None for path in paths], dtype=object)
        runs['domain'] = runs['info'].map(lambda info: info['environment_info']['domain_name'])
        if not runs.domain.isin(parameters['domains']).all():
            raise ValueError('A source run contains a domain outside the declared cohort')

        # 2. Expand observed simulations and join tasks by their original run and ID.
        attempts = runs[['source_file', 'simulations']].explode('simulations').rename(columns={'simulations': 'simulation'})
        attempts['response_key'] = attempts.simulation.map(lambda row: row['id'])
        attempts['task_id'] = attempts.simulation.map(lambda row: str(row['task_id']))
        attempts['trial'] = attempts.simulation.map(lambda row: row['trial'] + 1)
        attempts['response'] = attempts.simulation.map(lambda row: (row.get('reward_info') or {}).get('reward'))
        if attempts.response_key.isna().any() or attempts.response_key.duplicated().any():
            raise ValueError('Missing or repeated original simulation IDs require source review')
        if not attempts.response.dropna().isin([0, 1]).all():
            raise ValueError('A recorded reward is outside the declared binary scale')
        items = runs[['source_file', 'tasks']].explode('tasks').rename(columns={'tasks': 'task'})
        items['task_id'] = items.task.map(lambda row: str(row['id']))
        items['item_key'] = items.source_file + '::' + items.task_id
        attempts = attempts.merge(items[['source_file', 'task_id', 'item_key']], on=['source_file', 'task_id'],
            how='left', validate='many_to_one')
        if attempts.item_key.isna().any():
            raise ValueError('An original simulation has no corresponding task definition')
        items = items.loc[items.item_key.isin(attempts.item_key)].merge(
            runs[['source_file', 'info', 'domain']], on='source_file', validate='many_to_one')

        # 3. Identify subjects by their literal model label and recorded agent settings.
        subjects = runs[['source_file']].rename(columns={'source_file': 'subject_key'})
        subjects['raw_label'] = [(row.submission or {}).get('model_name') or row.info['agent_info']['llm']
            for row in runs.itertuples()]
        configurations = [dict(agent_info=row.info['agent_info'], git_commit=row.info.get('git_commit'),
            source_submission=row.source_submission) for row in runs.itertuples()]
        subjects['features'] = [dict(harness=labels['harness'], configuration_scope=labels['configuration_scope'],
            agent_configuration_sha256=hashlib.sha256(json.dumps(configuration, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
            recorded_agent_model=row.info['agent_info']['llm'], implementation=row.info['agent_info']['implementation'],
            recorded_harness_revision=row.info.get('git_commit') or 'unknown')
            for row, configuration in zip(runs.itertuples(), configurations)]

        # 4. Keep complete task inputs, policies, tools and grading rules separately.
        items['raw_item_id'] = items.domain + '/' + items.task_id
        items['content'] = [json.dumps(dict(task={key: value for key, value in row.task.items() if key != 'evaluation_criteria'},
            environment_info=row.info['environment_info'],
            user_guidelines=row.info['user_info'].get('global_simulation_guidelines')),
            ensure_ascii=False, sort_keys=True, allow_nan=False) for row in items.itertuples()]
        items['features'] = [dict(domain=row.domain, source_task_id=row.task_id, input_scope=labels['input_scope'])
            for row in items.itertuples()]
        items['grading_criterion'] = [dict(rule=json.dumps(dict(rule=self.grading['rule'],
            evaluation_criteria=row.task['evaluation_criteria']), ensure_ascii=False, sort_keys=True)) for row in items.itertuples()]
        specifications = [json.dumps(dict(self.grading['verifiers']['native'], recorded_harness_revision=row.info.get('git_commit')),
            sort_keys=True) for row in items.itertuples()]
        items['verifier'] = [Judge(judged_by='llm', spec=spec) if 'NL_ASSERTION' in row.task['evaluation_criteria']['reward_basis']
            else ExactMatcher(spec=spec) for row, spec in zip(items.itertuples(), specifications)]

        # 5. Retain the run conditions and both reports of the user simulator.
        runs['test_condition'] = [json.dumps(dict(source_file=row.source_file,
            recorded_settings={key: value for key, value in row.info.items()
                if key not in {'agent_info', 'user_info', 'environment_info'}}), sort_keys=True) for row in runs.itertuples()]
        runs['interactors'] = [json.dumps(dict(recorded_user_info={key: value for key, value in row.info['user_info'].items()
            if key != 'global_simulation_guidelines'}, submission_user_simulator=(row.submission or {}).get('methodology', {}).get('user_simulator')),
            sort_keys=True) for row in runs.itertuples()]
        attempts = attempts.merge(runs[['source_file', 'info', 'timestamp', 'submission', 'test_condition', 'interactors']],
            on='source_file', validate='many_to_one')
        attempts['subject_key'] = attempts.source_file

        # 6. Preserve every native simulation, including errors and full messages.
        traces = attempts[['response_key']].copy()
        traces['trace'] = [json.dumps(dict(source_file=row.source_file, run_timestamp=row.timestamp,
            run_info=row.info, submission=row.submission, simulation=row.simulation, submission_scope=labels['submission_scope']),
            ensure_ascii=False, allow_nan=False) for row in attempts.itertuples()]
        return {'subjects': subjects, 'items': items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            'responses': attempts[['response_key', 'subject_key', 'item_key', 'trial', 'response', 'test_condition', 'interactors']],
            'traces': traces}


if __name__ == '__main__':
    Tau2Bench(__file__).main_from_args()
