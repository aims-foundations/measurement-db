#!/usr/bin/env python3
"""Tabulate ProgramBench's published attempts, test verdicts and complete trajectories."""

import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from measurement_db.build_base import BenchmarkBuild, ExactMatcher


class ProgramBench(BenchmarkBuild):
    def download(self):
        return self.fetch_sources('*')

    def build_subject_item_response_rows(self):
        parameters = self.build_parameters

        # 1. Read the immutable registry and its published per-test verdicts.
        with tarfile.open(self.raw_dir / parameters['layout']['registry']) as archive:
            registry = {'/'.join(Path(member.name).parts[1:]): archive.extractfile(member).read()
                for member in archive if member.isfile()}
        ignored = json.loads(registry['ignored_tests.json'])
        manifests, score_frames = [], []
        for source_run in parameters['runs']:
            manifest = yaml.safe_load(registry[f'submissions/{source_run}/submission.yaml'])
            manifests.append(dict(source_run=source_run, display_name=manifest['system']['model'],
                provider=manifest['system']['provider'], grader_version=manifest['eval']['programbench_version']))
            scores = json.loads(registry[f'submissions/{source_run}/_stats/score.json'])
            frame = pd.Series(scores, name='published_tests', dtype=object).rename_axis('task_id').reset_index()
            score_frames.append(frame.assign(source_run=source_run))
        scores = pd.concat(score_frames, ignore_index=True)
        scores['active_tests'] = [{key: value for key, value in row.published_tests.items()
            if key not in set(ignored.get(row.task_id, []))} for row in scores.itertuples()]
        if not scores.published_tests.map(lambda tests: all(type(value) is bool for value in tests.values())).all():
            raise ValueError('Published test verdicts must be boolean')

        # 2. Join every recorded trajectory, including attempts without a grade.
        trajectory_rows, evaluation_rows = [], []
        for source_run, filename in parameters['runs'].items():
            with tarfile.open(self.raw_dir / filename) as archive:
                for member in archive:
                    path = Path(*Path(member.name).parts[1:])
                    if not member.isfile() or len(path.parts) != 2:
                        continue
                    if path.name.endswith('.traj.json'):
                        trajectory_rows.append(dict(source_run=source_run, task_id=path.parent.name,
                            trajectory=json.loads(archive.extractfile(member).read())))
                    elif path.name.endswith('.eval.json'):
                        evaluation_rows.append(dict(source_run=source_run, task_id=path.parent.name,
                            evaluation=json.loads(archive.extractfile(member).read())))
        calls = pd.DataFrame(trajectory_rows).merge(pd.DataFrame(evaluation_rows),
            on=['source_run', 'task_id'], how='outer', validate='one_to_one')
        calls = calls.merge(scores, on=['source_run', 'task_id'], how='outer', validate='one_to_one')
        calls = calls.merge(pd.DataFrame(manifests), on='source_run', validate='many_to_one')
        if calls.trajectory.isna().any() or not calls.trajectory.map(lambda value:
            [message.get('role') for message in value['messages'][:2]] == ['system', 'user']).all():
            raise ValueError('Each attempt must have its recorded initial system and user messages')
        calls['config'] = calls.trajectory.map(lambda value: value['info']['config'])
        calls['mini_version'] = calls.trajectory.map(lambda value: value['info']['mini_version'])

        # 3. Separate recorded model and harness configurations, excluding task paths.
        calls['policy'] = [dict(model=row.config['model'],
            agent={key: value for key, value in row.config['agent'].items() if key != 'output_path'},
            environment={key: value for key, value in row.config['environment'].items() if key != 'image'},
            model_type=row.config.get('model_type'), agent_type=row.config.get('agent_type'),
            environment_type=row.config.get('environment_type'), mini_version=row.mini_version)
            for row in calls.itertuples()]
        calls['policy_sha256'] = calls.policy.map(lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        calls['subject_key'] = calls.source_run + ':' + calls.policy_sha256
        subjects = calls.drop_duplicates('subject_key').copy()
        subjects['raw_label'] = subjects.config.map(lambda value: value["model"]["model_name"])
        subjects['features'] = [dict(source_run=row.source_run, configuration_sha256=row.policy_sha256,
            model_alias=row.config['model']['model_name'], provider=row.provider,
            harness=parameters['labels']['harness'], harness_version=row.mini_version,
            reasoning_effort=row.config['model'].get('model_kwargs', {}).get('reasoning_effort') or
                row.config['model'].get('model_kwargs', {}).get('reasoning', {}).get('effort'))
            for row in subjects.itertuples()]

        subject_ids = {
            row.subject_key: self.add_subject(row.raw_label, features=row.features)
            for row in subjects.itertuples()
        }

        # One execution owns the complete trajectory and native evaluation.
        # Each active behavioral test grades that same generated program.
        item_ids = {}
        current_run = None
        for row in calls.itertuples():
            if row.source_run != current_run:
                current_run = row.source_run
                print(f"[programbench] Converting {current_run}", flush=True)
            subject_id = subject_ids[row.subject_key]
            trace = dict(source_run=row.source_run, task_id=row.task_id,
                trajectory=row.trajectory,
                evaluation=row.evaluation if isinstance(row.evaluation, dict) else None,
                published_tests=row.published_tests if isinstance(row.published_tests, dict) else None)
            attempt_id = self.add_attempt(subject_id=subject_id,
                raw_attempt_id=row.source_run + '/' + row.task_id,
                trace=json.dumps(trace, ensure_ascii=False, allow_nan=False))
            # An unreleased/empty evaluation defines no individual test outcomes.
            # Preserve the attempt above rather than inventing a failed test.
            if not isinstance(row.active_tests, dict):
                continue
            content = json.dumps(dict(messages=row.trajectory['messages'][:2],
                environment=dict(image=row.config['environment']['image'], cwd=row.config['environment']['cwd'])),
                ensure_ascii=False, allow_nan=False)
            for test_name, passed in row.active_tests.items():
                key = (row.task_id, content, row.grader_version, test_name)
                if key not in item_ids:
                    spec = dict(**self.grading['verifiers']['published'],
                        grader_version=row.grader_version, test_name=test_name)
                    item_ids[key] = self.add_item(
                        raw_item_id=row.task_id + ':' + test_name,
                        testlet_id=row.task_id, content=content,
                        grading_criterion=dict(rule=self.grading['rule']),
                        verifier=ExactMatcher(spec=json.dumps(spec, sort_keys=True)))
                self.add_response(subject_id=subject_id, item_id=item_ids[key],
                    response=float(passed), attempt_id=attempt_id, trace=None,
                    test_condition=None)
        print(f"[programbench] Registered {len(item_ids):,} behavioral-test items", flush=True)


if __name__ == '__main__':
    ProgramBench(__file__).main_from_args()
