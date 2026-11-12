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
from measurement_db.build_base import ConvertedTrace, Judge, StructuredTraceBuild
from measurement_db.scripts.build_measurement_tables.trace_converters import mini_swe_agent


class ProgramBench(StructuredTraceBuild):
    def download(self):
        return self.fetch_sources('*')

    @staticmethod
    def _client_metadata(messages):
        """Keep client fields the common mini-swe-agent mapping does not expose."""
        block_fields = {
            'text': {'type', 'text'}, 'thinking': {'type', 'thinking'},
            'tool_use': {'type', 'id', 'name', 'input'},
            'tool_result': {'type', 'tool_use_id', 'content'},
            'redacted_thinking': {'type'},
        }

        def blocks(content, mapped=None):
            result = {}
            for index, block in enumerate(content if isinstance(content, list) else []):
                if not isinstance(block, dict):
                    continue
                used = block_fields.get(block.get('type'), set(block))
                rest = {key: value for key, value in block.items() if key not in used}
                if block.get('type') == 'tool_result' and block.get('text') == block.get('content'):
                    rest.pop('text', None)
                if mapped is not None:
                    other = mapped[index] if isinstance(mapped, list) and index < len(mapped) else {}
                    other = other if isinstance(other, dict) else {}
                    for key in used - {'type'}:
                        if key in block and block[key] != other.get(key):
                            rest[key] = block[key]
                if rest:
                    result[str(index)] = rest
            return result

        def calls(values):
            result = {}
            for index, call in enumerate(values or []):
                rest = {key: value for key, value in call.items() if key not in {'id', 'type', 'function'}}
                function = {key: value for key, value in (call.get('function') or {}).items()
                            if key not in {'name', 'arguments'}}
                if function:
                    rest['function'] = function
                if rest:
                    result[str(index)] = rest
            return result

        metadata = {}
        for index, message in enumerate(messages):
            fields = {key: message[key] for key in ('created_at', 'completed_at') if key in message}
            response = (message.get('extra') or {}).get('response') or {}
            for key in ('created', 'created_at', 'completed_at'):
                if key in response:
                    fields['response_' + key] = response[key]
            content_fields = blocks(message.get('content'))
            response_fields = blocks(response.get('content'), message.get('content'))
            if content_fields:
                fields['content_blocks'] = content_fields
            if response_fields and response_fields != content_fields:
                fields['response_content_blocks'] = response_fields
            call_fields = calls(message.get('tool_calls'))
            if call_fields:
                fields['tool_calls'] = call_fields
            choice_fields = {}
            for choice_index, choice in enumerate(response.get('choices') or []):
                extra_calls = calls((choice.get('message') or {}).get('tool_calls'))
                if extra_calls and extra_calls != call_fields:
                    choice_fields[str(choice_index)] = extra_calls
            if choice_fields:
                fields['response_tool_calls'] = choice_fields
            if fields:
                metadata[str(index)] = fields
        return metadata

    def convert_trace(self, source, *, context):
        native = {key: source[key] for key in ('source_run', 'task_id', 'published_tests')
                  if source[key] is not None}
        evaluation = source['evaluation']
        if evaluation is not None:
            tests, fields = [], {}
            for index, result in enumerate(evaluation.get('test_results') or []):
                if not all(isinstance(result.get(key), str) for key in ('branch', 'name', 'status')):
                    raise ValueError('Each native test result needs its branch, name and status')
                tests.append(dict(name=result['branch'] + '/' + result['name'], status=result['status']))
                # The branch remains explicit so a full name can be reversed
                # even when the original test name itself contains slashes.
                fields[str(index)] = {key: value for key, value in result.items() if key not in {'name', 'status'}}
            evaluation = {**evaluation, 'test_results': tests}
            if fields:
                native['test_result_fields'] = fields
        client = self._client_metadata(source['trajectory'].get('messages') or [])
        if client:
            native['client_metadata'] = client
        value, counts = mini_swe_agent.convert(
            source['trajectory'], evaluation=evaluation, native=native,
        )
        if source['published_tests'] is not None:
            # An explicitly empty published map is a graded zero; absence is
            # an ungraded attempt. The common helper omits empty dictionaries.
            value.setdefault('native', {})['published_tests'] = source['published_tests']
        return ConvertedTrace(value, counts)

    def build_tables(self) -> dict[str, pd.DataFrame]:
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
        scores['response'] = scores.active_tests.map(lambda tests: sum(tests.values()) / len(tests) if tests else 0.)

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
        calls['response_key'] = calls.index
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

        # 4. Preserve the actual initial request, named environment and grading scope.
        items = calls.copy()
        items['item_key'] = items.response_key
        items['active_test_names'] = items.active_tests.map(lambda value: sorted(value) if isinstance(value, dict) else None)
        items['test_set_sha256'] = items.active_test_names.map(lambda value: hashlib.sha256(json.dumps(value).encode()).hexdigest())
        items['raw_item_id'] = items.task_id + ':' + items.test_set_sha256.str[:16]
        items['content'] = [json.dumps(dict(messages=row.trajectory['messages'][:2],
            environment=dict(image=row.config['environment']['image'], cwd=row.config['environment']['cwd'])),
            ensure_ascii=False, allow_nan=False) for row in items.itertuples()]
        items['features'] = items[['task_id', 'grader_version', 'test_set_sha256']].to_dict('records')
        items['grading_criterion'] = [dict(rule=self.grading['rule']) for _ in items.index]
        items['verifier'] = [Judge(spec=json.dumps(dict(**self.grading['verifiers']['published'],
            grader_version=row.grader_version, active_test_names=row.active_test_names), sort_keys=True))
            for row in items.itertuples()]

        # 5. Pass native records to the shared trace/1 conversion boundary.
        traces = calls[['response_key']].copy()
        traces['trace'] = [dict(source_run=row.source_run, task_id=row.task_id,
            trajectory=row.trajectory, evaluation=row.evaluation if isinstance(row.evaluation, dict) else None,
            published_tests=row.published_tests if isinstance(row.published_tests, dict) else None)
            for row in calls.itertuples()]
        calls['item_key'] = calls.response_key
        calls['test_condition'] = 'run=' + calls.source_run + ';grader=' + calls.grader_version
        return dict(subjects=subjects[['subject_key', 'raw_label', 'features']],
            items=items[['item_key', 'raw_item_id', 'content', 'features', 'grading_criterion', 'verifier']],
            responses=calls[['response_key', 'subject_key', 'item_key', 'response', 'test_condition']], traces=traces)


if __name__ == '__main__':
    ProgramBench(__file__).main_from_args()
