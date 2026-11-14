"""Small native inputs that exercise inline conversion through real build_tables()."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
from zipfile import ZipFile

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


def legacy_traces(self, traces):
    """Write ordinary JSON strings when testing the legacy serialization boundary."""
    return traces.assign(trace=[json.dumps(result.value) for result in traces.trace])


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def _archive(path, documents):
    with tarfile.open(path, 'w:gz') as archive:
        for name, document in documents.items():
            content = (yaml.safe_dump(document) if name.endswith('.yaml') else json.dumps(document)).encode()
            entry = tarfile.TarInfo('release/' + name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))


def build_trace(slug, source):
    """Prepare a real builder's tables using only a tiny temporary native release."""
    source = deepcopy(source)
    with tempfile.TemporaryDirectory(prefix='inline-trace-fixture-') as scratch:
        bench = Path(scratch) / slug
        raw = bench / 'raw'
        raw.mkdir(parents=True)
        metadata = yaml.safe_load((ROOT / f'benchmarks/{slug}/metadata.yaml').read_text())
        parameters = metadata['build']['parameters']
        if slug == 'programbench':
            run, task = source['source_run'], source['task_id']
            parameters['runs'] = {run: 'run.tar.gz'}
            trajectory = source['trajectory']
            scores = {'companion': {}}
            records = {f'{task}/{task}.traj.json': trajectory,
                       'companion/companion.traj.json': trajectory,
                       'companion/companion.eval.json': {'test_results': []}}
            if source['published_tests'] is not None:
                scores[task] = source['published_tests']
            if source['evaluation'] is not None:
                records[f'{task}/{task}.eval.json'] = source['evaluation']
            _archive(raw / 'run.tar.gz', records)
            _archive(raw / parameters['layout']['registry'], {
                'ignored_tests.json': {},
                f'submissions/{run}/submission.yaml': {'system': {'model': 'fixture', 'provider': 'fixture'},
                                                     'eval': {'programbench_version': 'fixture'}},
                f'submissions/{run}/_stats/score.json': scores,
            })
        elif slug == 'theagentcompany':
            parameters['models'] = {'fixture': 'fixture'}
            parameters['harnesses'] = {'fixture': 'OpenHands'}
            run = raw / parameters['layout']['runs'] / 'fixture'
            _write(run / 'results/eval_task.json', {'final_score': {'result': 1, 'total': 1}})
            path = run / 'trajectories/traj_task.txt'
            path.parent.mkdir(parents=True)
            path.write_text(source if isinstance(source, str) else json.dumps(source))
            task = raw / parameters['layout']['tasks'] / 'task'
            task.mkdir(parents=True)
            (task / 'task.md').write_text('Original task')
            (task / 'evaluator.py').write_text('Original evaluator')
        elif slug == 'os_harm':
            folder = raw / 'drive/results_main_category_1_misuse/pyautogui/screenshot_a11y_tree/model/editor/task'
            record = source['record']
            _write(folder / 'better_log.json', record)
            _write(folder / 'judgment/gpt-4.1/aer/v3/judgment.json', source['judgment'])
            _write(folder / parameters['layout']['caption'], source['caption'])
            (folder / parameters['layout']['trajectory']).write_text(
                '\n'.join(json.dumps(row) for row in source['trajectory']))
        elif slug == 'taubench':
            task = {'user_id': 'fixture', 'instruction': 'Original task', 'actions': [], 'outputs': []}
            result = source['result']
            sierra = isinstance(result, dict) and 'traj' in result
            native = {**result, 'info': {'task': task}, 'trial': 0, 'task_id': 0,
                      'reward': result.get('reward', 0)} if sierra else {
                          'info': {'task': task}, 'trial': 0, 'task_id': 0, 'reward': 0, 'traj': []}
            parameters['sierra_domains'] = {'fixture.json': 'airline'}
            parameters['sierra_models'] = {'fixture.json': source['configuration'].get('model', 'fixture')}
            _write(raw / 'sierra/historical_trajectories/fixture.json', [native])
            salt = b'inline-fixture'
            encoding = parameters['hal_encoding']
            key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                            iterations=int(encoding['iterations']))
            cipher = Fernet(base64.urlsafe_b64encode(key.derive(encoding['public_password'].encode())))
            config = source['configuration']
            if 'agent_args' not in config:
                config = {**config, 'agent_args': {'model_name': config.get('model', 'fixture')}}
            hal_result = {'task': task, 'reward': 0} if sierra else (
                {**result, 'task': task} if isinstance(result, dict) else result)
            # An ungraded API error borrows the definition from a separate attempt.
            results = {'0': hal_result, 'definition': {'task': task, 'reward': 0}}
            logs = [{**row, 'weave_task_id': '0'} for row in source.get('logs', [])]
            payload = {'config': config, 'raw_eval_results': results, 'raw_logging_results': logs or [
                {'weave_task_id': 'definition', 'inputs': {}, 'output': {}}]}
            encrypted = {'salt': base64.b64encode(salt).decode(),
                         'encrypted_data': base64.b64encode(cipher.encrypt(json.dumps(payload).encode())).decode()}
            path = raw / 'hal/archives/taubench-fixture.zip'
            path.parent.mkdir(parents=True)
            with ZipFile(path, 'w') as archive:
                archive.writestr('run.json', json.dumps(encrypted))
            if isinstance(hal_result, str):
                definition = {'config': config, 'raw_eval_results': {'0': {'task': task, 'reward': 0}},
                              'raw_logging_results': [{'weave_task_id': '0', 'inputs': {}, 'output': {}}]}
                encrypted['encrypted_data'] = base64.b64encode(cipher.encrypt(json.dumps(definition).encode())).decode()
                with ZipFile(path.with_name('taubench-definition.zip'), 'w') as archive:
                    archive.writestr('run.json', json.dumps(encrypted))
        else:
            raise ValueError(slug)
        (bench / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
        namespace = runpy.run_path(str(ROOT / f'benchmarks/{slug}/build.py'))
        cls = next(value for value in namespace.values()
                   if isinstance(value, type) and value.__module__ == '<run_path>')
        builder = cls(str(bench / 'build.py'))
        tables = builder.build_tables()
        if slug == 'programbench':
            return next(result for result in tables['traces'].trace if result.value['native']['task_id'] == task)
        if slug == 'taubench':
            suffix = 'sierra/historical_trajectories/fixture.json::0' if sierra else 'hal/archives/taubench-fixture.zip::0'
            return tables['traces'].set_index('response_key').loc[suffix, 'trace']
        return tables['traces'].trace.iloc[0]


def assert_source_contains(actual, expected):
    """Check source evidence while allowing the fixture's task/linkage metadata."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual
            assert_source_contains(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected):
            assert_source_contains(actual_row, expected_row)
    else:
        assert actual == expected
