"""Independent native checks for encrypted tau-bench and replayed CORE-Bench."""

import ast
import base64
import copy
import hashlib
import json
import math
import tarfile
import unicodedata
from zipfile import ZipFile

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from scipy.stats import t


def source_attempts(directory, metadata):
    if directory.name == 'taubench':
        yield from _tau(directory, metadata)
    else:
        yield from _core(directory, metadata)


def _tau(directory, metadata):
    raw, parameters = directory / 'raw', metadata['build']['parameters']
    runs, bank = [], {}
    for path in sorted(raw.glob(parameters['layout']['sierra_runs'])):
        rows = json.loads(path.read_text())
        runs.append((str(path.relative_to(raw)), parameters['sierra_domains'][path.name],
            [(str(r['task_id']), r, r['info']['task']) for r in rows]))
    for path in sorted(raw.glob(parameters['layout']['hal_runs'])):
        with ZipFile(path) as archive:
            envelope = json.loads(archive.read(archive.namelist()[0]))
        encoding = parameters['hal_encoding']
        key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
            salt=base64.b64decode(envelope['salt']), iterations=int(encoding['iterations']))
        run = json.loads(Fernet(base64.urlsafe_b64encode(key.derive(encoding['public_password'].encode())))
            .decrypt(base64.b64decode(envelope['encrypted_data'])))
        rows = list(run['raw_eval_results'].items())
        for identifier, record in rows:
            if isinstance(record, dict) and record.get('task'):
                if identifier in bank:
                    assert bank[identifier] == record['task']
                bank[identifier] = record['task']
        runs.append((str(path.relative_to(raw)), 'airline', [(i, r, None) for i, r in rows]))
    for filename, domain, rows in runs:
        for position, (identifier, record, task) in enumerate(rows):
            task = task or bank[identifier]
            content = json.dumps(dict(user_id=task['user_id'], instruction=task['instruction']),
                sort_keys=True, ensure_ascii=False)
            group = domain + '/' + hashlib.sha256(unicodedata.normalize('NFC', content).encode()).hexdigest()
            score = record.get('reward') if isinstance(record, dict) else None
            details = (((record.get('info') or {}).get('reward_info') or {}).get('info') or {}) if isinstance(record, dict) else {}
            checks = []
            if 'r_actions' in details:
                checks.append((group + '::database', group, details['r_actions']))
            for output, met in (details.get('outputs') or {}).items():
                assert type(met) is bool and output in task['outputs']
                checks.append((group + '::output:' + output, group, float(met)))
            if not checks:
                checks = [(group, group, score)]
            yield filename + '::' + str(position), score, checks


def _core(directory, metadata):
    from ..build_measurement_tables.load_source_files import read_gpg_json
    root = directory / 'raw/core-bench'
    tasks = json.loads((root / 'benchmark/dataset/core_train.json').read_text())
    tasks += read_gpg_json(root / 'benchmark/dataset/core_test.json.gpg',
        password=metadata['build']['parameters']['task_bank']['public_password'], scratch_dir=directory)
    tasks = {task['capsule_id']: task for task in tasks}
    function = next(node for node in ast.parse((root / 'benchmark/evaluations.py').read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == 'eval_result_json')
    namespace = dict(Dict=dict, np=np, math=math, t=t)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'native_evaluator', 'exec'), namespace)
    evaluate = namespace['eval_result_json']

    # Instrument the native increments, independently of the builder's prefix replay.
    class Capture(ast.NodeTransformer):
        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id.startswith('correct_'):
                return [node, ast.parse('captured.add(key)').body[0]]
            return node
    instrumented = ast.fix_missing_locations(Capture().visit(copy.deepcopy(function)))
    replay = dict(namespace, captured=set())
    exec(compile(ast.Module(body=[instrumented], type_ignores=[]), 'instrumented_native_evaluator', 'exec'), replay)
    with tarfile.open(root / 'agent_results.tar.gz') as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            if not member.isfile() or not member.name.endswith('.json') or member.name in metadata['build']['parameters']['excluded_files']:
                continue
            difficulty = member.name.removesuffix('.json').rsplit('_', 1)[1]
            for index, record in enumerate(json.load(archive.extractfile(member))['capsule_results']):
                group = record['capsule_id']
                raw_item = group + '__' + difficulty
                references = tasks[group]['results']
                score = (record['correct_written_answers'] + record['correct_vision_answers']) / (
                    record['total_written_questions'] + record['total_vision_questions'])
                counts = evaluate(references, copy.deepcopy(record['result_report']))
                if all(counts[field] == record[field] for field in counts):
                    replay['captured'].clear()
                    assert replay['eval_result_json'](references, copy.deepcopy(record['result_report'])) == counts
                    checks = [(raw_item + '::' + hashlib.sha256(question.encode()).hexdigest(), group,
                        float(question in replay['captured'])) for question, answer in references[0].items()
                        if isinstance(answer, (int, float, list, str))]
                    assert sum(c[2] for c in checks) == counts['correct_written_answers'] + counts['correct_vision_answers']
                else:
                    checks = [(raw_item, group, score)]
                yield member.name + ':' + str(index), score, checks
