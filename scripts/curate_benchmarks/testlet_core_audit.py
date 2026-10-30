"""Reconcile CORE-Bench subquestions, original prompts, reports and run settings."""

import hashlib
import json
import tarfile

import pandas as pd


def verify(directory, tables, metadata):
    from .batch2_audits import verify_batch2
    from .native_result_audits import _check
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.load_source_files import read_gpg_json
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='CORE-Bench complete native component evidence')
    root = directory / 'raw/core-bench'
    tasks = json.loads((root / 'benchmark/dataset/core_train.json').read_text())
    tasks += read_gpg_json(root / 'benchmark/dataset/core_test.json.gpg',
        password=metadata['build']['parameters']['task_bank']['public_password'], scratch_dir=directory)
    bank = {task['capsule_id']: task for task in tasks}
    templates = json.loads((root / 'benchmark/benchmark_prompts.json').read_text())
    protocol = metadata['grading']['verifiers']['provider_result']
    parents, mapping = {}, {}
    for item in tables['items'].to_dict('records'):
        parent, _, component = item['raw_item_id'].partition('::')
        capsule, difficulty = parent.split('__')
        task = bank[capsule]
        content = templates['codeocean_' + difficulty].replace('{task_prompt}', task['task_prompt'])
        content = content.replace('{json_fields}', str(task['results'][0].keys()))
        content += '\n\nScientific code capsule: ' + capsule + '\nCapsule DOI: ' + task['capsule_doi']
        _check((item['testlet_id'], item['content'], item['item_features']),
            (capsule, content, 'difficulty=' + difficulty), 'CORE original capsule input and difficulty')
        _check(pd.isna(item['asset_manifest']), True, 'CORE no invented capsule attachments')
        base = dict(reference_answer=json.dumps(task['results'], ensure_ascii=False, sort_keys=True),
            rule=metadata['grading']['rule'], response_scale=dict(kind='interval', min=0, max=1, direction='higher_is_better'))
        expected, spec = base, protocol
        if component:
            questions = [q for q, answer in task['results'][0].items()
                if isinstance(answer, (int, float, list, str)) and hashlib.sha256(q.encode()).hexdigest() == component]
            _check(len(questions), 1, 'CORE exact released subquestion identity')
            question = questions[0]
            expected = dict(reference_answer=json.dumps([result[question] for result in task['results']], ensure_ascii=False),
                rule=json.dumps(dict(question=question,
                    rule='Exact replay of the pinned evaluator, preserving reported answer order and early-abort behavior'), ensure_ascii=False, sort_keys=True),
                response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
            spec = dict(source=protocol['source'], function='eval_result_json', selected_question=question)
        _check(json.loads(item['grading_criterion']), expected, 'CORE full original question, references and response scale')
        _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(spec, sort_keys=True)},
            'CORE original evaluator provenance')
        parents.setdefault(parent, dict(item, item_id=parent, raw_item_id=parent,
            grading_criterion=json.dumps(base), verifier=json.dumps({'class': 'exact_matcher', 'spec': json.dumps(protocol, sort_keys=True)})))
        mapping[item['item_id']] = parent
    originals = {}
    with tarfile.open(root / 'agent_results.tar.gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.json') or member.name in metadata['build']['parameters']['excluded_files']:
                continue
            for index, record in enumerate(json.load(archive.extractfile(member))['capsule_results']):
                originals[member.name + ':' + str(index)] = record
    _check(set(originals), set(tables['attempts'].raw_attempt_id), 'CORE every released run retained once')
    for attempt in tables['attempts'].itertuples():
        trace = json.loads(attempt.trace)
        _check(json.loads(trace['source_trace']), originals[attempt.raw_attempt_id]['result_report'], 'CORE complete original report including empty reports')
    projected = parent_view(tables, list(parents.values()), mapping)
    # The original native audit enumerates nonempty report dictionaries. The
    # complete report audit above additionally checks empty and null reports.
    projected['traces'] = projected['traces'][['response_id', 'trace']].copy()
    projected['traces'] = projected['traces'].loc[projected['traces'].trace.map(
        lambda value: isinstance(json.loads(value), dict) and bool(json.loads(value)))]
    return verify_batch2(directory, tables=projected)
