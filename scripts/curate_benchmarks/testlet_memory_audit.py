"""Reconcile recorded memory-check verdicts with native aggregate units."""

import hashlib
import json
import math
import re
import unicodedata


def source_attempts(directory, metadata):
    from .native_result_audits import _goodai_ltm_sources
    source = _goodai_ltm_sources(directory, metadata)
    for run, record in source['records'].items():
        definition = source['expected'][run]
        stimulus = json.dumps(definition['stimulus'], sort_keys=True, ensure_ascii=False)
        task = hashlib.sha256(unicodedata.normalize('NFC', stimulus).encode()).hexdigest()
        family = definition['stimulus']['task_type']
        reasons, targets = record['reasoning'], record['expected_responses']
        grades = []
        if family in {'Delayed Recall', 'Instruction Recall', 'Trigger Response'} and len(reasons) == len(targets):
            positive = {'Checklist correct',
                'The answer makes reference to the expected answer, which is not numerical.',
                'The answer makes reference to the expected answer, which is numerical, and the numbers match.'}
            negative = {'Checklist Incorrect',
                'The answer does not make reference to the expected answer, which is not numerical.',
                'The answer makes reference to the expected answer, which is numerical, and the numbers do not match.'}
            for i, reason in enumerate(reasons):
                yes, no = set(positive), set(negative)
                if family == 'Trigger Response':
                    yes.add(f"'{targets[i]}' is in the response.")
                    no.add(f"'{targets[i]}' is not in the response.")
                grades.append(float(reason in yes) if reason in yes | no else None)
            if None in grades or not grades or not math.isclose(sum(grades) * record['max_score'] / len(grades), record['score'], abs_tol=1e-9):
                grades = []
        elif family == 'NameList' and len(reasons) == 1:
            match = re.fullmatch(r'(\d+) names out of (\d+) were found\. Names (.+) were not in the response\.', reasons[0])
            if match and len(set(targets)) == len(targets) == int(match[2]) == record['max_score']:
                missed = match[3].split(', ')
                if set(missed) <= set(targets):
                    grades = [float(name not in missed) for name in targets]
                    if sum(grades) != record['score'] or sum(grades) != int(match[1]):
                        grades = []
        elif family == 'Spy Meeting' and len(targets) == 3 and len(reasons) >= 3:
            for i in range(3):
                reason = reasons[i]
                grades.append(1. if reason == 'Answer contains expected keyword(s)' else
                    0. if reason == str(targets[i]) + ' not found in answer.' else None)
            extra = reasons[3:]
            if (None in grades or not all(re.fullmatch(r'Answer also contains `.+`, which indicates that the agent has recalled something incorrect\.', r) for r in extra)
                    or not math.isclose(max((sum(grades) - bool(extra)) / 3, 0.), record['score'], abs_tol=1e-9)):
                grades = []
            if grades:
                grades.append(float(not extra))
        checks = [(task + '::' + ('no_conflicting_keywords' if family == 'Spy Meeting' and i == 3 else str(i)), task, grade)
            for i, grade in enumerate(grades)]
        if not checks:
            checks = [(task, task, record['score'])]
        yield run, record['score'], checks


def verify(directory, tables, metadata, source=None):
    """Preserve the original audit's input, subject, asset and full-trace checks."""
    from collections import Counter
    import pandas as pd
    from .native_result_audits import _check, _features, _goodai_ltm_sources
    from .testlet_audits import verify_component_results
    from ..build_measurement_tables.hash_measurement_ids import canonical_grading_criterion

    source = _goodai_ltm_sources(directory, metadata) if source is None else source
    verify_component_results(directory, tables=tables)
    subjects, roster = {}, Counter()
    for row in tables['subjects'].itertuples():
        extra = _features(row.subject_features_extra)
        session = extra['source_session']
        expected = source['sessions'][session]
        _check((row.display_name, row.harness, extra), (expected['model'], expected['harness'], expected['features']), 'GoodAI subject configuration')
        _check(all(pd.isna(getattr(row, field)) for field in ['harness_version', 'reasoning_effort', 'access_date']), True, 'GoodAI unrecorded settings')
        subjects[row.subject_id] = session
        roster[session] += 1
    _check(roster, Counter({key: 1 for key in source['sessions']}), 'GoodAI subject roster')
    _check(json.loads(tables['benchmarks'].iloc[0].response_scale), dict(kind='mixed'), 'GoodAI mixed scales')
    assets = tables['assets'].set_index('asset_id').to_dict('index')
    items = tables['items'].set_index('item_id').to_dict('index')
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    for name, lookup in [('assets', assets), ('items', items), ('attempts', attempts)]:
        _check(len(lookup), len(tables[name]), 'GoodAI unique ' + name)
    for key, asset in assets.items():
        _check((hashlib.sha256(asset['data']).hexdigest(), len(asset['data'])), (key, asset['byte_size']), 'GoodAI asset bytes')
    for attempt in attempts.values():
        name = attempt['raw_attempt_id']
        original = source['records'][name]
        trace = json.loads(attempt['trace'])
        _check(trace['recorded_task_score'], original['score'], 'GoodAI original aggregate')
        _check(json.loads(trace['source_trace']), dict(source_file=name, record=original), 'GoodAI full original record')
        _check(subjects[attempt['subject_id']], source['expected'][name]['session'], 'GoodAI attempt subject')
    seen_items, seen_assets = set(), set()
    checked = set()
    for row in tables['responses'].itertuples():
        attempt = attempts[row.attempt_id]
        name = attempt['raw_attempt_id']
        expected, original = source['expected'][name], source['records'][name]
        _check(row.subject_id, attempt['subject_id'], 'GoodAI observation subject')
        _check((row.trial, row.test_condition, pd.isna(row.interactors)), (expected['trial'], name, True), 'GoodAI source occasion')
        item = items[row.item_id]
        seen_items.add(row.item_id)
        if (row.item_id, name) in checked:
            continue
        checked.add((row.item_id, name))
        content = json.dumps(expected['stimulus'], sort_keys=True, ensure_ascii=False)
        task = hashlib.sha256(unicodedata.normalize('NFC', content).encode()).hexdigest()
        _check(item['testlet_id'], task, 'GoodAI shared source stimulus')
        _check(item['content'], unicodedata.normalize('NFC', content).strip(), 'GoodAI original stimulus')
        _check(_features(item['item_features']), expected['features'], 'GoodAI source task features')
        if item['raw_item_id'] == task:
            criterion = expected['criterion']
            verifier = dict(**{'class': expected['verifier_class']}, spec=json.dumps(expected['protocol'], sort_keys=True, ensure_ascii=False))
        else:
            _check(item['raw_item_id'].startswith(task + '::'), True, 'GoodAI native check identity')
            component = item['raw_item_id'][len(task) + 2:]
            family = expected['stimulus']['task_type']
            reference = ('No incorrect keyword from the native distractor lists' if component == 'no_conflicting_keywords'
                else original['expected_responses'][int(component)])
            criterion = dict(reference_answer=json.dumps(reference, ensure_ascii=False),
                rule=json.dumps(dict(task=family, native_check=component,
                    rule='Explicit released check explanation, reconciled with the published task score')),
                response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
            verifier = dict(**{'class': 'judge'}, spec=json.dumps(dict(kind='published_check_explanation',
                source_program=metadata['build']['parameters']['task_programs'][family],
                program_revision=metadata['build']['parameters']['labels']['program_revision'])))
        _check(json.loads(item['grading_criterion']), json.loads(canonical_grading_criterion(criterion)), 'GoodAI exact reference, rule and scale')
        _check(json.loads(item['verifier']), verifier, 'GoodAI check provenance')
        manifest = []
        for ordinal, (path, body) in enumerate(expected['resources'], 1):
            identity = hashlib.sha256(body).hexdigest()
            manifest.append(dict(asset_id=identity, path=path, role='task_program_reference', ordinal=ordinal,
                media_type='application/json' if path.endswith('.json') else 'text/x-python'))
            _check(assets[identity]['data'], body, 'GoodAI original program or menu')
            seen_assets.add(identity)
        _check(json.loads(item['asset_manifest']), manifest, 'GoodAI resource manifest')
    _check(seen_items, set(items), 'GoodAI item coverage')
    _check(seen_assets, set(assets), 'GoodAI asset coverage')
    return dict(source_subjects=len(subjects), source_items=len(items), source_responses=len(tables['responses']),
        source_traces=len(attempts), source_assets=len(assets), **source['counts'])
