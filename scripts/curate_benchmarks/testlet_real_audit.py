"""Reconcile REAL's complete provider records, task definitions and run settings."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _features
    from ..build_measurement_tables.register_measurements import _subject_row

    raw = directory / 'raw'
    settings_file = raw / 'subject_settings.json'
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    definitions = {path.stem: json.loads(path.read_text()) for path in (raw / 'tasks').glob('*.json')}
    native, inputs, subjects, index = {}, {}, {}, 0
    for model in json.loads((raw / 'model_data.json').read_text())['aiModels']:
        if not model.get('name'):
            continue
        subject = _subject_row(model['name'], settings.get(model['name'], {}), None)
        subjects.setdefault(subject['subject_id'], subject)
        for site in model.get('websites') or []:
            website = site.get('websiteId') or site.get('name')
            for record in site.get('tasks') or [None]:
                run = str(index)
                index += 1
                if record is None or record.get('id') is None or record.get('accuracy') is None:
                    continue
                task = str(record['id'])
                inputs.setdefault((task, website), record.get('prompt') or task)
                native[run] = record, task, website, subject['subject_id']
    observed_subjects = tables['subjects'].astype(object).where(tables['subjects'].notna(), None).set_index('subject_id').to_dict('index')
    _check(observed_subjects, {key: {k: v for k, v in row.items() if k != 'subject_id'} for key, row in subjects.items()},
        'REAL every named subject and its recorded settings')
    items = tables['items'].set_index('item_id').to_dict('index')
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    groups = tables['responses'].groupby('attempt_id', sort=False)
    _check(set(groups.groups), set(attempts), 'REAL every original execution is observed')
    seen, used, trials = set(), set(), Counter()
    for attempt in tables['attempts'].itertuples():
        record, task, website, subject = native[attempt.raw_attempt_id]
        _check(attempt.raw_attempt_id not in seen, True, 'REAL unique source-row identity')
        seen.add(attempt.raw_attempt_id)
        evidence = json.loads(attempt.trace)
        _check(json.loads(evidence['source_trace']), record, 'REAL complete unmodified provider task record and answer')
        _check(attempt.subject_id, subject, 'REAL original model attribution')
        checks = (definitions.get(task) or {}).get('evals')
        verifier = dict(**{'class': 'exact_matcher'}, spec=json.dumps(metadata['grading']['verifiers'][
            'task_checks' if checks else 'provider_result'], sort_keys=True))
        if any(check.get('type') == 'llm_boolean' for check in checks or []):
            verifier.update({'class': 'judge', 'judged_by': 'llm'})
        trials[subject, task] += 1
        for row in groups.get_group(attempt.attempt_id).itertuples():
            item = items[row.item_id]
            parent, _, component = item['raw_item_id'].partition('::')
            _check((parent, item['testlet_id'], item['content']), (task, task, inputs[task, website]), 'REAL first captured full task input')
            _check(_features(item['item_features']), {'website': website} if website else {}, 'REAL original website')
            _check((row.subject_id, row.trial), (subject, trials[subject, task]), 'REAL original model and consecutive run occurrence')
            _check(pd.isna(row.test_condition) and pd.isna(row.interactors) and pd.isna(item['asset_manifest']), True,
                'REAL no invented configuration, interactor or environment snapshot')
            _check(json.loads(item['verifier']), verifier, 'REAL exact source checker description')
            rule = json.dumps(dict(check_id=component,
                rule='Explicit membership in the provider passed/failed lists; conflicting verdicts remain ungraded',
                source_task_checks=checks, definition_mapping='Native check ID; individual rule mapping is not asserted'), sort_keys=True) if component else (
                json.dumps(checks, sort_keys=True) if checks else metadata['grading']['fallback_rule'])
            _check(json.loads(item['grading_criterion']), dict(reference_answer=None, rule=rule), 'REAL complete check definition and explicit mapping limits')
            used.add(row.item_id)
    _check((seen, used), (set(native), set(items)), 'REAL complete execution and definition coverage')
    return dict(source_records=len(native), source_full_answers=sum(bool(row[0].get('retrievedAnswer')) for row in native.values()),
        source_subject_configurations=len(subjects), source_task_contexts=len(inputs),
        source_unreleased_task_definitions=len({task for task, _ in inputs if task not in definitions}))
