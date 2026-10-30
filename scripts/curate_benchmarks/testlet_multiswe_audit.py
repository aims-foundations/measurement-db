"""Reconcile Multi-SWE inputs, grading assets, configurations and full patches."""

from collections import Counter
import hashlib
import json
import re

import pandas as pd


def _records(path):
    if not path.is_file():
        return []
    text = path.read_text()
    if text.startswith('version https://git-lfs.github.com/spec/v1'):
        return []
    return [json.loads(line) for line in text.split('\n') if line.strip()]


def _tests(record, primary, fallback):
    value = record.get(primary)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = None
    if isinstance(value, list) and value:
        return list(map(str, value))
    return sorted(record[fallback]) if isinstance(record.get(fallback), dict) else []


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _features
    from ..build_measurement_tables.register_measurements import _subject_row

    raw = directory / 'raw'
    tasks = {}
    for path in sorted(raw.glob('*.jsonl')):
        for record in _records(path):
            task = record.get('instance_id') or f"{record['org']}__{record['repo']}-{record['number']}"
            _check(task not in tasks, True, 'Multi-SWE unique original task definition')
            tasks[task] = record
    fallback = pd.read_parquet(raw / 'swebench_verified.parquet').set_index('instance_id').problem_statement.to_dict()
    definitions = {}
    for task, record in tasks.items():
        issues = [value for value in record.get('resolved_issues') or [] if isinstance(value, dict)]
        texts = [((issue.get('title') or '').strip() + '\n' + (issue.get('body') or '').strip()).strip() for issue in issues]
        issue = '\n\n'.join(value for value in texts if value) or (
            (record.get('title') or '').strip() + '\n' + (record.get('body') or '').strip()).strip() or fallback.get(task, '').strip()
        content = json.dumps(dict(issue=issue, repository_context=dict(repository='/'.join(str(record.get(k) or '') for k in ('org', 'repo')),
            base=record.get('base'), base_commit=record.get('base_commit'))), ensure_ascii=False, sort_keys=True)
        passed, failed = _tests(record, 'PASS_TO_PASS', 'p2p_tests'), _tests(record, 'FAIL_TO_PASS', 'f2p_tests')
        protocol = dict(test_patch=record.get('test_patch') or None, fail_to_pass=failed, pass_to_pass=passed, n_pass_to_pass=len(passed))
        reference = record.get('fix_patch') or record.get('patch') or None
        payload = json.dumps(dict(test_protocol=protocol, reference_patch=reference), ensure_ascii=False).encode()
        definitions[task] = content, protocol, reference, payload, hashlib.sha256(payload).hexdigest()

    unescape = lambda value: re.sub(r'_x([0-9a-f]{2})_', lambda match: chr(int(match[1], 16)), value)
    runs, dates = {}, {}
    for path in sorted((raw / 'results').glob('*.json'), key=lambda p: (unescape(p.name), p.name)):
        label = unescape(path.stem)
        language, name = label.split('__', 1)
        dated = re.fullmatch(r'(\d{4})(\d{2})(\d{2})_(.+)', name)
        date = '-'.join(dated.groups()[:3]) if dated else None
        name = dated[4] if dated else name
        harness, _, model = name.partition('_')
        if not model:
            model, harness = harness, None
        record = json.loads(path.read_text())
        outcomes = {}
        for value, identifiers in [(1., record.get('resolved') or record.get('resolved_ids') or []),
                                   (0., record.get('unresolved_ids') or record.get('unresolved') or [])]:
            for identifier in identifiers:
                task = re.sub(r'^([^/]+)/([^:]+):pr-(\d+)$', r'\1__\2-\3', identifier)
                _check(task not in outcomes, True, 'Multi-SWE unique original task outcome')
                outcomes[task] = value
        if not outcomes:
            continue
        key = model, harness
        dates[key] = min(v for v in [dates.get(key), date] if v is not None) if dates.get(key) or date else None
        predictions = {r['instance_id']: r['model_patch'] for r in _records(raw / 'preds' / (path.stem + '.jsonl'))
            if r.get('instance_id') is not None and isinstance(r.get('model_patch'), str)}
        runs[label] = dict(path=path, model=model, harness=harness, language=language, outcomes=outcomes, predictions=predictions)
    settings_file = raw / 'subject_settings.json'
    settings = json.loads(settings_file.read_text()) if settings_file.is_file() else {}
    expected_subjects = {}
    for run in runs.values():
        features = {'harness': run['harness']} if run['harness'] else {}
        subject = _subject_row(run['model'], dict(features, **settings.get(run['model'], {})), dates[run['model'], run['harness']])
        run['subject'] = subject['subject_id']
        expected_subjects.setdefault(subject['subject_id'], subject)
    subjects = tables['subjects'].astype(object).where(tables['subjects'].notna(), None).set_index('subject_id').to_dict('index')
    _check(subjects, {key: {k: v for k, v in row.items() if k != 'subject_id'} for key, row in expected_subjects.items()},
        'Multi-SWE exact model, harness and earliest recorded access date')
    items = tables['items'].set_index('item_id').to_dict('index')
    assets = tables['assets'].set_index('asset_id').data.to_dict()
    groups = tables['responses'].groupby('attempt_id', sort=False)
    seen, used, used_assets, trials, reports = set(), set(), set(), Counter(), 0
    for attempt in tables['attempts'].itertuples():
        label, task = attempt.raw_attempt_id.split('::', 1)
        run = runs[label]
        _check(attempt.subject_id, run['subject'], 'Multi-SWE original model/run association')
        _check((label, task) not in seen, True, 'Multi-SWE unique original run')
        seen.add((label, task))
        content, protocol, reference, payload, digest = definitions[task]
        folder = raw / 'checks' / run['path'].stem / task
        report = json.loads((folder / 'report.json').read_text()) if (folder / 'report.json').exists() else None
        reports += report is not None
        evidence = json.loads(attempt.trace)
        _check(evidence['recorded_task_score'], run['outcomes'][task], 'Multi-SWE retained task success')
        _check(json.loads(evidence['source_trace']), dict(source_result_file=run['path'].relative_to(raw).as_posix(), source_run=label,
            instance_id=task, model_patch=run['predictions'].get(task), test_report=report,
            test_report_source=(folder / 'report.json').relative_to(raw).as_posix() if report is not None else None),
            'Multi-SWE complete original submitted patch and test report')
        trials[attempt.subject_id, task] += 1
        for row in groups.get_group(attempt.attempt_id).itertuples():
            item = items[row.item_id]
            parent, _, check = item['raw_item_id'].partition('::')
            _check((parent, item['testlet_id'], item['content']), (task, task, content), 'Multi-SWE full issue and repository revision')
            _check(_features(item['item_features']), {'lang': run['language']}, 'Multi-SWE recorded language')
            _check((row.subject_id, row.trial), (attempt.subject_id, trials[attempt.subject_id, task]), 'Multi-SWE original subject and trial')
            _check(pd.isna(row.test_condition) and pd.isna(row.interactors), True, 'Multi-SWE no invented observation settings')
            _check(json.loads(item['asset_manifest']), [dict(asset_id=digest, path='test-environment.json',
                role='grading_reference', media_type='application/json', ordinal=1)], 'Multi-SWE linked full grading environment')
            if digest not in used_assets:
                _check(assets[digest], payload, 'Multi-SWE complete reference patch and test environment bytes')
                used_assets.add(digest)
            criterion = dict(reference_answer=reference, rule=metadata['grading']['rule'])
            spec = protocol
            if check:
                criterion = dict(reference_answer=None, rule=json.dumps(dict(test_name=check,
                    rule='Named test result after applying the exact submitted patch and reference test patch',
                    test_environment_asset='test-environment.json')))
                spec = dict(kind='native_fix_patch_result', test_environment_asset='test-environment.json')
            _check(json.loads(item['grading_criterion']), criterion, 'Multi-SWE exact original check or retained aggregate')
            _check(json.loads(item['verifier']), dict(**{'class': 'exact_matcher'}, spec=json.dumps(spec)), 'Multi-SWE complete checker specification')
            used.add(row.item_id)
    _check(seen, {(label, task) for label, run in runs.items() for task in run['outcomes']}, 'Multi-SWE complete source run census')
    _check((used, used_assets), (set(items), set(assets)), 'Multi-SWE complete task and asset coverage')
    return dict(source_full_runs=len(seen), source_test_reports=reports, source_subject_configurations=len(subjects), source_grading_assets=len(assets))
