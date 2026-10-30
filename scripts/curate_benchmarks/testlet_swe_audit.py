"""Preserve SWE task environments, test identities and complete native run audits."""

from collections import defaultdict
import hashlib
import json
import re

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _swe_live_sources, _swe_live_tabular, _swe_poly_sources, _swe_poly_tabular
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='SWE complete native test and execution evidence')
    live = directory.name == 'swebench_live'
    source = (_swe_live_sources if live else _swe_poly_sources)(directory, metadata)
    protocol = metadata['grading']['verifiers']['published_report']
    definitions = {}
    for (_, task), entry in source['native'].items():
        parent = (entry['track'] if live else entry['split']) + '/' + task
        fix = entry['evaluator_fix'] or None if live else None
        record = entry['task']['record']
        spec = dict(protocol, recorded_evaluator_fix=fix) if live else protocol
        key = parent, json.dumps(spec, sort_keys=True)
        if key in definitions:
            _check(definitions[key], record, 'SWE one complete source task for each grading context')
        definitions[key] = record
    environments = {}
    for key, record in definitions.items():
        checks = {field: record[field] for field in protocol['task_fields'] if field in record}
        payload = json.dumps(checks, sort_keys=True).encode()
        base = dict(reference_answer=record['patch'], rule=json.dumps(dict(rule=metadata['grading']['rule'], checks=checks), sort_keys=True))
        environments[key] = payload, hashlib.sha256(payload).hexdigest(), base, json.dumps(base), json.dumps(key)
    assets = tables['assets'].set_index('asset_id').data.to_dict()
    parents, mapping, used_assets = {}, {}, set()
    for item in tables['items'].to_dict('records'):
        parent, _, component = item['raw_item_id'].partition('::')
        verifier = json.loads(item['verifier'])
        key = parent, verifier['spec']
        record = definitions[key]
        _check(verifier, {'class': 'exact_matcher', 'spec': key[1]}, 'SWE original native test verifier')
        task = parent.split('/', 1)[1]
        _check(item['testlet_id'], parent if live else task, 'SWE original issue testlet across evaluation variants')
        payload, digest, base, serialized_base, identity = environments[key]
        _check(json.loads(item['asset_manifest']), [dict(asset_id=digest, path='test-environment.json',
            role='grading_reference', media_type='application/json', ordinal=1)], 'SWE exact shared grading environment attachment')
        if digest not in used_assets:
            _check(assets[digest], payload, 'SWE complete original test environment bytes')
            used_assets.add(digest)
        if component:
            criterion = json.loads(item['grading_criterion'])
            rule = json.loads(criterion['rule'])
            name = rule['test_name']
            if live:
                group = rule['test_group']
                _check(group in {'FAIL_TO_PASS', 'PASS_TO_PASS', 'FAIL_TO_FAIL', 'PASS_TO_FAIL'}, True, 'SWE-Live native test group')
                identifier = hashlib.sha256(json.dumps([group, name]).encode()).hexdigest()
                expected = dict(test_name=name, test_group=group,
                    rule='Released named test verdict; missing checks are not inferred',
                    test_environment_asset='test-environment.json', repo=record['repo'], base_commit=record['base_commit'])
            else:
                identifier = hashlib.sha256(name.encode()).hexdigest()
                expected = dict(test_name=name, rule='Recorded post-patch verdict; contradictory named verdicts remain ungraded',
                    test_environment_asset='test-environment.json', repo=record['repo'], base_commit=record['base_commit'])
            _check(component, identifier, 'SWE exact released test identifier')
            _check(criterion, dict(reference_answer=None, rule=json.dumps(expected, sort_keys=True)), 'SWE complete named test definition')
        else:
            _check(json.loads(item['grading_criterion']), base, 'SWE retained original task-level grading contract')
        reconstructed = dict(item, item_id=identity, raw_item_id=parent,
            grading_criterion=serialized_base)
        if identity in parents:
            _check(reconstructed, parents[identity], 'SWE every test retains the same full input and source attributes')
        parents[identity] = reconstructed
        mapping[item['item_id']] = identity
    _check(set(assets), used_assets, 'SWE no unused or invented grading assets')
    projected = parent_view(tables, list(parents.values()), mapping)
    if live:
        _verify_live_reports(directory, tables, source, metadata)
        # The original audit predates the explicit test-report field. Reports
        # are checked above; its existing prediction and trajectory checks remain.
        def original_trace(value):
            trace = json.loads(value)
            del trace['test_reports']
            return json.dumps(trace, ensure_ascii=False)
        projected['traces']['trace'] = projected['traces'].trace.map(original_trace)
    return (_swe_live_tabular if live else _swe_poly_tabular)(directory, projected, metadata, source=source)


def _verify_live_reports(directory, tables, source, metadata):
    from .native_result_audits import _check

    release = directory / 'raw' / metadata['build']['parameters']['layout']['release']
    runs = sorted({key[0] for key in source['native']}, key=len, reverse=True)
    reports = defaultdict(dict)
    for path in sorted(release.rglob('*.json')):
        if not (path.name == 'report.json' or path.name.endswith(('.report.json', '.results.json'))):
            continue
        name = path.relative_to(release).as_posix()
        run = next((run for run in runs if name.startswith(run + '/')), None)
        if run is None:
            continue
        match = re.search(r'(?:^|/)([^/]+?__[^/]+?-\d+)(?=/|\.|$)', name[len(run) + 1:])
        if match and (run, match[1]) in source['native']:
            reports[run, match[1]][name] = json.loads(path.read_text())
    for attempt in tables['attempts'].itertuples():
        trace = json.loads(json.loads(attempt.trace)['source_trace'])
        key = trace['source_run'], trace['instance_id']
        _check(attempt.raw_attempt_id, key[0] + '::' + key[1], 'SWE-Live original run and issue identify the execution')
        observed = {entry['source_file']: entry['record'] for entry in trace['test_reports']}
        _check(len(observed), len(trace['test_reports']), 'SWE-Live unique report associations')
        _check(observed, reports[key], 'SWE-Live every complete original test report and alias')
