"""Check Aider's final unit-test evidence and complete original executions."""

from collections import Counter
import json
import re

import pandas as pd


def catch_roster(files):
    """Tokenize captured C++ test declarations independently of the builder."""
    result = {}
    pattern = r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*|[A-Za-z_]\w*|[^\s]'
    for filename, source in files.items():
        tokens = [match.group() for match in re.finditer(pattern, source, re.S)
            if not match.group().startswith(('//', '/*'))]
        for index, token in enumerate(tokens[:-2]):
            if token == 'TEST_CASE' and tokens[index + 1] == '(':
                name = json.loads(tokens[index + 2])
                assert name not in result
                result[name] = filename
    return result


def recorded_tests(original, roster):
    """Read complete final-cycle rosters, or only explicitly named failures."""
    lines = original['history'].split('\n')
    blocks = []
    for index in range(len(lines) - 1):
        if lines[index] != '```' or not lines[index + 1].startswith(('-- ', '[', 'CMake ', 'make')):
            continue
        # Aider can append its next status message directly to the closing fence.
        end = next((offset for offset in range(index + 2, len(lines)) if lines[offset].startswith('```')), None)
        if end is not None:
            blocks.append(lines[index + 1:end])
    if len(blocks) != len(original['native_record']['tests_outcomes']):
        return {}
    final = blocks[-1]
    success = next((line for line in final if line.startswith('All tests passed (')), None)
    if success:
        count = int(re.search(r' in (\d+) test cases?\)', success).group(1))
        return dict.fromkeys(roster, 1.) if count == len(roster) else {}
    summary = next((line for line in final if line.startswith('test cases: ')), None)
    if summary is None or 'failed' not in summary:
        return {}
    fields = summary.split('|')
    total = int(fields[0].split(':')[1])
    failures = int(next(field.strip().split()[0] for field in fields[1:] if 'failed' in field))
    names = {' '.join(name.split()): name for name in roster}
    if len(names) != len(roster):
        return {}
    failed, unknown = set(), False
    index = 0
    while index < len(final):
        line = final[index]
        if len(line) >= 20 and set(line) == {'-'}:
            end = next((offset for offset in range(index + 1, len(final)) if len(final[offset]) >= 20
                and set(final[offset]) == {'-'}), None)
            if end is None:
                break
            header = final[index + 1:end]
            key = ' '.join(' '.join(header).split())
            if key not in names and len(header) > 1 and all(value.startswith('  ') for value in header[1:]):
                key = ' '.join(header[0].split())
            if key in names:
                failed.add(names[key])
            else:
                unknown = True
            index = end
        index += 1
    if len(failed) > failures:
        return {}
    if total == len(roster) and len(failed) == failures and not unknown:
        return {name: float(name not in failed) for name in roster}
    return {name: 0. for name in roster if name in failed}


def verify(directory, tables, metadata):
    from .native_result_audits import _aider_source_records, _check, _features
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='Aider component source audit')
    _check(len(tables.get('assets', [])), 0, 'Aider complete inline source files without invented assets')
    native, definitions = _aider_source_records(directory, metadata)
    rosters = {task: catch_roster(definition['test_files']) for task, definition in definitions.items()}
    observations, traces = attempt_view(tables)
    subjects = tables['subjects'].set_index('subject_id').to_dict('index')
    items = tables['items'].set_index('item_id').to_dict('index')
    children = {key: frame for key, frame in tables['responses'].groupby('attempt_id')}
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    protocol = metadata['grading']['verifiers']['functional_tests']
    seen, used_items, used_subjects = Counter(), set(), set()
    for row in observations.itertuples():
        trace = json.loads(traces[row.response_id])
        key = trace['source_member']
        original = native[key]
        evidence = {name: original[name] for name in ['native_record', 'published_record', 'history',
            'additional_instructions_md', 'final_files']}
        evidence.update(source_member=key, source_archive=metadata['build']['parameters']['layout']['archive'])
        _check(trace, evidence, 'Aider complete original conversation, output and native metadata')
        _check(attempts[row.attempt_id]['raw_attempt_id'], key, 'Aider original execution identity')
        _check(row.response, original['grade'], 'Aider original final task verdict')
        _check(row.trial, original['trial'], 'Aider original repeated-run order')
        _check(pd.isna(row.test_condition) and pd.isna(row.interactors), True, 'Aider no invented evaluation context')
        subject = subjects[row.subject_id]
        features = _features(subject['subject_features_extra'])
        for field in ['harness', 'harness_version', 'reasoning_effort']:
            if pd.notna(subject[field]):
                features[field] = subject[field]
        _check(features, original['features'], 'Aider complete model configuration and prompting variant')
        _check(subject['display_name'], original['native_record']['model'], 'Aider recorded model label')
        task = original['task']
        definition = definitions[task]
        outcomes = recorded_tests(original, rosters[task])
        parent = 'cpp/' + task
        expected = {parent + '::' + name: (name, grade) for name, grade in outcomes.items()}
        if not expected:
            expected[parent] = None, original['grade']
        observed = Counter()
        for child in children[row.attempt_id].itertuples():
            item = items[child.item_id]
            name, grade = expected[item['raw_item_id']]
            observed[item['raw_item_id']] += 1
            _check(child.response, grade, 'Aider native final test-case verdict')
            _check(item['testlet_id'], parent, 'Aider original exercise membership')
            _check(json.loads(item['content']), definition['content'], 'Aider full supplied instructions and initial files')
            _check(_features(item['item_features']), dict(lang='cpp'), 'Aider original task language')
            _check(pd.isna(item['asset_manifest']), True, 'Aider no invented file links')
            criterion = json.loads(item['grading_criterion'])
            _check(json.loads(criterion['reference_answer']), definition['references'], 'Aider original reference files')
            rule = metadata['grading']['rule'] if name is None else json.dumps(dict(test_case=name,
                test_file=rosters[task][name], rule='Native Catch test-case result in the final recorded test-feedback cycle'))
            _check(criterion['rule'], rule, 'Aider precise test identity and scoring rule')
            verifier = json.loads(item['verifier'])
            _check(verifier['class'], 'exact_matcher', 'Aider native test verifier')
            _check(json.loads(verifier['spec']), dict(**protocol, **{key: definition[key] for key in [
                'test_files', 'cmake', 'postbuild']}), 'Aider complete captured test and build configuration')
            used_items.add(child.item_id)
        _check(observed, Counter({key: 1 for key in expected}), 'Aider no omitted, duplicated or invented tests')
        seen[key] += 1
        used_subjects.add(row.subject_id)
    _check(seen, Counter({key: 1 for key in native}), 'Aider complete original execution census')
    _check(used_items, set(items), 'Aider no unused component definitions')
    _check(used_subjects, set(subjects), 'Aider no unused subject configurations')
    _check(len(subjects), len({tuple(sorted(record['features'].items())) for record in native.values()}), 'Aider distinct source configurations')
    return dict(source_attempts=len(native), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=tables['items'].testlet_id.nunique(),
        source_test_cycles=sum(len(record['native_record']['tests_outcomes']) for record in native.values()))
