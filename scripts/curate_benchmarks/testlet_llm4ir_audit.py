"""Verify individual reported assertion labels without relabeling them correctness."""

import csv
import hashlib
import io
import json
from collections import Counter

import pandas as pd


def verify(directory, tables, metadata):
    """Audit source assessment groups without asserting shared execution links."""
    from .native_result_audits import _check
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset
    validate_dataset(tables, context='LLM4IR individual predictions with unestablished execution links')
    _check('attempts' not in tables and tables['responses'].attempt_id.isna().all(), True,
        'LLM4IR does not infer a physical execution from a collection of assertion answers')
    responses = tables['responses'].copy()
    linked = responses.set_index('response_id')
    groups, assignments, remaining = {}, {}, []
    for row in tables['traces'].itertuples():
        trace = json.loads(row.trace)
        if 'source_assessment_key' not in trace:
            remaining.append(row._asdict())
            continue
        key = trace['source_assessment_key']
        identity = hashlib.sha256(key.encode()).hexdigest()
        original = json.loads(trace['source_assessment_trace'])
        position = trace['assertion_row']
        expected_output = original['output'] if position is None else list(csv.DictReader(io.StringIO(original['output'])))[position].get('comment')
        _check(trace['output'], expected_output, 'LLM4IR complete individual model answer')
        assessment = dict(attempt_id=identity, raw_attempt_id=key, benchmark_id=directory.name,
            subject_id=linked.loc[row.response_id, 'subject_id'], trace=json.dumps(dict(
                recorded_task_score=trace['recorded_program_score'], source_trace=trace['source_assessment_trace'])))
        if identity in groups:
            _check(assessment, groups[identity], 'LLM4IR exact common source assessment evidence')
        groups[identity] = assessment
        assignments[row.response_id] = identity
    responses['attempt_id'] = responses.response_id.map(assignments)
    # This temporary audit projection checks the published grouping. It is
    # never written to the dataset as an execution assertion.
    projected = dict(tables, responses=responses, attempts=pd.DataFrame(groups.values())[
        ['attempt_id', 'benchmark_id', 'subject_id', 'raw_attempt_id', 'trace']],
        traces=pd.DataFrame(remaining).drop(columns=['Index'], errors='ignore'))
    result = _verify_assessment_groups(directory, projected, metadata)
    result.update(source_assessment_collections=result['source_attempts'], source_attempts=0)
    return result


def _verify_assessment_groups(directory, tables, metadata):
    from .native_result_audits import _check, _llm4ir, _llm4ir_source_records
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='LLM4IR released per-assertion predictions')
    source = _llm4ir_source_records(directory, metadata)
    native = {key: row for key, row in source['native'].items() if row['task'] == 'execution'}
    targets = {row['item']: row for row in native.values()}
    protocol = metadata['grading']['verifiers']['execution']
    item_table = tables['items'].astype(object).where(tables['items'].notna(), None)
    items = item_table.set_index('item_id').to_dict('index')
    linked = tables['responses'].loc[tables['responses'].attempt_id.notna()]
    groups = linked.groupby('attempt_id', sort=False)
    _check(set(groups.groups), set(tables['attempts'].attempt_id), 'LLM4IR all program assessments retain observations')
    parents, mapping, used, seen = {}, {}, set(), set()
    prediction_count = unknown = 0
    for attempt in tables['attempts'].itertuples():
        key = attempt.raw_attempt_id
        _check(key not in seen and key in native, True, 'LLM4IR original recorded assessment identity')
        seen.add(key)
        original = native[key]
        evidence = json.loads(attempt.trace)
        _check(evidence['recorded_task_score'], original['grade'], 'LLM4IR original program summary retained as evidence')
        _check(json.loads(evidence['source_trace']), original['trace'], 'LLM4IR full source rows and complete assertion output')
        expected, occurrences = {}, Counter()
        if original['trace']['output'] is not None:
            for record in csv.DictReader(io.StringIO(original['trace']['output'])):
                statement = record['assert_statement']
                occurrence = occurrences[statement]
                occurrences[statement] += 1
                identity = hashlib.sha256(json.dumps([statement, occurrence]).encode()).hexdigest()
                value = 1. if record['final_result'] == 'pass' else 0. if record['final_result'] == 'fail' else None
                expected[identity] = statement, occurrence, value
        actual = Counter()
        for row in groups.get_group(attempt.attempt_id).itertuples():
            item = items[row.item_id]
            parent, _, component = item['raw_item_id'].partition('::')
            _check(parent, original['item'], 'LLM4IR original input form and program')
            _check(item['testlet_id'], original['program'], 'LLM4IR shared source-program testlet')
            _check((row.subject_id, row.trial), (attempt.subject_id, original['trial']), 'LLM4IR original model and published repetition')
            _check(json.loads(row.test_condition), original['condition'], 'LLM4IR original assessment condition')
            _check(pd.isna(row.interactors), True, 'LLM4IR no invented actors')
            base = dict(reference_answer=original['reference'], rule=protocol['rule'], response_scale=protocol['response_scale'])
            original_verifier = dict(**{'class': 'judge'}, spec=json.dumps(protocol, sort_keys=True))
            if expected:
                statement, occurrence, value = expected[component]
                criterion = dict(reference_answer=statement,
                    rule=json.dumps(dict(occurrence=occurrence,
                        rule='Record the parsed assertion prediction: pass=1, fail=0; other labels are unknown. This is the model\'s reported judgment, not an executed test verdict.'), sort_keys=True),
                    response_scale=dict(kind='discrete', values=[0, 1], direction='unordered',
                        meanings={'0': 'The model predicted fail.', '1': 'The model predicted pass.'}))
                verifier = dict(**{'class': 'judge'}, spec=json.dumps(dict(kind='reported_assertion_prediction', original_protocol=protocol), sort_keys=True))
                _check(json.loads(item['grading_criterion']), criterion, 'LLM4IR exact assertion and unordered prediction meaning')
                _check(json.loads(item['verifier']), verifier, 'LLM4IR preserved source-reported prediction protocol')
                _check(None if pd.isna(row.response) else row.response, value, 'LLM4IR original parsed label including unavailable labels')
                actual[component] += 1
                prediction_count += 1
                unknown += value is None
            else:
                _check(component, '', 'LLM4IR summary-only assessment does not invent assertion results')
                _check((json.loads(item['grading_criterion']), json.loads(item['verifier'])), (base, original_verifier),
                    'LLM4IR original fallback protocol')
                _check(len(groups.get_group(attempt.attempt_id)), 1, 'LLM4IR single retained aggregate')
            reconstructed = dict(item, item_id=parent, raw_item_id=parent,
                grading_criterion=json.dumps(base), verifier=json.dumps(original_verifier))
            if parent in parents:
                _check(reconstructed, parents[parent], 'LLM4IR every assertion retains the complete original program input')
            parents[parent] = reconstructed
            mapping[row.item_id] = parent
            used.add(row.item_id)
        if expected:
            _check(actual, Counter({name: 1 for name in expected}), 'LLM4IR complete original assertion-row coverage')
    _check(seen, set(native), 'LLM4IR all original execution-reasoning assessments')
    selected = dict(tables, responses=linked, items=tables['items'].loc[tables['items'].item_id.isin(used)])
    projected = parent_view(selected, list(parents.values()), mapping)
    projected['items'] = pd.concat([projected['items'], tables['items'].loc[~tables['items'].item_id.isin(used)]], ignore_index=True)
    projected['responses'] = pd.concat([projected['responses'], tables['responses'].loc[tables['responses'].attempt_id.isna()]], ignore_index=True)
    projected['traces'] = pd.concat([projected['traces'], tables['traces']], ignore_index=True)
    details = _llm4ir(directory, projected, metadata, source=source)
    return dict(details, source_attempts=len(tables['attempts']), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=int(tables['items'].testlet_id.nunique()),
        source_assertion_predictions=prediction_count, source_unavailable_assertion_labels=unknown)
