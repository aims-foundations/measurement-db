"""Validate OpenBioRQ rubric items and complete original execution evidence."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _features, _jsonl
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='OpenBioRQ component source audit')
    raw = directory / 'raw'
    parameters = metadata['build']['parameters']
    layout = parameters['layout']
    protocol = metadata['grading']['verifiers']['checklist']
    questions = _jsonl(raw / layout['tasks'])
    rubrics = _jsonl(raw / layout['rubrics'])
    bank = {row['task_id']: row for row in questions}
    rubric = {row['task_id']: row for row in rubrics}
    _check(len(bank), len(questions), 'OpenBioRQ unique questions')
    _check(len(rubric), len(rubrics), 'OpenBioRQ unique rubrics')
    _check(set(bank), set(rubric), 'OpenBioRQ complete rubric coverage')
    expected_items = {}
    for task, question in bank.items():
        _check(question['self_contained_question'], rubric[task]['question'], 'OpenBioRQ original rubric question')
        for criterion in rubric[task]['criteria']:
            key = task + '::' + str(criterion['id'])
            _check(key in expected_items, False, 'OpenBioRQ unique criterion identity')
            expected_items[key] = task, criterion
    items = tables['items'].set_index('item_id').to_dict('index')
    _check(set(tables['items'].raw_item_id), set(expected_items), 'OpenBioRQ all and only native criteria')
    _check(len(items), len(expected_items), 'OpenBioRQ no duplicated definitions')
    for item in items.values():
        task, criterion = expected_items[item['raw_item_id']]
        _check(item['testlet_id'], task, 'OpenBioRQ original question membership')
        _check(item['content'], bank[task]['self_contained_question'], 'OpenBioRQ complete question')
        _check(pd.isna(item['asset_manifest']) and pd.isna(item['item_features']), True, 'OpenBioRQ no invented inputs or features')
        grading = json.loads(item['grading_criterion'])
        _check(json.loads(grading['reference_answer']), bank[task]['gold_answer'], 'OpenBioRQ complete reference grounding')
        _check(json.loads(grading['rule']), dict(rule=protocol['criterion_rule'], criterion=criterion), 'OpenBioRQ exact criterion, weight and text')
        verifier = json.loads(item['verifier'])
        _check(verifier['class'], 'judge', 'OpenBioRQ verifier class')
        _check(verifier['judged_by'], 'llm', 'OpenBioRQ grader type')
        _check(json.loads(verifier['spec']), dict(protocol, selected_criterion_id=criterion['id']), 'OpenBioRQ complete grading configuration')

    subjects = tables['subjects'].set_index('subject_id').to_dict('index')
    attempts = tables['attempts'].set_index('raw_attempt_id').to_dict('index')
    _check(len(attempts), len(tables['attempts']), 'OpenBioRQ unique physical executions')
    expected, seen, subject_ids = Counter(), set(), set()
    successes = ungraded = 0
    for model, label in parameters['models'].items():
        folder = raw / layout['results'] / model
        predictions = _jsonl(folder / 'predictions.jsonl')
        verdicts = _jsonl(folder / 'checklist.jsonl')
        checks = {row['task_id']: row for row in verdicts}
        _check(len(checks), len(verdicts), 'OpenBioRQ unique judgments')
        _check({row['task_id'] for row in predictions}, set(bank), 'OpenBioRQ full prediction coverage')
        _check(len(predictions), len(bank), 'OpenBioRQ one original attempt per question')
        _check(set(checks), set(bank), 'OpenBioRQ full judgment coverage')
        summary = json.loads((folder / 'summary.json').read_text())
        _check(summary['n_predictions'], len(predictions), 'OpenBioRQ reported attempt count')
        mode = 'enabled' if summary['tools'] else 'disabled'
        _check(mode, parameters['tool_access'][model], 'OpenBioRQ recorded tool configuration')
        scores = []
        for prediction in predictions:
            task = prediction['task_id']
            native = checks[task]
            score = native.get('checklist_score')
            aggregate = None if score is None else float(score >= protocol['success_threshold'])
            key = model + '::' + task
            attempt = attempts[key]
            evidence = json.loads(attempt['trace'])
            _check(evidence['recorded_task_score'], aggregate, 'OpenBioRQ original aggregate')
            trace = json.loads(evidence['source_trace'])
            _check(trace, dict(prediction=prediction, checklist_score=score,
                verdicts=native.get('verdicts') or []), 'OpenBioRQ full answer, tools, rationale and native judgments')
            subject = subjects[attempt['subject_id']]
            _check(subject['display_name'], parameters.get('subject_labels', {}).get(label, label), 'OpenBioRQ original model label')
            _check(subject['harness'], parameters['subject_features']['harness'], 'OpenBioRQ harness')
            _check(_features(subject['subject_features_extra']), dict(tool_mode=mode), 'OpenBioRQ complete tool settings')
            weights = {criterion['id']: criterion['weight'] for criterion in rubric[task]['criteria']}
            if score is not None:
                _check({v['id'] for v in native['verdicts']}, set(weights), 'OpenBioRQ graded checklist coverage')
                _check(len(native['verdicts']), len(weights), 'OpenBioRQ one verdict per criterion')
                reconstructed = sum(weights[v['id']] * protocol['verdict_values'][v['v']]
                    for v in native['verdicts']) / sum(weights.values())
                if abs(reconstructed - score) > .000501:
                    raise ValueError('OpenBioRQ weighted verdicts disagree with the original aggregate')
                scores.append(score)
                successes += int(aggregate)
            else:
                ungraded += 1
            flags = {verdict['id']: protocol['verdict_values'][verdict['v']] for verdict in native.get('verdicts') or []}
            for identifier in weights:
                expected[attempt['attempt_id'], attempt['subject_id'], task + '::' + str(identifier),
                    flags.get(identifier), 1, parameters['conditions']['test_condition'], None] += 1
            seen.add(key)
            subject_ids.add(attempt['subject_id'])
        _check(summary['n_judged'], len(scores), 'OpenBioRQ graded count')
        if scores and abs(100 * sum(s >= .5 for s in scores) / len(scores) - summary['full_core_657']['solve@0.5_pct']) > .050001:
            raise ValueError('OpenBioRQ task outcomes disagree with the published solve rate')
    actual = Counter((r.attempt_id, r.subject_id, items[r.item_id]['raw_item_id'],
        None if pd.isna(r.response) else r.response, r.trial, r.test_condition,
        None if pd.isna(r.interactors) else r.interactors) for r in tables['responses'].itertuples())
    _check(actual, expected, 'OpenBioRQ every native outcome and original occasion')
    _check(seen, set(attempts), 'OpenBioRQ complete execution census, including ungraded runs')
    _check(subject_ids, set(subjects), 'OpenBioRQ no unused subject configuration')
    _check(len(subjects), len(parameters['models']), 'OpenBioRQ distinct native configurations')
    return dict(source_attempts=len(attempts), source_component_observations=sum(expected.values()),
        imported_items=len(items), source_observed_testlets=len(bank), source_successes=successes,
        source_ungraded_observations=ungraded)
