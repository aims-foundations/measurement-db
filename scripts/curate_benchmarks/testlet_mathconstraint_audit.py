"""Independently reconcile native SAT decisions, witness checks and full attempts."""

from collections import Counter, defaultdict
import hashlib
import json
import unicodedata

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='MathConstraint decision and witness checks')
    raw = directory / 'raw'
    definitions = {}
    for path in sorted(raw.glob('*_instances/*.json')):
        record = json.loads(path.read_text())
        if not isinstance(record.get('prompt'), str) or not record['prompt']:
            continue
        name = record.get('name') or path.stem
        _check(type(record['satisfiable']), bool, 'MathConstraint native problem satisfiability')
        reference = (json.dumps(record['solution'], sort_keys=True) if record['solution'] is not None else 'SATISFIABLE') if record['satisfiable'] else 'UNSATISFIABLE'
        definition = record['prompt'], reference, record['satisfiable']
        _check(definitions.get(name, definition), definition, 'MathConstraint identical overlapping task definitions')
        definitions[name] = definition
    source, skipped = {}, 0
    parameters = metadata['build']['parameters']
    for path in sorted(raw.glob('results/*/*/*.json')):
        if path.parent.name not in parameters['splits'] or path.stem not in parameters['conditions']:
            continue
        run = json.loads(path.read_text())
        source_file = path.relative_to(raw).as_posix()
        for position, record in enumerate(run['results']):
            task = record['problem_name']
            if task not in definitions or record['correct'] is None:
                skipped += 1
                continue
            _check(record['is_satisfiable'], definitions[task][2], 'MathConstraint original problem/result association')
            for field in ('correct', 'satisfiability_correct', 'solution_correct'):
                _check(type(record[field]), bool, 'MathConstraint explicit native Boolean judgment')
            source[source_file + ':' + str(position)] = dict(task=task, record=record,
                model=run.get('model') or path.parent.parent.name,
                condition='split=' + path.parent.name + ';tools=' + parameters['conditions'][path.stem],
                trace=dict(source_file=source_file, source_row=position, source_record=record,
                    run_metadata={key: value for key, value in run.items() if key != 'results'}))
    observed_tasks = {row['task'] for row in source.values()}
    items = tables['items'].set_index('item_id').to_dict('index')
    roster, item_checks = Counter(), {}
    testlets = {name: hashlib.sha256(unicodedata.normalize('NFC', definition[0]).strip().encode()).hexdigest()
        for name, definition in definitions.items()}
    for item_id, item in items.items():
        task, field = item['raw_item_id'].split('::')
        prompt, reference, satisfiable = definitions[task]
        _check(item['testlet_id'], testlets[task], 'MathConstraint original problem stimulus testlet')
        _check(item['content'], unicodedata.normalize('NFC', prompt).strip(), 'MathConstraint complete original problem prompt')
        _check(pd.isna(item['item_features']) and pd.isna(item['asset_manifest']), True,
            'MathConstraint no invented task attributes or assets')
        _check(field in ('satisfiability_correct', 'solution_correct') and (satisfiable or field != 'solution_correct'),
            True, 'MathConstraint applicable source-declared check')
        rule = metadata['grading']['verifiers']['components'][field]['rule']
        answer = reference if field == 'solution_correct' else ('SATISFIABLE' if satisfiable else 'UNSATISFIABLE')
        _check(json.loads(item['grading_criterion']), dict(reference_answer=answer, rule=rule),
            'MathConstraint exact decision or witness criterion')
        verifier = dict(metadata['grading']['verifiers']['solver'], selected_field=field, component_rule=rule)
        _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(verifier, sort_keys=True)},
            'MathConstraint native grading provenance')
        item_checks[item_id] = testlets[task], field, answer
        roster[item_checks[item_id]] += 1
    expected_roster = Counter({(testlets[task], field,
        definitions[task][1] if field == 'solution_correct' else ('SATISFIABLE' if definitions[task][2] else 'UNSATISFIABLE')): 1
        for task in observed_tasks for field in ['satisfiability_correct'] + (['solution_correct'] if definitions[task][2] else [])})
    _check(roster, expected_roster, 'MathConstraint exact source check coverage, with no vacuous UNSAT witness')

    subjects = tables['subjects'].set_index('subject_id').to_dict('index')
    _check(Counter(row['display_name'] for row in subjects.values()),
        Counter({row['model']: 1 for row in source.values()}), 'MathConstraint complete original model labels')
    for subject in subjects.values():
        _check(all(pd.isna(subject[field]) for field in ('harness', 'harness_version', 'reasoning_effort', 'access_date', 'subject_features_extra')),
            True, 'MathConstraint no invented model configuration')
    original, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    seen, expected, trials = Counter(), Counter(), defaultdict(list)
    for row in original.itertuples():
        key = attempts[row.attempt_id]['raw_attempt_id']
        native = source[key]
        task, record = native['task'], native['record']
        _check(json.loads(traces[row.response_id]), native['trace'],
            'MathConstraint complete native answer, reasoning, tool chain, configuration and validation details')
        _check((subjects[row.subject_id]['display_name'], items[row.item_id]['testlet_id'], row.test_condition, row.response),
            (native['model'], testlets[task], native['condition'], int(record['correct'])),
            'MathConstraint original model, problem, condition and aggregate verdict')
        _check(pd.isna(row.interactors), True, 'MathConstraint no invented interacting subjects')
        for field in ['satisfiability_correct'] + (['solution_correct'] if definitions[task][2] else []):
            answer = definitions[task][1] if field == 'solution_correct' else ('SATISFIABLE' if definitions[task][2] else 'UNSATISFIABLE')
            expected[row.attempt_id, (testlets[task], field, answer), int(record[field])] += 1
        trials[row.subject_id, testlets[task], row.test_condition].append(row.trial)
        seen[key] += 1
    actual = Counter((r.attempt_id, item_checks[r.item_id], r.response) for r in tables['responses'].itertuples())
    _check(actual, expected, 'MathConstraint exact native judgments for every recorded execution')
    _check(seen, Counter({key: 1 for key in source}), 'MathConstraint no missing or duplicated executions')
    for values in trials.values():
        _check(sorted(values), list(range(1, len(values) + 1)), 'MathConstraint original repeat occasions')
    return dict(source_attempts=len(source), source_component_observations=len(actual),
        imported_items=len(items), source_observed_testlets=len({testlets[task] for task in observed_tasks}),
        source_task_aliases=len(observed_tasks), source_excluded_records=skipped,
        source_successes=sum(row['record']['correct'] for row in source.values()))
