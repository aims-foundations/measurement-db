"""Check individual HealthAdmin evaluations and both policies on each original run."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _healthadmin, _healthadmin_source_records
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='HealthAdmin shared executions and complete evaluator evidence')
    source = _healthadmin_source_records(directory, metadata)
    native, tasks, flags, website, _ = source
    protocol = metadata['grading']['verifiers']
    items = tables['items'].set_index('item_id').to_dict('index')
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    rows = tables['responses'].groupby('attempt_id', sort=False)
    _check(set(rows.groups), set(attempts), 'HealthAdmin every run has linked observations')
    projected_items, projected_rows, projected_traces = {}, [], []
    seen, used = Counter(), set()

    for attempt_id, observations in rows:
        attempt = attempts[attempt_id]
        evidence = json.loads(attempt['trace'])
        trace = json.loads(evidence['source_trace'])
        record = trace['source_record']
        name, path = record['run_name'], trace['task_path']
        _check(attempt['raw_attempt_id'], name + ':pass_orig', 'HealthAdmin original physical run identity')
        _check(trace['source_field'], 'pass_orig', 'HealthAdmin one original execution for both grading policies')
        _check(path, native[name]['path'], 'HealthAdmin original task association')
        task = tasks[path]
        first = observations.iloc[0]
        prototype = items[first.item_id]
        raw_id = record['domain'] + '/' + record['task_id']
        condition = 'obs=' + record['input_type'] + ';prompt=' + record['prompt_type']
        _check(evidence['recorded_task_score'], float(record['pass_orig']), 'HealthAdmin retained original strict grade')
        available = bool(website[name]['trajectory_json'])

        for row in observations.itertuples():
            item = items[row.item_id]
            _check((row.subject_id, row.trial, row.test_condition, pd.isna(row.interactors)),
                (attempt['subject_id'], 1, condition, True), 'HealthAdmin unchanged subject, occasion and recorded configuration')
            _check(item['testlet_id'], path, 'HealthAdmin explicit source task testlet')
            for field in ['content', 'item_features', 'asset_manifest']:
                left, right = item[field], prototype[field]
                _check(None if pd.isna(left) else left, None if pd.isna(right) else right,
                    'HealthAdmin every child preserves the complete task ' + field)
            if available:
                prefix, separator, index = item['raw_item_id'].rpartition('::')
                _check((prefix, separator), (raw_id, '::'), 'HealthAdmin named task check')
                position = int(index)
                _check(index, str(position), 'HealthAdmin exact native check index')
                criterion = task['evals'][position]
                rule = dict(criterion=criterion, eval_index=position,
                    removed_from_revised_policy=position in flags.get(path, {}))
                _check(json.loads(item['grading_criterion']), dict(reference_answer=None,
                    rule=json.dumps(rule, ensure_ascii=False, sort_keys=True)), 'HealthAdmin full criterion and policy applicability')
                verifier = dict(spec=json.dumps(dict(protocol['task_evaluator'], eval_index=position), sort_keys=True))
                verifier['class'] = 'judge' if criterion['type'] == 'llm_judge' else 'exact_matcher'
                if verifier['class'] == 'judge':
                    verifier['judged_by'] = 'llm'
                _check(json.loads(item['verifier']), verifier, 'HealthAdmin native checker provenance')
            used.add(row.item_id)

        # Re-expose the two historical assessments to the unchanged native audit.
        # Every retained input and trace field comes from the converted tables.
        for field, policy in metadata['build']['parameters']['protocols'].items():
            selected = [value for index, value in enumerate(task['evals'])
                if field == 'pass_orig' or index not in flags.get(path, {})]
            rule = metadata['grading']['rule'] + '\n' + json.dumps(dict(protocol=policy, evals=selected), ensure_ascii=False)
            verifier = dict(spec=json.dumps(dict(aggregation=protocol[policy],
                task_evaluator=protocol['task_evaluator']), sort_keys=True))
            verifier['class'] = 'judge' if any(value['type'] == 'llm_judge' for value in selected) else 'exact_matcher'
            if verifier['class'] == 'judge':
                verifier['judged_by'] = 'llm'
            parent_id = json.dumps([path, record['prompt_type'], policy])
            parent = dict(prototype, item_id=parent_id, raw_item_id=raw_id,
                grading_criterion=json.dumps(dict(reference_answer=None, rule=rule)), verifier=json.dumps(verifier))
            if not available:
                candidates = [items[row.item_id] for row in observations.itertuples()
                    if json.loads(items[row.item_id]['grading_criterion']) == json.loads(parent['grading_criterion'])]
                _check(len(candidates), 1, 'HealthAdmin both aggregate-only policy definitions retained')
                original = candidates[0]
                _check(json.loads(original['verifier']), verifier, 'HealthAdmin retained aggregate checker')
                _check(original['raw_item_id'], raw_id, 'HealthAdmin retained aggregate task identity')
            if parent_id in projected_items:
                _check(parent, projected_items[parent_id], 'HealthAdmin consistent task input across runs')
            projected_items[parent_id] = parent
            response_id = name + ':' + field
            observation = first.to_dict()
            observation.update(response_id=response_id, item_id=parent_id, attempt_id=None, response=float(record[field]))
            projected_rows.append(observation)
            projected_traces.append(dict(response_id=response_id, subject_id=first.subject_id,
                item_id=parent_id, trial=first.trial, test_condition=first.test_condition,
                trace=json.dumps(dict(trace, source_field=field))))
        seen[name] += 1

    _check(seen, Counter({name: 1 for name in native}), 'HealthAdmin all physical runs exactly once')
    _check(used, set(items), 'HealthAdmin every imported definition checked')
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(projected_items.values()), responses=pd.DataFrame(projected_rows),
        traces=pd.DataFrame(projected_traces))
    return _healthadmin(directory, projected, metadata, source=source)
