"""Validate native company checkpoints, complete tasks and original trajectories."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .batch2_audits import verify_batch2
    from .native_result_audits import _check
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='TheAgentCompany complete checkpoint and execution evidence')
    layout = metadata['build']['parameters']['layout']
    tasks = directory / 'raw' / layout['tasks']
    runs = directory / 'raw' / layout['runs']
    protocol = metadata['grading']['verifiers']['checkpoint']
    parents, mapping = {}, {}
    actual = Counter()
    for item in tables['items'].to_dict('records'):
        task, _, component = item['raw_item_id'].partition('::')
        _check(item['testlet_id'], '1.0.0/' + task, 'Company original task version and identity')
        _check(item['content'], (tasks / task / 'task.md').read_text().strip(), 'Company complete original task prompt')
        code = (tasks / task / 'evaluator.py').read_text()
        base_spec = dict(protocol, grading_code=code)
        base_rule = dict(reference_answer=None, rule=metadata['grading']['rule'])
        if component == 'ungraded_task_outcome':
            rule, spec = base_rule, base_spec
        else:
            kind, index, unit, total = component.split(':')
            _check((kind, unit), ('checkpoint', 'out-of'), 'Company native checkpoint identity and available credit')
            index, total = int(index), int(total)
            _check(index > 0 and total > 0, True, 'Company positive native checkpoint indices and credit')
            rule = dict(reference_answer=None, rule=json.dumps(dict(rule=protocol['checkpoint_rule'],
                checkpoint_index=index, available_credit=total), sort_keys=True))
            spec = dict(base_spec, checkpoint_index=index, available_credit=total)
        _check(json.loads(item['grading_criterion']), rule, 'Company native checkpoint criterion')
        _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(spec, sort_keys=True)},
            'Company complete evaluator code and checkpoint provenance')
        _check(pd.isna(item['asset_manifest']) and pd.isna(item['item_features']), True,
            'Company no invented task assets or attributes')
        parents.setdefault(task, dict(item, item_id=task, raw_item_id=task,
            grading_criterion=json.dumps(base_rule), verifier=json.dumps({'class': 'exact_matcher', 'spec': json.dumps(base_spec, sort_keys=True)})))
        mapping[item['item_id']] = task
        actual[task, component] += 1

    expected = set()
    for attempt in tables['attempts'].itertuples():
        trace = json.loads(json.loads(attempt.trace)['source_trace'])
        filename = attempt.raw_attempt_id.rsplit('/', 1)[-1]
        if filename.startswith('eval_'):
            native = json.loads((runs / attempt.raw_attempt_id).read_text())
            task = filename.removeprefix('eval_').removesuffix('.json').removesuffix('-image')
            _check(trace['evaluation'], native, 'Company complete original evaluator result object')
            expected.update((task, f"checkpoint:{index}:out-of:{value['total']}")
                for index, value in enumerate(native['checkpoints'], 1) if value['total'])
        else:
            _check(trace['evaluation'], None, 'Company no fabricated evaluation for an ungraded run')
            task = filename.removeprefix('traj_').removesuffix('.gz').removesuffix('.json').removesuffix('.txt').removesuffix('-image')
            expected.add((task, 'ungraded_task_outcome'))
    _check(actual, Counter({key: 1 for key in expected}), 'Company complete native checkpoint roster')
    projected = parent_view(tables, list(parents.values()), mapping)
    projected['traces'] = projected['traces'][['response_id', 'trace']].copy()
    projected['traces']['trace'] = projected['traces'].trace.map(lambda text: json.loads(text)['trajectory'])
    return verify_batch2(directory, tables=projected)
