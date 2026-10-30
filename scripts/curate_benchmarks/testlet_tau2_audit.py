"""Preserve tau2's complete input, grading and execution checks after expansion."""

import hashlib
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _tau2_bench
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset
    from ..build_measurement_tables.normalize_evaluation_settings import canonicalize_features, features_string

    validate_dataset(tables, context='tau2 complete native component evidence')
    raw = directory / 'raw'
    paths = sorted((raw / 'repo_results').glob('*.json')) + sorted(
        path for path in (raw / 's3_trajectories').glob('*/*.json') if path.name != 'submission.json')
    runs = {str(path.relative_to(raw)): json.loads(path.read_text()) for path in paths}
    tasks = {(filename, str(task['id'])): task for filename, run in runs.items() for task in run['tasks']}
    items = tables['items'].set_index('item_id').to_dict('index')
    originals, traces = attempt_view(tables)
    parents, associations, verified = {}, {}, set()
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    contexts = {}
    for row in originals.itertuples():
        trace = json.loads(traces[row.response_id])
        _check(attempts[row.attempt_id]['raw_attempt_id'], trace['simulation']['id'], 'tau2 original simulation identity')
        contexts[row.attempt_id] = trace['source_file'], str(trace['simulation']['task_id'])
    for row in tables['responses'].itertuples():
        filename, task_id = contexts[row.attempt_id]
        key = row.item_id, filename, task_id
        if key in verified:
            continue
        verified.add(key)
        task, info = tasks[filename, task_id], runs[filename]['info']
        domain = info['environment_info']['domain_name']
        parent_raw = domain + '/' + task_id
        item = items[row.item_id]
        original, _, component = item['raw_item_id'].partition('::')
        _check((original, item['testlet_id']), (parent_raw, parent_raw), 'tau2 original task and testlet')
        _check(json.loads(item['content']), dict(task={k: v for k, v in task.items() if k != 'evaluation_criteria'},
            environment_info=info['environment_info'], user_guidelines=info['user_info'].get('global_simulation_guidelines')),
            'tau2 every check retains its complete task, policy, tools and user guidelines')
        expected_features = canonicalize_features(dict(domain=domain, source_task_id=task_id,
            input_scope=metadata['build']['parameters']['labels']['input_scope']))
        _check(item['item_features'], features_string(expected_features), 'tau2 unchanged task provenance')
        _check(pd.isna(item['asset_manifest']), True, 'tau2 no invented input assets')
        base_rule = json.dumps(dict(rule=metadata['grading']['rule'], evaluation_criteria=task['evaluation_criteria']),
            ensure_ascii=False, sort_keys=True)
        base_spec = dict(metadata['grading']['verifiers']['native'], recorded_harness_revision=info.get('git_commit'))
        base_verifier = {'class': 'exact_matcher', 'spec': json.dumps(base_spec, sort_keys=True)}
        if 'NL_ASSERTION' in task['evaluation_criteria']['reward_basis']:
            base_verifier.update({'class': 'judge', 'judged_by': 'llm'})
        criterion = json.loads(item['grading_criterion'])
        if component and component != 'ungraded_task_outcome':
            kind, digest = component.split(':', 1)
            rule = json.loads(criterion['rule'])
            _check(digest, hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest(), 'tau2 exact native checker definition')
            _check(rule['checker'], kind, 'tau2 checker identity')
            _check(criterion, dict(reference_answer=None, rule=criterion['rule']), 'tau2 native binary check')
            spec = json.dumps(dict(base_spec, checker=kind, evaluation_criteria=task['evaluation_criteria']), sort_keys=True)
            verifier = {'class': 'exact_matcher', 'spec': spec}
            if kind in {'NL_ASSERTION', 'COMMUNICATE'}:
                verifier.update({'class': 'judge', 'judged_by': 'llm'})
            _check(json.loads(item['verifier']), verifier, 'tau2 complete original grading criteria and checker provenance')
        else:
            _check(criterion, dict(reference_answer=None, rule=base_rule), 'tau2 unchanged aggregate fallback criterion')
            _check(json.loads(item['verifier']), base_verifier, 'tau2 unchanged aggregate fallback verifier')
        parent = dict(item, raw_item_id=parent_raw,
            grading_criterion=json.dumps(dict(reference_answer=None, rule=base_rule)), verifier=json.dumps(base_verifier))
        identity = {k: parent[k] for k in ('content', 'grading_criterion', 'verifier', 'item_features')}
        parent_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        parents.setdefault(parent_id, dict(parent, item_id=parent_id))
        associations[key] = parent_id
    _check({key[0] for key in verified}, set(items), 'tau2 no orphaned checks')
    originals['item_id'] = [associations[(row.item_id, *contexts[row.attempt_id])] for row in originals.itertuples()]
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    return _tau2_bench(directory, projected, metadata)
