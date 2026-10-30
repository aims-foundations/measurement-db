"""Validate CHI checkpoints and preserve its complete native run audit."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _chi_bench, _chi_source_records
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='CHI complete native checkpoint evidence')
    source = _chi_source_records(directory, metadata)
    parents, mapping = {}, {}
    for item in tables['items'].to_dict('records'):
        task, _, check = item['raw_item_id'].partition('::')
        native = source['tasks'][task]
        _check(item['testlet_id'], task, 'CHI original task testlet across grader settings')
        verifier = json.loads(item['verifier'])
        protocol = json.loads(verifier['spec'])
        base = dict(reference_answer=None, rule=metadata['grading']['rule'] +
            f" Expected terminal status: {native['expected_target_status']}. Verifier contract: {native['verifier_contract']}.")
        expected = base if not check else dict(reference_answer=None, rule=json.dumps(dict(check=check,
            expected_terminal_status=native['expected_target_status'], verifier_contract=native['verifier_contract'],
            rule='Released named checkpoint verdict; not-applicable checks remain ungraded')))
        _check(json.loads(item['grading_criterion']), expected, 'CHI exact native checkpoint definition')
        parent_verifier = {'class': 'judge', 'spec': json.dumps(protocol, sort_keys=True), 'judged_by': 'llm'}
        if protocol['judge_model'] is not None:
            parent_verifier['judge'] = protocol['judge_model']
        expected_verifier = parent_verifier if not check or check.startswith('judge.') else {'class': 'exact_matcher', 'spec': parent_verifier['spec']}
        _check(verifier, expected_verifier, 'CHI deterministic versus model-judged checkpoint')
        identity = json.dumps([task, protocol], sort_keys=True)
        parent = dict(item, item_id=identity, raw_item_id=task, grading_criterion=json.dumps(base), verifier=json.dumps(parent_verifier))
        if identity in parents:
            _check(parent, parents[identity], 'CHI full instruction and metadata retained on every checkpoint')
        parents[identity] = parent
        mapping[item['item_id']] = identity
    projected = parent_view(tables, list(parents.values()), mapping)
    return _chi_bench(directory, projected, metadata, source_records=source)
