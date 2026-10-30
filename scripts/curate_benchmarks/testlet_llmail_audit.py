"""Validate each LLMail objective and retain the full native endpoint audit."""

from collections import Counter
import hashlib
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _llmail_inject, _llmail_source_records
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='LLMail complete native objective evidence')
    source = _llmail_source_records(directory, metadata)
    _, inputs, _, scenarios, _ = source
    parameters = metadata['build']['parameters']
    names = json.loads((directory / 'raw' / parameters['layout']['root'] / parameters['paths']['objectives']).read_text())
    parents, mapping, roster = {}, {}, Counter()
    for item in tables['items'].to_dict('records'):
        task, _, component = item['raw_item_id'].partition('::')
        content = json.loads(item['content'])
        key = content['scenario_key'], content['email_subject'], content['email_body']
        _check(task, inputs[key], 'LLMail first native input alias')
        _check(item['testlet_id'], hashlib.sha256(item['content'].encode()).hexdigest(), 'LLMail exact original input testlet')
        scenario = scenarios[key[0]]
        _check(component in names, True, 'LLMail native objective identity')
        _check(json.loads(item['grading_criterion']), dict(reference_answer=None,
            rule=json.dumps(dict(objective=component, description=names[component], attack_objective=scenario['task']), sort_keys=True)),
            'LLMail complete objective definition')
        _check(json.loads(item['verifier']), {'class': 'exact_matcher',
            'spec': json.dumps(metadata['grading']['verifiers']['objectives'], sort_keys=True)}, 'LLMail original objective grader')
        _check(pd.isna(item['item_features']), True, 'LLMail no invented input attributes')
        parent = dict(item, item_id=task, raw_item_id=task, grading_criterion=json.dumps(dict(reference_answer=None,
            rule=json.dumps(dict(attack_objective=scenario['task'], success_rule=metadata['grading']['rule'])))))
        if task in parents:
            _check(parent, parents[task], 'LLMail every objective retains identical complete input and corpus attachments')
        parents[task] = parent
        mapping[item['item_id']] = task
        roster[task, component] += 1
    _check(roster, Counter({(alias, name): 1 for alias in inputs.values() for name in names}), 'LLMail all five native objective definitions')
    return _llmail_inject(directory, parent_view(tables, list(parents.values()), mapping), metadata, source_records=source)
