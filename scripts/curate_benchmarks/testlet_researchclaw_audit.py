"""Reconcile every ResearchClaw rubric, input asset and original research run."""

from collections import Counter
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _features, _researchclawbench, _researchclawbench_sources
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='ResearchClaw complete native component evidence')
    source = _researchclawbench_sources(directory, metadata)
    protocol = metadata['grading']['verifiers']['reported']
    parents, mapping, components = {}, {}, Counter()
    for item in tables['items'].to_dict('records'):
        task, _, component = item['raw_item_id'].partition('::')
        native = source['tasks'][task]
        _check(item['testlet_id'], task, 'ResearchClaw original task testlet')
        _check(item['content'], native['content'], 'ResearchClaw complete instruction on every criterion')
        _check(json.loads(item['asset_manifest']), native['asset_manifest'], 'ResearchClaw original inputs on every criterion')
        _check(_features(item['item_features']), dict(domain=task.split('_')[0],
            input_scope='published_template_and_pinned_task_assets'), 'ResearchClaw unchanged task attributes')
        base_rule = json.dumps(dict(rule=metadata['grading']['rule'], published_checklist=native['checklist']), ensure_ascii=False)
        base_verifier = {'class': 'judge', 'judged_by': 'llm', 'spec': json.dumps(protocol, sort_keys=True)}
        if component:
            index = int(component)
            _check(component, str(index), 'ResearchClaw native criterion index')
            _check(json.loads(item['grading_criterion']), dict(reference_answer=None, rule=json.dumps(dict(
                criterion=native['checklist'][index], criterion_index=index, instruction=protocol['criterion_rule']),
                ensure_ascii=False, sort_keys=True)), 'ResearchClaw complete criterion content, type and weight')
            _check(json.loads(item['verifier']), dict(base_verifier,
                spec=json.dumps(dict(protocol, criterion_index=index), sort_keys=True)), 'ResearchClaw criterion grading provenance')
            components[task, index] += 1
        else:
            _check(json.loads(item['grading_criterion']), dict(reference_answer=None, rule=base_rule),
                'ResearchClaw unchanged aggregate-only rubric')
            _check(json.loads(item['verifier']), base_verifier, 'ResearchClaw unchanged aggregate-only verifier')
        parents.setdefault(task, dict(item, item_id=task, raw_item_id=task,
            grading_criterion=json.dumps(dict(reference_answer=None, rule=base_rule)), verifier=json.dumps(base_verifier)))
        mapping[item['item_id']] = task
    _check(components, Counter({(task, index): 1 for task, native in source['tasks'].items()
        for index in range(len(native['checklist']))}), 'ResearchClaw complete published rubric roster')
    projected = parent_view(tables, list(parents.values()), mapping)
    return _researchclawbench(directory, projected, metadata, source=source)
