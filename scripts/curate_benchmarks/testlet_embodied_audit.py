"""Validate conditional symbolic goals while keeping historical grades unknown."""

from collections import Counter, defaultdict
import hashlib
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _embodied_agent_interface, _embodied_agent_interface_sources, _features
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='EAI conditional symbolic checks')
    prompts, conditions, records = source = _embodied_agent_interface_sources(directory, metadata)
    protocol = metadata['grading']['verifiers']['official_pipeline']
    definitions = {}
    for task, annotation in conditions.items():
        for goal, alternatives in annotation['goal_conditions'].items():
            occurrences = defaultdict(list)
            for alternative, predicates in enumerate(alternatives):
                for position, predicate in enumerate(predicates):
                    occurrences[json.dumps(predicate, ensure_ascii=False, sort_keys=True)].append(
                        dict(alternative=alternative, position=position))
            for encoded, positions in occurrences.items():
                key = str(goal) + ':' + hashlib.sha256(encoded.encode()).hexdigest()
                definitions[task + '::' + key] = dict(goal_id=str(goal), condition=json.loads(encoded),
                    goal_alternatives=alternatives, positions=positions)
    items = tables['items'].set_index('item_id').to_dict('index')
    _check(Counter(item['raw_item_id'] for item in items.values()), Counter({key: 1 for key in definitions}),
        'EAI every distinct native goal condition exactly once')
    parents, mapping = {}, {}
    for item_id, item in items.items():
        task, _ = item['raw_item_id'].split('::')
        check = definitions[item['raw_item_id']]
        _check(item['testlet_id'], task, 'EAI original task testlet')
        _check(item['content'], prompts[task], 'EAI complete original prompt for every check')
        _check(_features(item['item_features']), dict(simulator='behavior', module='goal_interpretation'), 'EAI source module')
        _check(pd.isna(item['asset_manifest']), True, 'EAI no invented assets')
        _check(json.loads(item['grading_criterion']), dict(reference_answer=json.dumps(check['condition'], ensure_ascii=False),
            rule=json.dumps(dict(protocol=protocol['component_rule'], **check), ensure_ascii=False, sort_keys=True)),
            'EAI exact predicates, conditional alternatives and native multiplicities')
        _check(json.loads(item['verifier']), {'class': 'judge', 'spec': json.dumps(dict(protocol,
            goal_id=check['goal_id'], selected_condition=check['condition']), ensure_ascii=False, sort_keys=True)},
            'EAI symbolic protocol is documented without executing a grader')
        parents.setdefault(task, dict(item, item_id=task, raw_item_id=task,
            grading_criterion=json.dumps(dict(reference_answer=json.dumps(conditions[task]['goal_conditions'], ensure_ascii=False, sort_keys=True),
                rule=metadata['grading']['rule'])), verifier=json.dumps({'class': 'judge', 'spec': json.dumps(protocol, sort_keys=True)})))
        mapping[item_id] = task
    _check(tables['responses'].response.isna().all(), True, 'EAI no fabricated predicate verdicts')
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    expected = Counter()
    by_task = defaultdict(list)
    for key in definitions:
        by_task[key.split('::')[0]].append(key)
    for key, attempt in attempts.items():
        saved = json.loads(attempt['trace'])
        _check(saved['recorded_task_score'], None, 'EAI original task grade also unavailable')
        trace = json.loads(saved['source_trace'])
        locator = trace['source_member'], trace['source_row']
        _check(attempt['raw_attempt_id'], locator[0] + ':' + str(locator[1]), 'EAI original answer identity')
        task = records[locator][1]['identifier']
        expected.update((key, raw_item) for raw_item in by_task[task])
    _check(Counter((row.attempt_id, items[row.item_id]['raw_item_id']) for row in tables['responses'].itertuples()),
        expected, 'EAI each answer links to its exact conditional-check roster')
    projected = parent_view(tables, list(parents.values()), mapping)
    projected['benchmarks'] = projected['benchmarks'].assign(response_type='continuous_bounded',
        response_scale=json.dumps(dict(kind='interval', min=0, max=1, direction='higher_is_better')))
    stats = _embodied_agent_interface(directory, projected, metadata, source=source)
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents))
