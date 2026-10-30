"""Check every block with the pinned native matcher and audit complete plans."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _features, _phyblock
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='PhyBlock reference-block checks')
    layout = metadata['build']['parameters']['layout']
    with ZipFile(directory / 'raw' / layout['archive']) as archive:
        members = {name.split('/', 1)[1]: name for name in archive.namelist() if not name.endswith('/')}
        code = archive.read(members['evaluate_block_construction.py'])
        _check(hashlib.sha256(code).hexdigest(), '512171b70c0eefa1ce61b8b39ca5e4a3cd00eb11310dc990cfe2f5e43739ff85',
            'PhyBlock reviewed native matcher')
        names = {'extract_message_content', 'are_blocks_equal', 'is_place_legal'}
        definitions = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name in names]
        _check({node.name for node in definitions}, names, 'PhyBlock exact reviewed native helpers')
        native = {'json': json, 'Path': Path}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'reviewed_phyblock_helpers', 'exec'), native)
        goals, candidates, outputs = {}, {}, {}
        for name, member in members.items():
            if re.fullmatch(layout['goals_pattern'], name):
                scene = Path(name).stem
                goals[scene] = json.loads(archive.read(member))
                candidates[scene] = json.loads(archive.read(members[layout['candidates'].format(scene=scene)]))
            elif re.fullmatch(layout['outputs_pattern'], name):
                outputs[name] = archive.read(member).decode('utf-8')

    items = tables['items'].set_index('item_id').to_dict('index')
    parents, parent_ids, item_checks, rosters = {}, {}, {}, Counter()
    for item_id, item in items.items():
        scene, check = item['raw_item_id'].split('::')
        condition, name = check.split(':', 1)
        features = _features(item['item_features'])
        _check((item['testlet_id'], features['grading_condition']), (scene, condition),
            'PhyBlock original scene and grading condition')
        key = scene, item['content'], item['item_features']
        parent = parents.setdefault(key, dict(item))
        parent_ids.setdefault(key, item_id)
        _check(item['asset_manifest'], parent['asset_manifest'], 'PhyBlock every block retains complete input images')
        kind = 'block' if name.startswith('block:') else 'extra_placement_count'
        if kind == 'block':
            _check(int(name.split(':')[1]) in [b['order'] for b in goals[scene]['blocks']], True,
                'PhyBlock original reference-block identifier')
        else:
            _check(name, kind, 'PhyBlock declared error-count check')
        component = metadata['grading']['verifiers']['components'][kind]
        reference = json.dumps(dict(goal=goals[scene], candidates=candidates[scene]))
        _check(json.loads(item['grading_criterion']), dict(reference_answer=reference,
            rule=json.dumps(dict(interpretation=component['rule'], selected_check=name), sort_keys=True),
            response_scale=component['response_scale']), 'PhyBlock full geometry, selected check and native scale')
        spec = dict(metadata['grading']['verifiers'][condition], selected_check=name)
        _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(spec, sort_keys=True)},
            'PhyBlock pinned matching protocol and selected check')
        item_checks[item_id] = key, condition, name
        rosters[key, name] += 1
    _check(rosters, Counter((key, name) for key in parents for name in
        ['block:' + str(b['order']) for b in goals[key[0]]['blocks']] + ['extra_placement_count']),
        'PhyBlock complete reference roster per input and grading condition')

    responses = tables['responses']
    _check(responses.attempt_id.isna().any(), False, 'PhyBlock each check has its original execution')
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    _check(set(attempts), set(responses.attempt_id), 'PhyBlock exact observed executions')
    contexts = responses[['attempt_id', 'subject_id', 'trial', 'interactors']].drop_duplicates()
    _check(contexts.attempt_id.is_unique, True, 'PhyBlock all conditions share one model and occasion')
    expected, seen, projected_responses, projected_traces = Counter(), Counter(), [], []
    for attempt_id, group in responses.groupby('attempt_id', sort=False):
        execution = attempts[attempt_id]
        evidence = json.loads(execution['trace'])
        trace = json.loads(evidence['source_trace'])
        file = execution['raw_attempt_id']
        text = outputs[file]
        _check((trace['source_file'], trace['native_record']), (file, text), 'PhyBlock complete original execution bytes')
        _check(group.subject_id.unique().tolist(), [execution['subject_id']], 'PhyBlock execution subject linkage')
        scene = Path(file).stem
        message = native['extract_message_content'](json.loads(text))
        sequence = [candidates[scene][int(value) - 1] for value in re.findall(r'block with index (\d+)', message)
            if 0 < int(value) <= len(candidates[scene])]
        conditions = ['pose_constrained'] + (['topology_only'] if goals[scene]['level'] == 2 and len(goals[scene]['blocks']) < 7 else [])
        _check(set(group.test_condition), set(conditions), 'PhyBlock only native applicable grading conditions')
        derived = {}
        for condition in conditions:
            blocks = [dict(block, placed=False) for block in goals[scene]['blocks']]
            for candidate in sequence:
                for block in blocks:
                    if not block['placed'] and (condition == 'topology_only' or native['is_place_legal'](block, blocks)) and native['are_blocks_equal'](
                            candidate, block, evaluate_mode='A' if condition == 'pose_constrained' else 'B'):
                        block['placed'] = True
                        break
            tp = sum(block['placed'] for block in blocks)
            fp, fn = len(sequence) - tp, len(blocks) - tp
            precision, recall = tp / len(sequence) if sequence else 0, tp / len(blocks) if blocks else 0
            f1 = round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0
            derived[condition] = dict(condition=condition, TP=tp, FP=fp, FN=fn, F1=f1)
            assessment = group.loc[group.test_condition.eq(condition)]
            row = assessment.iloc[0].to_dict()
            key = item_checks[row['item_id']][0]
            _check(key[0], scene, 'PhyBlock recorded plan belongs to its original scene')
            for block in blocks:
                expected[attempt_id, key, condition, 'block:' + str(block['order']), int(block['placed'])] += 1
            expected[attempt_id, key, condition, 'extra_placement_count', fp] += 1
            row.update(item_id=parent_ids[key], response=f1, attempt_id=None)
            projected_responses.append(row)
            projected_traces.append(dict(response_id=row['response_id'], trace=json.dumps(dict(source_file=file,
                native_record=text, derived_grade=derived[condition]))))
        _check(trace['derived_grades'], derived, 'PhyBlock original F1 assessments retained once per execution')
        _check(evidence['recorded_task_score'], derived['pose_constrained']['F1'], 'PhyBlock original primary aggregate')
        seen[file] += 1
    actual = Counter((r.attempt_id, *item_checks[r.item_id], r.response) for r in responses.itertuples())
    _check(actual, expected, 'PhyBlock exact per-block and excess-placement outcomes')
    _check(seen, Counter({file: 1 for file in outputs}), 'PhyBlock multiple graders do not create additional executions')

    parent_rows = []
    for key, parent in parents.items():
        scene = key[0]
        condition = _features(parent['item_features'])['grading_condition']
        parent.update(item_id=parent_ids[key], raw_item_id=scene,
            grading_criterion=json.dumps(dict(reference_answer=json.dumps(dict(goal=goals[scene], candidates=candidates[scene])),
                rule=metadata['grading']['rule'])),
            verifier=json.dumps({'class': 'exact_matcher', 'spec': json.dumps(metadata['grading']['verifiers'][condition], sort_keys=True)}))
        parent_rows.append(parent)
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parent_rows), responses=pd.DataFrame(projected_responses), traces=pd.DataFrame(projected_traces))
    stats = _phyblock(directory, projected, metadata)
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(responses),
        imported_items=len(items), source_observed_testlets=len({key[0] for key in parents}))
