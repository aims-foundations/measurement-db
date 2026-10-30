"""Check native engineering rubrics, then rerun the complete original audit."""

import ast
from collections import Counter
import json
import re
import zipfile

import pandas as pd


def _details(text):
    """Independent reader for literal fields, ignoring opaque simulation objects."""
    try:
        return json.loads(text).get('evaluation_result') or {}
    except json.JSONDecodeError:
        first = re.sub(r'<matlab\.object object at 0x[0-9a-fA-F]+>', 'None', text.strip().splitlines()[0])
        tree = ast.parse(first, mode='eval').body
    root = dict(zip([ast.literal_eval(k) for k in tree.keys], tree.values))

    def literal(node):
        if isinstance(node, ast.Dict):
            return {ast.literal_eval(k): literal(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name):
            return {'true': True, 'false': False, 'null': None}.get(node.id)
        if isinstance(node, ast.Attribute) and ast.unparse(node) in {'np.True_', 'np.False_'}:
            return node.attr == 'True_'
        if isinstance(node, ast.Call) and ast.unparse(node.func) in {
                'np.float64', 'np.float32', 'np.int64', 'np.int32', 'np.bool_'}:
            return ast.literal_eval(node.args[0])
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return None
    return literal(root.get('evaluation_result')) or {}


def _grade(details, name, spec):
    def get(path):
        result = details
        for key in path.split('/'):
            if not isinstance(result, dict) or key not in result:
                return None
            result = result[key]
        return result
    value = get(spec.get('field', name))
    if value is None and 'failed_field' in spec and get(spec['failed_field']) == spec['failed_value']:
        value = 0
    if value is None:
        return None
    kind = spec['format']
    if kind == 'boolean':
        assert type(value) is bool
        return int(value)
    if kind == 'false_passes':
        assert type(value) is bool
        return int(not value)
    if kind == 'zero_passes':
        assert type(value) in (int, float) and value >= 0
        return int(value == 0)
    if kind == 'points':
        assert type(value) in (int, float) and 0 <= value <= spec['maximum']
        return value
    if kind == 'earned_points':
        numerator, denominator = value.split('/')
        assert float(denominator) == spec['maximum']
        return float(numerator)
    if kind == 'pattern':
        values = re.findall(spec['pattern'], value)
        assert len(values) == 1
        return spec['values'][values[0]]
    if kind == 'labels':
        return spec['values'][value]
    if kind == 'status':
        return {'passed': 1, 'failed': 0}.get(value.lower())
    if kind == 'pass_label':
        label = value.split(' (')[0]
        labels = {'Fail': 0, 'Partially Pass': 1, 'Pass': 2} if spec.get('partial') else {'Fail': 0, 'Pass': 1}
        return labels[label]
    raise ValueError('Undeclared engineering check format')


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _engdesign
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='EngDesign native engineering checks')
    grading = metadata['grading']
    items = tables['items'].set_index('item_id').to_dict('index')
    parents = tables['items'].drop_duplicates('testlet_id').copy()
    parent_ids = dict(zip(parents.testlet_id, parents.item_id))
    parent_rows = parents.set_index('testlet_id').to_dict('index')
    source_codes = {}
    with zipfile.ZipFile(directory / 'raw' / metadata['build']['parameters']['layout']['archive']) as archive:
        for name in archive.namelist():
            parts = name.split('/')
            if len(parts) == 4 and parts[1] == 'tasks' and parts[-1] == 'evaluate.py':
                source_codes[parts[2]] = archive.read(name).decode().replace('\r\n', '\n').replace('\r', '\n')
    for item in items.values():
        task, name = item['raw_item_id'].split('::', 1)
        _check(item['testlet_id'], task, 'EngDesign original design task membership')
        parent = parent_rows[task]
        for field in ['content', 'asset_manifest', 'item_features']:
            actual, expected = item[field], parent[field]
            _check(None if pd.isna(actual) else actual, None if pd.isna(expected) else expected,
                'EngDesign every check retains the complete stimulus and assets')
        spec = {'format': 'boolean'} if name == 'task_verdict' else grading['verifiers']['named_checks']['tasks'][task][name]
        scale = dict(kind='discrete', values=[0, 1], direction='higher_is_better')
        if spec['format'] in {'points', 'earned_points'}:
            scale = dict(kind='interval', min=0, max=spec['maximum'], direction='higher_is_better')
        elif spec.get('partial'):
            scale = dict(kind='discrete', values=[0, 1, 2], direction='higher_is_better',
                meanings={'0': 'Fail', '1': 'Partially Pass', '2': 'Pass'})
        interpretation = grading['rule'] if name == 'task_verdict' else grading['verifiers']['named_checks']['rule']
        _check(json.loads(item['grading_criterion']), dict(reference_answer=None, response_scale=scale,
            rule=json.dumps(dict(interpretation=interpretation, selected_check=name,
                extraction=spec, released_evaluator=source_codes[task]), sort_keys=True)),
            'EngDesign complete criterion, native field and scale')
        verifier = {k: v for k, v in grading['verifiers']['native_pass'].items() if k != 'source_template'}
        verifier.update(source=grading['verifiers']['native_pass']['source_template'].format(task=task),
            selected_check=name, extraction=spec)
        _check(json.loads(item['verifier']), {'class': 'exact_matcher', 'spec': json.dumps(verifier, sort_keys=True)},
            'EngDesign source-specific grading provenance')

    originals, traces = attempt_view(tables)
    expected = Counter()
    for row in originals.itertuples():
        task = items[row.item_id]['testlet_id']
        checks = grading['verifiers']['named_checks']['tasks'].get(task, {})
        details = _details(json.loads(traces[row.response_id])['native_record']) if checks else {}
        values = {name: _grade(details, name, spec) for name, spec in checks.items()}
        if not any(value is not None for value in values.values()):
            values = {'task_verdict': None if pd.isna(row.response) else row.response}
        for name, value in values.items():
            expected[row.attempt_id, task + '::' + name, value] += 1
    actual = Counter((r.attempt_id, items[r.item_id]['raw_item_id'], None if pd.isna(r.response) else r.response)
        for r in tables['responses'].itertuples())
    _check(actual, expected, 'EngDesign exact named outcomes, missing checks and fallback verdicts')
    _check(set(tables['responses'].item_id), set(items), 'EngDesign no invented or orphaned checks')

    # Collapse only for the unchanged original input/configuration/trace audit.
    parents['raw_item_id'] = parents.testlet_id
    parents['grading_criterion'] = [json.dumps(dict(reference_answer=None, rule=json.dumps(dict(
        interpretation=grading['rule'], released_evaluator=source_codes[task])))) for task in parents.testlet_id]
    base = grading['verifiers']['native_pass']
    parents['verifier'] = [json.dumps({'class': 'exact_matcher', 'spec': json.dumps(dict(
        **{k: v for k, v in base.items() if k != 'source_template'},
        source=base['source_template'].format(task=task)), sort_keys=True)}) for task in parents.testlet_id]
    originals['item_id'] = originals.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    projected = {name: frame for name, frame in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=originals,
        traces=pd.DataFrame(dict(response_id=list(traces), trace=list(traces.values()))))
    stats = _engdesign(directory, projected, metadata)
    return dict(stats, source_attempts=len(originals), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents))
