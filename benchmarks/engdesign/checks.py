"""Read declared native checks without executing simulation objects or model code."""

import ast
import json
import math
import re


def _scalar(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in {'true', 'false', 'null'}:
        return {'true': True, 'false': False, 'null': None}[node.id]
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == 'np' and node.attr in {'True_', 'False_'}):
        return node.attr == 'True_'
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'np'
            and node.func.attr in {'float64', 'float32', 'int64', 'int32', 'bool_'}
            and len(node.args) == 1 and not node.keywords):
        return _scalar(node.args[0])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _scalar(node.operand)
        if type(value) in (int, float):
            return -value if isinstance(node.op, ast.USub) else value
    raise ValueError('Unsupported representation of a selected EngDesign check')


def _field(tree, path):
    for part in path.split('/'):
        if not isinstance(tree, ast.Dict):
            return None
        fields = {ast.literal_eval(key): value for key, value in zip(tree.keys, tree.values)}
        if part not in fields:
            return None
        tree = fields[part]
    return _scalar(tree)


def scale(spec):
    if spec['format'] in {'points', 'earned_points'}:
        return dict(kind='interval', min=0, max=spec['maximum'], direction='higher_is_better')
    if spec.get('partial'):
        return dict(kind='discrete', values=[0, 1, 2], direction='higher_is_better',
            meanings={'0': 'Fail', '1': 'Partially Pass', '2': 'Pass'})
    return dict(kind='discrete', values=[0, 1], direction='higher_is_better')


def read_checks(text, specifications):
    """Extract only declared scoring fields; unknown and absent values stay null."""
    if not specifications:
        return {}
    # Native logs mix JSON and Python reprs. Inspect syntax, never evaluate calls.
    try:
        document = json.loads(text)
        tree = ast.parse(repr(document), mode='eval').body
    except json.JSONDecodeError:
        tree = ast.parse(text.strip().splitlines()[0], mode='eval').body
    if not isinstance(tree, ast.Dict):
        raise ValueError('Expected a native EngDesign record')
    fields = {ast.literal_eval(k): v for k, v in zip(tree.keys, tree.values)}
    result = fields.get('evaluation_result')
    grades = {}
    for name, spec in specifications.items():
        value = _field(result, spec.get('field', name))
        if value is None and spec.get('failed_field'):
            if _field(result, spec['failed_field']) == spec['failed_value']:
                value = 0
        if value is None:
            continue
        kind = spec['format']
        if kind in {'boolean', 'false_passes'}:
            if type(value) is not bool:
                raise ValueError(f'{name}: expected a native Boolean')
            grade = float(value if kind == 'boolean' else not value)
        elif kind == 'zero_passes':
            if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
                raise ValueError(f'{name}: expected a nonnegative collision count')
            grade = float(value == 0)
        elif kind in {'points', 'earned_points'}:
            if kind == 'earned_points':
                numerator, denominator = value.split('/')
                if float(denominator) != spec['maximum']:
                    raise ValueError(f'{name}: changed native point scale')
                value = float(numerator)
            if type(value) not in (int, float) or not 0 <= value <= spec['maximum']:
                raise ValueError(f'{name}: invalid native points')
            grade = float(value)
        elif kind in {'labels', 'pattern'}:
            if kind == 'pattern':
                matches = re.findall(spec['pattern'], value)
                if len(matches) != 1:
                    raise ValueError(f'{name}: ambiguous native sub-check label')
                value = matches[0]
            grade = spec['values'][value]
        elif kind == 'pass_label':
            match = re.match(r'^(Partially Pass|Pass|Fail)(?:$| \()', value)
            if not match:
                raise ValueError(f'{name}: unknown native pass label')
            values = {'Fail': 0, 'Partially Pass': 1, 'Pass': 2} if spec.get('partial') else {'Fail': 0, 'Pass': 1}
            grade = values[match[1]]
        elif kind == 'status':
            if value.lower() not in {'passed', 'failed'}:
                continue  # An exception message is evidence, not a literal verdict.
            grade = float(value.lower() == 'passed')
        else:
            raise ValueError(f'Unknown EngDesign check format: {kind}')
        grades[name] = grade
    return grades
