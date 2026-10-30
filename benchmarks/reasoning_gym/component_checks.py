"""Recover explicitly scored positions from released answers, without inference.

These native scorers compare named mappings, output positions or polynomial roots.
Other scalar rewards remain whole-answer observations. A native format failure
that skips the comparisons returns no outcomes; its original reward survives.
"""

import ast
import json
import math


SOURCE_FILES = {
    'polynomial_equations': 'algebra/polynomial_equations.py',
    'cryptarithm': 'algorithmic/cryptarithm.py',
    'game_of_life': 'algorithmic/game_of_life.py',
    'letter_jumble': 'algorithmic/letter_jumble.py',
    'sentence_reordering': 'algorithmic/sentence_reordering.py',
    'figlet_font': 'cognition/figlet_fonts.py',
    'emoji_mystery': 'games/emoji_mystery.py',
    'futoshiki': 'games/futoshiki.py',
    'mini_sudoku': 'games/mini_sudoku.py',
    'sudoku': 'games/sudoku.py',
}
GRID_TASKS = {'futoshiki', 'mini_sudoku', 'sudoku'}


def reference_checks(dataset, reference):
    """List the native comparison coordinates and their exact reference values."""
    if dataset == 'polynomial_equations':
        roots = parsed_roots(reference)
        return {**{f'root:{i}': dict(position=i, expected=value) for i, value in enumerate(roots)},
            'extra_roots': dict(expected=0, measurement='unmatched predicted roots')}
    if dataset == 'cryptarithm':
        mapping = dict((name, int(value)) for name, value in (pair.split('=') for pair in reference.split(',')))
        return {'mapping:' + name: dict(letter=name, expected=value) for name, value in mapping.items()}
    if dataset in GRID_TASKS:
        board = [[int(c) for c in line if c in '123456789'] for line in reference.splitlines()]
        board = [row for row in board if row]
        if not board or any(len(row) != len(board) for row in board):
            raise ValueError('The released solution does not match its native square-grid format')
        return {f'cell:{i}:{j}': dict(row=i, column=j, expected=value)
            for i, row in enumerate(board) for j, value in enumerate(row)}
    if dataset == 'game_of_life':
        board = json.loads(reference)
        if board and isinstance(board[0], list):
            return {f'cell:{i}:{j}': dict(row=i, column=j, expected=value)
                for i, row in enumerate(board) for j, value in enumerate(row)}
        return {f'cell:{i}': dict(position=i, expected=value) for i, value in enumerate(board)}
    if dataset in {'letter_jumble', 'sentence_reordering'}:
        words = reference.strip().lower().split() if dataset == 'letter_jumble' else reference.split()
        return {f'word:{i}': dict(position=i, expected=word) for i, word in enumerate(words)}
    if dataset in {'figlet_font', 'emoji_mystery'}:
        text = reference.strip().lower() if dataset == 'figlet_font' else reference
        return {f'character:{i}': dict(position=i, expected=character) for i, character in enumerate(text)}
    return {}


def comparison_outcomes(dataset, answer, reference, configuration=None):
    """Return comparison flags, or None when the native scorer exits beforehand."""
    checks = reference_checks(dataset, reference)
    if dataset == 'polynomial_equations':
        return root_outcomes(answer, reference, configuration or {})
    if not checks or not isinstance(answer, str):
        return None
    if dataset == 'cryptarithm':
        if len(answer.split(',')) != len(checks):
            return None
        try:
            predicted = dict((name, int(value)) for name, value in
                (pair.strip().split('=') for pair in answer.split(',')))
        except ValueError:
            return None
        if len(predicted) != len(checks):
            return None
        return {key: float(predicted.get(check['letter']) == check['expected']) for key, check in checks.items()}
    if dataset == 'game_of_life':
        try:
            predicted = json.loads(answer)
        except (ValueError, TypeError):
            return None
        flags = {}
        for key, check in checks.items():
            try:
                actual = predicted[check['row']][check['column']] if 'row' in check else predicted[check['position']]
                flags[key] = float(actual == check['expected'])
            except IndexError:
                flags[key] = 0.
            except TypeError:
                if 'row' not in check:
                    return None
                flags[key] = 0.
        return flags
    if dataset in GRID_TASKS:
        width = max(check['column'] for check in checks.values()) + 1
        if dataset != 'futoshiki' and not answer:
            return None
        if '\n'.join(line.rstrip() for line in answer.split('\n')) == '\n'.join(line.rstrip() for line in reference.split('\n')):
            return dict.fromkeys(checks, 1.)
        rows = [[int(c) for c in line if c in '123456789'] for line in answer.split('\n')]
        rows = [row for row in rows if len(row) == width][:width]
        return {key: float(check['row'] < len(rows) and rows[check['row']][check['column']] == check['expected'])
            for key, check in checks.items()}
    if dataset in {'letter_jumble', 'sentence_reordering'}:
        words = answer.strip().lower().split() if dataset == 'letter_jumble' else answer.split()
        if dataset == 'sentence_reordering' and len(words) != len(checks):
            return None
        return {key: float(check['position'] < len(words)
            and words[check['position']].lower() == check['expected'].lower()) for key, check in checks.items()}
    if dataset == 'figlet_font':
        answer = answer.replace(' ', '').strip().lower()
    elif dataset == 'emoji_mystery' and len(answer) != len(checks):
        return None
    return {key: float(check['position'] < len(answer) and answer[check['position']] == check['expected'])
        for key, check in checks.items()}


def parsed_roots(text):
    """The native parser sorts numeric fields and ignores nonnumeric fields."""
    roots = []
    for field in (text or '').split(','):
        try:
            roots.append(float(field.strip()))
        except ValueError:
            continue
    return sorted(roots)


def root_configuration(source):
    """Read the three scalar defaults from the pinned scorer, without importing it."""
    config = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
        and n.name == 'PolynomialEquationsConfig')
    names = {'exp_distance_factor', 'penalty_missing_factor', 'penalty_extra_factor'}
    return {n.targets[0].id: ast.literal_eval(n.value) for n in config.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}


def root_outcomes(answer, reference, configuration):
    """Retain native greedy matches, individual rewards and the extra-root count."""
    remaining = list(enumerate(parsed_roots(reference)))
    result = {f'root:{i}': 0. for i, _ in remaining}
    result['extra_roots'] = 0.
    for predicted in parsed_roots(answer):
        eligible = [(abs(predicted - value), index, position) for position, (index, value) in enumerate(remaining)
            if abs(predicted - value) < float('inf')]
        if eligible:
            distance, index, position = min(eligible)
            remaining.pop(position)
            result[f'root:{index}'] = math.exp(distance * configuration['exp_distance_factor'])
        else:
            result['extra_roots'] += 1.
    return result


def component_scale(protocol, dataset, component):
    if dataset == 'polynomial_equations':
        return protocol['extra_count_scale'] if component == 'extra_roots' else protocol['aggregate_scale']
    return protocol['binary_scale']
