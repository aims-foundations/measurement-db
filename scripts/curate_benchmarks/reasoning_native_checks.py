"""Observe comparisons inside pinned Reasoning Gym scorer implementations.

The instrumentation preserves each expression's operands and return value. It
records comparisons already performed by the native scorer; it neither invokes
models nor evaluates code supplied in a model answer.
"""

import ast
import json
import math
from types import MethodType, SimpleNamespace


class NativeComparisons:
    def __init__(self, dataset, source):
        self.dataset = dataset
        self.events = []
        self.whole = False
        self.compared = False
        functions = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)
            and node.name in {'score_answer', 'partial', '_parse_score_to_list'}]

        class Observe(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'math.exp':
                    return ast.copy_location(ast.Call(func=ast.Name(id='_root_reward', ctx=ast.Load()),
                        args=[node.args[0], ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[]), node)
                return node

            def visit_Compare(self, node):
                self.generic_visit(node)
                if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                    return ast.copy_location(ast.Call(func=ast.Name(id='_compare', ctx=ast.Load()),
                        args=[node.left, node.comparators[0], ast.Constant(ast.unparse(node)),
                            ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[]), node)
                return node

            def visit_Return(self, node):
                self.generic_visit(node)
                return ast.copy_location(ast.Return(value=ast.Call(func=ast.Name(id='_finish', ctx=ast.Load()),
                    args=[node.value, ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])],
                    keywords=[])), node)

        namespace = dict(json=json, math=math, _compare=self._compare, _finish=self._finish, _root_reward=self._root_reward)
        body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        body.extend(Observe().visit(node) for node in functions)
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            '<observed-native-comparisons>', 'exec'), namespace)
        defaults = {}
        if dataset == 'polynomial_equations':
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'PolynomialEquationsConfig')
            for node in cls.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in {'exp_distance_factor', 'penalty_missing_factor', 'penalty_extra_factor'}:
                        defaults[node.targets[0].id] = ast.literal_eval(node.value)
        self.defaults = defaults
        self.native = SimpleNamespace(config=SimpleNamespace(**defaults))
        for node in functions:
            setattr(self.native, node.name, MethodType(namespace[node.name], self.native))

    def _compare(self, left, right, expression, context):
        result = left == right
        if expression in {'answer == oracle_answer', 'expected_answer == answer', 'answer == correct_word', "answer == entry['answer']",
                'answer_stripped == oracle_answer_stripped'} and result:
            self.whole = True
        coordinate = None
        if expression == 'predict_mapping[alphabet] == number':
            coordinate = 'mapping:' + context['alphabet']
        elif expression.startswith('ans_arr[') and expression.endswith(' == expected_value'):
            coordinate = f"cell:{context['i']}" + (f":{context['j']}" if 'j' in context else '')
        elif expression == 'word == model_words[i]':
            coordinate = 'word:' + str(context['i'])
        elif expression == 'goal_word.lower() == answer_word.lower()':
            coordinate = 'word:' + str(len(self.events))
        elif expression == 'a == b':
            if self.dataset in {'mini_sudoku', 'sudoku', 'futoshiki'}:
                row = context['row']
                column = sum(key.startswith(f'cell:{row}:') for key, _ in self.events)
                coordinate = f'cell:{row}:{column}'
            else:
                coordinate = 'character:' + str(len(self.events))
        if coordinate is not None:
            self.events.append((coordinate, float(result)))
        return result

    def _root_reward(self, exponent, context):
        reward = math.exp(exponent)
        position = self.remaining_roots.pop(context['matched_distance_index'])
        self.events.append(('root:' + str(position), reward))
        return reward

    def _finish(self, value, context):
        if self.dataset == 'polynomial_equations':
            if 'predicted_solutions' in context:
                self.events.append(('extra_roots', float(context.get('extra_solutions', 0))))
                self.compared = True
            return value
        # These locals are assigned only after the relevant native format gates.
        markers = {'cryptarithm': 'total', 'game_of_life': 'total_cells',
            'mini_sudoku': 'num_matching', 'sudoku': 'num_matching', 'futoshiki': 'num_matching',
            'sentence_reordering': 'credit', 'letter_jumble': 'partial_score',
            'figlet_font': 'correct_count', 'emoji_mystery': 'score'}
        self.compared |= markers[self.dataset] in context
        return value

    def score(self, answer, reference, checks, configuration=None):
        self.events.clear()
        self.whole = self.compared = False
        entry = dict(answer=reference)
        if self.dataset == 'polynomial_equations':
            self.remaining_roots = list(range(sum(k.startswith('root:') for k in checks)))
            self.native.config = SimpleNamespace(**dict(self.defaults, **(configuration or {})))
        if self.dataset in {'mini_sudoku', 'sudoku', 'futoshiki'}:
            board = [[int(c) for c in line if c in '123456789'] for line in reference.splitlines()]
            entry['metadata'] = dict(solution=[row for row in board if row])
        grade = self.native.score_answer(answer, entry)
        if self.whole:
            return grade, dict.fromkeys(checks, 1.)
        if self.compared or self.events:
            flags = dict.fromkeys(checks, 0.)
            for key, value in self.events:
                if key not in checks:
                    raise ValueError('A native comparison has no reference coordinate: ' + key)
                flags[key] = value
            return grade, flags
        return grade, None
