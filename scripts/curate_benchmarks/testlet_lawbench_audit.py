"""Reconcile LawBench components with native parser state and all original records."""

import ast
from collections import Counter
import importlib
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd


def _native_parsers(archive, prefix):
    """Observe original set comparisons without reimplementing their parsers."""
    parsers = {}
    for task, module, name, predicted in [
        ('2-3', 'wbfl', 'compute_wbfl', 'predict_set'),
        ('3-1', 'ljp_article', 'compute_ljp_article', 'pred_set'),
        ('3-3', 'ljp_accusation', 'compute_ljp_accusation', 'pred_set'),
    ]:
        tree = ast.parse(archive.read(prefix + 'evaluation/evaluation_functions/' + module + '.py'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)

        class ObserveSets(ast.NodeTransformer):
            def visit_Expr(self, node):
                call = node.value
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name) and call.func.value.id == 'score_list'
                        and call.func.attr == 'append'):
                    capture = ast.parse('observed_sets.append((gt_set.copy(), ' + predicted + '.copy()))').body[0]
                    return [node, capture]
                return node

        namespace = dict(vars(importlib.import_module('evaluation_functions.' + module)))
        namespace['observed_sets'] = []
        instrumented = ast.fix_missing_locations(ast.Module(body=[ObserveSets().visit(function)], type_ignores=[]))
        exec(compile(instrumented, '<observed-native-lawbench-parser>', 'exec'), namespace)
        parsers[task] = namespace[name], namespace['observed_sets']
    entity_module = importlib.import_module('utils.comprehension_scores')
    tree = ast.parse(archive.read(prefix + 'evaluation/evaluation_functions/xxcq.py'))
    vocabulary, = [ast.literal_eval(node) for node in ast.walk(tree) if isinstance(node, ast.Set)]
    return parsers, entity_module, vocabulary


def _source_checks(task, record, parsers, entities, vocabulary):
    if task in parsers:
        grader, captured = parsers[task]
        captured.clear()
        grader([record])
        truth, predicted = captured.pop()
        checks = [('reference:' + str(value), 'reference_hit', value, float(value in predicted)) for value in sorted(truth)]
    elif task == '2-6':
        truth = entities.__extract_entities_ref(record['refr'])
        predicted = entities.__extract_entities_pred(record['prediction'], vocabulary)
        if not truth:
            return []
        checks = [('entity:' + name, 'entity_similarity', dict(entity_type=name, value=value),
            entities.CJRCEvaluator.compute_f1(value, predicted[name]) if name in predicted else 0.)
            for name, value in sorted(truth.items())]
    else:
        return []
    return checks + [('extra_prediction_count', 'extra_prediction_count', None, len(set(predicted) - set(truth)))]


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _lawbench, _lawbench_source_records
    from .testlet_audit_views import parent_view
    from ..build_measurement_tables.hash_measurement_ids import canonical_grading_criterion
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    _check(sys.version_info[:3], (3, 12, 12), 'LawBench declared native scorer runtime')
    validate_dataset(tables, context='LawBench native reference checks and original executions')
    source = _lawbench_source_records(directory, metadata)
    native = source[0]
    parameters = metadata['build']['parameters']
    archive_path = Path(directory) / 'raw' / parameters['paths']['archive']
    prefix = parameters['paths']['prefix']
    grader_path = str(archive_path) + '/' + prefix + parameters['paths']['grader_directory']
    sys.path.insert(0, grader_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            parsers, entities, vocabulary = _native_parsers(archive, prefix)
        expectations = {}
        for key, original in native.items():
            checks = _source_checks(original['identity'][1], original['record'], parsers, entities, vocabulary)
            if not checks:
                checks = [('ungraded_task_outcome' if original['grade'] is None else '',
                    'unknown' if original['grade'] is None else 'aggregate', None, original['grade'])]
            expectations[key] = checks
    finally:
        sys.path.remove(grader_path)

    import hashlib
    import unicodedata
    parents, mapping, actual_items = {}, {}, Counter()
    items = tables['items'].set_index('item_id').to_dict('index')
    for item_id, item in items.items():
        parent, _, component = item['raw_item_id'].partition('::')
        original = native[parent]
        setting, task, content, reference = original['identity']
        normalized = unicodedata.normalize('NFC', content).strip()
        _check(item['content'], content, 'LawBench every check preserves the exact original prompt')
        _check(item['testlet_id'], hashlib.sha256(json.dumps([task, normalized], ensure_ascii=False).encode()).hexdigest(),
            'LawBench exact complete source stimulus testlet')
        protocol = metadata['grading']['verifiers'][task]
        base_criterion = dict(reference_answer=json.dumps(reference, ensure_ascii=False),
            rule=protocol['rule'].format(task=task, setting=setting), response_scale=protocol['response_scale'])
        base_verifier = dict(**{'class': 'judge'}, spec=json.dumps(protocol, sort_keys=True))
        checks = {check[0]: check for check in expectations[parent]}
        _check(component in checks, True, 'LawBench source-defined component')
        _, kind, value, _ = checks[component]
        if kind in {'aggregate', 'unknown'}:
            criterion, verifier = base_criterion, base_verifier
        else:
            component_protocol = metadata['grading']['verifiers']['components'][kind]
            criterion = dict(reference_answer=json.dumps(value, ensure_ascii=False),
                rule=component_protocol['rule'], response_scale=component_protocol['response_scale'])
            verifier = dict(**{'class': 'exact_matcher'}, spec=json.dumps(dict(native_metric=protocol,
                component_id=component, component=component_protocol), sort_keys=True))
        _check(json.loads(item['grading_criterion']), json.loads(canonical_grading_criterion(criterion)),
            'LawBench exact reference definition and native component scale')
        _check(json.loads(item['verifier']), verifier, 'LawBench original parser and selected component')
        parent_item = dict(item, item_id=parent, raw_item_id=parent,
            grading_criterion=canonical_grading_criterion(base_criterion), verifier=json.dumps(base_verifier))
        if parent in parents:
            _check(parent_item, parents[parent], 'LawBench unchanged metadata and assets on every child')
        parents[parent] = parent_item
        mapping[item_id] = parent
        actual_items[parent, component] += 1
    roster = Counter({(parent, check[0]): 1 for parent in parents for check in expectations[parent]})
    _check(actual_items, roster, 'LawBench complete named check roster')
    attempts = tables['attempts'].set_index('attempt_id').to_dict('index')
    _check(Counter(row['raw_attempt_id'] for row in attempts.values()), Counter({key: 1 for key in native}),
        'LawBench every original prediction exactly once')
    expected = Counter()
    for attempt_id, attempt in attempts.items():
        key = attempt['raw_attempt_id']
        for component, _, _, grade in expectations[key]:
            expected[attempt_id, component, grade] += 1
    actual = Counter((row.attempt_id, items[row.item_id]['raw_item_id'].partition('::')[2],
        None if pd.isna(row.response) else row.response) for row in tables['responses'].itertuples())
    _check(actual, expected, 'LawBench native per-reference outcomes, extra counts and missing grades')
    projected = parent_view(tables, list(parents.values()), mapping)
    stats = _lawbench(directory, projected, metadata, source=source)
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=tables['items'].testlet_id.nunique())
