"""Expose the named reference checks behind four pinned LawBench F1 metrics.

No model is called. Parsers follow the released deterministic evaluators; text
similarity remains continuous, and unsupported predicted labels are counted
separately instead of creating hundreds of easy negative classification items.
"""

import ast
import importlib
import re



TASKS = {'2-3', '2-6', '3-1', '3-3'}


def load_native_tools(archive, prefix):
    """Read the local label vocabulary and retain pinned native entity parsers."""
    tree = ast.parse(archive.read(prefix + 'evaluation/evaluation_functions/wbfl.py'))
    options = next(ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'option_list' for target in node.targets))
    return dict(classification_options=options,
        accusations=importlib.import_module('evaluation_functions.ljp_accusation').option_list,
        entities=importlib.import_module('utils.comprehension_scores'))


def component_scores(task, record, tools):
    """Return reference hit/similarity checks and an extra-prediction count."""
    prediction, reference = record['prediction'], record['refr']
    if task == '2-3':
        truth = set(reference[3:-1].split('、'))
        predicted = {label for label in tools['classification_options'] if label in prediction}
    elif task == '3-3':
        truth = set(reference.replace('罪名:', '').split(';'))
        predicted = {label for label in tools['accusations'] if label in prediction}
    elif task == '3-1':
        import cn2an
        truth = {int(value) for value in reference.replace('法条:刑法第', '').replace('条', '').split('、')}
        predicted = set()
        for chunk in prediction.split('、'):
            chunk = re.sub(r'第(.*?)款', '', chunk.replace('万元', '元'))
            chunk = re.sub(r'第(.*?)条', lambda match: match[1], chunk)
            numbers = re.findall(r'\d+', cn2an.transform(chunk, 'cn2an'))
            if numbers:
                predicted.add(int(numbers[0]))
    elif task == '2-6':
        native = tools['entities']
        truth = native.__extract_entities_ref(reference)
        # This exact vocabulary belongs to the pinned xxcq evaluator.
        predicted = native.__extract_entities_pred(prediction, {'犯罪嫌疑人', '受害人', '被盗货币',
            '物品价值', '盗窃获利', '被盗物品', '作案工具', '时间', '地点', '组织机构'})
        if not truth:
            return []  # Preserve the native empty-reference whole-answer judgment.
        checks = [dict(component_id='entity:' + name, kind='entity_similarity',
            reference=dict(entity_type=name, value=value),
            response=native.CJRCEvaluator.compute_f1(value, predicted[name]) if name in predicted else 0.)
            for name, value in sorted(truth.items())]
        return checks + [dict(component_id='extra_prediction_count', kind='extra_prediction_count',
            reference=None, response=len(set(predicted) - set(truth)))]
    else:
        return []
    checks = [dict(component_id='reference:' + str(label), kind='reference_hit', reference=label,
        response=float(label in predicted)) for label in sorted(truth)]
    return checks + [dict(component_id='extra_prediction_count', kind='extra_prediction_count',
        reference=None, response=len(predicted - truth))]
