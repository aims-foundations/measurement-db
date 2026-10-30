"""Read released InfiBench checks; never execute submitted code or custom graders."""


def released_checks(config, detail):
    """Yield source check IDs, definitions, native grades and their separate scales."""
    grading = config['grading']
    binary = dict(kind='discrete', values=[0, 1], direction='higher_is_better')
    families = [
        ('keyword', 'keywords', [rule for rule in grading.get('keywords', [])
            if not isinstance(rule, dict) or 'post_handler' not in rule]),
        ('blank', 'blank_filling_detail', grading.get('blank_filling', {}).get('targets', [])),
        ('test', 'unit_test_detail', grading.get('unit_test', {}).get('tests', [])),
        ('similarity', 'similarity_detail', grading.get('similarity', [])),
    ]
    for kind, field, rules in families:
        if field not in detail:
            continue
        values = detail[field]
        if not isinstance(values, list) or len(values) != len(rules):
            raise ValueError('Native InfiBench check definitions and outcomes do not align: ' + field)
        for index, (rule, value) in enumerate(zip(rules, values)):
            scale = dict(binary)
            if kind == 'keyword':
                if value not in ('match', 'unmatch'):
                    raise ValueError('Unknown keyword verdict')
                grade = float(value == 'match')
                if isinstance(rule, dict) and rule.get('neg'):
                    scale['direction'] = 'lower_is_better'
            elif kind == 'blank':
                if not isinstance(value, str) or not value.startswith(('matched:', 'unmatched:')):
                    raise ValueError('Unknown blank-filling verdict')
                grade = float(value.startswith('matched:'))
            elif kind == 'test':
                if type(value.get('passed')) is not bool:
                    raise ValueError('Unknown test verdict')
                grade = float(value['passed'])
            else:
                grade = value['normalized_score']
                scale = dict(kind='interval', min=0, max=1, direction='higher_is_better')
            yield kind + ':' + str(index), dict(kind=kind, index=index, definition=rule), grade, scale
    if 'custom_score' in detail:
        # The custom evaluator exposes only its total, not consistently named
        # subchecks. Keep that native point score as one check, without guessing.
        yield 'custom', dict(kind='custom', definition=grading['customized']), detail['custom_score'], dict(
            kind='interval', min=None, max=None, direction='higher_is_better')
