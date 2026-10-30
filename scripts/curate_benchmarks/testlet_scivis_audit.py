"""Check SciVis component definitions and preserve its full original report audit."""

import json

from bs4 import BeautifulSoup


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _scivis_sources, _scivisagentbench
    from .testlet_audit_views import parent_view

    source = _scivis_sources(directory, metadata)
    items = tables['items'].set_index('item_id').to_dict('index')
    executions = tables['attempts'].set_index('attempt_id').trace.map(json.loads)
    parents, mapping = {}, {}
    for attempt, children in tables['responses'].groupby('attempt_id', sort=False):
        trace = json.loads(executions[attempt]['source_trace'])
        original = source['native'][trace['source_file'], trace['case_id']]
        node = BeautifulSoup(original['trace']['case_html'], 'html.parser')
        identity = None
        for child in children.itertuples():
            item = items[child.item_id]
            raw, _, component = item['raw_item_id'].partition('::')
            _check((raw, item['testlet_id']), (original['raw_item_id'], original['raw_item_id']), 'SciVis original case testlet')
            if component:
                kind, position = component.split(':', 1)
                if kind == 'vision':
                    element = node.select('.rubric-item')[int(position)]
                    description = element.select_one('.rubric-criterion').get_text(' ', strip=True)
                    score = element.select_one('.rubric-score').get_text(strip=True)
                elif kind == 'rule':
                    element = node.find_all('span', style='font-weight: 600; color: #333;')[int(position)]
                    description = element.get_text(' ', strip=True)
                    score = element.find_next_sibling('span').get_text(strip=True)
                else:
                    _check(kind, 'metric', 'SciVis native component kind')
                    element, = [value for value in node.select('.metric-box')
                        if value.select_one('.metric-label').get_text(strip=True) == position]
                    description = position
                    score = element.select_one('.metric-value').get_text(strip=True)
                maximum = float(score.split('/')[1].split('(')[0])
                expected = dict(reference_answer=original['criterion']['reference_answer'], rule=json.dumps(dict(
                    criterion=description, source_component=component, rule='Released component points on the original scale'), ensure_ascii=False),
                    response_scale=dict(kind='interval', min=0, max=maximum, direction='higher_is_better'))
            else:
                expected = dict(original['criterion'], response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
            _check(json.loads(item['grading_criterion']), expected, 'SciVis exact original component rule and scale')
            identity = json.dumps([item['content'], original['criterion'], item['item_features'], item['asset_manifest']], sort_keys=True)
            parent = dict(item, item_id=identity, raw_item_id=raw, grading_criterion=json.dumps(original['criterion']))
            if identity in parents:
                _check(parent, parents[identity], 'SciVis full input and grading context on every component')
            parents[identity] = parent
        mapping[attempt] = identity
    return _scivisagentbench(directory, parent_view(tables, list(parents.values()), execution_parents=mapping), metadata, source=source)
