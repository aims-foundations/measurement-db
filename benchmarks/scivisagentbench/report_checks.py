"""Read explicit component scores from released HTML reports."""

import re

from bs4 import BeautifulSoup


def scored_checks(html):
    page = BeautifulSoup(html, 'html.parser')
    checks = []

    def add(identity, criterion, text):
        match = re.fullmatch(r'([0-9.]+)/([0-9.]+)(?:\s*\([0-9.]+%\))?', text)
        if match is None:
            raise ValueError('Unrecognized explicit component score: ' + text)
        score, maximum = map(float, match.groups())
        if score == maximum == 0:
            return  # The report declares no scoring opportunities for this metric.
        if not 0 <= score <= maximum or maximum <= 0:
            raise ValueError('Invalid component score bounds')
        checks.append(dict(component_id=identity, criterion=criterion, score=score, maximum=maximum))

    for index, node in enumerate(page.select('.rubric-item')):
        add('vision:' + str(index), node.select_one('.rubric-criterion').get_text(' ', strip=True),
            node.select_one('.rubric-score').get_text(strip=True))
    for index, node in enumerate(page.find_all('span', style='font-weight: 600; color: #333;')):
        add('rule:' + str(index), node.get_text(' ', strip=True), node.find_next_sibling('span').get_text(strip=True))
    for node in page.select('.metric-box'):
        name = node.select_one('.metric-label').get_text(strip=True)
        if name == 'Visualization Quality' and any(c['component_id'].startswith('vision:') for c in checks):
            continue
        if name == 'Rule-Based Score' and any(c['component_id'].startswith('rule:') for c in checks):
            continue
        if name in {'Visualization Quality', 'Rule-Based Score', 'Output Generation', 'Efficiency', 'Text Q&A Score'}:
            # Narrative Q&A judgments do not expose a stable per-question score table.
            # Keep that explicitly reported subtotal until its mapping is unambiguous.
            add('metric:' + name, name, node.select_one('.metric-value').get_text(strip=True))
    return checks
