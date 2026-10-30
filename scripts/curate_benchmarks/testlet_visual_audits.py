"""Reconcile released per-dimension visual ratings without re-running judges."""

import io
import json
import re
import zipfile

import pandas as pd
from bs4 import BeautifulSoup


def source_attempts(directory, metadata):
    if directory.name == 'risebench':
        yield from _rise(directory, metadata)
    elif directory.name == 'scivisagentbench':
        yield from _scivis(directory, metadata)
    else:
        raise ValueError(directory.name)


def _rise(directory, metadata):
    raw, parameters = directory / 'raw', metadata['build']['parameters']
    run = 0
    for profile in [*parameters['subjects'], *parameters['current_subjects']]:
        if profile in parameters['subjects']:
            with zipfile.ZipFile(raw / 'results' / (profile + '.zip')) as archive:
                sheet = pd.read_excel(io.BytesIO(archive.read(profile + '/bagel_judge.xlsx')), keep_default_na=False)
        else:
            path, = sorted((raw / 'current' / profile).glob('*_judge.xlsx'))
            sheet = pd.read_excel(path, keep_default_na=False)
        for row in sheet.to_dict('records'):
            task, checks = row['index'], []
            grade = float(row['complete']) if row['match_log'] == 'succeed' else None
            for name in ('Reasoning', 'ApprConsistency', 'VisualPlausibility'):
                value = row[name] if grade is not None else None
                value = None if value in ('', None) else value
                assert value is None or value in (1, 2, 3, 4, 5)
                checks.append((task + '::' + name, task, value))
            yield str(run), grade, checks
            run += 1


def _scivis(directory, metadata):
    raw, parameters = directory / 'raw', metadata['build']['parameters']
    for pattern in parameters['report_patterns'].values():
        for path in sorted(raw.glob(pattern)):
            suite = parameters['suites'][path.relative_to(raw).parts[2]]
            for case in BeautifulSoup(path.read_text(), 'html.parser').select('.case-section'):
                task = suite + '/' + case['id']
                checks = []
                for index, node in enumerate(case.select('.rubric-item')):
                    value = float(node.select_one('.rubric-score').get_text().split('/')[0])
                    checks.append((task + '::vision:' + str(index), task, value))
                rule_nodes = case.find_all('span', style='font-weight: 600; color: #333;')
                for index, node in enumerate(rule_nodes):
                    value = float(node.find_next_sibling('span').get_text().split('/')[0])
                    checks.append((task + '::rule:' + str(index), task, value))
                for node in case.select('.metric-box'):
                    name = node.select_one('.metric-label').get_text(strip=True)
                    if name == 'Visualization Quality' and case.select('.rubric-item'):
                        continue
                    if name == 'Rule-Based Score' and rule_nodes:
                        continue
                    if name in {'Visualization Quality', 'Rule-Based Score', 'Output Generation', 'Efficiency', 'Text Q&A Score'}:
                        value, maximum = map(float, node.select_one('.metric-value').get_text(strip=True).split('/'))
                        if maximum:
                            checks.append((task + '::metric:' + name, task, value))
                percent = float(re.search(r'\(([0-9.]+)%\)', case.select_one('.case-score').get_text())[1])
                grade = float(percent >= metadata['grading']['verifiers']['reported_case']['pass_percent'])
                if not checks:
                    checks = [(task, task, grade)]
                yield str(path.relative_to(raw)) + '#' + case['id'], grade, checks
