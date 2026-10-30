"""Reconcile every native InfiBench detail, independently of component conversion."""

from pathlib import Path
import re
import tarfile
from zipfile import ZipFile

import yaml


def check_definition(config, component):
    """Resolve a component against the independent native grading specification."""
    grading = config['grading']
    binary = dict(kind='discrete', values=[0, 1], direction='higher_is_better')
    if component == 'custom':
        return dict(kind='custom', definition=grading['customized']), dict(
            kind='interval', min=None, max=None, direction='higher_is_better')
    kind, ordinal = component.split(':')
    index = int(ordinal)
    assert index >= 0
    if kind == 'keyword':
        choices = [value for value in grading['keywords'] if not isinstance(value, dict) or 'post_handler' not in value]
    elif kind == 'blank':
        choices = grading['blank_filling']['targets']
    elif kind == 'test':
        choices = grading['unit_test']['tests']
    elif kind == 'similarity':
        choices = grading['similarity']
        binary = dict(kind='interval', min=0, max=1, direction='higher_is_better')
    else:
        raise ValueError('Unknown native InfiBench component ' + component)
    definition = choices[index]
    if kind == 'keyword' and isinstance(definition, dict) and definition.get('neg'):
        binary['direction'] = 'lower_is_better'
    return dict(kind=kind, index=index, definition=definition), binary


def source_attempts(directory, metadata):
    raw = directory / 'raw'
    layout = metadata['build']['parameters']['layout']
    parts = metadata['grading']['verifiers']['legacy_dev']['score_components']
    with tarfile.open(raw / layout['legacy']) as archive:
        root = archive.getmembers()[0].name.split('/')[0] + '/'
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            name = member.name.removeprefix(root)
            if member.isfile() and re.fullmatch(layout['legacy_results'], name):
                records = yaml.load(archive.extractfile(member).read(), Loader=yaml.CSafeLoader)
                yield from _records('legacy_dev', name, records, parts)
    with ZipFile(raw / layout['results']) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('.yaml'):
                records = yaml.load(archive.read(name), Loader=yaml.CSafeLoader)
                yield from _records('v2_1', name, records, parts)


def _records(version, filename, records, parts):
    for case, record in records.items():
        task = version + ':' + case
        for index, detail in enumerate(record['detail']):
            numerator = sum(detail.get(part + '_score', 0) for part in parts)
            denominator = detail.get('max_score', sum(detail.get(part + '_totscore', 0) for part in parts))
            if 'max_score' in detail:
                numerator = min(numerator, detail['max_score'])
            if 'min_score' in detail:
                numerator = max(numerator, detail['min_score'])
            reconstructed = numerator / denominator * record['full_score']
            aggregate = reconstructed if version == 'legacy_dev' else record['all_scores'][index]
            assert abs(aggregate - reconstructed) <= 1e-12
            checks = []
            for field, kind in [('keywords', 'keyword'), ('blank_filling_detail', 'blank'),
                    ('unit_test_detail', 'test'), ('similarity_detail', 'similarity')]:
                for ordinal, value in enumerate(detail.get(field, [])):
                    if kind == 'keyword':
                        assert value in {'match', 'unmatch'}
                        grade = float(value == 'match')
                    elif kind == 'blank':
                        assert value.startswith(('matched:', 'unmatched:'))
                        grade = float(value.startswith('matched:'))
                    elif kind == 'test':
                        assert type(value['passed']) is bool
                        grade = float(value['passed'])
                    else:
                        grade = value['normalized_score']
                    checks.append((task + '::' + kind + ':' + str(ordinal), task, grade))
            if 'custom_score' in detail:
                checks.append((task + '::custom', task, detail['custom_score']))
            if not checks:
                checks = [(task, task, aggregate)]
            yield version + ':' + filename + ':' + case + ':' + str(index + 1), aggregate, checks
