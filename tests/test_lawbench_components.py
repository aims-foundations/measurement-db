"""Native set comparisons preserve misses, extra predictions and parser quirks."""

import pytest

pytest.importorskip('cn2an')
from benchmarks.lawbench.components import component_scores


def test_multilabel_checks_do_not_turn_extra_labels_into_successes():
    checks = component_scores('2-3', dict(refr='类别:有子女、有财产。', prediction='有子女，有债务'),
        dict(classification_options=['有子女', '有财产', '有债务']))
    assert {c['component_id']: c['response'] for c in checks} == {
        'reference:有子女': 1., 'reference:有财产': 0., 'extra_prediction_count': 1}


def test_accusations_preserve_native_substring_overlap_and_duplicates():
    checks = component_scores('3-3', dict(refr='罪名:盗窃', prediction='盗窃、盗窃、抢劫'),
        dict(accusations=['盗窃', '抢劫']))
    assert {c['component_id']: c['response'] for c in checks} == {'reference:盗窃': 1., 'extra_prediction_count': 1}


def test_statute_parser_preserves_paragraph_removal_and_first_number_quirks():
    checks = component_scores('3-1', dict(refr='法条:刑法第10、20条',
        prediction='第十条第二款、第20条与第30条、第20条、第99条'), {})
    assert {c['component_id']: c['response'] for c in checks} == {
        'reference:10': 0., 'reference:20': 1., 'extra_prediction_count': 1}


def test_blank_prediction_keeps_each_required_statute_as_a_miss():
    checks = component_scores('3-1', dict(refr='法条:刑法第10、20条', prediction=''), {})
    assert [c['response'] for c in checks] == [0., 0., 0]
