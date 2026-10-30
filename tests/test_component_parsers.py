"""Guard native grading semantics that can be lost during component extraction."""

import pytest

from benchmarks.infibench.component_grades import released_checks
from benchmarks.scivisagentbench.report_checks import scored_checks
from benchmarks.swebench_live.test_reports import named_checks
from scripts.build_measurement_tables import register_measurements as registrations


def test_swe_report_layouts_and_conflicting_aliases():
    statuses = {'FAIL_TO_PASS': {'success': ['works', 'conflict', 'works'], 'failure': ['fails', 'conflict']}}
    expected = {('FAIL_TO_PASS', 'works'): 1., ('FAIL_TO_PASS', 'fails'): 0., ('FAIL_TO_PASS', 'conflict'): None}
    assert named_checks({'issue': {'tests_status': statuses}}, 'issue') == expected
    assert named_checks({'instance_id': 'issue', **statuses}, 'issue') == expected
    assert named_checks({'instance_id': 'issue', 'resolved': True}, 'issue') == {}
    with pytest.raises(ValueError, match='different tasks'):
        named_checks({'instance_id': 'another'}, 'issue')
    with pytest.raises(ValueError, match='vocabulary'):
        named_checks({'FAIL_TO_PASS': {'unknown_status': ['test']}}, 'issue')


def test_infi_retains_negative_keywords_and_native_point_scores():
    config = {'grading': {'keywords': [{'post_handler': 'strip'}, {'content': 'forbidden', 'neg': True}],
        'unit_test': {'tests': ['assert result == 2']}, 'customized': {'rule': 'native total'}}}
    checks = list(released_checks(config, {'keywords': ['match'], 'unit_test_detail': [{'passed': False}], 'custom_score': -2.5}))
    assert [(r[0], r[2]) for r in checks] == [('keyword:0', 1.), ('test:0', 0.), ('custom', -2.5)]
    assert checks[0][3]['direction'] == 'lower_is_better'
    assert checks[-1][3] == dict(kind='interval', min=None, max=None, direction='higher_is_better')
    with pytest.raises(ValueError, match='do not align'):
        list(released_checks(config, {'keywords': []}))
    with pytest.raises(ValueError, match='test verdict'):
        list(released_checks(config, {'unit_test_detail': [{'passed': 'false'}]}))


def test_visual_ratings_do_not_duplicate_subtotals_or_invent_zero_denominator_checks():
    html = '''<div class="rubric-item"><div class="rubric-score">7/10</div>
      <div class="rubric-criterion">Color fidelity</div></div>
      <div class="metric-box"><div class="metric-label">Visualization Quality</div><div class="metric-value">7/10</div></div>
      <div class="metric-box"><div class="metric-label">Text Q&amp;A Score</div><div class="metric-value">0/0</div></div>
      <div class="metric-box"><div class="metric-label">Efficiency</div><div class="metric-value">3/10</div></div>'''
    checks = scored_checks(html)
    assert [(c['component_id'], c['score'], c['maximum']) for c in checks] == [('vision:0', 7., 10.), ('metric:Efficiency', 3., 10.)]


def test_subject_identity_can_be_computed_before_registering_aliases(monkeypatch):
    entry = dict(model='owner/model', company='provider', release_date='2025-01')
    monkeypatch.setattr(registrations, '_model_registry', {'owner/model': entry, 'owner__model': entry})
    monkeypatch.setattr(registrations, '_subjects', {})
    first = registrations.subject_id_for('owner/model', features={'harness': 'agent'})
    assert first == registrations.subject_id_for('owner__model', features={'harness': 'agent'})
    assert registrations._subjects == {}
    assert first == registrations.resolve_subject('owner__model', features={'harness': 'agent'})
    assert first != registrations.subject_id_for('owner/model', features={'harness': 'another agent'})
    assert first != registrations.subject_id_for('owner/model', features={'harness': 'agent'}, access_date='2026-01')
