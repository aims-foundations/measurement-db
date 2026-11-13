"""Browser snapshots precede actions; annotations and media stay unchanged."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_measurement_tables.trace_converters import browsergym
from scripts.build_measurement_tables.trace_format import validate
from scripts.curate_benchmarks.trace_source_audit import recover_browsergym


def source():
    return {'source_file': 'native.json', 'annotation': {'trajectory_success': 'Unsuccessful'},
            'record': {'model': 'recorded-model', 'summary_info': {'err_msg': 'Recorded interruption',
                       'stack_trace': 'Recorded traceback'}, 'steps': [
                {'num': 0, 'url': 'https://initial.example', 'last_action_error': '',
                 'reasoning': 'Inspect the page', 'action': "click('4')",
                 'stats': {'input_tokens': 100, 'output_tokens': 20, 'cost': .01},
                 'axtree': 'Full observation ' + '測定' * 20000,
                 'bounding_boxes': {'4': [0, 1, 2, 3]}, 'screenshot_path': 'recorded.png'},
                {'num': 1, 'url': 'https://initial.example', 'last_action_error': 'Element is detached',
                 'reasoning': 'Try the next element', 'action': "click('5')",
                 'stats': {'input_tokens': 120, 'output_tokens': 30, 'cost': 0}},
                {'num': 2, 'url': 'https://final.example', 'last_action_error': '',
                 'reasoning': None, 'action': None, 'stats': None}]}}


def test_snapshot_and_last_action_error_precede_the_current_agent_output():
    native = source(); before = deepcopy(native)
    raw = json.dumps(native, ensure_ascii=False, indent=2) + '\n'
    value, counts = browsergym.convert(raw)
    validate(value)
    assert native == before and recover_browsergym(value) == raw
    assert [s['type'] for s in value['steps']] == ['event', 'llm', 'event', 'event', 'llm', 'event']
    assert value['steps'][3]['content'] == 'Element is detached'
    assert counts['llm'] == 2 and counts['observations'] == 3 and counts['action_errors'] == 1
    assert value['steps'][1]['usage'] == {'prompt': 100, 'completion': 20}
    assert value['steps'][1]['cost'] == .01 and 'cost' not in value['steps'][4]
    assert counts['zero_cost_with_tokens_omitted'] == 1
    assert value['run']['exception'] == 'Recorded interruption'
    assert 'timing' not in value and not any('toolCalls' in s for s in value['steps'])


@pytest.mark.parametrize('change', ['action', 'reasoning', 'url', 'error', 'exception', 'drop', 'order'])
def test_independent_audit_rejects_changed_or_missing_displayed_values(change):
    value, _ = browsergym.convert(json.dumps(source()))
    if change == 'action': value['steps'][1]['content'] = 'Invented action'
    elif change == 'reasoning': value['steps'][1]['reasoning'] = 'Invented reasoning'
    elif change == 'url': value['steps'][0]['content'] = 'https://another.example'
    elif change == 'error': value['steps'][3]['content'] = 'Invented error'
    elif change == 'exception': value['run']['exception'] = 'Invented exception'
    elif change == 'drop': value['steps'].pop()
    else: value['steps'][0], value['steps'][1] = value['steps'][1], value['steps'][0]
    with pytest.raises(ValueError): recover_browsergym(value)


@pytest.mark.parametrize('fixture', ['WorkArenaAuditTests', 'VisualWebArenaAuditTests'])
def test_full_builder_keeps_measurements_assets_and_grading_channels_byte_identical(fixture):
    # Reuse the existing native-source fixture, including disagreeing repeated
    # human ratings and VisualWebArena's exact archived input-image bytes.
    tests = importlib.import_module('test_native_source_audits')
    case = getattr(tests, fixture)(methodName='runTest')
    case.setUp()
    try:
        from measurement_db.build_base import BenchmarkBuild
        import measurement_db.build_base as api
        original = case.directory.parent / 'tables'
        class Legacy(case.builder):
            _prepare_traces = BenchmarkBuild._prepare_traces
            _write_trace_table = BenchmarkBuild._write_trace_table
        api._tables.reload()
        baseline = case.directory.parent / 'legacy'
        Legacy(str(case.directory / 'build.py')).main_from_args(
            ['--source', str(case.directory / 'raw'), '--output', str(baseline)])
        for path in original.glob('*.parquet'):
            if path.name != 'traces.parquet':
                assert path.read_bytes() == (baseline / path.name).read_bytes(), path.name
    finally:
        case.doCleanups()
