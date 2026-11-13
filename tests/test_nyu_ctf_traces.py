"""Released multi-agent logs retain source boundaries and measured-turn roles."""
import importlib
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from scripts.build_measurement_tables.trace_format import validate
from scripts.curate_benchmarks.trace_source_audit import recover_nyu_ctf


Builder = runpy.run_path(str(ROOT/'benchmarks/nyu_ctf_bench/build.py'))['NYUCTFBench']


def convert(source):
    return Builder.__new__(Builder).convert_trace(source, context=None)


def source():
    call = {'id': 'native-id', 'function': {'name': 'run_command', 'arguments': '{"command":"file input"}'}}
    return {'task_id': 'task', 'summary': {'results': {'task': None}}, 'released_traces': [
        {'source_member': 'baseline/attempt-1.json', 'source_record': {'args': {'model': 'solver'}, 'messages': [
            ['2024-01-01T00:00:00+00:00', {'role': 'system', 'content': 'System', 'extra': {'type': 'SystemMessage'}}],
            ['2024-01-01T00:00:00+00:00', {'role': 'assistant', 'content': 'Demonstration', 'extra': {'type': 'AssistantMessage', 'response': {'data': None}}}],
            ['2024-01-01T00:00:01+00:00', {'type': 'text', 'text': 'Raw provider copy'}],
            ['2024-01-01T00:00:01+00:00', {'role': 'assistant', 'content': 'Actual output', 'tool_calls': [call], 'extra': {'type': 'UnparsedToolCalls', 'response': {'data': {'id': 'provider-id'}}}}],
            ['2024-01-01T00:00:01+00:00', {'role': 'tool', 'tool_call_id': 'native-id', 'name': 'run_command', 'content': 'Recorded result Ω\n'*3000}]]}},
        {'source_member': 'components/attempt-2.json', 'source_record': {'planner_model': 'planner-model', 'executor_model': 'executor-model', 'total_cost': 100,
            'planner': [{'role': 'MessageRole.ASSISTANT', 'content': 'Plan', 'index': 1}],
            'executors': [[{'role': 'MessageRole.KNOWLEDGE', 'content': 'Retrieved context'},
                {'role': 'MessageRole.ASSISTANT', 'content': 'Execute', 'tool_call': {'name': 'run_command', 'parsed_args': {'command': 'file input'}}},
                {'role': 'MessageRole.OBSERVATION', 'content': None, 'tool_result': {'name': 'run_command', 'result': {'stdout': 'ELF', 'returncode': 0}}}]]}},
        {'source_member': 'enigma/attempt-3.traj', 'source_record': {
            'trajectory': [{'response': 'Actual history output', 'action': 'ls'}], 'info': {'model_stats': {'cost': 100}},
            'history': [{'role': 'assistant', 'is_demo': True, 'content': 'Demo', 'action': 'demo'},
                {'role': 'assistant', 'agent': 'primary', 'content': 'Actual history output', 'thought': 'Reason', 'action': 'ls'}]}}]}


def test_provider_copies_demonstrations_and_component_agents_remain_distinct():
    raw = json.dumps(source(), ensure_ascii=False, indent=2)
    result = convert(raw); validate(result.value)
    assert recover_nyu_ctf(result.value) == raw
    turns = [s for s in result.value['steps'] if s['type'] == 'llm']
    assert len(turns) == 4
    assert [s['content'] for s in turns] == ['Actual output', 'Plan', 'Execute', 'Actual history output']
    assert [s.get('agent') for s in turns] == [None, 'planner', 'executors[0]', 'primary']
    assert turns[0]['toolCalls'][0]['id'] == 'native-id'
    assert 'id' not in turns[2]['toolCalls'][0]
    assert sum(s.get('name') == 'released_log_file' for s in result.value['steps']) == 3
    assert sum(s.get('name') == 'recorded_action' for s in result.value['steps']) == 1
    assert 'timing' not in result.value and 'run' not in result.value
    assert not any('usage' in s or 'cost' in s or 'start' in s for s in turns)


def test_summary_only_null_assessment_does_not_invent_an_attempt():
    native = source(); native['released_traces'] = []
    result = convert(native); validate(result.value)
    assert recover_nyu_ctf(result.value) == native
    assert len(result.value['steps']) == 1 and result.value['steps'][0]['content'] == 'null'
    assert not result.counts['llm']


@pytest.mark.parametrize('change', ['clip', 'drop', 'order', 'demo', 'knowledge', 'agent', 'metrics', 'invented_id'])
def test_independent_audit_rejects_source_and_attribution_corruption(change):
    value = convert(source()).value
    turns = [s for s in value['steps'] if s['type'] == 'llm']
    if change == 'clip': turns[0]['content'] = 'Clipped'
    elif change == 'drop': value['steps'].pop()
    elif change == 'order': value['steps'][1], value['steps'][2] = value['steps'][2], value['steps'][1]
    elif change == 'demo': next(s for s in value['steps'] if s.get('name') == 'demonstration')['type'] = 'llm'
    elif change == 'knowledge': next(s for s in value['steps'] if s.get('name') == 'retrieved_context')['type'] = 'llm'
    elif change == 'agent': turns[2]['agent'] = 'planner'
    elif change == 'metrics': turns[0]['cost'] = 100
    else: turns[2]['toolCalls'][0]['id'] = 'invented'
    with pytest.raises(ValueError): recover_nyu_ctf(value)


def test_full_builder_preserves_measurements_aggregate_links_and_asset_bytes():
    tests = importlib.import_module('test_native_source_audits')
    case = tests.NYUCTFAuditTests(methodName='runTest'); case.setUp()
    try:
        from measurement_db.build_base import BenchmarkBuild
        import measurement_db.build_base as api
        cls = runpy.run_path(str(ROOT/'benchmarks/nyu_ctf_bench/build.py'))['NYUCTFBench']
        class Legacy(cls):
            _prepare_traces = BenchmarkBuild._prepare_traces
            _write_trace_table = BenchmarkBuild._write_trace_table
        api._tables.reload(); baseline = case.directory/'legacy'
        Legacy(str(case.directory/'build.py')).main_from_args(['--source',str(case.directory/'raw'),'--output',str(baseline)])
        for path in (case.directory/'test_tables').glob('*.parquet'):
            if path.name != 'traces.parquet': assert path.read_bytes() == (baseline/path.name).read_bytes(), path.name
    finally:
        case.doCleanups()
