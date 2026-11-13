"""The OS-Harm adapter preserves execution, state and auxiliary-model roles."""
import importlib
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from scripts.build_measurement_tables.trace_format import validate
from scripts.curate_benchmarks.trace_source_audit import recover_os_harm
Builder = runpy.run_path(str(ROOT / 'benchmarks/os_harm/build.py'))['OSHarm']


def source():
    return {'record': {'params': {'model': 'solver'}, 'steps': [
        {'a11y_tree': 'Complete tree Ω\n'*3000, 'screenshot_file': 'external.png',
         'response': 'The recorded reply\n```python\npyautogui.click(1,2)\n```',
         'actions': ['pyautogui.click(1,2)', 'DONE']}]},
        'caption': {'caption': 'The final screen', 'response': {'model': 'captioner', 'created': 123,
                    'usage': {'prompt_tokens': 1000}}},
        'judgment': {'reasoning': 'The recorded judge explanation', 'safety': True, 'success': False},
        'trajectory': [{'action_timestamp': '20250605@085627', 'reward': 0, 'done': True}]}


def test_recorded_actions_and_auxiliary_models_do_not_become_invented_solver_calls():
    native = source(); raw = json.dumps(native, ensure_ascii=False, indent=2)
    result = Builder.__new__(Builder).convert_trace(raw, context=None)
    validate(result.value)
    assert recover_os_harm(result.value) == raw
    assert [s['type'] for s in result.value['steps']] == ['event', 'llm', 'event', 'event', 'event']
    assert result.counts['llm'] == 1 and result.counts['recorded_actions'] == 2
    solver = result.value['steps'][1]
    assert solver['model'] == 'solver' and not any(k in solver for k in ['usage','cost','toolCalls','start','end'])
    assert result.value['evaluation']['judge']['content'] == native['judgment']['reasoning']
    assert 'timing' not in result.value and 'run' not in result.value


@pytest.mark.parametrize('change', ['response', 'action', 'caption', 'judge', 'drop', 'order', 'attribution'])
def test_independent_audit_rejects_changed_or_misattributed_display_values(change):
    value = Builder.__new__(Builder).convert_trace(json.dumps(source()), context=None).value
    if change == 'response': value['steps'][1]['content'] = 'Clipped'
    elif change == 'action': value['steps'][2]['content'] = 'Invented action'
    elif change == 'caption': value['steps'][-1]['content'] = 'Invented screen'
    elif change == 'judge': value['evaluation']['judge']['content'] = 'Invented explanation'
    elif change == 'drop': value['steps'].pop()
    elif change == 'order': value['steps'][0],value['steps'][1] = value['steps'][1],value['steps'][0]
    else: value['steps'][1]['usage'] = {'prompt': 1000}
    with pytest.raises(ValueError): recover_os_harm(value)


def test_full_builder_keeps_all_measurement_files_byte_identical():
    tests = importlib.import_module('test_native_source_audits')
    case = tests.OSHarmNativeAuditTests(methodName='runTest'); case.setUp()
    try:
        from measurement_db.build_base import BenchmarkBuild
        import measurement_db.build_base as api
        class Legacy(case.builder):
            _prepare_traces = BenchmarkBuild._prepare_traces
            _write_trace_table = BenchmarkBuild._write_trace_table
        api._tables.reload(); baseline = case.directory.parent/'legacy'
        Legacy(str(case.directory/'build.py')).main_from_args(['--source',str(case.directory/'raw'),'--output',str(baseline)])
        for path in (case.directory.parent/'tables').glob('*.parquet'):
            if path.name != 'traces.parquet': assert path.read_bytes() == (baseline/path.name).read_bytes(), path.name
    finally:
        case.doCleanups()
