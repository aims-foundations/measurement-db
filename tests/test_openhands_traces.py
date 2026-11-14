"""OpenHands variants preserve source events, provenance and measurements."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest

from trace_builder_fixtures import legacy_traces
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from scripts.build_measurement_tables.trace_conversion import convert_trace
from trace_builder_fixtures import build_trace
from scripts.build_measurement_tables.trace_format import validate
from scripts.curate_benchmarks.trace_source_audit import recover_theagentcompany

Builder = runpy.run_path(str(ROOT / 'benchmarks/theagentcompany/build.py'))['TheAgentCompany']


def convert(source, **options):
    result = convert_trace(source, family="openhands", **options)
    return result.value, result.counts


def events():
    response = {'id': 'recorded-response', 'created': 1733471787, 'model': 'native-model',
                'choices': [{'message': {'role': 'assistant', 'content': 'Read both files.',
                 'tool_calls': [dict(id=name, type='function', function={'name': 'read',
                               'arguments': json.dumps({'path': name})}) for name in ['a', 'b']]}}],
                'usage': {'prompt_tokens': 30, 'completion_tokens': 10,
                          'prompt_tokens_details': {'cached_tokens': 20}}}
    source = [dict(id=0, source='user', action='message', args={'content': 'Solve the task.'},
                   timestamp='2024-12-06T07:56:25.192436')]
    for number, name in enumerate(['a', 'b']):
        metadata = dict(function_name='read', tool_call_id=name, model_response=deepcopy(response))
        source += [dict(id=number*2+1, source='agent', action='read', args={'path': name},
                        tool_call_metadata=deepcopy(metadata)),
                   dict(id=number*2+2, source='agent', observation='read', cause=number*2+1,
                        content='完整 output ' + name * 18000, tool_call_metadata=metadata)]
    return source


def test_parallel_actions_and_copied_responses_do_not_duplicate_model_turns():
    source = events(); original = deepcopy(source)
    raw = json.dumps(source, ensure_ascii=False, indent=2) + '\n'
    value, counts = convert(raw)
    validate(value)
    assert source == original
    assert recover_theagentcompany(value) == raw
    assert counts['llm'] == 1 and counts['tool'] == 2
    assert counts['duplicate_response_events'] == 1
    assert not counts['unmatched_tool_results']
    assert [s['callId'] for s in value['steps'] if s['type'] == 'tool'] == ['a', 'b']
    assert value['steps'][1]['usage'] == {'prompt': 30, 'completion': 10, 'cacheRead': 20}
    assert value['timing'] == 'partial'
    assert value['steps'][1]['start'] == 1733471787
    assert not any('end' in step for step in value['steps'])
    assert 'apiCalls' not in value.get('run', {})


@pytest.mark.parametrize('change', ['content', 'arguments', 'association', 'drop', 'order'])
def test_independent_audit_rejects_display_and_association_corruption(change):
    value, _ = convert(json.dumps(events()))
    if change == 'content': value['steps'][1]['content'] = 'A different answer'
    elif change == 'arguments': value['steps'][2]['arguments'] = '{}'
    elif change == 'association': value['steps'][2]['callId'] = 'b'
    elif change == 'drop': value['steps'].pop()
    else: value['steps'][2], value['steps'][3] = value['steps'][3], value['steps'][2]
    with pytest.raises(ValueError): recover_theagentcompany(value)


def test_conflicting_provider_response_and_dangling_tool_result_fail():
    source = events()
    source[3]['tool_call_metadata']['model_response']['choices'][0]['message']['content'] = 'conflicting'
    with pytest.raises(ValueError, match='conflicting'): convert(source)
    source = events(); source[2]['cause'] = 999
    _, counts = convert(source)
    assert counts['unmatched_tool_results'] == 1


def test_harness_controls_and_unassociated_errors_are_not_invented_model_calls():
    source = [dict(id=0, source='agent', action='system', args={'content': 'Policy'}),
              dict(id=1, source='user', action='recall', args={'query': 'Task'}, message='Recall'),
              dict(id=2, source='environment', observation='recall', cause=1, content='Context', message='Context'),
              dict(id=3, source='agent', observation='error', content='Unknown function'),
              dict(id=4, source='agent', action='condensation', message='Dropped history')]
    value, counts = convert(json.dumps(source))
    validate(value)
    assert [s['type'] for s in value['steps']] == ['message'] + ['event']*4
    assert not counts['llm'] and not counts['unmatched_tool_results']
    assert recover_theagentcompany(value) == json.dumps(source)


def test_native_action_without_provider_metadata_retains_execution_arguments():
    source = [dict(id=1, source='agent', action='run', message='Execute the command',
                   args={'command': 'ls', 'thought': 'Inspect files'}),
              dict(id=2, source='agent', observation='run', cause=1, content='Failed', extras={'exit_code': 2})]
    value, counts = convert(source)
    validate(value)
    assert counts['llm'] == counts['tool'] == 1
    assert 'callId' not in value['steps'][1]
    assert value['steps'][1]['exitCode'] == 2
    assert recover_theagentcompany(value) == source


def test_roleplay_and_text_transcript_variants_do_not_invent_identifiers_or_turns():
    source = json.dumps([{'user': 'Instruction', 'assistant': 'Solution', 'tool_calls': [
        {'func_name': 'execute_code', 'args': {'code': 'print(1)'}, 'result': '1'}]}])
    value = build_trace('theagentcompany', source).value
    validate(value)
    assert [s['type'] for s in value['steps']] == ['message', 'llm', 'tool']
    assert 'callId' not in value['steps'][-1]
    assert recover_theagentcompany(value) == source
    raw = '01. SYSTEM MESSAGE\nUnstructured native transcript\n' + '測定' * 20000
    value = build_trace('theagentcompany', raw).value
    validate(value)
    assert len(value['steps']) == 1 and recover_theagentcompany(value) == raw
    with pytest.raises(json.JSONDecodeError): build_trace('theagentcompany', '[broken JSON')


def test_complete_builder_and_native_audit_preserve_nontrace_files_and_ungraded_attempt(tmp_path):
    from measurement_db.build_base import BenchmarkBuild
    from scripts.build_measurement_tables import reload
    from measurement_db.scripts.curate_benchmarks.batch2_audits import verify_batch2
    reload()
    metadata = yaml.safe_load((ROOT / 'benchmarks/theagentcompany/metadata.yaml').read_text())
    metadata['build']['parameters']['models'] = {'run-a': 'model-a', 'run-b': 'model-b'}
    metadata['build']['parameters']['harnesses'] = {'run-a': 'OpenHands', 'run-b': 'MUSE'}
    bench = tmp_path / 'theagentcompany'; bench.mkdir()
    (bench / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    raw = bench / 'raw'
    task = raw / 'tasks/task-a'; task.mkdir(parents=True)
    (task / 'task.md').write_text('Solve the task.')
    (task / 'evaluator.py').write_text('released evaluator')
    for run in ['run-a', 'run-b']:
        run_dir = raw / 'experiments/evaluation/1.0.0' / run
        (run_dir / 'trajectories').mkdir(parents=True)
        if run == 'run-a':
            (run_dir / 'results').mkdir()
            (run_dir / 'results/eval_task-a.json').write_text(json.dumps({'final_score': {'result': 1, 'total': 2}}))
            text = json.dumps(events(), ensure_ascii=False, indent=2)
            (run_dir / 'trajectories/traj_task-a.json').write_text(text)
            (run_dir / 'trajectories/traj_task-a.json.gz').write_bytes(gzip.compress(text.encode()))
        else:
            (run_dir / 'trajectories/traj_task-a.txt').write_text('Full text transcript\n' + 'Ω' * 18000)
    output = tmp_path / 'structured'
    Builder(str(bench / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
    audit = verify_batch2(bench, output)
    assert audit['source_traces'] == 2 and audit['source_ungraded_observations'] == 1
    assert pd.read_parquet(output / 'responses.parquet').response.isna().sum() == 1
    class Legacy(Builder):
        _prepare_traces = legacy_traces
        _write_trace_table = BenchmarkBuild._write_trace_table
    reload()
    baseline = tmp_path / 'legacy'
    Legacy(str(bench / 'build.py')).main_from_args(['--source', str(raw), '--output', str(baseline)])
    for name in ['benchmarks', 'items', 'subjects', 'responses']:
        assert (output / f'{name}.parquet').read_bytes() == (baseline / f'{name}.parquet').read_bytes()
    altered = pd.read_parquet(output / 'traces.parquet')
    value = json.loads(altered.loc[0, 'trace']); value['steps'][1]['content'] = 'Corrupted'
    altered.loc[0, 'trace'] = json.dumps(value)
    altered.to_parquet(output / 'traces.parquet', index=False)
    with pytest.raises(ValueError): verify_batch2(bench, output)
    reload()
