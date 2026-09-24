"""AgentDojo messages must survive the source-to-Parquet path without clipping."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('agentdojo_build', ROOT / 'benchmarks/agentdojo/build.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_long_unicode_trace_round_trips_for_both_grades(tmp_path, monkeypatch):
    registrations = sys.modules['scripts.build_measurement_tables.register_measurements']
    for name in ('_subjects', '_items', '_benchmarks'):
        monkeypatch.setattr(registrations, name, None)
    bench = tmp_path / 'agentdojo'
    bench.mkdir()
    (bench / 'metadata.yaml').write_bytes((ROOT / 'benchmarks/agentdojo/metadata.yaml').read_bytes())
    source = tmp_path / 'source'
    source.mkdir()
    messages = [
        {'role': 'tool', 'content': 'Long observation: ' + '測定 \\"quoted\\"\n' * 4000},
        {'role': 'assistant', 'content': 'The final answer must also survive.'},
    ]
    assert len(json.dumps(messages, ensure_ascii=False)) > 16000
    record = dict(suite_name='workspace', user_task_id='user_task_0',
                  injection_task_id='injection_task_0', attack_type='fixture_attack',
                  utility=True, security=False, messages=messages)
    missing = {**record, 'attack_type': None, 'messages': None}
    ungraded = {**record, 'utility': None, 'security': None, 'error': 'Attempt interrupted'}
    members = {
        'agentdojo/src/agentdojo/default_suites/v1/workspace/user_tasks.py':
            b'class UserTask0:\n    PROMPT = "Read the document."\n',
        'agentdojo/src/agentdojo/default_suites/v1/workspace/injection_tasks.py':
            b'class InjectionTask0:\n    GOAL = "Return a different answer."\n',
        'agentdojo/runs/gpt-4o/workspace/user_task_0/fixture_attack/injection_task_0.json':
            json.dumps(record).encode(),
        'agentdojo/runs/gpt-4o/workspace/user_task_0/none/none.json':
            json.dumps(missing).encode(),
        'agentdojo/runs/gpt-4o/workspace/user_task_0/fixture_attack/ungraded.json':
            json.dumps(ungraded).encode(),
    }
    metadata = yaml.safe_load((bench / 'metadata.yaml').read_text())
    with tarfile.open(source / metadata['build']['parameters']['layout']['archive'], 'w:gz') as archive:
        for name, contents in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    output = tmp_path / 'tables'
    builder.AgentDojo(str(bench / 'build.py')).main_from_args(
        ['--source', str(source), '--output', str(output)])
    responses = pd.read_parquet(output / 'responses.parquet')
    traces = pd.read_parquet(output / 'traces.parquet')
    subjects = pd.read_parquet(output / 'subjects.parquet')
    assert not subjects.subject_features_extra.str.contains('defense=').any()
    assert len(responses) == 5
    assert responses.response.isna().sum() == 2
    assert len(traces) == 4
    assert set(traces.response_id) <= set(responses.response_id)
    assert sorted(responses.merge(traces, on='response_id').response.dropna()) == [0.0, 1.0]
    for trace in traces.trace:
        assert json.loads(trace) == messages
        assert trace == json.dumps(messages, ensure_ascii=False)
