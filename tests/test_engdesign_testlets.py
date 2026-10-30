"""Engineering checks preserve native partial scores and complete executions."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_engdesign_audit import verify


@pytest.fixture
def release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / 'benchmarks/engdesign/build.py'
    spec = importlib.util.spec_from_file_location('engineering_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'engdesign'
    raw = directory / 'raw'
    metadata = yaml.safe_load(path.with_name('metadata.yaml').read_text())
    archive = raw / metadata['build']['parameters']['layout']['archive']
    archive.parent.mkdir(parents=True)
    record = dict(response='Full original answer ' * 1000, passed=True,
        evaluation_result={'passband below target attenuation appropriate': True,
                           'target frequency attenuation appropriate': False})
    with zipfile.ZipFile(archive, 'w') as output:
        for task in ['AV_02', 'TB_01', 'AB_02']:
            output.writestr(f'release/tasks/{task}/LLM_prompt.txt', f'Design {task}.')
            output.writestr(f'release/tasks/{task}/output_structure.py', 'class Response_structure: pass\n')
            output.writestr(f'release/tasks/{task}/evaluate.py', '# Full original evaluator retained.\n')
            output.writestr(f'release/tasks/{task}/images/input.svg', '<svg xmlns="http://www.w3.org/2000/svg"/>')
        for trial, row in enumerate([record,
                dict(record, passed=False, evaluation_result={'error': 'Compilation failed'}),
                dict(record, passed='Evaluation failed', evaluation_result=None)]):
            text = repr(row).replace("appropriate': True", "appropriate': np.True_")
            output.writestr(f'release/tasks/AV_02/logs/design_log_4o_{trial}.jsonl', text)
        output.writestr('release/tasks/TB_01/logs/design_log_4o_0.jsonl', json.dumps(dict(
            record, passed=False, evaluation_result={'sub_scores': dict(dc_gain=20, ugf=5, pm=15, icmr=0, io_err=5, idc=10)})))
        output.writestr('release/tasks/AB_02/logs/design_log_4o_0.jsonl', json.dumps(dict(
            record, evaluation_result={'iou_score': 0.9, 'temp_file_cleaned': True})))
        folder = 'release/iterative_result/iterative_gpt_4o/AV_02/'
        output.writestr(folder + 'design_log_4o_iteration_1.jsonl', repr(dict(record, iteration=1)))
        output.writestr(folder + 'design_responses.txt', 'Attempt 1:\nComplete refinement output.\n')
        output.writestr(folder + 'design_evaluations.txt', 'Attempt 1:\nComplete refinement feedback.\n')
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        target = tmp_path / 'converted'
        module.EngDesign(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(target)])
        tables = {p.stem: pd.read_parquet(p) for p in target.glob('*.parquet')}
        yield directory, tables, metadata
    finally:
        reload()


def test_native_checks_subtotals_missing_grades_and_dependent_iterations(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 6
    assert stats['source_component_observations'] == 19
    assert stats['source_observed_testlets'] == 3
    assert stats['source_iterative_chains'] == 1
    assert stats['source_ungraded_observations'] == 1
    tables = release[1]
    assert 20 in tables['responses'].response.values
    assert tables['responses'].response.isna().sum() == 7
    assert not tables['items'].raw_item_id.str.contains('temp_file_cleaned').any()
    assert tables['attempts'].trace.str.len().min() > 16000


@pytest.mark.parametrize('damage', ['grade', 'testlet', 'field', 'scale', 'answer', 'aggregate', 'image', 'occasion'])
def test_lost_or_invented_evidence_is_rejected(release, damage):
    directory, tables, metadata = release
    if damage == 'grade':
        tables['responses'].loc[0, 'response'] = 0.25
    elif damage == 'testlet':
        tables['items'].loc[0, 'testlet_id'] = 'wrong-design'
    elif damage in {'field', 'scale'}:
        criterion = json.loads(tables['items'].loc[0, 'grading_criterion'])
        if damage == 'field':
            rule = json.loads(criterion['rule'])
            rule['selected_check'] = 'other-check'
            criterion['rule'] = json.dumps(rule)
        else:
            criterion['response_scale'] = dict(kind='interval', min=0, max=100, direction='higher_is_better')
        tables['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
    elif damage == 'image':
        tables['assets'].loc[0, 'data'] = b'Changed input'
    elif damage == 'occasion':
        tables['responses'].loc[0, 'trial'] = 99
    else:
        trace = json.loads(tables['attempts'].loc[0, 'trace'])
        if damage == 'aggregate':
            trace['recorded_task_score'] = 0.25
        else:
            native = json.loads(trace['source_trace'])
            native['native_record'] = native['native_record'].replace('Full original answer', 'Changed answer')
            trace['source_trace'] = json.dumps(native)
        tables['attempts'].loc[0, 'trace'] = json.dumps(trace)
    with pytest.raises((ValueError, KeyError, AssertionError, RuntimeError)):
        verify(directory, tables, metadata)


def test_selected_fields_never_execute_python(tmp_path):
    from benchmarks.engdesign.checks import read_checks
    target = tmp_path / 'executed'
    text = "{'evaluation_result': {'check': __import__('pathlib').Path(" + repr(str(target)) + ").touch()}}"
    with pytest.raises(ValueError, match='Unsupported representation'):
        read_checks(text, {'check': {'format': 'boolean'}})
    assert not target.exists()
