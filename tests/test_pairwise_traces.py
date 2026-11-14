"""Pairwise traces keep their own side and never attribute judge costs to it."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from scripts.build_measurement_tables.trace_converters import pairwise
from scripts.build_measurement_tables.trace_format import validate
from scripts.curate_benchmarks.trace_source_audit import recover_alpacaeval

REFERENCE = 'gpt4_1106_preview'


def record(side=2):
    own, other = ('2', '1') if side == 2 else ('1', '2')
    return {'native_record': {f'generator_{own}': 'evaluated-model', f'generator_{other}': REFERENCE,
            f'output_{own}': 'Complete candidate answer Ω\n'*2000, f'output_{other}': 'Fixed reference answer',
            'annotator': 'weighted_alpaca_eval_gpt4_turbo', 'preference': 1.25,
            'raw_completion': {'message': {'content': 'M'}, 'total_tokens': 2000},
            'price_per_example': .02, 'time_per_example': .5}, 'source_file': 'annotations.json', 'source_row': 0}


@pytest.mark.parametrize('side', [1, 2])
def test_evaluated_side_judge_output_and_native_counterpart_remain_distinct(side):
    source = record(side); raw = json.dumps(source, ensure_ascii=False, indent=2)
    value, counts = pairwise.convert(raw, reference=REFERENCE)
    validate(value)
    assert recover_alpacaeval(value) == raw and counts['comparison_sides'] == 1
    assert len(value['steps']) == 1 and value['steps'][0]['model'] == 'evaluated-model'
    assert value['steps'][0]['content'] == source['native_record'][f'output_{side}']
    assert value['evaluation']['judge']['content'] == 'M'
    assert 'usage' not in value['steps'][0] and 'cost' not in value['steps'][0] and 'run' not in value


@pytest.mark.parametrize('change', ['answer', 'judge', 'wrong_side', 'judge_cost', 'extra_side'])
def test_independent_audit_rejects_crossed_sides_and_changed_judge_output(change):
    value, _ = pairwise.convert(json.dumps(record()), reference=REFERENCE)
    if change == 'answer': value['steps'][0]['content'] = 'Clipped'
    elif change == 'judge': value['evaluation']['judge']['content'] = 'm'
    elif change == 'wrong_side':
        step = value['steps'][0]; step['model'] = REFERENCE; step['content'] = 'Fixed reference answer'
        for ref in step['native']['source_fields']:
            ref['source'][-1] = ref['source'][-1].replace('_2', '_1')
    elif change == 'judge_cost': value['steps'][0]['cost'] = .02
    else: value['steps'].append(deepcopy(value['steps'][0]))
    with pytest.raises(ValueError): recover_alpacaeval(value)


def test_unavailable_judgment_has_no_invented_vote_or_evaluation():
    source = record(); source['native_record'].update(preference=-1, raw_completion=None)
    value, counts = pairwise.convert(source, reference=REFERENCE)
    validate(value)
    assert recover_alpacaeval(value) == source and counts['judge_outputs'] == 0
    assert 'evaluation' not in value


def test_complete_alpaca_builder_preserves_soft_preferences_ungraded_rows_and_all_other_files(tmp_path):
    from measurement_db.build_base import BenchmarkBuild
    import measurement_db.build_base as api
    from measurement_db.scripts.curate_benchmarks.native_result_audits import _alpacaeval
    api._tables.reload()
    bench = tmp_path / 'alpacaeval'; bench.mkdir()
    metadata = yaml.safe_load((ROOT / 'benchmarks/alpacaeval/metadata.yaml').read_text())
    (bench / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    layout = metadata['build']['parameters']['layout']; raw = bench / 'raw'; release = raw / layout['release']
    bank = [{'instruction': f'Question {i}', 'output': f'Reference {i}', 'dataset': 'fixture'} for i in range(2)]
    p = release / layout['reference']; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(bank))
    p = release / layout['judge_prompt']; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('Frozen judge prompt')
    source = {}
    for candidate, prefs in [('model-a', [1.25, 1.75]), ('model-b', [None, 1.5])]:
        rows = []
        for number, pref in enumerate(prefs):
            row = dict(generator_1=REFERENCE, generator_2=candidate, instruction=bank[number]['instruction'],
                       output_1=bank[number]['output'], output_2='Full candidate output ' + candidate + 'Ω'*18000,
                       dataset='fixture', annotator='weighted_alpaca_eval_gpt4_turbo',
                       preference=pref if pref is not None else -1,
                       raw_completion={'message': {'content': 'M'}} if pref is not None else None)
            rows.append(row)
        p = release / f'results/{candidate}/weighted_alpaca_eval_gpt4_turbo/annotations.json'
        p.parent.mkdir(parents=True); p.write_text(json.dumps(rows, ensure_ascii=False))
        source.update({(str(p.relative_to(raw)), i): row for i, row in enumerate(rows)})
    Builder = runpy.run_path(str(ROOT / 'benchmarks/alpacaeval/build.py'))['AlpacaEval']
    output = tmp_path / 'structured'
    Builder(str(bench / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
    tables = {p.stem: pd.read_parquet(p) for p in output.glob('*.parquet')}
    assert len(tables['responses']) == 4 and tables['responses'].response.isna().sum() == 1
    assert sorted(tables['responses'].response.dropna()) == [.25, .5, .75]
    assert _alpacaeval(bench, tables, metadata, source_records=(source, bank, 0))['source_graded'] == 3
    class Legacy(Builder):
        _prepare_traces = BenchmarkBuild._prepare_traces
        _write_trace_table = BenchmarkBuild._write_trace_table
    api._tables.reload(); baseline = tmp_path / 'legacy'
    Legacy(str(bench / 'build.py')).main_from_args(['--source', str(raw), '--output', str(baseline)])
    for p in output.glob('*.parquet'):
        if p.name != 'traces.parquet': assert p.read_bytes() == (baseline / p.name).read_bytes(), p.name
    api._tables.reload()
