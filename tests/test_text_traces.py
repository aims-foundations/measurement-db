"""Text pilot preserves literal answers, sparse trace links, grades and media."""
import base64
import json
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest

from trace_builder_fixtures import legacy_traces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from scripts.build_measurement_tables.trace_conversion import convert_trace
from scripts.build_measurement_tables.trace_format import validate
from scripts.curate_benchmarks.trace_source_audit import recover_text


@pytest.mark.parametrize('source', ['42', '[1, 2]', '{"answer": true}', '  complete proof Ω\n'*5000, ''])
def test_literal_text_is_not_parsed_as_json_or_split_into_invented_turns(source):
    value, counts = convert(source)
    validate(value)
    assert counts == {'llm': 1, 'text_records': 1}
    assert value['steps'][0]['content'] == source
    assert recover_text(value) == source
    assert 'evaluation' not in value and 'run' not in value
    value['steps'][0]['content'] = 'Changed output'
    with pytest.raises(ValueError): recover_text(value)


def convert(source, **options):
    result = convert_trace(source, family="text", **options)
    return result.value, result.counts


def test_full_matharena_builder_retains_sparse_trace_links_grading_and_image_bytes(tmp_path):
    from measurement_db.build_base import BenchmarkBuild
    import measurement_db.build_base as api
    api._tables.reload()
    bench = tmp_path / 'matharena'; bench.mkdir()
    (bench / 'metadata.yaml').write_bytes((ROOT / 'benchmarks/matharena/metadata.yaml').read_bytes())
    raw = bench / 'raw'
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
    prompt = json.dumps([{'type': 'text', 'text': 'Read this mathematical figure.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}])
    final = dict(problem_idx='1', problem='Original question', user_message=prompt, image=png,
                 model_name='Test Model', model_config='test/model', idx_answer=0, correct=True,
                 gold_answer='[1,2]', answer='[1, 2]', parsed_answer=None)
    fallback = {**final, 'idx_answer': 1, 'correct': False, 'answer': None, 'parsed_answer': 'Recorded fallback'}
    proof = {**final, 'problem_idx': '2', 'user_message': 'Prove the claim.', 'image': None,
             'correct': None, 'gold_answer': None, 'answer': 'Complete proof\n' + '測定' * 10000,
             'grading_details_judge_1': json.dumps([{'title': 'First criterion', 'points': 1, 'max_points': 3},
                                                   {'title': 'Second criterion', 'points': 2, 'max_points': 4}])}
    for competition, records in [('aime_2025', [final, fallback]), ('putnam_2025', [proof])]:
        p = raw / f'sources/{competition}/data/fixture.parquet'; p.parent.mkdir(parents=True)
        pd.DataFrame(records).to_parquet(p, index=False)
    Builder = runpy.run_path(str(ROOT / 'benchmarks/matharena/build.py'))['MathArenaBuild']
    output = tmp_path / 'structured'
    Builder(str(bench / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
    responses = pd.read_parquet(output / 'responses.parquet')
    traces = pd.read_parquet(output / 'traces.parquet')
    assert len(responses) == 4 and len(traces) == 3
    assert sorted(responses.response) == [0., 1/3, .5, 1.]
    assert set(traces.response_id) < set(responses.response_id)
    assert {recover_text(value) for value in traces.trace} == {final['answer'], fallback['parsed_answer'], proof['answer']}
    class Legacy(Builder):
        _prepare_traces = legacy_traces
        _write_trace_table = BenchmarkBuild._write_trace_table
    api._tables.reload()
    baseline = tmp_path / 'legacy'
    Legacy(str(bench / 'build.py')).main_from_args(['--source', str(raw), '--output', str(baseline)])
    for p in output.glob('*.parquet'):
        if p.name != 'traces.parquet':
            assert p.read_bytes() == (baseline / p.name).read_bytes(), p.name
    api._tables.reload()
