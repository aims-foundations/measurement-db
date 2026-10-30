"""A shared plan produces condition-specific block checks, not extra executions."""

import base64
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml


def test_sequential_dependencies_and_shared_grading_conditions(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'benchmarks/phyblock/build.py'
    spec = importlib.util.spec_from_file_location('phyblock_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = yaml.safe_load(path.with_name('metadata.yaml').read_text())
    directory = tmp_path / 'phyblock'
    raw = directory / 'raw'
    archive = raw / metadata['build']['parameters']['layout']['archive']
    archive.parent.mkdir(parents=True)
    blocks = [dict(order=1, type='cube', color='red', euler=[0, 0, 0], depend=[0]),
        dict(order=2, type='cube', color='blue', euler=[0, 0, 0], depend=[1]),
        dict(order=3, type='cube', color='green', euler=[0, 0, 0], depend=[2])]
    candidates = [{k: b[k] for k in ('type', 'color', 'euler')} for b in blocks]
    candidates.append(dict(type='cube', color='red', euler=[0, 0, 90]))
    # Blue arrives before its dependency and red has the wrong orientation.
    # Pose matching places no blocks; topology matching places red and blue.
    text = 'Move block with index 2\nMove block with index 4\nMove block with index 2'
    records = [dict(model='gpt-4o', choices=[dict(message=dict(content=text))]),
        dict(model='claude-3-7-sonnet', content=[dict(thinking='reasoning'), dict(text=text)]),
        dict(model_version='gemini-2.5-pro', candidates=[dict(content=dict(parts=[dict(text=text)]))]),
        dict(model='fixture-open-model', predict=text, instruction='Build this scene.',
            image=['SCENEs/001_1.png', 'SCENEs_cand/001_16_cand.png'])]
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=')
    with ZipFile(archive, 'w') as output:
        output.writestr('release/data/SCENEs_400_Goal_Jsons/001.json', json.dumps(dict(level=2, blocks=blocks)))
        output.writestr('release/data/SCENEs_400_Cand_Jsons/001_16_cand_blocks.json', json.dumps(candidates))
        output.writestr('release/data/SCENEs_400_Goal_Imgs/001_1.png', png)
        output.writestr('release/data/SCENEs_400_Cand_Imgs/001_16_cand.png', png)
        for index, record in enumerate(records):
            output.writestr(f'release/outputs/model-{index}/001.json', json.dumps(record))
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        target = tmp_path / 'converted'
        module.PhyBlock(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(target)])
        tables = {p.stem: pd.read_parquet(p) for p in target.glob('*.parquet')}
    finally:
        reload()
    assert len(tables['attempts']) == 4
    assert len(tables['responses']) == 32
    assert set(tables['items'].testlet_id) == {'001'}
    assert tables['responses'].groupby('attempt_id').test_condition.nunique().eq(2).all()
    joined = tables['responses'].merge(tables['items'][['item_id', 'raw_item_id']], on='item_id')
    pose = joined.loc[joined.test_condition.eq('pose_constrained')]
    topology = joined.loc[joined.test_condition.eq('topology_only')]
    assert pose.loc[pose.raw_item_id.str.contains(':block:'), 'response'].eq(0).all()
    assert pose.loc[pose.raw_item_id.str.endswith('extra_placement_count'), 'response'].eq(3).all()
    assert topology.loc[topology.raw_item_id.str.endswith('block:1'), 'response'].eq(1).all()
    assert topology.loc[topology.raw_item_id.str.endswith('block:2'), 'response'].eq(1).all()
    assert topology.loc[topology.raw_item_id.str.endswith('block:3'), 'response'].eq(0).all()
    assert topology.loc[topology.raw_item_id.str.endswith('extra_placement_count'), 'response'].eq(1).all()
    for encoded in tables['attempts'].trace:
        trace = json.loads(encoded)
        evidence = json.loads(trace['source_trace'])
        assert trace['recorded_task_score'] == 0
        assert evidence['derived_grades']['topology_only']['F1'] == 0.667
        assert json.loads(evidence['native_record']) in records
