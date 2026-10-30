"""Named rubric ratings are recoverable without changing historical averages."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
from zipfile import ZipFile

import pandas as pd
import pytest
import yaml

from benchmarks.imgedit.ratings import read_ratings
from scripts.curate_benchmarks.testlet_imgedit_audit import verify


def test_explicit_ratings_not_explanation_numbers():
    names = ['Prompt Compliance', 'Visual Naturalness', 'Physical & Detail Integrity']
    text = ('Brief reasoning: There are 4 objects.\nPrompt Compliance: 4\n'
        '**Visual Naturalness:** 3 — Slight blur.\n### Physical & Detail Integrity:\n**2** - Distortion.\n'
        '- **Prompt Compliance:** This is an explanation, not another score.')
    assert read_ratings(text, names, {}) == dict(zip(names, [4., 3., 2.]))
    assert read_ratings('Unable to evaluate the missing image.', names, {}) == dict.fromkeys(names)
    with pytest.raises(ValueError):
        read_ratings('Prompt Compliance: 4.5', names, {})
    with pytest.raises(ValueError):
        read_ratings('Prompt Compliance: 4\nPrompt Compliance: 5', names, {})


@pytest.fixture
def release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_file = root / 'benchmarks/imgedit/build.py'
    spec = importlib.util.spec_from_file_location('imgedit_fixture', build_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'imgedit'
    raw = directory / 'raw'
    raw.mkdir(parents=True)
    metadata = yaml.safe_load(build_file.with_name('metadata.yaml').read_text())
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    layout = metadata['build']['parameters']['layout']
    names = ['Prompt Compliance', 'Visual Naturalness', 'Physical & Detail Integrity']
    rubric = 'Edit: <edit_prompt>.\n' + ''.join(name + ': A number from 1 to 5.\n' for name in names)
    tasks = {str(index): dict(id='input.jpg', prompt='Edit ' + str(index), edit_type='replace') for index in range(3)}
    with tarfile.open(raw / layout['inputs'], 'w') as archive:
        for name, data in [(layout['tasks'], json.dumps(tasks).encode()),
            (layout['rubrics'], json.dumps(dict(replace=rubric)).encode()),
            (layout['input_prefix'] + 'input.jpg', b'original input image bytes')]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    judgments = {'0': '\n'.join(f'{name}: {score}' for name, score in zip(names, [4, 3, 2])),
        '1': '\n'.join(f'**{name}:** {score} — explanation' for name, score in zip(names, [5, 4, 3])),
        '2': 'Unable to evaluate the missing edited image.'}
    with ZipFile(raw / layout['results'], 'w') as archive:
        prefix = layout['output_prefix']
        for name, value in [('result.json', judgments), ('average_score.json', {'0': 3.}),
            ('typescore.json', dict(replace=3., overall=3.))]:
            archive.writestr(prefix + name, json.dumps(value))
        archive.writestr(prefix + 'result.jsonl', '\n'.join(json.dumps(dict(key=key, result=value)) for key, value in judgments.items()))
        for key in tasks:
            archive.writestr(prefix + key + '.png', ('original output ' + key).encode())
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.ImgEdit(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        tables = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}
        yield directory, tables, metadata
    finally:
        reload()


def test_formatted_ratings_recovered_and_refusal_unknown(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 3
    assert stats['source_component_observations'] == 9
    assert stats['source_saved_grades'] == 1
    assert stats['source_ungraded_attempts'] == 2
    assert stats['source_ungraded_components'] == 3


@pytest.mark.parametrize('damage', ['grade', 'image', 'judge_text', 'aggregate', 'testlet', 'occasion'])
def test_corrupt_input_or_execution_is_rejected(release, damage):
    directory, tables, metadata = release
    if damage == 'grade':
        tables['responses'].loc[0, 'response'] = 1.
    elif damage == 'image':
        tables['assets'].loc[0, 'data'] = b'different image bytes'
    elif damage == 'testlet':
        tables['items'].loc[0, 'testlet_id'] = 'other task'
    elif damage == 'occasion':
        tables['responses'].loc[0, 'trial'] = 2
    else:
        record = json.loads(tables['attempts'].loc[0, 'trace'])
        if damage == 'aggregate':
            record['recorded_task_score'] = 5.
        else:
            trace = json.loads(record['source_trace'])
            trace['judge_text'] = 'different grading evidence'
            record['source_trace'] = json.dumps(trace)
        tables['attempts'].loc[0, 'trace'] = json.dumps(record)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        verify(directory, tables, metadata)
