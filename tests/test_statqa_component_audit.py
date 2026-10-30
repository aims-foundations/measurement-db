"""Native column/method scores must remain separate and source-associated."""

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
import yaml

from scripts.curate_benchmarks.testlet_statqa_audit import target_score, verify


def test_native_targets_and_parser_defect():
    record = dict(extracted_answer='{"columns": [" Age "], "methods": ["wrong"]}',
        ground_truth="{'columns': ['age'], 'methods': ['mean']}")
    assert target_score(record, 'columns') == 1.
    assert target_score(record, 'methods') == 0.
    record['ground_truth'] = '''{"columns": ["patient's age"], "methods": ["mean"]}'''
    assert target_score(record, 'columns') == 0.  # Preserve the native quote-replacement defect.
    assert target_score(dict(extracted_answer='{}', ground_truth='{}'), 'columns') == 0.


@pytest.fixture
def release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_file = root / 'benchmarks/statqa/build.py'
    spec = importlib.util.spec_from_file_location('statqa_fixture', build_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'statqa'
    raw = directory / 'raw'
    metadata = yaml.safe_load(build_file.with_name('metadata.yaml').read_text())
    p = metadata['build']['parameters']
    layout = p['layout']
    source = raw / layout['release']
    source.mkdir(parents=True)
    def csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    words = dict(PROMPT_TASK_DESCRIPTION='Choose', PROMPT_INSTRUCTION='Select sets',
        PROMPT_CLASSIFICATION='mean', PROMPT_RESPONSE=p['prompt']['response_suffix'])
    (source / 'prompt_wording.py').write_text('\n'.join(f'{key} = {value!r}' for key, value in words.items()))
    csv(source / layout['column_metadata'] / 'Example_col_meta.csv', [dict(dataset='Example', column='age', type='quant')])
    question = dict(dataset='Example', refined_question='Find the mean age',
        ground_truth="{'columns': ['age'], 'methods': ['mean']}")
    csv(source / layout['bank'], [dict(question, task='descriptive', difficulty='easy')])
    prefix = ('### Task Description: Choose\n### Instruction: Select sets\n### Classification List: \nmean'
        '\n### Column Information: \ncolumn: age; type: quantitative.')
    prompt = prefix + p['prompt']['question_separator'] + question['refined_question'] + p['prompt']['response_separator'] + p['prompt']['response_suffix']
    csv(source / layout['prompts'], [dict(question, prompt=prompt)])
    archive_path = source / layout['answers']
    archive_path.parent.mkdir(parents=True)
    with ZipFile(archive_path, 'w') as archive:
        for scope, name, answers in [
            ('General Analysis', 'gpt-4_zero-shot', [
                '{"columns": ["age"], "methods": ["wrong"]}', 'malformed',
                '{"columns": ["age"], "methods": ["mean"]}']),
            ('Qualitative Analysis', 'gpt-4_zero-shot-Explain', ['No released judgment'])]:
            rows = [dict(question, extracted_answer=answer, model_answer='Complete output: ' + answer) for answer in answers]
            archive.writestr(scope + '/' + name + '.csv', pd.DataFrame(rows).to_csv(index=False))
    (directory / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    from scripts.build_measurement_tables import reload
    reload()
    try:
        output = tmp_path / 'converted'
        module.StatQA(str(directory / 'build.py')).main_from_args(['--source', str(raw), '--output', str(output)])
        tables = {path.stem: pd.read_parquet(path) for path in output.glob('*.parquet')}
        yield directory, tables, metadata
    finally:
        reload()


def test_partial_selection_missing_judgments_and_original_joint_score(release):
    stats = verify(*release)
    assert stats['source_attempts'] == 4
    assert stats['source_component_observations'] == 8
    assert stats['imported_items'] == 2
    assert stats['source_correct'] == 1
    assert stats['source_ungraded'] == 1
    _, tables, _ = release
    assert tables['responses'].response.isna().sum() == 2
    assert tables['responses'].response.sum() == 3


@pytest.mark.parametrize('damage', ['grade', 'answer', 'target', 'occasion', 'reference'])
def test_damaged_import_is_rejected(release, damage):
    directory, tables, metadata = release
    if damage == 'grade':
        tables['responses'].loc[0, 'response'] = 1 - tables['responses'].loc[0, 'response']
    elif damage == 'answer':
        evidence = json.loads(tables['attempts'].loc[0, 'trace'])
        original = json.loads(evidence['source_trace'])
        original['source_record']['model_answer'] = 'Different output'
        evidence['source_trace'] = json.dumps(original)
        tables['attempts'].loc[0, 'trace'] = json.dumps(evidence)
    elif damage == 'occasion':
        tables['responses'].loc[0, 'trial'] = 5
    else:
        criterion = json.loads(tables['items'].loc[0, 'grading_criterion'])
        if damage == 'target':
            rule = json.loads(criterion['rule'])
            rule['target'] = 'invented check'
            criterion['rule'] = json.dumps(rule)
        else:
            criterion['reference_answer'] = '{}'
        tables['items'].loc[0, 'grading_criterion'] = json.dumps(criterion)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        verify(directory, tables, metadata)
