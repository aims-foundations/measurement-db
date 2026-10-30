"""Audit named image-edit ratings and every original input, output and mean."""

from collections import Counter
from copy import deepcopy
import json
import re
import tarfile
from zipfile import ZipFile

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _imgedit
    from .testlet_audit_views import attempt_view
    from ..build_measurement_tables.response_scales import canonical_response_scale
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='ImgEdit rubric component audit')
    parameters = metadata['build']['parameters']
    layout, labels = parameters['layout'], parameters['labels']
    protocol = metadata['grading']['verifiers']['native_judge']
    with tarfile.open(directory / 'raw' / layout['inputs']) as archive:
        tasks = json.load(archive.extractfile(layout['tasks']))
        rubrics = json.load(archive.extractfile(layout['rubrics']))
    dimensions = {category: [line.split(':', 1)[0] for line in prompt.splitlines()
        if line.endswith(': A number from 1 to 5.')] for category, prompt in rubrics.items()}
    _check(all(len(names) == 3 for names in dimensions.values()), True, 'ImgEdit three declared rubric dimensions')
    with ZipFile(directory / 'raw' / layout['results']) as archive:
        judgments = json.loads(archive.read(layout['output_prefix'] + parameters['files']['judgments']))
    items = tables['items'].set_index('item_id').to_dict('index')
    definitions = Counter()
    for item in items.values():
        parent, dimension = item['raw_item_id'].rsplit('::', 1)
        task = tasks[parent]
        _check(item['testlet_id'], parent, 'ImgEdit original editing task')
        _check(dimension in dimensions[task['edit_type']], True, 'ImgEdit named rubric dimension')
        _check(json.loads(item['grading_criterion']), dict(reference_answer=None, rule=json.dumps(dict(
            instruction=metadata['grading']['rule'], dimension=dimension,
            judge_prompt=rubrics[task['edit_type']].replace('<edit_prompt>', task['prompt'])))),
            'ImgEdit complete rubric and selected dimension')
        _check(json.loads(item['verifier']), {'class': 'judge', 'judge': labels['judge'], 'judged_by': 'llm',
            'spec': json.dumps(dict(protocol, selected_dimension=dimension), sort_keys=True)},
            'ImgEdit original judge and explicit parsing provenance')
        definitions[parent, dimension] += 1
    _check(definitions, Counter({(key, name): 1 for key, task in tasks.items() for name in dimensions[task['edit_type']]}),
        'ImgEdit every task and rubric dimension exactly once')

    observations, traces = attempt_view(tables)
    attempts = tables['attempts'].set_index('attempt_id')
    expected = Counter()
    for row in observations.itertuples():
        key = json.loads(traces[row.response_id])['source_key']
        _check(attempts.loc[row.attempt_id, 'raw_attempt_id'], key, 'ImgEdit source execution identity')
        names = dimensions[tasks[key]['edit_type']]
        aliases = {name: name for name in names}
        aliases.update({alias: name for alias, name in protocol['dimension_aliases'].items() if name in names})
        # Independent, anchored extraction of literal named scores. A rubric
        # heading may precede its numeric score on the immediately following line.
        text = judgments[key].replace('**', '')
        pattern = (r'(?m)^[ \t]*(?:#{1,6}[ \t]+|[-*][ \t]+|[1-3]\.[ \t]+)?('
            + '|'.join(re.escape(name) for name in aliases)
            + r'):[ \t]*(?:\n[ \t]*)?([1-5])(?=$|[ \t]|\.(?:$|[ \t]))')
        grades = {}
        for match in re.finditer(pattern, text):
            name = aliases[match[1]]
            _check(name not in grades, True, 'ImgEdit one literal rating per dimension')
            grades[name] = float(match[2])
        for name in names:
            expected[row.attempt_id, key + '::' + name, grades.get(name)] += 1
    actual = Counter((row.attempt_id, items[row.item_id]['raw_item_id'],
        None if pd.isna(row.response) else row.response) for row in tables['responses'].itertuples())
    _check(actual, expected, 'ImgEdit exact named scores, Markdown recovery and missing judgments')

    parents = tables['items'].drop_duplicates('testlet_id').copy()
    parent_rows = parents.set_index('testlet_id').to_dict('index')
    for item in items.values():
        for field in ['content', 'asset_manifest', 'item_features']:
            _check(item[field], parent_rows[item['testlet_id']][field], 'ImgEdit unchanged shared input: ' + field)
    parents['raw_item_id'] = parents.testlet_id
    parent_ids = dict(zip(parents.testlet_id, parents.item_id))
    observations['item_id'] = observations.item_id.map(lambda key: parent_ids[items[key]['testlet_id']])
    observations['attempt_id'] = None
    parents['grading_criterion'] = [json.dumps(dict(reference_answer=None, rule=json.dumps(dict(
        instruction=protocol['aggregate_rule'], judge_prompt=rubrics[tasks[key]['edit_type']].replace('<edit_prompt>', tasks[key]['prompt'])))))
        for key in parents.raw_item_id]
    parents['verifier'] = json.dumps({'class': 'judge', 'judge': labels['judge'], 'judged_by': 'llm',
        'spec': json.dumps(protocol, sort_keys=True)})
    old_metadata = deepcopy(metadata)
    old_metadata['grading']['rule'] = protocol['aggregate_rule']
    scale = dict(kind='interval', min=1, max=5, direction='higher_is_better')
    old_metadata['benchmark'].update(response_type='continuous', response_scale=scale, categorical=False)
    projected = {name: table for name, table in tables.items() if name != 'attempts'}
    projected.update(items=parents, responses=observations,
        traces=observations[['response_id', 'subject_id', 'item_id', 'benchmark_id', 'trial', 'test_condition', 'interactors']].assign(
            trace=observations.response_id.map(traces)),
        benchmarks=tables['benchmarks'].assign(response_type='continuous', categorical=False,
            response_scale=canonical_response_scale(scale), n_items=len(parents),
            n_responses=len(observations), n_response_values=observations.response.nunique(),
            coverage=observations.response.notna().sum() / len(parents)))
    stats = _imgedit(directory, projected, old_metadata)
    return dict(stats, source_attempts=len(attempts), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len(parents),
        source_ungraded_components=int(tables['responses'].response.isna().sum()))
