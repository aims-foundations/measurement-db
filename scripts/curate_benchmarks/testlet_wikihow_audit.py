"""Check new rubric items while retaining the complete conversation/source audit."""

import copy
import json

import pandas as pd


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _wikihow_agent
    from .testlet_audits import verify_component_results
    from ..build_measurement_tables.hash_measurement_ids import canonical_grading_criterion
    from ..build_measurement_tables.response_scales import canonical_response_scale
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='WikiHowAgent component audit')
    counts = verify_component_results(directory, tables=tables)
    rubrics = json.loads((directory / 'raw' / metadata['build']['parameters']['layout']['rubrics']).read_text())
    items = tables['items'].set_index('item_id').to_dict('index')
    completion_items = {key: row for key, row in items.items() if row['raw_item_id'].endswith('::Completion Achieved')}
    completion = tables['responses'].loc[tables['responses'].item_id.isin(completion_items)].copy()
    parents = {row['testlet_id']: row for row in completion_items.values()}
    _check(len(parents), len(completion_items), 'WikiHowAgent one completion item per tutorial')
    for row in items.values():
        parent = parents[row['testlet_id']]
        for field in ['content', 'item_features', 'asset_manifest']:
            _check(None if pd.isna(row[field]) else row[field], None if pd.isna(parent[field]) else parent[field], 'WikiHowAgent consistent shared tutorial ' + field)
        name = row['raw_item_id'].removeprefix(row['testlet_id'] + '::')
        if name == 'Completion Achieved':
            criterion = dict(rule=metadata['grading']['verifiers']['completion']['rule'],
                response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
        else:
            criterion = dict(rule=json.dumps(dict(criterion=name, **rubrics[name]), ensure_ascii=False),
                response_scale=dict(kind='interval', min=1, max=5, direction='higher_is_better'))
            verifier = json.loads(row['verifier'])
            _check(verifier['class'], 'judge', 'WikiHowAgent native rubric judge')
            _check(json.loads(verifier['spec']), metadata['grading']['verifiers']['rubric'], 'WikiHowAgent rubric provenance')
        _check(json.loads(row['grading_criterion']), json.loads(canonical_grading_criterion(criterion)), 'WikiHowAgent full rubric and scale')
    occasions = tables['responses'].merge(completion[['attempt_id', 'trial', 'test_condition', 'interactors']],
        on='attempt_id', validate='many_to_one', suffixes=('', '_completion'))
    for column in ['trial', 'test_condition', 'interactors']:
        _check(occasions[column].fillna('').tolist(), occasions[column + '_completion'].fillna('').tolist(), 'WikiHowAgent shared source occasion')

    # The completion observation is unchanged. Reuse its existing full native
    # audit after projecting away the additional rubric observations.
    legacy = {name: frame.copy(deep=True) for name, frame in tables.items() if name not in {'attempts', 'items', 'responses'}}
    legacy['responses'] = completion.assign(attempt_id=None)
    legacy['items'] = tables['items'].loc[tables['items'].item_id.isin(completion_items)].copy()
    legacy['items']['raw_item_id'] = legacy['items'].raw_item_id.str.removesuffix('::Completion Achieved')
    legacy['items']['grading_criterion'] = canonical_grading_criterion(dict(rule=metadata['grading']['verifiers']['completion']['rule']))
    traces = completion.drop(columns='response').merge(tables['attempts'][['attempt_id', 'trace']], on='attempt_id', validate='one_to_one')
    traces['trace'] = traces.trace.map(lambda text: json.loads(text)['source_trace'])
    legacy['traces'] = traces[['response_id', 'subject_id', 'item_id', 'benchmark_id', 'trial', 'test_condition', 'interactors', 'trace']]
    profile = copy.deepcopy(metadata)
    profile['benchmark'].update(response_type='binary', categorical=True,
        response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
    legacy['benchmarks']['response_scale'] = canonical_response_scale(profile['benchmark']['response_scale'])
    legacy['benchmarks']['response_type'] = 'binary'
    legacy['benchmarks']['categorical'] = True
    # Derived counts belong to the projected completion view.
    legacy['benchmarks']['n_items'] = len(legacy['items'])
    legacy['benchmarks']['n_responses'] = len(completion)
    legacy['benchmarks']['n_response_values'] = completion.response.nunique()
    observed_pairs = completion[['subject_id', 'item_id']].drop_duplicates().shape[0]
    legacy['benchmarks']['coverage'] = observed_pairs / (len(legacy['items']) * len(legacy['subjects']))
    original = _wikihow_agent(directory, legacy, profile)
    return {**original, **counts}
