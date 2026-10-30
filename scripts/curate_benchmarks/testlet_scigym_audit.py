"""Verify individual reaction comparisons and all original SciGym metrics."""

import hashlib
import json
from xml.dom import minidom

import pandas as pd
import pyarrow.parquet as pq


def _signatures(xml, modifiers):
    document = minidom.parseString(xml)
    result = set()
    for reaction in document.getElementsByTagNameNS('*', 'reaction'):
        roles = []
        for name, tag in [('listOfReactants', 'speciesReference'), ('listOfProducts', 'speciesReference'), ('listOfModifiers', 'modifierSpeciesReference')]:
            groups = [child for child in reaction.childNodes if child.localName == name]
            roles.append(tuple(sorted({child.getAttribute('species') for group in groups for child in group.childNodes if child.localName == tag})))
        if not modifiers:
            roles[-1] = ()
        result.add(tuple(roles))
    document.unlink()
    return result


def verify(directory, tables, metadata):
    from .native_result_audits import _check, _scigym
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='SciGym complete original execution and reaction evidence')
    bank = {r['folder_name']: r for r in pq.read_table(directory / 'raw/tasks.parquet').to_pylist()}
    items = tables['items'].set_index('item_id').to_dict('index')
    assets = tables['assets'].set_index('asset_id').data.to_dict()
    groups = tables['responses'].groupby('attempt_id', sort=False)
    _check(set(groups.groups), set(tables['attempts'].attempt_id), 'SciGym every execution remains observed')
    parents, responses, traces, used, used_assets = {}, [], [], set(), set()
    protocol = metadata['grading']['verifiers']['reaction_recovery']
    for attempt in tables['attempts'].itertuples():
        children = groups.get_group(attempt.attempt_id)
        first = children.iloc[0]
        prototype = items[first.item_id]
        task = prototype['testlet_id']
        original = bank[task]
        evidence = json.loads(attempt.trace)
        trace = json.loads(evidence['source_trace'])
        reference = json.dumps({name: original[name] for name in ['truth_xml', 'partial', 'truth_sedml']})
        digest = hashlib.sha256(reference.encode()).hexdigest()
        _check(assets[digest], reference.encode(), 'SciGym complete source reference and simulation specification')
        used_assets.add(digest)
        expected = {}
        for condition in ['with_modifiers', 'without_modifiers']:
            true, partial, predicted = [_signatures(xml, condition == 'with_modifiers')
                for xml in [original['truth_xml'], original['partial'], trace['final_model']]]
            missing, added = true - partial, predicted - partial
            for signature in missing:
                target = json.dumps(dict(zip(('reactants', 'products', 'modifiers'), signature)), sort_keys=True)
                name = condition + ':reaction:' + hashlib.sha256(target.encode()).hexdigest()
                criterion = dict(reference_answer=target, rule=json.dumps(dict(condition=condition,
                    rule='The final model recovers this missing reference reaction under the native species-set comparison.', reference_asset='reaction-reference.json'), sort_keys=True),
                    response_scale=dict(kind='discrete', values=[0, 1], direction='higher_is_better'))
                expected[name] = float(signature in added), criterion, dict(kind='native_reaction_set_comparison', condition=condition, original_protocol=protocol)
            criterion = dict(reference_answer=None, rule=json.dumps(dict(condition=condition,
                rule='Number of added reaction signatures absent from the missing reference set.', reference_asset='reaction-reference.json'), sort_keys=True),
                response_scale=dict(kind='interval', min=0, max=None, direction='lower_is_better'))
            expected[condition + ':extra'] = float(len(added - missing)), criterion, dict(kind='native_reaction_set_comparison', condition=condition, original_protocol=protocol)
        ste = metadata['grading']['verifiers']['trajectory_error']
        expected['ste'] = None, dict(reference_answer=reference, rule=metadata['grading']['rule'], response_scale=ste['response_scale']), dict(ste, native_metric='ste')
        actual = {}
        for row in children.itertuples():
            item = items[row.item_id]
            parent, _, component = item['raw_item_id'].partition('::')
            _check((parent, item['testlet_id']), (task, task), 'SciGym source biomodel testlet')
            _check((row.subject_id, row.trial), (attempt.subject_id, 1), 'SciGym original execution and trial')
            _check(pd.isna(row.test_condition) and pd.isna(row.interactors), True, 'SciGym original observation context')
            for field in ['content', 'item_features']:
                _check(item[field], prototype[field], 'SciGym complete common input ' + field)
            _check(json.loads(item['asset_manifest']), [dict(asset_id=digest, path='reaction-reference.json',
                role='grading_reference', media_type='application/json', ordinal=1)], 'SciGym complete grading reference association')
            grade, criterion, spec = expected[component]
            _check(json.loads(item['grading_criterion']), criterion, 'SciGym original reaction comparison and native scale')
            _check(json.loads(item['verifier']), dict(**{'class': 'exact_matcher'}, spec=json.dumps(spec, sort_keys=True)), 'SciGym original structural comparator')
            _check(component not in actual, True, 'SciGym no repeated component')
            actual[component] = None if pd.isna(row.response) else row.response
            used.add(row.item_id)
        _check(actual, {key: values[0] for key, values in expected.items()}, 'SciGym exact native reaction outcomes and extra counts')
        metrics = {value['native_metric']: value for value in trace['measurements']}
        _check(len(metrics), len(trace['measurements']), 'SciGym unique original metrics')
        _check(set(metrics), set(metadata['build']['parameters']['metrics']), 'SciGym all original measurements retained')
        _check(evidence['recorded_task_score'], metrics['rpm_f1']['score'], 'SciGym retained primary task score')
        for metric, value in metrics.items():
            spec = ste if metric == 'ste' else protocol
            identity = task + ':' + metric
            parent = dict(prototype, item_id=identity, raw_item_id=task, asset_manifest=None,
                grading_criterion=json.dumps(dict(reference_answer=reference, rule=metadata['grading']['rule'], response_scale=spec['response_scale'])),
                verifier=json.dumps(dict(**{'class': 'exact_matcher'}, spec=json.dumps(dict(spec, native_metric=metric), sort_keys=True))))
            if identity in parents:
                _check(parent, parents[identity], 'SciGym full original input and metric identity')
            parents[identity] = parent
            response = first.to_dict()
            response.update(response_id=attempt.attempt_id + ':' + metric, item_id=identity, attempt_id=None, response=value['score'])
            responses.append(response)
            traces.append(dict(response_id=response['response_id'], trace=json.dumps(dict(
                source_rows=trace['source_rows'], chat_history=trace['chat_history'], final_model=trace['final_model'],
                native_metric=metric, grade_origin=value['grade_origin']))))
    _check((used, used_assets), (set(items), set(assets)), 'SciGym full definition and reference coverage')
    projected = {key: frame for key, frame in tables.items() if key not in {'assets', 'attempts'}}
    projected.update(items=pd.DataFrame(parents.values()), responses=pd.DataFrame(responses), traces=pd.DataFrame(traces))
    details = _scigym(directory, projected, metadata)
    return dict(details, source_attempts=len(tables['attempts']), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=len({item['testlet_id'] for item in items.values()}))
