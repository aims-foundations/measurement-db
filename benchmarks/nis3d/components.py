"""Expose source-backed nucleus checks while retaining every published aggregate."""

import json
from zipfile import ZipFile

import pandas as pd

from measurement_db.build_base import ExactMatcher
from .segmentation_checks import volume_reference, prediction_array, match_objects, agrees_with_paper


def convert(tables, matrix, parameters, grading, raw):
    parents = tables['items'].set_index('item_key').to_dict('index')
    traces = tables['traces'].set_index('response_key').trace.to_dict()
    definitions, observations, attempts = {}, [], []
    layout = parameters['layout']
    with ZipFile(raw/layout['volumes_archive']) as bank, ZipFile(raw/layout['results_archive']) as outputs:
        for folder, records in matrix.groupby('folder', sort=False):
            reference = None
            for row in records.itertuples():
                key = folder + ':' + row.subject_key
                originals = tables['responses'].loc[tables['responses'].subject_key.eq(row.subject_key)
                    & tables['responses'].item_key.str.startswith(folder+':')]
                measurements = [dict(metric=r.item_key.split(':', 1)[1], grade=r.response,
                    source_trace=traces[r.response_key]) for r in originals.itertuples()]
                artifact = row.prediction_artifact
                replay = None
                if isinstance(artifact, dict):
                    if reference is None:
                        reference = volume_reference(bank, parameters, folder)
                    labels = prediction_array(outputs, artifact['member'], reference['gt'].shape)
                    replay = match_objects(labels, reference)
                reconciled = replay is not None and agrees_with_paper(replay['scores'], row.published_record)
                attempts.append(dict(attempt_key=key, raw_attempt_id=key, subject_key=row.subject_key,
                    trace=json.dumps(dict(published_measurements=measurements, prediction_artifact=artifact,
                        replay_metrics=replay['scores'] if replay else None, replay_matches_paper=reconciled), allow_nan=False)))
                retained = originals.loc[originals.item_key.eq(folder+':W-IoU')] if reconciled else originals
                for original in retained.itertuples():
                    definitions[original.item_key] = dict(item_key=original.item_key, **parents[original.item_key])
                    observations.append(dict(response_key=original.response_key, subject_key=row.subject_key,
                        item_key=original.item_key, attempt_key=key, response=original.response,
                        test_condition=parameters['labels']['condition']))
                if not reconciled:
                    continue
                parent = parents[folder+':W-Recall']
                protocol = grading['components']
                native_reference = json.loads(parent['grading_criterion']['reference_answer'])
                for label, target in reference['references'].items():
                    for measure, grade in [('detected', float(replay['recovered'][label])), ('iou', float(replay['iou'][label]))]:
                        component = 'nucleus:'+str(target['original_label'])+':'+measure
                        identity = folder+'::'+component
                        definitions[identity] = dict(parent, item_key=identity, raw_item_id=identity,
                            grading_criterion=dict(reference_answer=json.dumps(dict(target, source=native_reference), sort_keys=True),
                                rule=protocol[measure]['rule'], response_scale=protocol[measure]['response_scale']),
                            verifier=ExactMatcher(spec=json.dumps(dict(kind='native_nucleus_matching', measure=measure,
                                **protocol['implementation']), sort_keys=True)))
                        observations.append(dict(response_key=key+'::'+component, subject_key=row.subject_key,
                            item_key=identity, attempt_key=key, response=grade, test_condition=parameters['labels']['condition']))
                identity = folder+'::false_positives'
                definitions[identity] = dict(parent, item_key=identity, raw_item_id=identity,
                    grading_criterion=dict(reference_answer=None, **protocol['false_positives']),
                    verifier=ExactMatcher(spec=json.dumps(dict(kind='native_nucleus_matching', measure='false_positives',
                        **protocol['implementation']), sort_keys=True)))
                observations.append(dict(response_key=key+'::false_positives', subject_key=row.subject_key,
                    item_key=identity, attempt_key=key, response=float(replay['false_positives']),
                    test_condition=parameters['labels']['condition']))
    return dict(subjects=tables['subjects'], items=pd.DataFrame(definitions.values()),
        responses=pd.DataFrame(observations), attempts=pd.DataFrame(attempts))
