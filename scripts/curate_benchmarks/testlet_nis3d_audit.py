"""Independent dense replay of NIS3D's released MATLAB object matcher."""

import io
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy.io
from scipy.ndimage import find_objects
import tifffile

def relabel(array):
    array = array.astype(np.int32, copy=True)
    counts = np.bincount(array.ravel())
    valid = list(np.flatnonzero(counts[1:]) + 1)
    missing = list(np.flatnonzero(counts[1:] == 0) + 1)[::-1]
    mapping = np.arange(len(counts), dtype=np.int32)
    while valid and missing and missing[-1] < valid[-1]:
        mapping[valid.pop()] = missing.pop()
    return mapping[array]


def reference_context(original, confidence):
    low = (confidence < 2) & (confidence > 0)
    gt = original.copy()
    gt[low] = 0
    gt = relabel(gt)
    counts = np.bincount(gt.ravel())
    weights = np.zeros(len(counts))
    references = {}
    for label, region in enumerate(find_objects(gt), 1):
        if region is None:
            continue
        section = gt[region]
        first = np.flatnonzero(section.ravel(order='F') == label)[0]
        local = np.unravel_index(first, section.shape, order='F')
        point = tuple(axis.start + coordinate for axis, coordinate in zip(region, local))
        weights[label] = (float(confidence[point]) - 1) / 3
        references[label] = dict(original_label=int(original[point]), compact_label=label,
            voxel_count=int(counts[label]), confidence_weight=float(weights[label]),
            bounding_box_yxz=[[axis.start, axis.stop] for axis in region])
    return gt, low, counts, weights, references


def evaluate(prediction, original, confidence, context=None):
    """Use a dense overlap matrix, independently of the builder's sparse matcher."""
    gt, low, gold_sizes, weights, references = context or reference_context(original, confidence)
    labels = relabel(prediction)
    predicted_sizes = np.bincount(labels.ravel())
    ng, nl = len(gold_sizes)-1, len(predicted_sizes)-1
    ignored = np.bincount(labels[low], minlength=nl+1) > .5*predicted_sizes
    ignored[0] = True
    both = (labels > 0) & (gt > 0)
    pairs, intersections = np.unique(labels[both].astype(np.int64)*(ng+1)+gt[both], return_counts=True)
    pids, gids = pairs//(ng+1), pairs%(ng+1)
    matrix = np.zeros((nl+1, ng+1))
    matrix[pids, gids] = intersections/(predicted_sizes[pids]+gold_sizes[gids]-intersections)
    pg, gp = matrix.argmax(axis=1), matrix.argmax(axis=0)
    matched = [i for i in range(1, nl+1) if not ignored[i] and pg[i] > 0 and gp[pg[i]] == i]
    recovered = np.zeros(ng+1, dtype=bool)
    recovered[pg[matched]] = True
    ious = np.zeros(ng+1)
    ious[pg[matched]] = matrix[matched, pg[matched]]
    fp = int(sum(not ignored[i] for i in range(1, nl+1))-len(matched))
    tp, fn = float(weights[recovered].sum()), float(weights[~recovered].sum())
    valid = (labels > 0) & ~ignored[labels]
    votes = confidence.copy()
    votes[low] = 0
    scores = dict(W_F1=2*tp/(2*tp+fp+fn), W_Precision=tp/(tp+fp), W_Recall=tp/(tp+fn),
        W_SEG=float((ious*weights).sum())/(tp+fp+fn),
        W_IoU=float(votes[valid].sum())/(float(votes.sum())+int((valid & (confidence == 0)).sum())))
    targets = {ref['original_label']:dict(reference=ref, detected=float(recovered[label]), iou=float(ious[label]))
        for label, ref in references.items()}
    return dict(scores=scores, targets=targets, false_positives=fp)


def source_comparisons(directory, metadata, source):
    """Resolve every saved output and compare it to the published paper precision."""
    parameters = metadata['build']['parameters']
    layout = parameters['layout']
    results = {}
    with ZipFile(directory/'raw'/layout['volumes_archive']) as bank, ZipFile(directory/'raw'/layout['results_archive']) as archive:
        for folder in source['volumes']:
            available = [(method, artifact) for (volume, method), artifact in source['artifacts'].items() if volume == folder]
            if not available:
                continue
            prefix = layout['volumes_prefix']+folder+'/'
            gt = tifffile.imread(io.BytesIO(bank.read(prefix+parameters['ground_truth_files'][folder]))).transpose(1, 2, 0)
            vot = tifffile.imread(io.BytesIO(bank.read(prefix+parameters['confidence_files'][folder]))).transpose(1, 2, 0)
            context = reference_context(gt, vot)
            for method, artifact in available:
                blob = archive.read(artifact['member'])
                try:
                    loaded = scipy.io.loadmat(io.BytesIO(blob))
                    values = [value for key, value in loaded.items() if not key.startswith('__')]
                    if len(values) != 1:raise ValueError('Expected one native prediction array')
                    prediction = values[0]
                except NotImplementedError:
                    import h5py
                    with h5py.File(io.BytesIO(blob), 'r') as data:
                        names = [key for key in data if not key.startswith('#')]
                        if len(names) != 1:raise ValueError('Expected one native prediction array')
                        prediction = np.asarray(data[names[0]]).transpose(2, 1, 0)
                if prediction.shape[:2] != gt.shape[:2] or prediction.shape[2] < gt.shape[2]:
                    raise ValueError('Original prediction and reference dimensions differ')
                result = evaluate(prediction[:, :, -gt.shape[2]:], gt, vot, context)
                model = next(label for label, key in source['methods'].items() if key == method)
                paper = source['observations'][folder, model]
                result['reconciled'] = all(abs(value-float(paper[metric])) <= .5*10**(-len(paper[metric].partition('.')[2]))+1e-10
                    for metric, value in result['scores'].items())
                results[folder, method] = result
                print('NIS3D native object audit', folder, method, result['reconciled'], flush=True)
    return results


def verify(directory, tables, metadata, source=None, comparisons=None):
    from .native_result_audits import _check, _nis3d, _nis3d_sources
    from ..build_measurement_tables.validate_measurement_tables import validate_dataset

    validate_dataset(tables, context='NIS3D native object outcomes and complete paper records')
    source = source or _nis3d_sources(directory, metadata)
    comparisons = comparisons if comparisons is not None else source_comparisons(directory, metadata, source)
    items = tables['items'].astype(object).where(tables['items'].notna(), None).set_index('item_id').to_dict('index')
    subjects = tables['subjects'].set_index('subject_id').display_name.to_dict()
    groups = tables['responses'].groupby('attempt_id', sort=False)
    _check(set(groups.groups), set(tables['attempts'].attempt_id), 'NIS3D every segmentation result remains observed')
    protocol = metadata['grading']['verifiers']['components']
    metrics = {value['name']: key for key, value in metadata['grading']['verifiers'].items() if key.startswith('W_')}
    used, seen, parents, originals, traces = set(), set(), {}, [], []
    for attempt in tables['attempts'].itertuples():
        folder, method = attempt.raw_attempt_id.split(':')
        label = subjects[attempt.subject_id]
        _check(method, source['methods'][label], 'NIS3D unchanged source method association')
        key = folder, method
        _check(key not in seen, True, 'NIS3D one source result per method and volume')
        seen.add(key)
        paper = source['observations'][folder, label]
        evidence = json.loads(attempt.trace)
        replay = comparisons.get(key)
        reconciled = bool(replay and replay['reconciled'])
        _check(evidence['replay_matches_paper'], reconciled, 'NIS3D component extraction requires all paper metrics to agree')
        if replay:
            for metric, value in replay['scores'].items():
                _check(np.isclose(evidence['replay_metrics'][metric], value, rtol=0, atol=1e-12), True,
                    'NIS3D independently replayed native metric '+metric)
        else:
            _check(evidence['replay_metrics'], None, 'NIS3D unavailable output is not regenerated')
        _check(evidence['prediction_artifact'], source['artifacts'].get(key), 'NIS3D unchanged complete prediction archive association')
        children = groups.get_group(attempt.attempt_id)
        prototype = items[children.iloc[0].item_id]
        expected = {}
        for metric, source_metric in metrics.items():
            if reconciled and metric != 'W-IoU':continue
            p = metadata['grading']['verifiers'][source_metric]
            criterion = dict(reference_answer=json.dumps(source['volumes'][folder]['reference'], sort_keys=True),
                rule=metadata['grading']['rule']+' '+p['rule'], response_scale=dict(kind='interval', min=0, max=1, direction='higher_is_better'))
            expected[folder+':'+metric] = float(paper[source_metric]), criterion, dict(p, implementation=source['implementation'])
        if reconciled:
            for original_label, target in replay['targets'].items():
                for measure in ['detected', 'iou']:
                    raw_id = folder+'::nucleus:'+str(original_label)+':'+measure
                    criterion = dict(reference_answer=json.dumps(dict(target['reference'], source=source['volumes'][folder]['reference']), sort_keys=True),
                        rule=protocol[measure]['rule'], response_scale=protocol[measure]['response_scale'])
                    expected[raw_id] = target[measure], criterion, dict(kind='native_nucleus_matching', measure=measure, **protocol['implementation'])
            expected[folder+'::false_positives'] = float(replay['false_positives']), dict(reference_answer=None, **protocol['false_positives']), dict(
                kind='native_nucleus_matching', measure='false_positives', **protocol['implementation'])
        actual = set()
        for child in children.itertuples():
            item = items[child.item_id]
            raw_id = item['raw_item_id']
            _check(raw_id not in actual and raw_id in expected, True, 'NIS3D exact native component roster')
            actual.add(raw_id);used.add(child.item_id)
            grade, criterion, spec = expected[raw_id]
            _check(np.isclose(child.response, grade, atol=1e-12, rtol=0), True, 'NIS3D exact object outcome')
            _check(json.loads(item['grading_criterion']), criterion, 'NIS3D original mask label, position, weight and scale')
            _check(json.loads(item['verifier']), dict(**{'class':'exact_matcher'}, spec=json.dumps(spec, sort_keys=True)), 'NIS3D original deterministic grading context')
            _check(item['testlet_id'], folder, 'NIS3D original volume testlet')
            for field in ['content', 'item_features', 'asset_manifest']:
                _check(item[field], prototype[field], 'NIS3D complete unchanged shared image context '+field)
            _check((child.subject_id, child.trial, child.test_condition, pd.isna(child.interactors)),
                (attempt.subject_id, 1, 'published_baseline', True), 'NIS3D source model and observation conditions')
        _check(actual, set(expected), 'NIS3D no omitted native objects or retained metrics')
        measurements = {r['metric']:r for r in evidence['published_measurements']}
        _check(len(measurements), len(evidence['published_measurements']), 'NIS3D original metrics retained once')
        _check(set(measurements), set(metrics), 'NIS3D every original metric survives conversion')
        for metric, record in measurements.items():
            _check(record['grade'], float(paper[metrics[metric]]), 'NIS3D unchanged rounded paper value')
            p = metadata['grading']['verifiers'][metrics[metric]]
            identity = folder+':'+metric
            parent = dict(prototype, item_id=identity, raw_item_id=identity,
                grading_criterion=json.dumps(dict(reference_answer=json.dumps(source['volumes'][folder]['reference'], sort_keys=True),
                    rule=metadata['grading']['rule']+' '+p['rule'])),
                verifier=json.dumps(dict(**{'class':'exact_matcher'}, spec=json.dumps(dict(p, implementation=source['implementation']), sort_keys=True))))
            if identity in parents:_check(parent, parents[identity], 'NIS3D shared original grading identity')
            parents[identity] = parent
            response = children.iloc[0].to_dict();response.update(response_id=attempt.attempt_id+':'+metric,
                item_id=identity, attempt_id=None, response=record['grade'])
            originals.append(response)
            traces.append({**{name:response[name] for name in ['response_id','subject_id','item_id','benchmark_id','trial','test_condition','interactors']}, 'trace':record['source_trace']})
    _check(used, set(items), 'NIS3D no unused component definitions')
    _check(seen, {(folder, source['methods'][label]) for folder,label in source['observations']}, 'NIS3D all 36 original results')
    projected = {name:frame for name,frame in tables.items() if name != 'attempts'}
    projected.update(items=pd.DataFrame(parents.values()), responses=pd.DataFrame(originals), traces=pd.DataFrame(traces))
    stats = _nis3d(directory, projected, metadata, source=source)
    return dict(stats, source_attempts=len(seen), source_component_observations=len(tables['responses']),
        imported_items=len(items), source_observed_testlets=tables['items'].testlet_id.nunique(),
        source_reconciled_prediction_arrays=sum(r['reconciled'] for r in comparisons.values()))
