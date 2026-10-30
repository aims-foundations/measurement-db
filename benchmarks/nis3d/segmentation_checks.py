"""Replay the released object matcher on saved labels; no segmentation is run."""

import io
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import find_objects
import tifffile


def compact_labels(array):
    """Mirror the native rule: fill each lowest gap with the highest used label."""
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or (array < 0).any() or not np.equal(array, np.floor(array)).all():
            raise ValueError('Native segmentation labels must be nonnegative integers')
        array = array.astype(np.int64)
    counts = np.bincount(array.ravel())
    used = list(np.flatnonzero(counts[1:]) + 1)
    missing = list(np.flatnonzero(counts[1:] == 0) + 1)[::-1]
    mapping = np.arange(len(counts), dtype=np.int32)
    while used and missing and missing[-1] < used[-1]:
        mapping[used.pop()] = missing.pop()
    inverse = {int(mapping[label]): int(label) for label in np.flatnonzero(counts[1:]) + 1}
    return mapping[array], inverse


def volume_reference(archive, parameters, folder):
    prefix = parameters['layout']['volumes_prefix'] + folder + '/'
    gt = tifffile.imread(io.BytesIO(archive.read(prefix + parameters['ground_truth_files'][folder]))).transpose(1, 2, 0)
    confidence = tifffile.imread(io.BytesIO(archive.read(prefix + parameters['confidence_files'][folder]))).transpose(1, 2, 0)
    low = (confidence > 0) & (confidence < 2)
    gt[low] = 0
    gt, originals = compact_labels(gt)
    counts = np.bincount(gt.ravel())
    weights = np.zeros(len(counts))
    references = {}
    for label, region in enumerate(find_objects(gt), 1):
        if region is None:
            continue
        local = gt[region]
        index = np.unravel_index(np.flatnonzero(local.ravel(order='F') == label)[0], local.shape, order='F')
        point = tuple(axis.start + value for axis, value in zip(region, index))
        weights[label] = (float(confidence[point]) - 1) / 3
        references[label] = dict(original_label=originals[label], compact_label=label,
            voxel_count=int(counts[label]), confidence_weight=float(weights[label]),
            bounding_box_yxz=[[axis.start, axis.stop] for axis in region])
    return dict(gt=gt, confidence=confidence, low=low, counts=counts, weights=weights, references=references)


def prediction_array(archive, member, shape):
    blob = archive.read(member)
    try:
        data = loadmat(io.BytesIO(blob))
        values = [value for key, value in data.items() if not key.startswith('__')]
        if len(values) != 1:
            raise ValueError('Expected one saved prediction label array')
        labels = values[0]
    except NotImplementedError:
        import h5py
        with h5py.File(io.BytesIO(blob), 'r') as data:
            names = [key for key in data if not key.startswith('#')]
            if len(names) != 1:
                raise ValueError('Expected one saved HDF5 prediction label array')
            labels = np.asarray(data[names[0]]).transpose(2, 1, 0)
    if labels.shape[:2] != shape[:2] or labels.shape[2] < shape[2]:
        raise ValueError('Prediction and reference spatial dimensions disagree')
    return labels[:, :, -shape[2]:]


def pair_counts(labels, gt, radix):
    """Count overlaps after compressing identical consecutive voxel pairs."""
    present = (labels > 0) & (gt > 0)
    codes = labels[present].astype(np.int64) * radix + gt[present]
    if not len(codes):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(codes)])
    keys, inverse = np.unique(codes[starts], return_inverse=True)
    counts = np.zeros(len(keys), dtype=np.int64)
    np.add.at(counts, inverse, lengths)
    return keys, counts


def match_objects(prediction, reference):
    """Return native mutual best-IoU object outcomes and all five volume metrics."""
    gt, confidence, low = (reference[key] for key in ('gt', 'confidence', 'low'))
    labels, _ = compact_labels(prediction)
    predicted_sizes = np.bincount(labels.ravel())
    reference_sizes = reference['counts']
    ignored = np.bincount(labels[low], minlength=len(predicted_sizes)) > .5 * predicted_sizes
    ignored[0] = True
    keys, intersections = pair_counts(labels, gt, len(reference_sizes))
    pids, gids = keys // len(reference_sizes), keys % len(reference_sizes)
    overlap = intersections / (predicted_sizes[pids] + reference_sizes[gids] - intersections)
    # Sorting descending IoU, then ascending label, gives MATLAB's first-index tie.
    pred_order = np.lexsort((gids, -overlap, pids))
    gold_order = np.lexsort((pids, -overlap, gids))
    _, pfirst = np.unique(pids[pred_order], return_index=True)
    _, gfirst = np.unique(gids[gold_order], return_index=True)
    best_pred = pred_order[pfirst]
    best_gold = {int(gids[pos]): int(pids[pos]) for pos in gold_order[gfirst]}
    recovered = np.zeros(len(reference_sizes), dtype=bool)
    iou = np.zeros(len(reference_sizes))
    for pos in best_pred:
        pred, target = int(pids[pos]), int(gids[pos])
        if not ignored[pred] and best_gold[target] == pred:
            recovered[target] = True
            iou[target] = overlap[pos]
    fp = int((~ignored).sum() - recovered.sum())
    weights = reference['weights']
    tp, fn = float(weights[recovered].sum()), float(weights[~recovered].sum())
    valid = (labels > 0) & ~ignored[labels]
    confidence = confidence.copy()
    confidence[low] = 0
    scores = dict(W_F1=2*tp/(2*tp+fp+fn), W_Precision=tp/(tp+fp), W_Recall=tp/(tp+fn),
        W_SEG=float((weights*iou).sum())/(tp+fp+fn),
        W_IoU=float(confidence[valid].sum())/(float(confidence.sum())+int((valid & (reference['confidence']==0)).sum())))
    return dict(recovered=recovered, iou=iou, false_positives=fp, scores=scores)


def agrees_with_paper(scores, record):
    return all(abs(value-float(record[name])) <= .5*10**(-len(record[name].partition('.')[2]))+1e-10
        for name, value in scores.items())
