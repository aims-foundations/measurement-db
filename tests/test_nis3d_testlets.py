"""Native object identity, ignored-region matching and continuous overlap survive."""

import io
from zipfile import ZipFile

import numpy as np
import tifffile

from benchmarks.nis3d.segmentation_checks import compact_labels, volume_reference, match_objects
from scripts.curate_benchmarks.testlet_nis3d_audit import evaluate


def reference(gt, votes):
    stream = io.BytesIO()
    with ZipFile(stream, 'w') as archive:
        for name, array in [('gt.tif', gt), ('votes.tif', votes)]:
            payload = io.BytesIO()
            tifffile.imwrite(payload, array.transpose(2, 0, 1), photometric='minisblack')
            archive.writestr('volume/'+name, payload.getvalue())
    stream.seek(0)
    parameters = dict(layout=dict(volumes_prefix=''), ground_truth_files={'volume':'gt.tif'},
        confidence_files={'volume':'votes.tif'})
    with ZipFile(stream) as archive:
        return volume_reference(archive, parameters, 'volume')


def test_compaction_preserves_native_first_label_ties():
    labels = np.array([0, 1, 5, 7, 7])
    compact, originals = compact_labels(labels)
    assert compact.tolist() == [0, 1, 3, 2, 2]
    assert originals == {1:1, 3:5, 2:7}


def test_sparse_matcher_agrees_with_independent_dense_matcher():
    gt = np.array([[1, 1, 0, 4], [1, 1, 4, 4]], dtype=np.uint16)[:, :, None]
    votes = np.where(gt > 0, 4, 0).astype(np.uint8)
    labels = np.array([[3, 3, 0, 3], [0, 0, 6, 6]], dtype=np.uint16)[:, :, None]
    original_gt, original_votes = gt.copy(), votes.copy()
    expected = evaluate(labels, gt, votes)
    context = reference(gt, votes)
    actual = match_objects(labels, context)
    np.testing.assert_array_equal(gt, original_gt)
    np.testing.assert_array_equal(votes, original_votes)
    assert actual['false_positives'] == expected['false_positives']
    for metric, value in expected['scores'].items():
        assert np.isclose(actual['scores'][metric], value)
    for native_label, target in context['references'].items():
        outcome = expected['targets'][target['original_label']]
        assert outcome['reference'] == target
        assert actual['recovered'][native_label] == outcome['detected']
        assert actual['iou'][native_label] == outcome['iou']


def test_ignored_prediction_still_competes_for_native_reference_match():
    gt = np.array([1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0], dtype=np.uint16)[None, :, None]
    votes = np.array([4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1], dtype=np.uint8)[None, :, None]
    labels = np.array([7, 7, 7, 8, 0, 9, 9, 7, 7, 7, 7], dtype=np.uint16)[None, :, None]
    expected = evaluate(labels, gt, votes)
    context = reference(gt, votes)
    actual = match_objects(labels, context)
    for metric, value in expected['scores'].items():
        assert np.isclose(actual['scores'][metric], value)
    assert actual['false_positives'] == expected['false_positives']


def test_saved_double_labels_are_integers_without_truncation():
    import pytest
    assert compact_labels(np.array([0., 2., 2.]))[0].tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        compact_labels(np.array([0., 1.5]))
