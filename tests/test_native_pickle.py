"""Exercise data fidelity and rejection of executable/invalid native containers."""

import io
import pickle
from zipfile import ZipFile

import numpy as np
import pytest

from measurement_db.scripts.curate_benchmarks.read_native_pickle import (
    _DataUnpickler, _numeric_scalar, _tensor_view, read_native_pickle,
)


@pytest.mark.parametrize("protocol", [2, 4])
def test_native_numeric_scalars_preserve_values_and_types(tmp_path, protocol):
    values = [np.float32(1.25), np.float64(1 / 3), np.int64(2**60 + 3),
              np.uint64(2**63 + 7), np.bool_(True), np.complex128(1 + 2j)]
    path = tmp_path / "scores.pkl"
    path.write_bytes(pickle.dumps(values, protocol=protocol))
    restored = read_native_pickle(path)
    for expected, actual in zip(values, restored):
        assert type(actual) is type(expected)
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype,data", [(np.dtype("O"), b"\0" * 8),
    (np.dtype("f8"), b"\0"), (np.dtype("U1"), b"\0" * 4), ("f8", b"\0" * 8)])
def test_non_numeric_or_malformed_scalar_data_is_rejected(dtype, data):
    with pytest.raises(pickle.UnpicklingError, match="numeric scalar bytes"):
        _numeric_scalar(dtype, data)


def test_original_archive_member_needs_no_extraction(tmp_path):
    path = tmp_path / "upstream.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("results.pkl", pickle.dumps((["complete\noutput"], [np.float64(.75)])))
        archive.writestr("unsupported.pkl", b"cbuiltins\neval\n.")
    before = path.read_bytes()
    with ZipFile(path) as archive:
        with archive.open("results.pkl") as source:
            outputs, scores = read_native_pickle(source)
            assert outputs == ["complete\noutput"]
            assert type(scores[0]) is np.float64 and scores[0] == .75
        with archive.open("unsupported.pkl") as source:
            with pytest.raises(pickle.UnpicklingError, match="Unsupported native data global"):
                read_native_pickle(source)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("protocol", [2, 4])
def test_original_arrays_and_text(tmp_path, protocol):
    data = {"texts": ["full text", "αβ\n"], "ids": np.array([3, 1, 8], dtype=np.int64),
        "mask": np.array([True, False, True]), "embedded_text": np.array(["x", "λ"], dtype=object)}
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps(data, protocol=protocol))
    result = read_native_pickle(path)
    assert result["texts"] == data["texts"]
    for field in ["ids", "mask", "embedded_text"]:
        np.testing.assert_array_equal(result[field], data[field])
        assert result[field].dtype == data[field].dtype


@pytest.mark.parametrize("global_name", [b"builtins\neval", b"os\nsystem", b"subprocess\nPopen"])
def test_unknown_global_is_never_called(tmp_path, global_name):
    path = tmp_path / "unsupported.pkl"
    path.write_bytes(b"c" + global_name + b"\n.")
    with pytest.raises(pickle.UnpicklingError, match="Unsupported native data global"):
        read_native_pickle(path)


def test_tensor_view_preserves_offset_and_stride():
    values = np.arange(12, dtype=np.int64)
    result = _tensor_view(values, 1, (2, 3), (6, 2), False, None)
    np.testing.assert_array_equal(result, [[1, 3, 5], [7, 9, 11]])


@pytest.mark.parametrize("offset,shape,strides", [(-1, (2,), (1,)), (0, (99,), (1,)),
    (0, (2,), (-1,)), (0, (2, 2), (1,)), (20, (1,), (1,))])
def test_invalid_tensor_reference(offset, shape, strides):
    with pytest.raises((ValueError, pickle.UnpicklingError)):
        _tensor_view(np.arange(4), offset, shape, strides, False, None)


def test_storage_length_and_byte_order():
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("original/data/0", np.array([5, 10], dtype=">i8").tobytes())
    with ZipFile(stream) as archive:
        reader = _DataUnpickler(io.BytesIO(), archive, "original/", ">")
        values = reader.persistent_load(("storage", np.dtype(">i8"), "0", "cuda:0", 2))
        np.testing.assert_array_equal(values, [5, 10])
        with pytest.raises(pickle.UnpicklingError, match="byte count"):
            reader.persistent_load(("storage", np.dtype(">i8"), "0", "cpu", 3))
        with pytest.raises(pickle.UnpicklingError):
            reader.persistent_load(("storage", np.dtype("O"), "0", "cpu", 2))


@pytest.mark.parametrize("byteorder", ["little", "big", None])
def test_torch_container_data(tmp_path, byteorder):
    path = tmp_path / "original.pt"
    with ZipFile(path, "w") as archive:
        archive.writestr("original/data.pkl", pickle.dumps(["one", "two"]))
        if byteorder:
            archive.writestr("original/byteorder", byteorder)
    assert read_native_pickle(path) == ["one", "two"]


def test_unknown_byte_order(tmp_path):
    path = tmp_path / "original.pt"
    with ZipFile(path, "w") as archive:
        archive.writestr("original/data.pkl", pickle.dumps([]))
        archive.writestr("original/byteorder", "unknown")
    with pytest.raises(ValueError, match="byte order"):
        read_native_pickle(path)
