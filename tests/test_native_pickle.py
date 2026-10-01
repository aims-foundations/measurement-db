"""Exercise data fidelity and rejection of executable/invalid native containers."""

import io
import json
import pickle
import struct
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from measurement_db.scripts.curate_benchmarks.read_native_pickle import (
    _DataUnpickler, _array_from_buffer, _legacy_storage_bytes, _numeric_scalar,
    _tensor_view, native_json_value, read_native_pickle,
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


def test_serialized_regex_is_preserved_without_compilation(tmp_path, monkeypatch):
    import re
    pattern = re.compile(r"\[\[([^\]]+)\]\]", re.IGNORECASE)
    path = tmp_path / "judgment_inputs.pkl"
    path.write_bytes(pickle.dumps({"pattern": pattern}, protocol=4))
    def forbidden(*args, **kwargs):
        raise AssertionError("A native pattern must not be compiled")
    monkeypatch.setattr(re, "_compile", forbidden)
    restored = native_json_value(read_native_pickle(path))
    assert restored["pattern"] == {"stored_type": "re._compile", "args": [pattern.pattern, pattern.flags],
                                   "kwargs": {}, "state": None}


def test_fastchat_template_is_inert_data_without_source_imports(tmp_path):
    import sys
    path = tmp_path / "conversation.pkl"
    path.write_bytes(b"cfastchat.conversation\nConversation\n(tR"
                     b"(Vname\nVmistral\nVsep_style\n"
                     b"cfastchat.conversation\nSeparatorStyle\n(I7\ntRdb.")
    before = set(sys.modules)
    assert native_json_value(read_native_pickle(path)) == {
        "stored_type": "fastchat.conversation.Conversation", "args": [], "kwargs": {},
        "state": {"name": "mistral", "sep_style": {
            "stored_type": "fastchat.conversation.SeparatorStyle", "args": [7],
            "kwargs": {}, "state": None}},
    }
    assert not any(name.startswith("fastchat") for name in set(sys.modules) - before)
    path.write_bytes(b"cfastchat.conversation\nget_conv_template\n.")
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


def _legacy_storage_fixture(values, *, descriptor=None, count=None, extra=b""):
    record = descriptor or ("storage", values.dtype, "1234", "cpu", values.size, None)
    headers = [119547037146038801333356, 1001,
        {"protocol_version": 1001, "little_endian": True,
         "type_sizes": {"short": 2, "int": 4, "long": 4}}, record, [record[2]]]
    return (b"".join(pickle.dumps(value, protocol=2) for value in headers)
            + struct.pack("<q", values.size if count is None else count)
            + values.tobytes() + extra)


@pytest.mark.parametrize("dtype", ["<f4", "<i8", "?"])
def test_legacy_numeric_storage_keeps_exact_bytes_without_torch(dtype):
    import sys
    values = np.array([0, 1, 3], dtype=dtype)
    before = set(sys.modules)
    restored = _legacy_storage_bytes(_legacy_storage_fixture(values))
    assert restored.dtype == values.dtype and restored.tobytes() == values.tobytes()
    assert not any(name.startswith("torch") for name in set(sys.modules) - before)
    assert _DataUnpickler(io.BytesIO()).find_class("torch.storage", "_load_from_bytes") is _legacy_storage_bytes


@pytest.mark.parametrize("changes", [{"count": 4}, {"extra": b"unexpected"},
    {"descriptor": ("storage", np.dtype("O"), "1234", "cpu", 3, None)},
    {"descriptor": ("storage", np.dtype("<f4"), "1234", "cpu", -1, None)},
    {"descriptor": ("storage", np.dtype("<f4"), "1234", "cpu", 3, (0, 3))}])
def test_legacy_invalid_storage_is_rejected(changes):
    with pytest.raises(pickle.UnpicklingError):
        _legacy_storage_bytes(_legacy_storage_fixture(np.arange(3, dtype="<f4"), **changes))


def test_legacy_truncated_and_executable_nested_data_are_rejected():
    payload = _legacy_storage_fixture(np.arange(3, dtype="<f4"))
    for truncated in [payload[:3], payload[:-1]]:
        with pytest.raises(pickle.UnpicklingError):
            _legacy_storage_bytes(truncated)
    malicious = b"cos\nsystem\n."
    with pytest.raises(pickle.UnpicklingError, match="Unsupported native data global"):
        _legacy_storage_bytes(malicious)


def test_legacy_nested_storage_is_decoded_by_the_public_reader(tmp_path):
    path = tmp_path / "native.pkl"
    values = np.array([1.25, -2, 3], dtype="<f4")
    data = _legacy_storage_fixture(values)
    prefix = b"\x80\x04ctorch.storage\n_load_from_bytes\nB"
    path.write_bytes(prefix + struct.pack("<I", len(data)) + data + b"\x85R.")
    assert read_native_pickle(path).tobytes() == values.tobytes()
    malicious = b"cos\nsystem\n."
    path.write_bytes(prefix + struct.pack("<I", len(malicious)) + malicious + b"\x85R.")
    with pytest.raises(pickle.UnpicklingError, match="Unsupported native data global"):
        read_native_pickle(path)


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


@pytest.mark.parametrize("protocol", [4, 5])
def test_pandas_numeric_and_object_columns_preserve_records(tmp_path, protocol):
    frame = pd.DataFrame({"name": pd.Series(["first", "second"], dtype=object),
        "value": [1.25, np.nan], "outputs": [np.array([1., 2.]), np.array([3.])]}, index=[0, 1])
    frame.columns = pd.Index(frame.columns, dtype=object)
    path = tmp_path / "native.pkl"
    path.write_bytes(pickle.dumps(frame, protocol=protocol))
    pd.testing.assert_frame_equal(read_native_pickle(path), frame)


def test_scientific_records_keep_constructor_and_state_without_source_imports(tmp_path):
    import sys
    path = tmp_path / "record.pkl"
    # A protocol-0 constructor and state assignment for a source-only class.
    path.write_bytes(b"case.atoms\nAtoms\n(Voriginal\ntR(Vstate\nVpreserved\ndb.")
    before = set(sys.modules)
    record = read_native_pickle(path)
    assert record.source_type == "ase.atoms.Atoms"
    assert record.args == ("original",)
    assert record.state == {"state": "preserved"}
    assert not any(name.startswith("ase") for name in set(sys.modules) - before)


@pytest.mark.parametrize("global_name", [b"pandas\nread_pickle", b"pandas.io.common\nget_handle",
    b"ase.calculators.calculator\nCalculator", b"mlip_arena.models.externals.other\nRun"])
def test_unreviewed_data_and_scientific_globals_fail_closed(tmp_path, global_name):
    path = tmp_path / "unsupported.pkl"
    path.write_bytes(b"c" + global_name + b"\n.")
    with pytest.raises(pickle.UnpicklingError, match="Unsupported native data global"):
        read_native_pickle(path)


@pytest.mark.parametrize("dtype,shape,order", [(np.dtype("O"), (1,), "C"),
    (np.dtype("f8"), (-1,), "C"), (np.dtype("f8"), (1,), "X"), (np.dtype("f8"), (2,), "C")])
def test_malformed_protocol_five_array_is_rejected(dtype, shape, order):
    with pytest.raises((ValueError, pickle.UnpicklingError)):
        _array_from_buffer(b"\0" * 8, dtype, shape, order)


def test_scientific_json_preserves_full_values_and_explicit_nonfinite_outputs():
    native = {"text": "complete trace\n" * 2000, "id": np.int64(2**60 + 3),
        "outputs": np.array([np.nan, np.inf, -np.inf, 1.25]), "literal": "nan", "bytes": b"\0\xff"}
    result = json.loads(json.dumps(native_json_value(native), allow_nan=False))
    assert result["text"] == native["text"] and result["id"] == 2**60 + 3
    assert result["outputs"] == [{"nonfinite_float": "nan"}, {"nonfinite_float": "inf"},
                                  {"nonfinite_float": "-inf"}, 1.25]
    assert result["literal"] == "nan" and result["bytes"] == {"base64": "AP8="}
    with pytest.raises(TypeError, match="Unsupported native JSON"):
        native_json_value(object())


def test_captured_defaultdict_of_boolean_frames(tmp_path):
    from collections import defaultdict
    import pandas as pd
    original = defaultdict(dict)
    original['dataset']['model'] = pd.DataFrame([[True, False], [False, True]],
        index=pd.Index(['crop_b', 'crop_a'], dtype=object),
        columns=pd.Index(['1', '0'], dtype=object))
    path = tmp_path / 'original_correctness.pkl'
    path.write_bytes(pickle.dumps(original, protocol=4))
    restored = read_native_pickle(path)
    assert isinstance(restored, defaultdict) and restored.default_factory is dict
    pd.testing.assert_frame_equal(restored['dataset']['model'], original['dataset']['model'])
