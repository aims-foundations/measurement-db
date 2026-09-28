"""Read captured message and graph data without importing serialized classes.

Only the data containers below are supported. Unknown globals fail closed; this
is not a general pickle loader. Torch ZIP tensors become NumPy views, and SDK /
PyG instances retain their serialized state as passive records.
"""

from collections import OrderedDict, defaultdict
import base64
import io
import json
import math
from pathlib import Path
import pickle
import struct
from zipfile import ZipFile, is_zipfile

import numpy as np

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # NumPy 1.x writes the same array representation.
    from numpy.core.multiarray import _reconstruct


class StoredRecord:
    def __new__(cls, *args, **kwargs):
        record = object.__new__(cls)
        record.args, record.kwargs = args, kwargs
        return record

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class _DataUnpickler(pickle.Unpickler):
    def __init__(self, stream, archive=None, prefix=None, endian="<"):
        super().__init__(stream)
        self.archive, self.prefix, self.endian = archive, prefix, endian

    def find_class(self, module, name):
        records = {
            ("openai.types.chat.chat_completion_message", "ChatCompletionMessage"),
            ("torch_geometric.data.data", "Data"),
            ("torch_geometric.data.data", "DataEdgeAttr"),
            ("torch_geometric.data.data", "DataTensorAttr"),
            ("torch_geometric.data.storage", "GlobalStorage"),
        }
        if (module, name) in records:
            return StoredRecord
        # These scientific objects remain inert records. In particular, never
        # import a serialized calculator or instantiate an upstream model.
        scientific_records = {
            ("ase.atoms", "Atoms"), ("ase.cell", "Cell"),
            ("ase.spacegroup.spacegroup", "Spacegroup"),
            ("ase.utils.forcecurve", "ForceFit"),
            ("pymatgen.core.units", "FloatWithUnit"),
            ("pymatgen.core.units", "Unit"),
            ("pymatgen.io.ase", "MSONAtoms"),
            ("mlip_arena.models", "MLIPEnum"),
            ("mlip_arena.models.externals.mace-mp", "MACE_MP_Medium"),
            ("mlip_arena.models.externals.sevennet", "SevenNet"),
        }
        if (module, name) in scientific_records:
            return type(name, (StoredRecord,), {"source_type": module + "." + name})
        if module == "builtins" and name in {"int", "float", "set", "frozenset", "slice"}:
            return {"int": int, "float": float, "set": set, "frozenset": frozenset, "slice": slice}[name]
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("collections", "defaultdict"):
            return defaultdict
        if module == "pandas" or module.startswith("pandas."):
            # A small explicit set of installed pandas data constructors, never
            # a module/name chosen dynamically by the input file.
            import pandas as pd
            from pandas.core.internals.managers import BlockManager
            from pandas._libs.internals import _unpickle_block
            from pandas.core.indexes.base import _new_Index
            containers = {
                ("pandas", "DataFrame"): pd.DataFrame,
                ("pandas", "Index"): pd.Index,
                ("pandas", "RangeIndex"): pd.RangeIndex,
                ("pandas.core.frame", "DataFrame"): pd.DataFrame,
                ("pandas.core.internals.managers", "BlockManager"): BlockManager,
                ("pandas._libs.internals", "_unpickle_block"): _unpickle_block,
                ("pandas.core.indexes.base", "_new_Index"): _new_Index,
                ("pandas.core.indexes.base", "Index"): pd.Index,
                ("pandas.core.indexes.range", "RangeIndex"): pd.RangeIndex,
            }
            if (module, name) in containers:
                return containers[module, name]
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _tensor_view
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            # Never call Torch's loader: its nested pickle can contain code.
            return _legacy_storage_bytes
        if module == "torch" and name in {"FloatStorage", "LongStorage", "BoolStorage"}:
            dtype = {"FloatStorage": "f4", "LongStorage": "i8", "BoolStorage": "?"}[name]
            return np.dtype(dtype).newbyteorder(self.endian)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            return _reconstruct
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "scalar":
            return _numeric_scalar
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            return _array_from_buffer
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if (module, name) == ("_codecs", "encode"):
            return _latin1_bytes
        raise pickle.UnpicklingError(f"Unsupported native data global: {module}.{name}")

    def persistent_load(self, identifier):
        if (not isinstance(identifier, tuple) or len(identifier) != 5 or identifier[0] != "storage"
                or not isinstance(identifier[1], np.dtype) or identifier[1].hasobject
                or not str(identifier[2]).isdigit() or type(identifier[4]) is not int
                or identifier[4] < 0 or self.archive is None):
            raise pickle.UnpicklingError("Unsupported native storage reference")
        _, dtype, key, _location, size = identifier
        data = self.archive.read(self.prefix + "data/" + str(key))
        if len(data) != size * dtype.itemsize:
            raise pickle.UnpicklingError("Storage byte count differs from declaration")
        return np.frombuffer(data, dtype=dtype)


class _LegacyStorageUnpickler(_DataUnpickler):
    def persistent_load(self, identifier):
        if (not isinstance(identifier, tuple) or len(identifier) != 6
                or identifier[0] != "storage"
                or not isinstance(identifier[1], np.dtype)
                or identifier[1] not in {np.dtype("<f4"), np.dtype("<i8"), np.dtype("?")}
                or not isinstance(identifier[2], str) or not identifier[2].isdigit()
                or identifier[3] != "cpu" or type(identifier[4]) is not int
                or identifier[4] < 0 or identifier[5] is not None):
            raise pickle.UnpicklingError("Unsupported legacy numeric storage reference")
        return identifier


def _legacy_storage_bytes(data):
    """Decode the recorded CPU storage framing, without calling torch.load."""
    if not isinstance(data, bytes):
        raise pickle.UnpicklingError("Expected legacy storage bytes")
    stream = io.BytesIO(data)
    try:
        magic = _DataUnpickler(stream).load()
        protocol = _DataUnpickler(stream).load()
        system = _DataUnpickler(stream).load()
        if (magic != 119547037146038801333356 or protocol != 1001
                or system != {"protocol_version": 1001, "little_endian": True,
                              "type_sizes": {"short": 2, "int": 4, "long": 4}}):
            raise pickle.UnpicklingError("Unsupported legacy storage framing")
        record = _LegacyStorageUnpickler(stream).load()
        if (not isinstance(record, tuple) or len(record) != 6
                or _DataUnpickler(stream).load() != [record[2]]):
            raise pickle.UnpicklingError("Unexpected legacy storage record")
        # Validate even a literal tuple that did not use a persistent ID.
        record = _LegacyStorageUnpickler(io.BytesIO()).persistent_load(record)
        count = struct.unpack("<q", stream.read(8))[0]
        payload = stream.read()
        if count != record[4] or len(payload) != count * record[1].itemsize:
            raise pickle.UnpicklingError("Legacy storage byte count differs from declaration")
    except (EOFError, struct.error) as exc:
        raise pickle.UnpicklingError("Truncated legacy numeric storage") from exc
    return np.frombuffer(payload, dtype=record[1])


def _latin1_bytes(text, encoding):
    if encoding != "latin1":
        raise pickle.UnpicklingError("Only NumPy's original Latin-1 bytes are supported")
    return text.encode("latin1")


def _numeric_scalar(dtype, data):
    """Decode original numeric bytes without allowing object-scalar restoration."""
    if (not isinstance(dtype, np.dtype) or dtype.hasobject or dtype.kind not in "biufc"
            or not isinstance(data, bytes) or len(data) != dtype.itemsize):
        raise pickle.UnpicklingError("Expected numeric scalar bytes with the declared size")
    return np.frombuffer(data, dtype=dtype)[0]


def _array_from_buffer(data, dtype, shape, order):
    """Read protocol-5 array bytes without restoring object pointers."""
    if (not isinstance(data, (bytes, bytearray)) or not isinstance(dtype, np.dtype)
            or dtype.hasobject or order not in {"C", "F"} or not isinstance(shape, tuple)
            or any(type(n) is not int or n < 0 for n in shape)):
        raise pickle.UnpicklingError("Invalid native array buffer, dtype, shape or order")
    return np.frombuffer(data, dtype=dtype).reshape(shape, order=order)


def _tensor_view(storage, offset, shape, strides, requires_grad, hooks):
    if (not isinstance(storage, np.ndarray) or storage.ndim != 1 or storage.dtype.hasobject
            or type(offset) is not int or offset < 0 or len(shape) != len(strides)
            or any(type(n) is not int or n < 0 for n in [*shape, *strides])):
        raise pickle.UnpicklingError("Invalid tensor dimensions or storage")
    # NumPy also checks that the offset, dimensions and strides fit the buffer.
    return np.ndarray(tuple(shape), dtype=storage.dtype, buffer=storage,
        offset=offset * storage.dtype.itemsize,
        strides=tuple(value * storage.dtype.itemsize for value in strides))


def read_native_pickle(path):
    """Read passive data from a path or seekable binary source, including ZIP members."""
    path = path if hasattr(path, "read") else Path(path)
    if not is_zipfile(path):
        if hasattr(path, "read"):
            path.seek(0)
            return _DataUnpickler(path).load()
        with path.open("rb") as stream:
            return _DataUnpickler(stream).load()
    with ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("Duplicate Torch archive members")
        pickles = [name for name in members if name.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise ValueError("Expected one original Torch data.pkl")
        prefix = pickles[0].removesuffix("data.pkl")
        byteorder = archive.read(prefix + "byteorder").decode() if prefix + "byteorder" in members else "little"
        if byteorder not in {"little", "big"}:
            raise ValueError("Unknown original storage byte order")
        return _DataUnpickler(io.BytesIO(archive.read(pickles[0])), archive, prefix,
            "<" if byteorder == "little" else ">").load()


def native_json_value(value):
    """Represent scientific source data in JSON without clipping or repr fallbacks.

    Nonfinite values remain explicitly tagged, rather than becoming null grades
    or invalid JSON tokens. Source-only objects retain their original type,
    constructor arguments and state; their methods are never called.
    """
    if isinstance(value, np.ndarray):
        return native_json_value(value.tolist())
    if isinstance(value, np.generic):
        return native_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"complex": [native_json_value(value.real), native_json_value(value.imag)]}
    if isinstance(value, (bytes, bytearray)):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return {"mapping": [[native_json_value(key), native_json_value(item)] for key, item in value.items()]}
        return {key: native_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((native_json_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, StoredRecord):
        return {"stored_type": value.source_type, "args": native_json_value(value.args),
                "kwargs": native_json_value(value.kwargs), "state": native_json_value(getattr(value, "state", None))}
    if isinstance(value, type) and issubclass(value, StoredRecord):
        return {"stored_class": value.source_type}
    raise TypeError(f"Unsupported native JSON data type: {type(value).__name__}")
