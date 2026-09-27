"""Read captured message and graph data without importing serialized classes.

Only the data containers below are supported. Unknown globals fail closed; this
is not a general pickle loader. Torch ZIP tensors become NumPy views, and SDK /
PyG instances retain their serialized state as passive records.
"""

from collections import OrderedDict
import io
from pathlib import Path
import pickle
from zipfile import ZipFile, is_zipfile

import numpy as np

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # NumPy 1.x writes the same array representation.
    from numpy.core.multiarray import _reconstruct


class StoredRecord:
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
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _tensor_view
        if module == "torch" and name in {"FloatStorage", "LongStorage", "BoolStorage"}:
            dtype = {"FloatStorage": "f4", "LongStorage": "i8", "BoolStorage": "?"}[name]
            return np.dtype(dtype).newbyteorder(self.endian)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            return _reconstruct
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "scalar":
            return _numeric_scalar
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
