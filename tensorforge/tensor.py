import math

import numpy as np

_DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "int64": np.int64,
    "bool": np.bool_,
}


class Storage:
    """A flat 1D buffer of numbers. Many tensors can share one Storage."""

    def __init__(self, flat_values, dtype: str = "float32"):
        if dtype not in _DTYPES:
            raise ValueError(f"Unknown dtype {dtype!r}")
        data = np.array(flat_values, dtype=_DTYPES[dtype])
        if data.ndim != 1:
            raise ValueError("Storage needs a flat list of numbers")
        self.data = data
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.data)


def _contiguous_strides(shape: tuple) -> tuple:
    """Row-major strides: last stride is 1, each earlier one is the
    next stride times the next size. Shape (2, 3, 4) -> (12, 4, 1)."""
    strides = [0] * len(shape)
    step = 1
    for k in range(len(shape) - 1, -1, -1):
        strides[k] = step
        step *= shape[k]
    return tuple(strides)


class Tensor:
    def __init__(self, storage: Storage, shape, strides=None, offset: int = 0):
        self.storage = storage
        self.shape = tuple(shape)
        if strides is None:
            strides = _contiguous_strides(self.shape)
        self.strides = tuple(strides)
        self.offset = offset

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def dtype(self) -> str:
        return self.storage.dtype
