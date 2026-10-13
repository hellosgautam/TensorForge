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

    def _flat_index(self, idx: tuple) -> int:
        """Position in storage of element idx, using
        pos = offset + sum(i_k * stride_k)."""
        if len(idx) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(idx)}")
        pos = self.offset
        for i, size, stride in zip(idx, self.shape, self.strides):
            if i < 0:
                i += size  # -1 means the last element
            if not 0 <= i < size:
                raise IndexError(f"Index {i} out of range for size {size}")
            pos += i * stride
        return pos

    def _get(self, *idx):
        """Read one element as a plain Python number."""
        return self.storage.data[self._flat_index(idx)].item()

    def item(self):
        """The value of a one-element tensor."""
        if self.numel != 1:
            raise ValueError("item() only works on tensors with one element")
        return self._get(*([0] * self.ndim))
    
    def tolist(self):
        """Nested Python lists, read by walking the strides."""
        def build(dim, pos):
            if dim == self.ndim:
                return self.storage.data[pos].item()
            return [build(dim + 1, pos + i * self.strides[dim])
                    for i in range(self.shape[dim])]
        return build(0, self.offset)

    def to_numpy(self):
        return np.array(self.tolist(), dtype=_DTYPES[self.dtype])

    def __getitem__(self, key):
        """Indexing and slicing. Returns a view: no data is copied."""
        if not isinstance(key, tuple):
            key = (key,)

        # How many keys use up a real dimension (None and ... don't).
        n_real = sum(1 for k in key if k is not None and k is not Ellipsis)
        n_ellipsis = sum(1 for k in key if k is Ellipsis)
        if n_ellipsis > 1:
            raise IndexError("Only one ... allowed")
        if n_real > self.ndim:
            raise IndexError(f"Too many indices for a {self.ndim}-d tensor")

        # Expand ... into full slices, then pad missing dimensions with ':'.
        if n_ellipsis:
            i = next(j for j, k in enumerate(key) if k is Ellipsis)
            fill = (slice(None),) * (self.ndim - n_real)
            key = key[:i] + fill + key[i + 1:]
        else:
            key = key + (slice(None),) * (self.ndim - n_real)

        offset = self.offset
        shape, strides = [], []
        dim = 0  # which original dimension we're on
        for k in key:
            if k is None:
                shape.append(1)
                strides.append(0)
                continue
            size, stride = self.shape[dim], self.strides[dim]
            if isinstance(k, int):
                if k < 0:
                    k += size
                if not 0 <= k < size:
                    raise IndexError(f"Index {k} out of range for size {size}")
                offset += k * stride  # dimension is dropped
            elif isinstance(k, slice):
                start, stop, step = k.indices(size)
                length = len(range(start, stop, step))
                if length > 0:
                    offset += start * stride
                shape.append(length)
                strides.append(stride * step)
            else:
                raise TypeError(f"Unsupported index {k!r}")
            dim += 1

        return Tensor(self.storage, shape, strides, offset)


def _infer_shape(data) -> tuple:
    """Shape of a nested list, read from its first elements."""
    shape = []
    while isinstance(data, (list, tuple)):
        shape.append(len(data))
        if len(data) == 0:
            break
        data = data[0]
    return tuple(shape)


def _flatten(data, shape: tuple, depth: int, out: list) -> None:
    """Copy a nested list into out in row-major order, checking it isn't ragged."""
    if depth == len(shape):
        if isinstance(data, (list, tuple)):
            raise ValueError("Ragged nested list")
        out.append(data)
        return
    if not isinstance(data, (list, tuple)) or len(data) != shape[depth]:
        raise ValueError("Ragged nested list")
    for item in data:
        _flatten(item, shape, depth + 1, out)


def _infer_dtype(values: list) -> str:
    if values and all(isinstance(v, bool) for v in values):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int64"
    return "float32"


def tensor(data, dtype: str | None = None) -> Tensor:
    """Build a tensor from a number or a nested list."""
    shape = _infer_shape(data)
    flat: list = []
    _flatten(data, shape, 0, flat)
    if dtype is None:
        dtype = _infer_dtype(flat)
    return Tensor(Storage(flat, dtype), shape)


def _shape_arg(shape) -> tuple:
    """Lets you write zeros(2, 3) or zeros((2, 3))."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def full(shape, value, dtype: str = "float32") -> Tensor:
    shape = tuple(shape)
    return Tensor(Storage([value] * math.prod(shape), dtype), shape)


def zeros(*shape, dtype: str = "float32") -> Tensor:
    return full(_shape_arg(shape), 0, dtype)


def ones(*shape, dtype: str = "float32") -> Tensor:
    return full(_shape_arg(shape), 1, dtype)


def arange(start, stop=None, step=1) -> Tensor:
    """Like range(), but floats are allowed. arange(5) is 0, 1, 2, 3, 4."""
    if stop is None:
        start, stop = 0, start
    count = max(0, math.ceil((stop - start) / step))
    values = [start + i * step for i in range(count)]
    return tensor(values)


def eye(n: int, dtype: str = "float32") -> Tensor:
    """n x n identity matrix: 1 on the diagonal, 0 elsewhere."""
    flat = [0] * (n * n)
    for i in range(n):
        flat[i * n + i] = 1
    return Tensor(Storage(flat, dtype), (n, n))
