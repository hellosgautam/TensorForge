import math

import numpy as np

from tensorforge.random import get_generator

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
        self.version = 0  # goes up on every in-place write

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
        # Placeholders for Phase 3 (autograd). Not used yet.
        self.requires_grad = False
        self.grad = None
        self._prev = ()
        self._op = ""

    # ---------- basic properties ----------

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def dtype(self) -> str:
        return self.storage.dtype

    @property
    def _version(self) -> int:
        return self.storage.version

    def __repr__(self) -> str:
        return f"tensor({self.tolist()}, dtype={self.dtype})"

    # ---------- element access ----------

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

    # ---------- helpers ----------

    def _norm_dim(self, d: int, ndim: int | None = None) -> int:
        n = self.ndim if ndim is None else ndim
        if d < 0:
            d += n
        if not 0 <= d < n:
            raise IndexError(f"Dimension {d} out of range for {n} dims")
        return d

    def _positions(self) -> list:
        """Storage positions of every element, in row-major order."""
        out = []

        def walk(dim, pos):
            if dim == self.ndim:
                out.append(pos)
                return
            for i in range(self.shape[dim]):
                walk(dim + 1, pos + i * self.strides[dim])

        walk(0, self.offset)
        return out

    def _flat_values(self) -> list:
        return [self.storage.data[p].item() for p in self._positions()]

    # ---------- output ----------

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

    # ---------- indexing and slicing (no copying) ----------

    def __getitem__(self, key):
        """Indexing and slicing. Returns a view: no data is copied."""
        if not isinstance(key, tuple):
            key = (key,)

        n_real = sum(1 for k in key if k is not None and k is not Ellipsis)
        n_ellipsis = sum(1 for k in key if k is Ellipsis)
        if n_ellipsis > 1:
            raise IndexError("Only one ... allowed")
        if n_real > self.ndim:
            raise IndexError(f"Too many indices for a {self.ndim}-d tensor")

        if n_ellipsis:
            i = next(j for j, k in enumerate(key) if k is Ellipsis)
            fill = (slice(None),) * (self.ndim - n_real)
            key = key[:i] + fill + key[i + 1:]
        else:
            key = key + (slice(None),) * (self.ndim - n_real)

        offset = self.offset
        shape, strides = [], []
        dim = 0
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
                offset += k * stride
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

    def __setitem__(self, key, value):
        """Write into the selected elements. Changes every tensor
        that shares this storage."""
        target = self[key]
        if isinstance(value, Tensor):
            if value.shape != target.shape:
                raise ValueError(f"Shape mismatch: {value.shape} vs {target.shape}")
            values = value._flat_values()
        else:
            values = [value] * target.numel
        for pos, v in zip(target._positions(), values):
            self.storage.data[pos] = v
        self.storage.version += 1

    # ---------- view ops (no copying) ----------

    def permute(self, *dims):
        """Reorder dimensions. permute(2, 0, 1) moves dim 2 to the front."""
        dims = [self._norm_dim(d) for d in _shape_arg(dims)]
        if sorted(dims) != list(range(self.ndim)):
            raise ValueError(f"permute needs each of 0..{self.ndim - 1} once")
        return Tensor(self.storage,
                      [self.shape[d] for d in dims],
                      [self.strides[d] for d in dims],
                      self.offset)

    def transpose(self, d0: int, d1: int):
        """Swap two dimensions."""
        order = list(range(self.ndim))
        d0, d1 = self._norm_dim(d0), self._norm_dim(d1)
        order[d0], order[d1] = order[d1], order[d0]
        return self.permute(*order)

    @property
    def T(self):
        """Reverse all dimensions."""
        return self.permute(*reversed(range(self.ndim)))

    def unsqueeze(self, dim: int):
        """Insert a new dimension of size 1 at position dim."""
        dim = self._norm_dim(dim, self.ndim + 1)
        shape, strides = list(self.shape), list(self.strides)
        shape.insert(dim, 1)
        strides.insert(dim, 0)
        return Tensor(self.storage, shape, strides, self.offset)

    def squeeze(self, dim: int | None = None):
        """Remove size-1 dimensions (all of them, or just dim)."""
        if dim is None:
            keep = [k for k in range(self.ndim) if self.shape[k] != 1]
        else:
            dim = self._norm_dim(dim)
            keep = [k for k in range(self.ndim)
                    if k != dim or self.shape[k] != 1]
        return Tensor(self.storage,
                      [self.shape[k] for k in keep],
                      [self.strides[k] for k in keep],
                      self.offset)

    def expand(self, *shape):
        """Repeat size-1 dims to a bigger size by giving them stride 0."""
        shape = list(_shape_arg(shape))
        if len(shape) < self.ndim:
            raise ValueError("expand can't remove dimensions")
        pad = len(shape) - self.ndim
        old_shape = [1] * pad + list(self.shape)
        old_strides = [0] * pad + list(self.strides)
        new_strides = []
        for k, (target, size) in enumerate(zip(shape, old_shape)):
            if target == -1:
                shape[k] = size
                new_strides.append(old_strides[k])
            elif target == size:
                new_strides.append(old_strides[k])
            elif size == 1:
                new_strides.append(0)
            else:
                raise ValueError(f"Can't expand size {size} to {target}")
        return Tensor(self.storage, shape, new_strides, self.offset)

    # ---------- contiguity and reshaping ----------

    def is_contiguous(self) -> bool:
        """True if the strides are the normal row-major ones."""
        if 0 in self.shape:
            return True
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size == 1:
                continue
            if stride != expected:
                return False
            expected *= size
        return True

    def contiguous(self):
        """Self if already contiguous, otherwise a row-major copy."""
        if self.is_contiguous():
            return self
        return Tensor(Storage(self._flat_values(), self.dtype), self.shape)

    def view(self, *shape):
        """New shape over the same storage. Needs a contiguous tensor."""
        shape = list(_shape_arg(shape))
        if shape.count(-1) > 1:
            raise ValueError("Only one -1 allowed")
        if -1 in shape:
            known = math.prod(s for s in shape if s != -1)
            if known == 0 or self.numel % known != 0:
                raise ValueError(f"Can't infer -1 for {self.numel} elements")
            shape[shape.index(-1)] = self.numel // known
        if math.prod(shape) != self.numel:
            raise ValueError(f"Shape {tuple(shape)} doesn't fit {self.numel} elements")
        if not self.is_contiguous():
            raise RuntimeError("view() needs a contiguous tensor; use reshape()")
        return Tensor(self.storage, shape, None, self.offset)

    def reshape(self, *shape):
        """Like view, but copies first if the tensor isn't contiguous."""
        return self.contiguous().view(*shape)

    def flatten(self):
        return self.reshape(-1)


# ---------- building tensors ----------

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


def rand(*shape) -> Tensor:
    """Uniform random numbers in [0, 1)."""
    shape = _shape_arg(shape)
    g = get_generator()
    return Tensor(Storage([g.uniform() for _ in range(math.prod(shape))]), shape)


def randn(*shape) -> Tensor:
    """Standard normal random numbers (mean 0, std 1)."""
    shape = _shape_arg(shape)
    g = get_generator()
    return Tensor(Storage([g.normal() for _ in range(math.prod(shape))]), shape)
