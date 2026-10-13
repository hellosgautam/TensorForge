import pytest

from tensorforge.tensor import Storage, Tensor, _contiguous_strides


def test_contiguous_strides():
    assert _contiguous_strides((2, 3, 4)) == (12, 4, 1)
    assert _contiguous_strides((5,)) == (1,)
    assert _contiguous_strides(()) == ()


def test_tensor_fields():
    s = Storage([0, 1, 2, 3, 4, 5])
    t = Tensor(s, (2, 3))
    assert t.shape == (2, 3)
    assert t.strides == (3, 1)
    assert t.offset == 0
    assert t.ndim == 2
    assert t.numel == 6
    assert t.dtype == "float32"


def test_two_tensors_share_storage():
    s = Storage([0, 1, 2, 3, 4, 5])
    a = Tensor(s, (2, 3))
    b = Tensor(s, (3, 2))
    assert a.storage is b.storage


def test_bad_dtype_raises():
    with pytest.raises(ValueError):
        Storage([1, 2], dtype="complex")

from tensorforge.tensor import (
    Storage, Tensor, _contiguous_strides, arange, eye, full, ones, tensor, zeros,
)

def test_tensor_from_nested_list():
    t = tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.strides == (3, 1)
    assert t._get(0, 0) == 1
    assert t._get(1, 2) == 6
    assert t._get(-1, -1) == 6


def test_3d_tensor():
    t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert t.shape == (2, 2, 2)
    assert t._get(1, 0, 1) == 6


def test_scalar_tensor():
    t = tensor(5.0)
    assert t.shape == ()
    assert t.item() == 5.0


def test_ragged_list_raises():
    with pytest.raises(ValueError):
        tensor([[1, 2], [3]])


def test_dtype_inference():
    assert tensor([1, 2]).dtype == "int64"
    assert tensor([1.5, 2]).dtype == "float32"
    assert tensor([True, False]).dtype == "bool"


def test_index_out_of_range_raises():
    t = tensor([1, 2, 3])
    with pytest.raises(IndexError):
        t._get(3)


def test_constructors():
    assert zeros(2, 3).shape == (2, 3)
    assert zeros((2, 3))._get(1, 2) == 0
    assert ones(4)._get(3) == 1
    assert full((2, 2), 7.0)._get(1, 1) == 7.0


def test_arange():
    t = arange(5)
    assert t.shape == (5,)
    assert [t._get(i) for i in range(5)] == [0, 1, 2, 3, 4]
    assert arange(0, 1, 0.25).shape == (4,)


def test_eye():
    t = eye(3)
    assert t._get(0, 0) == 1
    assert t._get(1, 1) == 1
    assert t._get(0, 1) == 0

import random

import numpy as np

def make(shape):
    """A TensorForge tensor and a NumPy array holding the same values 0, 1, 2, ..."""
    a = np.arange(int(np.prod(shape))).reshape(shape)
    return tensor(a.tolist()), a


def test_tolist_roundtrip():
    t, a = make((2, 3, 4))
    assert t.tolist() == a.tolist()


def test_integer_index_drops_dim():
    t, a = make((2, 3, 4))
    assert t[1].shape == (3, 4)
    assert t[1].tolist() == a[1].tolist()
    assert t[1, 2, 3].item() == a[1, 2, 3]


def test_basic_slices():
    t, a = make((4, 6))
    assert t[1:, ::2].tolist() == a[1:, ::2].tolist()
    assert t[:, 2:5].tolist() == a[:, 2:5].tolist()


def test_negative_step():
    t, a = make((5,))
    assert t[::-1].tolist() == a[::-1].tolist()
    assert t[4:0:-2].tolist() == a[4:0:-2].tolist()


def test_negative_integer_index():
    t, a = make((3, 4))
    assert t[-1, -2].item() == a[-1, -2]


def test_none_adds_dim():
    t, a = make((3, 4))
    assert t[None].shape == (1, 3, 4)
    assert t[:, None].shape == a[:, None].shape


def test_ellipsis():
    t, a = make((2, 3, 4))
    assert t[..., 1].tolist() == a[..., 1].tolist()
    assert t[0, ...].tolist() == a[0, ...].tolist()


def test_chained_slicing():
    t, a = make((4, 5, 6))
    assert t[1:][:, ::2][..., -1].tolist() == a[1:][:, ::2][..., -1].tolist()


def test_slice_shares_storage():
    t, _ = make((4, 4))
    view = t[1:3, ::2]
    assert view.storage is t.storage
    assert view.offset == 4
    assert view.strides == (4, 2)


def test_empty_slice():
    t, a = make((5,))
    assert t[2:2].shape == (0,)
    assert t[2:2].tolist() == []


def test_too_many_indices_raises():
    t, _ = make((2, 3))
    with pytest.raises(IndexError):
        t[0, 0, 0]


def _random_key(rng, shape):
    key = []
    for size in shape:
        if rng.random() < 0.3:
            key.append(rng.randrange(-size, size))
        else:
            start = rng.choice([None, rng.randrange(-size, size)])
            stop = rng.choice([None, rng.randrange(-size, size + 1)])
            step = rng.choice([None, 1, 2, 3, -1, -2])
            key.append(slice(start, stop, step))
    return tuple(key)


def test_random_slicing_matches_numpy():
    rng = random.Random(0)
    for _ in range(300):
        t, a = make((3, 4, 5))
        for _ in range(2):
            if t.ndim == 0 or 0 in t.shape:
                break
            key = _random_key(rng, t.shape)
            t, a = t[key], a[key]
        assert t.shape == a.shape
        assert t.tolist() == a.tolist()
