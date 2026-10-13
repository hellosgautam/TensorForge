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
