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
