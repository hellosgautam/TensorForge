from tensorforge.random import Generator, manual_seed, get_generator



def test_same_seed_gives_same_sequence():
    a = Generator(42)
    b = Generator(42)
    assert [a.next_uint32() for _ in range(5)] == [b.next_uint32() for _ in range(5)]


def test_different_seeds_give_different_sequences():
    a = Generator(1)
    b = Generator(2)
    assert [a.next_uint32() for _ in range(5)] != [b.next_uint32() for _ in range(5)]


def test_values_are_32_bit():
    g = Generator(7)
    for _ in range(1000):
        v = g.next_uint32()
        assert 0 <= v < 2**32
def test_uniform_stays_in_range():
    g = Generator(3)
    for _ in range(1000):
        v = g.uniform(-2.0, 5.0)
        assert -2.0 <= v < 5.0


def test_uniform_mean_is_about_half():
    g = Generator(0)
    values = [g.uniform() for _ in range(100_000)]
    mean = sum(values) / len(values)
    assert abs(mean - 0.5) < 0.01

def test_normal_mean_and_std():
    g = Generator(0)
    values = [g.normal() for _ in range(100_000)]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    assert abs(mean - 0.0) < 0.02
    assert abs(std - 1.0) < 0.02


def test_normal_shift_and_scale():
    g = Generator(1)
    values = [g.normal(mean=10.0, std=3.0) for _ in range(100_000)]
    mean = sum(values) / len(values)
    assert abs(mean - 10.0) < 0.1


def test_permutation_contains_each_number_once():
    g = Generator(5)
    p = g.permutation(100)
    assert sorted(p) == list(range(100))


def test_permutation_actually_shuffles():
    g = Generator(5)
    assert g.permutation(100) != list(range(100))


def test_manual_seed_is_reproducible():
    manual_seed(123)
    first = [get_generator().uniform() for _ in range(5)]
    manual_seed(123)
    second = [get_generator().uniform() for _ in range(5)]
    assert first == second
