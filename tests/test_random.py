from tensorforge.random import Generator


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
