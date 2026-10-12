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
