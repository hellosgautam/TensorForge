MASK64 = (1 << 64) - 1  # keeps numbers within 64 bits
import math


class Generator:
    def __init__(self, seed: int):
        self.state = seed & MASK64
        if self.state == 0:
            self.state = 0x9E3779B97F4A7C15  # xorshift breaks if the state is 0

    def next_uint32(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK64
        x ^= x >> 27
        self.state = x
        return ((x * 0x2545F4914F6CDD1D) & MASK64) >> 32

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        u = self.next_uint32() / 2**32  # a decimal from 0 up to (not including) 1
        return low + (high - low) * u

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        u1 = 1.0 - self.uniform()  # from just above 0 up to 1, so log(u1) is safe
        u2 = self.uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z

    def randint(self, n: int) -> int:
        """Random integer from 0 to n - 1."""
        return int(self.uniform() * n)

    def permutation(self, n: int) -> list[int]:
        """The numbers 0 to n - 1 in a random order (Fisher-Yates shuffle)."""
        items = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.randint(i + 1)
            items[i], items[j] = items[j], items[i]
        return items



_global_generator = Generator(0)


def manual_seed(seed: int) -> None:
    """Reset the global generator so results can be reproduced."""
    global _global_generator
    _global_generator = Generator(seed)


def get_generator() -> Generator:
    return _global_generator


