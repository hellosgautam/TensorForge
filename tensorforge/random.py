MASK64 = (1 << 64) - 1  # keeps numbers within 64 bits


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

