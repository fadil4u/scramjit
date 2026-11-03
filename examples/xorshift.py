"""A random number generator from shifts and masks: xorshift32, as written for Python, with each step masked to 32 bits.

`<<` and `>>` compile. `>>` is exactly Python's. `<<` wraps at 64 bits, as `*` does, so `(x << 13) & 0xFFFFFFFF` gives
Python's result for any x that fits in 64 bits: masking keeps only bits that wrapping leaves alone.
"""

import time

import numpy as np

from scramjit import scram


@scram
def xorshift32(seed: int, out):
    x = seed
    for i in range(len(out)):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x
    return x


if __name__ == '__main__':
    n = 1_000_000
    plain, compiled = np.zeros(n, np.uint32), np.zeros(n, np.uint32)

    start = time.perf_counter()
    want = xorshift32.__wrapped__(2463534242, plain)
    python_time = time.perf_counter() - start

    xorshift32(1, compiled[:1])            # compiles
    start = time.perf_counter()
    got = xorshift32(2463534242, compiled)
    scramjit_time = time.perf_counter() - start

    assert got == want and compiled.tobytes() == plain.tobytes()
    print(f'{n:,} numbers: Python {python_time:.2f} s, scramjit {scramjit_time * 1000:.2f} ms, {python_time / scramjit_time:.0f}x, same numbers')
    print('first few:', compiled[:4])
