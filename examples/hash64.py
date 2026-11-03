"""64-bit hashes and random numbers as they're written for Python ints: masked with & 0xFFFFFFFFFFFFFFFF at each step.

A value that can only be in [0, 2**64), a literal past int64 or a masked result, is held as an unsigned 64-bit int,
so these give exactly Python's numbers, from FNV-1a over a buffer to splitmix64 filling a uint64 array.
"""

import time

import numpy as np

from scramjit import scram

MASK = (1 << 64) - 1


@scram
def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ int(b)) * 0x100000001B3) & MASK
    return h


@scram
def splitmix64(state: int, out):
    for i in range(len(out)):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out[i] = z ^ (z >> 31)
    return state


@scram
def bucket_counts(keys, counts):
    """Fibonacci hashing of uint64 keys into a power-of-two table, with numpy's uint64 arithmetic: it wraps."""
    shift = np.uint64(64 - 10)                       # 1024 buckets
    for i in range(len(keys)):
        counts[int((keys[i] * np.uint64(0x9E3779B97F4A7C15)) >> shift)] += 1
    return counts.max()


def timed(f, *args):
    start = time.perf_counter()
    r = f(*args)
    return r, time.perf_counter() - start


if __name__ == '__main__':
    data = np.frombuffer(np.random.default_rng(0).bytes(1_000_000), np.uint8)
    want, python_time = timed(fnv1a64.__wrapped__, data)
    fnv1a64(data[:1])                                    # compiles
    got, scramjit_time = timed(fnv1a64, data)
    assert got == want
    print(f'FNV-1a 64 of 1 MB: Python {python_time:.2f} s, scramjit {scramjit_time * 1000:.2f} ms, '
          f'{python_time / scramjit_time:.0f}x, {got:#018x} both')

    plain, compiled = np.zeros(1_000_000, np.uint64), np.zeros(1_000_000, np.uint64)
    want, python_time = timed(splitmix64.__wrapped__, 2**63 + 1, plain)
    splitmix64(2**63, compiled[:1])                      # compiles: a seed past int64 is its own signature
    got, scramjit_time = timed(splitmix64, 2**63 + 1, compiled)
    assert got == want and compiled.tobytes() == plain.tobytes()
    print(f'splitmix64, 1,000,000 numbers: Python {python_time:.2f} s, scramjit {scramjit_time * 1000:.2f} ms, '
          f'{python_time / scramjit_time:.0f}x, same numbers')

    keys = compiled                                      # the splitmix64 numbers, as uint64 keys
    plain, mine = np.zeros(1024, np.int64), np.zeros(1024, np.int64)
    with np.errstate(over='ignore'):                     # numpy warns that the multiply wraps; compiled code doesn't
        want, python_time = timed(bucket_counts.__wrapped__, keys, plain)
    bucket_counts(keys[:1], np.zeros(1024, np.int64))    # compiles
    got, scramjit_time = timed(bucket_counts, keys, mine)
    assert got == want and plain.tobytes() == mine.tobytes()
    print(f'Fibonacci hashing of 1,000,000 uint64 keys: Python {python_time:.2f} s, scramjit {scramjit_time * 1000:.2f} ms, '
          f'{python_time / scramjit_time:.0f}x, fullest bucket {got}')
