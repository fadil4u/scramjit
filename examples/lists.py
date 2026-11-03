"""Plain Python lists in and out: a list of prices passed in, a list of rolling maxima returned, and a list of
counters updated in place. Each list is copied into C memory at the call and the changed elements copied back,
about 10 ms per 200,000 floats; numpy isn't needed at all."""

import random
import time

from scramjit import scram


@scram
def rolling_max(prices: list[float], window: int):
    out = [0.0] * (len(prices) - window + 1)
    for i in range(len(out)):
        m = prices[i]
        for j in range(i + 1, i + window):
            if prices[j] > m:
                m = prices[j]
        out[i] = m
    return out


@scram
def tally(rolls: list[int], counts: list[int]):
    for r in rolls:
        counts[r - 1] += 1
    return len(rolls)


if __name__ == '__main__':
    random.seed(3)
    prices = [100.0 + random.gauss(0, 1) for _ in range(200_000)]

    start = time.perf_counter()
    want = rolling_max.__wrapped__(prices, 50)
    python_time = time.perf_counter() - start
    rolling_max(prices[:60], 50)                         # compiles
    start = time.perf_counter()
    got = rolling_max(prices, 50)
    scramjit_time = time.perf_counter() - start
    assert got == want and type(got) is list
    print(f'rolling maximum of 200,000 prices: Python {python_time:.2f} s, scramjit {scramjit_time * 1000:.1f} ms, '
          f'{python_time / scramjit_time:.0f}x, copying the lists in and out included')

    rolls = [random.randint(1, 6) for _ in range(100_000)]
    counts, plain = [0] * 6, [0] * 6
    tally(rolls, counts)
    tally.__wrapped__(rolls, plain)
    assert counts == plain
    print('dice counts, updated in place:', counts)
