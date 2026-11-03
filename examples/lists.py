"""Plain Python lists in and out: a list of prices passed in, a list of rolling maxima returned, a list of lists
of distances updated in place, and a list of counters. Each list is copied into C memory at the call and the changed elements copied back,
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
def shortest_paths(d):
    """Floyd-Warshall, in place in a list of lists of distances."""
    n = len(d)
    for k in range(n):
        dk = d[k]                           # a reference to the row, as in Python
        for i in range(n):
            di = d[i]
            via = di[k]
            for j in range(n):
                if via + dk[j] < di[j]:
                    di[j] = via + dk[j]
    return d[0][n - 1]


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

    d = [[0.0 if i == j else random.uniform(1, 100) for j in range(150)] for i in range(150)]
    plain = [list(r) for r in d]
    start = time.perf_counter()
    want = shortest_paths.__wrapped__(plain)
    python_time = time.perf_counter() - start
    shortest_paths([list(r) for r in d])                 # compiles
    start = time.perf_counter()
    got = shortest_paths(d)
    scramjit_time = time.perf_counter() - start
    assert got == want and d == plain
    print(f'all shortest paths in a 150x150 list of lists: Python {python_time:.2f} s, scramjit {scramjit_time * 1000:.1f} ms, '
          f'{python_time / scramjit_time:.0f}x, the distances updated in place either way')

    rolls = [random.randint(1, 6) for _ in range(100_000)]
    counts, plain = [0] * 6, [0] * 6
    tally(rolls, counts)
    tally.__wrapped__(rolls, plain)
    assert counts == plain
    print('dice counts, updated in place:', counts)
