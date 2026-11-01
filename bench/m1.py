"""M1 benchmark: CPython against scramjit, by mode, on 1-D kernels. Run from the repository root:

    python3 bench/m1.py

Times are per element (ns), best of 5; CPython runs a smaller input once. "hand C" is the
kernel written directly in C and compiled by cimo: the ceiling for scramjit's generated code.
"""

import math
import os
import sys
import time
import warnings

os.environ.setdefault('CIMO_FMA', '0')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'cimo', 'python'))

import numpy as np  # noqa: E402

from scramjit import scram  # noqa: E402


def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]


def ratio_sum(x, y):
    """Divisions in the loop: Python checks each divisor for zero."""
    s = 0.0
    for i in range(len(x)):
        s += x[i] / y[i]
    return s


def count_below(x, limit: int):
    """int against float comparisons in the loop."""
    n = 0
    for i in range(len(x)):
        if limit > x[i]:
            n += 1
    return n


def prefix_max(x, out):
    best = -math.inf
    for i in range(len(x)):
        if x[i] > best:
            best = x[i]
        out[i] = best
    return best


def edit_distance(a, b, row):
    for j in range(len(b) + 1):
        row[j] = j
    for i in range(1, len(a) + 1):
        prev = row[0]
        row[0] = i
        for j in range(1, len(b) + 1):
            cur = row[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            row[j] = min(row[j] + 1, row[j - 1] + 1, prev + cost)
            prev = cur
    return row[len(b)]


def collatz_total(n):
    total = 0
    for k in range(1, n):
        m = k
        while m != 1:
            m = m // 2 if m % 2 == 0 else 3 * m + 1
            total += 1
    return total


HAND_EMA = r"""
#include <stdint.h>
void ema(const double *x, int64_t n, double *y, double a) {
    y[0] = x[0];
    for (int64_t i = 1; i < n; i++)
        y[i] = a * x[i] + (1 - a) * y[i - 1];
}
"""


def best(f, repeat=5):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    rng = np.random.default_rng(0)
    N, NPY = 2_000_000, 50_000
    x, y = rng.random(N) + 0.5, rng.random(N) + 0.5
    s = np.frombuffer(bytes(rng.integers(97, 100, 1500, dtype=np.uint8)), np.uint8).astype(np.int64)
    t = np.frombuffer(bytes(rng.integers(97, 100, 1500, dtype=np.uint8)), np.uint8).astype(np.int64)
    cases = [
        ('ema', ema, lambda f, n: f(x[:n], np.empty(n), 0.3), N, NPY),
        ('ratio_sum', ratio_sum, lambda f, n: f(x[:n], y[:n]), N, NPY),
        ('count_below', count_below, lambda f, n: f(x[:n] * 4, 2), N, NPY),
        ('prefix_max', prefix_max, lambda f, n: f(x[:n], np.empty(n)), N, NPY),
        ('edit_distance', edit_distance,
         lambda f, n: f(s[:n], t[:n], np.zeros(n + 1, np.int64)), 1500, 150),
        ('collatz', collatz_total, lambda f, n: f(n), 30_000, 3_000),
    ]
    modes = [('parity', {}), ('parity=False', {'parity': False}), ('no bounds', {'boundscheck': False}),
             ('both off', {'parity': False, 'boundscheck': False})]
    work = {'edit_distance': lambda n: n * n, 'collatz': lambda n: n * 100}
    print(f'{"kernel":14s} {"CPython":>9s}' + ''.join(f'{m:>13s}' for m, _ in modes) + '   speedup')
    for name, fn, run, n, npy in cases:
        units = work.get(name, lambda k: k)
        t_py = best(lambda: run(fn, npy), 1) / units(npy) * 1e9
        row = []
        for _, opts in modes:
            f = scram(fn, fallback=False, **opts)
            run(f, min(n, 1000))                     # compile outside the timing
            row.append(best(lambda: run(f, n)) / units(n) * 1e9)
        print(f'{name:14s} {t_py:8.1f}n' + ''.join(f'{v:12.2f}n' for v in row) + f'   {t_py / row[0]:7.0f}x')
    import cimo
    hand = cimo.compile(HAND_EMA).ema
    out = np.empty(N)
    print(f'{"ema, hand C":14s} {"":9s}{best(lambda: hand(x, N, out, 0.3)) / N * 1e9:12.2f}n')


if __name__ == '__main__':
    warnings.simplefilter('error')
    main()
