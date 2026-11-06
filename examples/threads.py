"""Compiled code releases the GIL, so threads run at the same time. Plain Python threads take turns. Two ways to use
that: separate calls in separate threads, and one loop split into parts, each a call on its own range of the same data
from a ThreadPoolExecutor."""

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scramjit import scram


@scram
def settle(xs, ys, lo, hi):
    """ys[i] from xs[i] by fifty steps of a contraction, for i in range(lo, hi): each i on its own, so any split works."""
    for i in range(lo, hi):
        s = xs[i]
        for _ in range(50):
            s = math.sin(s) + 0.5
        ys[i] = s


def split(fn, n, workers, pool, *data):
    """fn(*data, lo, hi) over range(n) in workers parts, at the same time."""
    bounds = [(k * n // workers, (k + 1) * n // workers) for k in range(workers)]
    list(pool.map(lambda b: fn(*data, *b), bounds))


@scram
def count_primes(lo: int, hi: int):
    count = 0
    for n in range(lo, hi):
        if n < 2:
            continue
        d = 2
        prime = True
        while d * d <= n:
            if n % d == 0:
                prime = False
                break
            d += 1
        if prime:
            count += 1
    return count


def in_threads(fn, ranges):
    results = [0] * len(ranges)

    def work(i, lo, hi):
        results[i] = fn(lo, hi)
    ts = [threading.Thread(target=work, args=(i, *r)) for i, r in enumerate(ranges)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sum(results), time.perf_counter() - t0


workers = min(4, os.cpu_count() or 1)
size = 1_000_000
ranges = [(10_000_000 + i * size, 10_000_000 + (i + 1) * size) for i in range(workers)]
count_primes(2, 3)                               # compile before timing

t0 = time.perf_counter()
one_by_one = sum(count_primes(*r) for r in ranges)
serial = time.perf_counter() - t0
together, parallel = in_threads(count_primes, ranges)
assert together == one_by_one
print(f'{together:,} primes in {workers} ranges')
print(f'scramjit, one after another: {serial:.2f} s')
print(f'scramjit, {workers} threads:         {parallel:.2f} s  ({serial / parallel:.1f}x)')

small = [(lo, lo + size // 50) for lo, _ in ranges]
t0 = time.perf_counter()
for r in small:
    count_primes.__wrapped__(*r)
serial = time.perf_counter() - t0
_, parallel = in_threads(count_primes.__wrapped__, small)
print(f'python, a fiftieth of the work: {serial:.2f} s one after another, {parallel:.2f} s in {workers} threads  ({serial / parallel:.1f}x)')

n = 1_000_000
xs = np.random.default_rng(1).random(n)
one, parts = np.empty(n), np.empty(n)
settle(xs, one, 0, 10)                           # compile before timing
t0 = time.perf_counter()
settle(xs, one, 0, n)
serial = time.perf_counter() - t0
with ThreadPoolExecutor(workers) as pool:
    split(settle, n, workers, pool, xs, parts)   # starts the threads
    t0 = time.perf_counter()
    split(settle, n, workers, pool, xs, parts)
    parallel = time.perf_counter() - t0
assert np.array_equal(one, parts)                 # the same bits: each element is computed the same way
print(f'one loop, {n:,} elements: one call {serial:.2f} s, split over {workers} threads {parallel:.2f} s  ({serial / parallel:.1f}x)')
