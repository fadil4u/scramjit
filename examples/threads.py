"""Compiled code releases the GIL, so threads run at the same time. Plain Python threads take turns."""

import os
import threading
import time

from scramjit import scram


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
