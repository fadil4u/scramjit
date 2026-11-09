"""Sorting: the busiest hour of a day of requests to a server, by merging overlapping intervals. Each
hour's requests are sorted by start time with `key=`, merged into the stretches when at least one was being served, and
the hours ranked with `sorted(..., reverse=True)`. Requests come in as numpy arrays; the ranking goes out as Python's
own list of tuples."""

import time

import numpy as np

from scramjit import scram


@scram
def busiest(starts, lengths, windows):
    ranked = []
    for w in range(windows):
        lo, hi = w * 3600.0, (w + 1) * 3600.0
        calls = []
        for i in range(len(starts)):
            if lo <= starts[i] < hi:
                calls.append((float(starts[i]), min(float(starts[i] + lengths[i]), hi)))
        calls.sort(key=lambda c: c[0])
        busy, end = 0.0, -1.0
        for s, e in calls:
            if s > end:
                busy += e - s
                end = e
            elif e > end:
                busy += e - end
                end = e
        ranked.append((busy, w))
    return sorted(ranked, reverse=True)


rng = np.random.default_rng(3)
n = 200_000
starts = rng.random(n) * 86400.0
lengths = rng.exponential(0.4, n)

t0 = time.perf_counter()
busiest(starts[:10], lengths[:10], 1)
first = time.perf_counter() - t0
t0 = time.perf_counter()
want = busiest.__wrapped__(starts, lengths, 24)
py = time.perf_counter() - t0
t0 = time.perf_counter()
got = busiest(starts, lengths, 24)
sj = time.perf_counter() - t0
assert got == want
print(f'busiest hour: {got[0][1]}:00, serving {got[0][0] / 36:.1f}% of it')
print(f'Python {py:.2f} s, scramjit {sj * 1e3:.1f} ms ({py / sj:.0f}x), first call {first:.2f} s')
