"""Levenshtein distance. Each cell depends on three cells before it, so it doesn't vectorize."""

import time

import numpy as np

from scramjit import scram


@scram
def edit_distance(a, b, prev, cur):
    """a, b: uint8 arrays. prev, cur: two int64 rows of len(b) + 1."""
    for j in range(len(b) + 1):
        prev[j] = j
    for i in range(1, len(a) + 1):
        cur[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        for j in range(len(b) + 1):
            prev[j] = cur[j]
    return prev[len(b)]


def distance(s, t, fn=edit_distance):
    a, b = np.frombuffer(s, np.uint8), np.frombuffer(t, np.uint8)
    return fn(a, b, np.zeros(len(b) + 1, np.int64), np.zeros(len(b) + 1, np.int64))


print('kitten -> sitting:', distance(b'kitten', b'sitting'))

rng = np.random.default_rng(0)
s = rng.choice(list(b'ACGT'), 1500).astype(np.uint8).tobytes()
t = rng.choice(list(b'ACGT'), 1500).astype(np.uint8).tobytes()

t0 = time.perf_counter()
native = distance(s, t)
t1 = time.perf_counter()
python = distance(s, t, edit_distance.__wrapped__)
t2 = time.perf_counter()

assert native == python
print(f'two 1,500-base sequences: distance {native}')
print(f'python {t2 - t1:.2f} s, scramjit {(t1 - t0) * 1000:.1f} ms, {(t2 - t1) / (t1 - t0):.0f}x')
