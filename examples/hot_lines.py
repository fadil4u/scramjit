"""Which lines run most: insertion sort, where almost all the work depends on the data.

Run it under preflight to see how often each line ran, and how often each branch was taken:

    scramjit preflight examples/hot_lines.py:insertion_sort --counts --run examples/hot_lines.py
    scramjit preflight --counts --live --run examples/hot_lines.py
"""

import time

import numpy as np

from scramjit import scram


@scram
def insertion_sort(a):
    """Sorts a in place and returns how many elements moved."""
    moved = 0
    for i in range(1, len(a)):
        v = a[i]
        j = i - 1
        while j >= 0 and a[j] > v:
            a[j + 1] = a[j]
            moved += 1
            j -= 1
        a[j + 1] = v
    return moved


rng = np.random.default_rng(42)
data = rng.random(20000)

a = data.copy()
t = time.perf_counter()
moved = insertion_sort(a)
native = time.perf_counter() - t

b = data[:2000].copy()
t = time.perf_counter()
expected = insertion_sort.__wrapped__(b)
python = (time.perf_counter() - t) * (len(data) / len(b)) ** 2     # sorting is quadratic here

c = data[:2000].copy()
assert insertion_sort(c) == expected and np.array_equal(b, c)
assert np.all(a[:-1] <= a[1:])
print(f'{moved:,} moves; scramjit {native * 1000:.0f} ms, Python about {python:.1f} s')
