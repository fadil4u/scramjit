"""A function that can't compile whole: it parses text and builds a report, which only Python can do. Its loop over
the numbers compiles on its own anyway, and the rest stays Python. `f.preflight()` shows which is which, line by line."""

import time
import warnings

import numpy as np

from scramjit import FallbackWarning, scram


@scram
def report(lines, values):
    title = lines[0].strip().upper()       # strings: Python
    total = 0.0
    largest = 0.0
    for i in range(len(values)):           # numbers only: compiled on its own
        v = values[i]
        total += v * v
        if v > largest:
            largest = v
    labels = {}
    for k, line in enumerate(lines[1:]):   # builds a dict of strings: stays Python
        labels[k] = line.split(',')[0]
    return {'title': title, 'energy': total, 'largest': largest, 'labels': labels}


lines = ['sensor log', 'north,1', 'south,2']
values = np.random.default_rng(7).random(2_000_000)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', FallbackWarning)
    report(lines, values)
print('warning:', caught[0].message)

t0 = time.perf_counter()
want = report.__wrapped__(lines, values)
t1 = time.perf_counter()
best = float('inf')
for _ in range(5):
    t = time.perf_counter()
    report(lines, values)
    best = min(best, time.perf_counter() - t)
assert report(lines, values) == want
print(f'Python {t1 - t0:.3f} s, partly compiled {best * 1000:.2f} ms, {(t1 - t0) / best:.0f}x\n')

report.preflight()
