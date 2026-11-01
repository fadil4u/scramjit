"""What happens when a function doesn't compile, how to find out why, and how to rewrite it."""

import time
import warnings

import numpy as np

from scramjit import CompileError, FallbackWarning, scram


@scram
def histogram_dict(values):
    """Dicts don't compile, so this runs as plain Python: correct, and no faster."""
    counts = {}
    for v in values:
        counts[int(v)] = counts.get(int(v), 0) + 1
    return counts


@scram
def histogram_array(values, counts):
    """The same count, with the keys known to be small integers. This compiles."""
    for v in values:
        counts[v] += 1


values = np.random.default_rng(3).integers(0, 16, 1_000_000)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', FallbackWarning)
    t0 = time.perf_counter()
    by_dict = histogram_dict(values)
    t1 = time.perf_counter()
print('warning:', caught[0].message)

counts = np.zeros(16, np.int64)
histogram_array(values[:1], counts)                 # the first call for these types compiles
counts[:] = 0
t2 = time.perf_counter()
histogram_array(values, counts)
t3 = time.perf_counter()

assert all(by_dict[k] == counts[k] for k in by_dict)
print(f'dict {t1 - t0:.3f} s, array {(t3 - t2) * 1000:.2f} ms, {(t1 - t0) / (t3 - t2):.0f}x\n')

# every line that keeps the function in Python, and what the compiler decided for the one that compiles
histogram_dict.preflight()
print()
histogram_array.preflight()
print()


# For code that has to be native, make falling back an error.
@scram(fallback=False)
def must_be_native(values):
    return {int(values[0]): 1}


try:
    must_be_native(values)
except CompileError as e:
    print('CompileError:', str(e).split(': ', 1)[1])
