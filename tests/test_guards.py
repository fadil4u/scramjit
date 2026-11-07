"""int() of a float takes one test on the way that matters, and an if's comparisons (chained too) prove an index in bounds
for its body. Neither may change a result, an exception, or the writes made before it."""

import numpy as np
import pytest

from scramjit import scram

from util import FLOATS, same


@scram(cache=False)
def to_int(x):
    return int(x)


@scram(cache=False)
def to_int_in_loop(xs, out):
    for i in range(len(xs)):
        out[i] = int(xs[i] * 3.0)


# what doesn't fit in 64 bits wraps, as documented ("Integers are 64-bit, and wrap"): -2**63 fits, 2**63 doesn't
FITS = [x for x in FLOATS if x != x or abs(x) == float('inf') or -2.0 ** 63 <= x < 2.0 ** 63]


@pytest.mark.parametrize('x', FITS + [2.0 ** 62, -(2.0 ** 62), 9.5, -9.5, 0.999, -0.999, 2.0 ** 63 - 1024])
def test_int_of_a_float(x):
    same(to_int, x)


def test_int_in_a_loop_raises_after_the_same_writes():
    for bad in (float('nan'), float('inf'), -float('inf')):
        same(to_int_in_loop, np.array([1.5, -2.25, bad, 4.0]), np.zeros(4, np.int64))
    same(to_int_in_loop, np.array([1.5, -2.25, 7.0, 4.0]), np.zeros(4, np.int64))


@scram(cache=False)
def histogram(x, counts, lo, hi):
    nb = len(counts)
    scale = nb / (hi - lo)
    for i in range(len(x)):
        k = int((x[i] - lo) * scale)
        if 0 <= k < nb:
            counts[k] += 1


@scram(cache=False)
def upto_and_including(a, ks):
    s = 0.0
    n = len(a)
    for k in ks:
        if 0 <= k <= n:                 # k == n is out of range: the check must stay
            s += a[k]
    return s


@scram(cache=False)
def reassigned_in_body(a, ks):
    s = 0.0
    for k in ks:
        if 0 <= k < len(a):
            k = k + 1                   # what the test proved no longer holds
            s += a[k]
    return s


@scram(cache=False)
def split_and(a, ks):
    s = 0.0
    for k in ks:
        if k >= 0 and k < len(a):
            s += a[k]
    return s


@scram(cache=False)
def negative_allowed(a, ks):
    s = 0.0
    for k in ks:
        if k < len(a):                  # no lower bound: a negative k counts from the end
            s += a[k]
    return s


def checks(f):
    entry = next(v for v in f._cache.values() if hasattr(v, 'source'))
    return entry.source.count('SJ_FAIL(1,') + entry.source.count('SJ_FAIL(1, ')


def test_histogram_matches_and_has_no_index_check():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.random(1000) * 1.4 - 0.2, [-0.0, 1.0, 0.999999]])
    same(histogram, x, np.zeros(16, np.int64), 0.0, 1.0)
    assert checks(histogram) == 0
    same(histogram, np.array([0.5, float('nan')]), np.zeros(16, np.int64), 0.0, 1.0)     # ValueError, after one count


@pytest.mark.parametrize('f', [upto_and_including, reassigned_in_body, split_and, negative_allowed])
def test_guards_keep_what_python_does(f):
    a = np.arange(5.0)
    for ks in ([0, 1, 4], [0, 4, 5], [-1, 2], [-6, 0], [5], [3, 4]):
        same(f, a, np.array(ks, np.int64))


def test_which_guards_prove_the_index():
    a, ks = np.arange(5.0), np.array([0, 1], np.int64)
    for f in (upto_and_including, reassigned_in_body, split_and, negative_allowed):
        f(a, ks)
    assert checks(split_and) == 0
    assert checks(upto_and_including) and checks(reassigned_in_body) and checks(negative_allowed)
