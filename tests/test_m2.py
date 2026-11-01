"""M2: iteration builtins, reductions, tuples, math, lists, narrow types."""

import itertools

import numpy as np

from scramjit import scram
from util import same

strict = scram(fallback=False)


# ---- loops over arrays, enumerate, zip, reversed

@strict
def total(a):
    s = 0.0
    for v in a:
        s += v
    return s


@strict
def weighted(a, b):
    s = 0.0
    for i, (u, v) in enumerate(zip(a, b)):
        s += i * u * v
    return s


@strict
def enum_start(a, k):
    s = 0
    for i, v in enumerate(a, k):
        s += i * 3 + int(v)
    return s


@strict
def rev(a, out):
    j = 0
    for v in reversed(a):
        out[j] = v
        j += 1
    return j


@strict
def rev_range(a, b, c):
    s = 0
    last = -1
    for i in reversed(range(a, b, c)):
        s = s * 3 + i
        last = i
    return s + last


@strict
def zip3(a, b, c):
    s = 0.0
    for x, y, z in zip(a, b, range(100)):
        s += x * y + z
    return s


@strict
def mutate_while_iterating(a):
    s = 0.0
    for i, v in enumerate(a):
        if i + 1 < len(a):
            a[i + 1] = a[i + 1] + v
        s += v
    return s


@strict
def after_loop(a):
    i = -5
    v = 0.0
    for i, v in enumerate(a):
        pass
    return i * 1000.0 + v


def test_iteration_builtins():
    rng = np.random.default_rng(5)
    for n, m in itertools.product([0, 1, 4, 9], [0, 1, 4, 9]):
        a, b = rng.random(n), rng.random(m)
        ai = rng.integers(-5, 5, n).astype(np.int64)
        same(total, a)
        same(total, ai)
        same(weighted, a, b)
        same(enum_start, ai, -3)
        same(enum_start, ai, 7)
        same(rev, a, np.zeros(n))
        same(zip3, a, b, 0)
        same(mutate_while_iterating, a)
        same(after_loop, a)
    for a, b, c in itertools.product([-6, 0, 5], [-6, 0, 5], [-2, -1, 1, 3]):
        same(rev_range, a, b, c)
