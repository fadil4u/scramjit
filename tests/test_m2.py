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


# ---- reductions: CPython's own algorithms (sum's int, compensated float and generic phases)

@strict
def sum_gen(a):
    return sum((v * v for v in a if v > 0.25), 0.0)


def sum_default_start(a):
    return sum(v * v for v in a)


@strict
def sum_start(a, s):
    return sum(a, s)


@strict
def sum_py_floats(a, k):
    return sum((float(v) * k for v in a), 0.0)


@strict
def sum_mixed(a, n):
    return sum((i for i in range(n)), 0.5) + sum(a, 0.0)


@strict
def sum_ints(a, n):
    return sum(range(n)) + sum(i * j for i in range(n) for j in range(i) if (i + j) % 3) + sum(a)


@strict
def shadow(a, v):
    s = sum((v + 1 for v in a), 0.0)
    return s + v           # the comprehension's v doesn't leak


@strict
def anyall(a, t):
    return any(v > t for v in a) * 2 + all(v < t for v in a)


@strict
def any_short_circuits(a):
    return any(1 / v > 1 for v in a)


@strict
def any_list_does_not(a):
    return any([1 / v > 1 for v in a])


@strict
def mm(a):
    return min(a) + max(v * 2 for v in a)


@strict
def mm_default(a):
    return min((v for v in a if v > 10), default=-1.0) + max(a, default=7.0)


def test_reductions():
    rng = np.random.default_rng(11)
    for n in (0, 1, 3, 50):
        a = rng.random(n)
        big = np.concatenate([a, a * 1e20, -a * 1e20, [1e-30] * 3])
        ai = rng.integers(-9, 9, n).astype(np.int64)
        for x in (a, big):
            same(sum_gen, x)
            same(sum_start, x, 0.0)
            same(sum_start, x, np.float64(0.5))
            same(sum_py_floats, x, 1e16)
            same(sum_mixed, x, n)
            same(shadow, x, 3.0)
            same(anyall, x, 0.5)
            same(mm_default, x)
            if n:
                same(mm, x)
        same(sum_start, ai, 0)
        same(sum_ints, ai, n)
        same(mm, ai) if n else None
    for x in ([0.0, -0.0], [-0.0], [np.inf, -np.inf], [np.nan, 1.0], [1.0, np.nan], [1e308, 1e308, -1e308]):
        x = np.array(x)
        same(sum_start, x, 0.0)
        same(sum_start, x, -0.0)
        same(sum_py_floats, x, 1.0)
        same(mm, x)
    same(mm, np.zeros(0))                                       # ValueError, same message and line
    same(any_short_circuits, np.array([0.5, 0.0]))              # True before reaching 1 / 0.0
    same(any_list_does_not, np.array([0.5, 0.0]))               # numpy: inf, no error either way
    same(any_short_circuits, np.array([2.0, 0.5]))


def test_sum_of_floats_needs_a_float_start():
    import pytest
    from scramjit import CompileError
    with pytest.raises(CompileError, match=r'write sum\(\.\.\., 0\.0\)'):
        scram(sum_default_start, fallback=False)(np.ones(3))


# ---- tuples

@strict
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@strict
def swap_elements(x, i, j):
    x[i], x[j] = x[j], x[i]


@strict
def minmax(a):
    lo = hi = 0.0
    lo, hi = a[0], a[0]
    for v in a:
        lo, hi = min(lo, v), max(hi, v)
    return lo, hi


@strict
def tuple_var(a, k):
    p = (k, a[0] * 2.0)
    q = p
    return q[0] + q[-1] + len(p), p[1]


@strict
def shape_arg(shape, x):
    h, w = shape
    return h * w + x


def test_tuples():
    for n in range(0, 40):
        same(fib, n)
    x = np.arange(6.0)
    for i, j in [(0, 5), (2, 2), (-1, 3), (6, 0), (1, -7)]:
        same(swap_elements, x, i, j)
    rng = np.random.default_rng(2)
    for n in (1, 2, 9):
        same(minmax, rng.random(n))
        same(minmax, rng.integers(-5, 5, n).astype(np.float64))
    same(tuple_var, np.array([1.5]), 3)
    same(tuple_var, np.array([1.5]), np.int64(3))
    same(shape_arg, (3, 4), 2)
    same(shape_arg, (3, 4.5), 2)


def unpack_wrong(n):
    a, b, c = (n, n)
    return a


def test_tuple_unpacking_mismatch_falls_back():
    import warnings
    from scramjit import FallbackWarning
    f = scram(unpack_wrong)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        same(f, 3)
