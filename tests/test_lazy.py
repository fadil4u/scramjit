"""map, filter and list comprehensions with several for clauses, against CPython."""

import math

import numpy as np
import pytest

from scramjit import FallbackWarning, scram
from util import same


def twice(a): return sum(map(lambda v: v * 2, a), 0.0)
def absd(a): return sum(map(abs, a), 0.0)
def roots(a): return max(map(math.sqrt, a))
def two(a, b): return sum(map(lambda x, y: x * y, a, b), 0.0)
def keep(a): return sum(filter(lambda v: v > 0, a), 0.0)
def truthy(a): return sum(filter(None, a), 0.0)
def fsum_map(a): return math.fsum(map(lambda v: v / 3, a))
def anym(a): return any(map(lambda v: v < 0, a))
def allm(a): return all(map(lambda v: v > -5, a))
def nested(a): return sum(map(lambda v: v + 1, filter(lambda v: v > 0, a)), 0.0)
def in_gen(a): return sum((x * x for x in filter(lambda v: v > 0, a)), 0.0)


def loop_map(a, out):
    n = 0
    for v in map(lambda x: x * 3, a):
        if v > 100:
            break
        out[n] = v
        n += 1
    return n


def loop_filter(a, out):
    n = 0
    for v in filter(lambda x: x != 2.0, a):
        out[n] = v
        n += 1
    return n


def raises(a): return sum(map(lambda v: 1.0 / v, a), 0.0)
def outer_var(a, k: float): return sum(map(lambda v: v * k, a), 0.0)
def shadow(a, v: float): return sum(map(lambda v: v + 1, a), v)


def pairs(a):
    p = [x * y for x in a for y in a]
    if len(p) == 0:
        return 0, -1.0, -1.0, 0.0
    return len(p), p[0], p[-1], sum(p, 0.0)


def triangular(n: int):
    p = [i * j for i in range(n) for j in range(i) if (i + j) % 2 == 0]
    s = 0
    for v in p:
        s += v
    return len(p), s


def big(n: int):
    p = [i + j for i in range(n) for j in range(n)]
    return len(p), p[-1]


def out_of_range(n: int):
    p = [i for i in range(n) for j in range(2)]
    return p[2 * n]


ARRAYS = [[], [1.0], [1.0, 2.0, -3.0, 0.0, 4.5], [-1.0, -2.0], [1e300, 1e300, -0.0], [math.inf, 1.0], [math.nan, 2.0]]


@pytest.mark.parametrize('f', [twice, absd, truthy, keep, fsum_map, anym, allm, nested, in_gen, raises])
def test_one_array(f):
    g = scram(f, fallback=False)
    for v in ARRAYS:
        same(g, np.array(v, float))


def test_pairs():
    g = scram(pairs, fallback=False)
    for v in ARRAYS[:6]:                                # (util compares tuples with ==: no NaN here)
        same(g, np.array(v, float))


def test_map_several_iterables():
    g = scram(two, fallback=False)
    for a, b in ([1.0, 2.0], [3.0, 4.0]), ([1.0, 2.0, 3.0], [4.0]), ([], [1.0]):
        same(g, np.array(a), np.array(b))


def test_roots_and_errors():
    g = scram(roots, fallback=False)
    for v in ([1.0, 4.0], [], [-1.0], [math.nan]):
        same(g, np.array(v, float))


@pytest.mark.parametrize('f', [loop_map, loop_filter])
def test_for_statements(f):
    g = scram(f, fallback=False)
    for v in ARRAYS + [[10.0, 40.0, 1.0], [2.0, 2.0]]:
        same(g, np.array(v, float), np.zeros(8))


def test_names():
    a = np.array([1.0, 2.0])
    same(scram(outer_var, fallback=False), a, 2.5)
    same(scram(shadow, fallback=False), a, 7.5)


def test_rebound_map_is_not_the_builtin():
    import lazy_rebound
    a = np.array([1.0, -2.0])
    with pytest.warns(FallbackWarning):                 # its map is the module's, which adds one
        assert scram(lazy_rebound.rebound)(a) == lazy_rebound.rebound(a) == 5.0


@pytest.mark.parametrize('n', [0, 1, 3, 40])
def test_several_for_clauses(n):
    same(scram(triangular, fallback=False), n)
    same(scram(big, fallback=False), n)
    same(scram(out_of_range, fallback=False), n)
