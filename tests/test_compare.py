"""Comparisons of composite values (_compare): tuples, lists, dicts, instances, nested, as Python compares them: the
same result, or the same exception (a comparison Python can't make stays Python, and raises there)."""

import math
import random
from dataclasses import dataclass, field

import pytest

from scramjit import scram
from scramjit._errors import CompileError


def same(f, *args):
    want = f(*args)
    got = scram(f, fallback=False, cache=False)(*args)
    assert got == want and repr(got) == repr(want), (got, want)
    return got


@dataclass
class P:
    x: float
    y: float


@dataclass(order=True)
class Job:
    pri: int
    name_id: int = field(compare=False)
    size: float = 0.0


class Plain:
    def __init__(self, v):
        self.v = v


def tuples(a, b):
    t, u = (a, (b, 1.0)), (a, (b, 2.0))
    v = (a, b)
    return (t == u, t != u, t < u, t <= u, t > u, t >= u, v < (a, b, 0), v == (a, b), (a, b) == (a, b, 0),
            v < (a, b), (1, 2) < (1, 2.5) < (1, 3), v == [a, b])


def lists(n):
    xs = [i % 3 for i in range(n)]
    ys = [float(i % 3) for i in range(n)]
    zs = [i % 3 for i in range(n + 1)]
    rows = [[i, i + 1] for i in range(n)]
    rows2 = [[i, i + 1] for i in range(n)]
    pairs = [(i, (i, -i)) for i in range(n)]
    return (xs == ys, xs != ys, xs < zs, zs < xs, xs <= ys, rows == rows2, rows < rows2, rows[1:] > rows,
            [1, 2] in rows, [5, 6] not in rows, (2, (2, -2)) in pairs, (2, (2, 2)) in pairs, xs == xs,
            [[1, 2], [3]] < [[1, 2, 0]], [] < [rows[0]], xs != [], rows[0] == [0, 1])


def same_object(n):
    xs = [math.nan] * n
    rows = []
    rows.append(xs)
    rows.append(xs)
    return rows[0] == rows[1], xs in rows, rows == rows      # the very list: equal, as Python finds


def dicts(a):
    d = {'x': a, 'y': [1, 2]}
    e = {'y': [1, 2], 'x': a}
    f = {'x': a, 'z': [1, 2]}
    return d == e, d != e, d == f, [d] == [e], d in [f, e]


def instances(a):
    p = P(a, 1.0)
    q = P(a, 1.0)
    r = P(a, 2.0)
    j = Job(1, 7, 2.0)
    k = Job(1, 8, 3.0)
    u = Plain(a)
    w = Plain(a)
    return p == q, p == r, p != r, j < k, j == Job(1, 9, 2.0), u == w, u == u, [p] == [q], p in [r, q]


def sorts(n):
    rnd = [(i * 7919) % 13 for i in range(n)]
    jobs = []
    for i in range(n):
        jobs.append(Job(rnd[i] % 4, i, float(rnd[i])))
    jobs.sort()
    rows = [[rnd[i] % 3, rnd[(i + 1) % n] % 2] for i in range(n)]
    rows.sort(reverse=True)
    ids = [j.name_id for j in jobs]
    return ids, rows


def test_tuples():
    for a, b in ((1, 2), (1, 1), (2, 1), (0, -1)):
        same(tuples, a, b)


def test_lists():
    for n in (2, 3, 5):
        same(lists, n)
    same(same_object, 3)


def test_dicts_and_instances():
    for a in (1.0, 2.5):
        same(dicts, a)
        same(instances, a)


def test_sorting_composites():
    for n in (1, 5, 40, 100):
        same(sorts, n)


def ordered_dicts(a):
    return {'x': a} < {'x': a}


def mixed(a):
    return (a, 1) < (a, (1,))


def test_what_python_refuses_stays_python():
    for f in (ordered_dicts, mixed):
        with pytest.raises(CompileError):
            scram(f, fallback=False, cache=False)(1)
        with pytest.raises(TypeError):
            scram(f)(1)
