"""An instance given to another compiled function or to a method: as a parameter, an element of a list of instances, or
self. It goes as the attributes the callee uses, and what the callee assigns comes back. Everything observable must
match CPython."""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from scramjit import CompileError, FallbackWarning, scram

from test_classes import same


@dataclasses.dataclass
class B:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    m: float = 1.0
    label: str = 'unused'

    def dist2(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy + 0.01

    def absorb(self, other):
        self.m += other.m
        other.m = 0.0


@scram(cache=False)
def pull(a, b, dt):
    r2 = a.dist2(b)
    f = b.m / (r2 * math.sqrt(r2))
    a.vx += (b.x - a.x) * f * dt
    a.vy += (b.y - a.y) * f * dt


@scram(cache=False)
def norm(p):
    return math.sqrt(p.x * p.x + p.y * p.y)


@scram(cache=False)
def move(p, dx):
    p.x += dx
    return p.x


@scram(cache=False)
def chase(a, b):
    a.x = b.x + 1.0          # assigns through one what it reads through the other


@scram(cache=False)
def raises(p, a):
    p.x = 42.0
    return a[10]


def nbody(bs, dt, steps):
    for _ in range(steps):
        for a in bs:
            for b in bs:
                pull(a, b, dt)
        for a in bs:
            a.x += a.vx * dt
            a.y += a.vy * dt
    return sum((b.x for b in bs), 0.0)


def norms(p, q):
    return norm(p) + norm(q)


def moves(p, q):
    move(p, 1.0)
    s = move(q, 2.0) + p.x
    return s


def method_with_other(p, q):
    return p.dist2(q) + q.dist2(p)


def absorbs(p, q):
    p.absorb(q)
    return p.m + q.m


def chases_in_list(bs):
    for a in bs:
        for b in bs:
            chase(a, b)
    return bs[0].x


def chases_two(p, q):
    chase(p, q)
    return p.x


def raises_in_callee(p, a):
    p.y = 3.0
    return raises(p, a)


def element_by_index(bs, i):
    q = bs[i]
    return norm(q) + move(bs[0], 1.0) * 0.0


def ps():
    return [B(math.cos(i), math.sin(1.3 * i), label=str(i)) for i in range(6)]


@pytest.mark.parametrize('f,make,compiled', [
    (nbody, lambda: [ps(), 0.01, 5], True),
    (norms, lambda: [B(3.0, 4.0), B(1.0, 0.0)], True),
    (moves, lambda: [B(3.0, 4.0), B(1.0, 0.0)], True),
    (method_with_other, lambda: [B(3.0, 4.0), B(1.0, 0.0)], True),
    (absorbs, lambda: [B(3.0, 4.0, m=2.0), B(1.0, 0.0, m=5.0)], True),
    (chases_in_list, lambda: [ps()], False),                 # a and b may be one instance
    (chases_two, lambda: [B(3.0, 4.0), B(1.0, 0.0)], True),  # two parameters are two objects, or it runs as Python
    (raises_in_callee, lambda: [B(1.0, 2.0), np.arange(3.0)], True),
    (element_by_index, lambda: [ps(), 2], True),
])
def test_passing_instances(f, make, compiled):
    same(f, make, compiled)


def test_one_object_for_both_parameters_runs_as_python():
    g = scram(cache=False)(chases_two)
    p = B(3.0, 4.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(p, p) == 4.0 and p.x == 4.0


def test_reason():
    with pytest.raises(CompileError, match='may be one instance'):
        scram(cache=False, fallback=False)(chases_in_list)(ps())


def test_writes_in_the_callee_before_it_raises():
    g = scram(cache=False)(raises_in_callee)
    p = B(1.0, 2.0)
    with pytest.raises(IndexError):
        g(p, np.arange(3.0))
    assert (p.x, p.y) == (42.0, 3.0) and g.signatures


def ignores(label, x, log):
    s = 0.0
    for v in x:
        s += v
    return s


def test_parameters_never_used_take_anything():
    g = scram(cache=False)(ignores)
    x = np.arange(4.0)
    assert g('a label', x, print) == 6.0 and g.signatures
    assert g(B(1.0, 2.0), x, None) == 6.0 and len(g.signatures) == 2
    assert g('another', x, print) == 6.0 and len(g.signatures) == 2
