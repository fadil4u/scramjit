"""Objects in attributes: p.pos.x, self.pos.norm(), p.a.b.c, written and returned. Each number on the way is a variable.
Everything observable must match CPython; one object reachable two ways runs as Python."""

import dataclasses
import math
import warnings

import pytest

from scramjit import FallbackWarning, scram

from test_classes import same


@dataclasses.dataclass
class V:
    x: float
    y: float

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def scale(self, k):
        self.x *= k
        self.y *= k


@dataclasses.dataclass
class Body:
    pos: V
    vel: V
    m: float = 1.0

    def step(self, dt):
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt
        return self.pos.norm()


@dataclasses.dataclass
class Deep:
    inner: Body
    tag: str = 'x'


class Prop:
    def __init__(self):
        self.v = V(1.0, 2.0)

    @property
    def w(self):
        return self.v


def run(b, n, dt):
    s = 0.0
    for _ in range(n):
        s += b.step(dt) + b.vel.norm() * b.m
    return s


def writes_inner(b):
    b.pos.x = 7.0
    b.vel.scale(2.0)
    return b.pos.x + b.vel.y


def returns_inner(b):
    b.pos.y += 1.0
    return b.pos


def deep(d):
    d.inner.pos.x += 1.0
    d.inner.vel.scale(0.5)
    return d.inner.step(0.1) + d.inner.m


def assigns_object(a, b):
    a.pos = b.pos
    return 0.0


def through_property(p):
    return p.w.x


def bodies():
    return Body(V(1.0, 2.0), V(0.5, -0.5))


@pytest.mark.parametrize('f,make,compiled', [
    (run, lambda: [bodies(), 50, 0.1], True),
    (writes_inner, lambda: [bodies()], True),
    (deep, lambda: [Deep(bodies())], True),
    (assigns_object, lambda: [bodies(), bodies()], False),
    (through_property, lambda: [Prop()], False),
])
def test_nested(f, make, compiled):
    same(f, make, compiled)


def test_returns_the_inner_object():
    b = bodies()
    g = scram(cache=False)(returns_inner)
    r = g(b)
    assert r is b.pos and b.pos.y == 3.0 and g.signatures


def test_one_inner_object_reachable_two_ways_is_one_set_of_variables():
    shared = V(1.0, 2.0)
    b = Body(shared, shared)
    g = scram(cache=False)(writes_inner)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        want = writes_inner(Body(*(lambda s: (s, s))(V(1.0, 2.0))))
        assert g(b) == want and b.pos is b.vel and g.signatures      # one set of variables for both names
    assert g(bodies()) == writes_inner(bodies()) and len(g.signatures) == 2
    a, c = bodies(), bodies()
    c.pos = a.pos

    def two(p, q):
        p.pos.x = 5.0
        return q.pos.x
    h = scram(cache=False, fallback=False)(two)
    assert h(a, c) == 5.0 and a.pos.x == 5.0 and c.pos is a.pos
    assert h(bodies(), bodies()) == 1.0 and len(h.signatures) == 2


@dataclasses.dataclass
class Ring:
    center: V              # sorts before radius: the order values are passed in must follow the parameters'
    radius: float
    weight: float = 2.0


def ring_area(r):
    r.center.x += 1.0
    return r.radius * r.weight + r.center.x * 10 + r.center.y * 100


def test_inner_object_before_a_number():
    same(ring_area, lambda: [Ring(V(1.0, 2.0), 3.0)], True)
    r = Ring(V(1.0, 2.0), 3.0)
    assert scram(cache=False)(ring_area)(r) == 3.0 * 2.0 + 20.0 + 200.0 and r == Ring(V(2.0, 2.0), 3.0)
