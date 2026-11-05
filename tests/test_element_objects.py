"""Objects in the attributes of a list's elements: b.pos.x for b in bs, methods of them, passing them on. Each number is
a list over the elements, by path. Everything observable must match CPython."""

import dataclasses
import math

import pytest

from scramjit import scram

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


@scram(cache=False)
def speed(v):
    return v.norm()


def step(bs, dt, n):
    for _ in range(n):
        for b in bs:
            b.pos.x += b.vel.x * dt
            b.pos.y += b.vel.y * dt
            if b.pos.y < 0.0:
                b.vel.y = -b.vel.y
    return bs[0].pos.x + bs[-1].pos.y


def by_index(bs):
    s = 0.0
    for i in range(len(bs)):
        s += bs[i].pos.x * bs[i].m
    return s


def methods_of_inner(bs):
    s = 0.0
    for b in bs:
        b.vel.scale(0.5)
        s += b.vel.norm()
    return s


def passes_inner(bs):
    s = 0.0
    for b in bs:
        s += speed(b.vel)
    return s


def bodies():
    return [Body(V(1.0, 2.0), V(3.0, -4.0)), Body(V(5.0, 0.5), V(-1.0, -2.0), 2.0), Body(V(0.0, 0.0), V(0.5, 0.5))]


def shared_inner():
    v = V(1.0, 1.0)
    return [[Body(V(1.0, 2.0), v), Body(V(5.0, 0.5), v)]]


@pytest.mark.parametrize('f,make,compiled', [
    (step, lambda: [bodies(), 0.1, 30], True),
    (by_index, lambda: [bodies()], True),
    (methods_of_inner, lambda: [bodies()], True),
    (passes_inner, lambda: [bodies()], True),
    (step, lambda: (lambda b: [[b, bodies()[1], b], 0.1, 5])(bodies()[0]), True),     # one element twice
    (methods_of_inner, shared_inner, True),           # one object in two elements: pooled
    (step, lambda: [shared_inner()[0], 0.1, 5], True),
])
def test_element_objects(f, make, compiled):
    same(f, make, compiled)
