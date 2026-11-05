"""New instances made by what a compiled function calls: a method or function returning one (a.add(b), make(x)),
operators (a + b through __add__), new instances inside expressions, and instances made here holding other objects.
Everything observable must match CPython."""

import dataclasses
import math
import warnings

import pytest

from scramjit import CompileError, FallbackWarning, scram

from test_classes import same


@dataclasses.dataclass
class V:
    x: float
    y: float

    def __add__(self, o):
        return V(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return V(self.x - o.x, self.y - o.y)

    def __mul__(self, k):
        return V(self.x * k, self.y * k)

    def __neg__(self):
        return V(-self.x, -self.y)

    def add(self, o):
        r = V(self.x + o.x, self.y + o.y)
        return r

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def unit(self):
        n = self.norm()
        return V(self.x / n, self.y / n)


@dataclasses.dataclass
class Body:
    pos: V
    vel: V


@dataclasses.dataclass
class Odd:
    x: float

    def __add__(self, o):
        if o.x < 0.0:
            return NotImplemented
        return Odd(self.x + o.x)


@scram(cache=False)
def make(a):
    return V(a, 2.0 * a)


def by_method(a, b):
    c = a.add(b)
    return c.x + c.y


def by_function(a):
    v = make(a)
    return v.x * v.y


def returns_call(a, b):
    return a.add(b)


def returns_operator(a, b):
    return a - b


def operators(a, b, n):
    s = 0.0
    for i in range(n):
        d = b - a
        u = d.unit()
        c = a + u * 0.5
        e = -c
        s += c.norm() + e.x
    return s


def accumulates(x, n):
    acc = V(0.0, 0.0)
    for i in range(n):
        acc = acc + V(x, 1.0)
    return acc


def inside_expressions(a, b):
    return (a - b).norm() + (a + b * 2.0).unit().x


def holds_new(a, n):
    b = Body(V(a, 1.0), V(0.5, 0.25))
    for i in range(n):
        b.pos.x += b.vel.x
        b.pos.y += b.vel.y
    return b.pos.x + b.pos.y


def holds_arguments(p, q):
    b = Body(p, q)
    b.pos.x = 7.0
    return p.x + b.vel.y


def returns_holding_new(a):
    b = Body(V(a, 1.0), V(0.5, 0.25))
    b.vel.y += 1.0
    return b


def returns_holding_arguments(p, a):
    b = Body(p, V(a, a))
    return b


@pytest.mark.parametrize('f,make_args', [
    (by_method, lambda: [V(1.0, 2.0), V(3.0, 4.0)]),
    (by_function, lambda: [1.5]),
    (returns_call, lambda: [V(1.0, 2.0), V(3.0, 4.0)]),
    (returns_operator, lambda: [V(1.0, 2.0), V(3.0, 4.0)]),
    (operators, lambda: [V(1.0, 2.0), V(4.0, 6.0), 20]),
    (accumulates, lambda: [0.5, 10]),
    (inside_expressions, lambda: [V(1.0, 2.0), V(3.0, 5.0)]),
    (holds_new, lambda: [1.5, 10]),
    (holds_arguments, lambda: [V(1.0, 2.0), V(3.0, 4.0)]),
    (returns_holding_new, lambda: [1.5]),
])
def test_new_instances(f, make_args):
    same(f, make_args, True)


def test_returned_objects():
    r = scram(cache=False, fallback=False)(returns_call)(V(1.0, 2.0), V(3.0, 4.0))
    assert type(r) is V and r == V(4.0, 6.0)
    p = V(1.0, 2.0)
    b = scram(cache=False, fallback=False)(returns_holding_arguments)(p, 3.0)
    assert type(b) is Body and b.pos is p and b.vel == V(3.0, 3.0)
    b = scram(cache=False, fallback=False)(returns_holding_new)(1.5)
    assert b == Body(V(1.5, 1.0), V(0.5, 1.25)) and type(b.pos) is V


def conditional(a, b, c):
    return (a + b).x if c else 0.0


def may_not_implement(a, b):
    c = a + b
    return c.x


@pytest.mark.parametrize('f,make_args,why', [
    (conditional, lambda: [V(1.0, 2.0), V(3.0, 4.0), True], 'one side of a condition'),
    (may_not_implement, lambda: [Odd(1.0), Odd(2.0)], ''),
])
def test_stays_python(f, make_args, why):
    same(f, make_args, False)
    with pytest.raises(CompileError, match=why):
        scram(cache=False, fallback=False)(f)(*make_args())



def drift(p, v, n, dt):
    for i in range(n):
        p = p + v * dt              # the parameter starts as the argument's numbers, then is each new instance
    return p.norm()


def reset(p, a):
    s = p.x
    p = V(a, a)
    return s + p.x


def rebound_after_write(p, v):
    p.x = 1.0
    p = p + v
    return p.x


def rebound_and_returned(p, v, n):
    for i in range(n):
        p = p + v
    return p


@pytest.mark.parametrize('f,make_args,compiled', [
    (drift, lambda: [V(1.0, 2.0), V(0.5, 0.5), 10, 0.1], True),
    (reset, lambda: [V(1.0, 2.0), 3.0], True),
    (rebound_after_write, lambda: [V(1.0, 2.0), V(1.0, 1.0)], False),      # the write must reach the argument
    (rebound_and_returned, lambda: [V(1.0, 2.0), V(1.0, 1.0), 0], False),  # n == 0 returns the argument itself
])
def test_a_parameter_given_a_new_instance(f, make_args, compiled):
    same(f, make_args, compiled)


@dataclasses.dataclass
class Mover:
    pos: V
    vel: V

    def step(self, dt):
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt
        return self.pos.x


@scram(cache=False)
def energy(m):
    return m.vel.x * m.vel.x + m.vel.y * m.vel.y + m.pos.y


def method_of_holder(a, n):
    m = Mover(V(a, 0.0), V(1.0, 2.0))
    s = 0.0
    for i in range(n):
        s += m.step(0.1)
    return s + m.pos.y


def method_of_holder_of_argument(p, v, n):
    m = Mover(p, v)
    for i in range(n):
        m.step(0.5)
    return p.x + v.y


def passes_holder(a):
    m = Mover(V(a, 1.0), V(3.0, 4.0))
    return energy(m)


@pytest.mark.parametrize('f,make_args', [
    (method_of_holder, lambda: [1.5, 10]),
    (method_of_holder_of_argument, lambda: [V(1.0, 2.0), V(0.5, -1.0), 4]),
    (passes_holder, lambda: [2.0]),
])
def test_instances_holding_objects(f, make_args):
    same(f, make_args, True)
