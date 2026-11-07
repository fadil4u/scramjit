"""Lists of instances: for p in ps, enumerate(ps), p = ps[i], ps[i].a, len(ps) and methods of elements compile, with a list
of numbers per attribute, copied in at the call and changed values written back. Everything observable must match CPython."""

import dataclasses
import traceback
import warnings

import numpy as np
import pytest

from scramjit import CompileError, FallbackWarning, scram

from test_classes import same


@dataclasses.dataclass
class P:
    x: float
    v: float = 0.0
    hits: int = 0
    alive: bool = True

    def kick(self, f, dt):
        self.v += f * dt

    def energy(self):
        return 0.5 * self.v * self.v

    def bump(self, a, i):
        self.hits += 1
        return a[i]


class Slots:
    __slots__ = ('x', 'v')

    def __init__(self, x, v=0.0):
        self.x, self.v = x, v


class WithClassAttr:
    k = 2.0

    def __init__(self, x):
        self.x = x


def ps(n=5, cls=P):
    return [cls(1.0 + i * 0.5) for i in range(n)]


def step(ps, dt, n):
    for _ in range(n):
        for p in ps:
            p.kick(-p.x, dt)
            p.x += p.v * dt
            if p.x < 1.5:
                p.hits += 1
    return sum((q.x for q in ps), 0.0)


def by_index(ps):
    s = 0.0
    for i in range(len(ps)):
        s += ps[i].x * i
        ps[i].v = s
    return s + ps[-1].x


def named_element(ps, i):
    q = ps[i]
    q.x += 1.0
    return q.x


def named_out_of_range(ps):
    ps[0].x = 9.0
    q = ps[10]
    return 1.0


def enumerated(ps):
    s = 0.0
    for i, p in enumerate(ps, 1):
        s += p.x * i
    for i, p in enumerate(ps):
        p.hits = i
    return s


def flags(ps):
    n = 0
    for p in ps:
        if p.alive:
            n += 1
        p.alive = p.x > 2.0
    return n


def methods(ps):
    e = 0.0
    for p in ps:
        p.kick(1.0, 0.5)
        e += p.energy()
    return e + ps[0].energy()


def method_raises(ps, a):
    for p in ps:
        p.x = 7.0
        p.bump(a, 10)
    return 1.0


def nested(ps):
    s = 0.0
    for p in ps:
        for q in ps:
            s += abs(p.x - q.x)
    return s


def class_attr(ps):
    s = 0.0
    for p in ps:
        s += p.x * p.k
    return s


def writes_class_attr(ps):
    for p in ps:
        p.k = 1.0
    return 0.0


def returns_element(ps):
    ps[0].x += 1.0
    return ps[0]


def passes_list(ps):
    return ps[0].x + len(str(ps))


def appends(ps):
    ps.append(ps[0])
    return 0


def nothing(ps):
    return len(ps)


@pytest.mark.parametrize('f,make,compiled', [
    (step, lambda: [ps(), 0.1, 20], True),
    (by_index, lambda: [ps()], True),
    (named_element, lambda: [ps(), 2], True),
    (named_element, lambda: [ps(), -1], True),
    (named_element, lambda: [ps(), 5], True),            # IndexError, as Python's list gives
    (named_out_of_range, lambda: [ps()], True),
    (enumerated, lambda: [ps()], True),
    (flags, lambda: [ps()], True),
    (methods, lambda: [ps()], True),
    (method_raises, lambda: [ps(), np.arange(3.0)], True),
    (nested, lambda: [ps()], True),
    (step, lambda: [ps(5, Slots), 0.1, 3], False),       # no kick on Slots: AttributeError
    (by_index, lambda: [ps(5, Slots)], True),
    (class_attr, lambda: [ps(4, WithClassAttr)], True),
    (writes_class_attr, lambda: [ps(4, WithClassAttr)], False),
    (by_index, lambda: [[]], False),
    (by_index, lambda: [ps(3) + [Slots(1.0)]], False),   # two classes
    (by_index, lambda: [[P(1), P(2.0)]], False),         # an int and a float
    (returns_element, lambda: [ps()], True),               # the element itself (test_instance_returns.py)
    (passes_list, lambda: [ps()], False),
    (appends, lambda: [ps()], True),                       # grown: held by reference (tests/test_glists.py), from 0.0.4
    (nothing, lambda: [ps()], False),
])
def test_lists_of_instances(f, make, compiled):
    same(f, make, compiled)


def test_one_instance_twice():
    g = scram(cache=False, fallback=False)(by_index)
    p = P(1.0)
    lst = [p, p]
    assert g(lst) == by_index([P(1.0)] * 2) and lst[0] is lst[1] and p.v == 1.0    # [P(1.0)] * 2 is one instance too
    assert g.signatures[0][0].name.endswith('with repeats')
    same(by_index, lambda: (lambda o: [[o, P(2.0), o]])(P(1.0)), True)


def element_and_object(ps, q):
    ps[0].x = 5.0
    q.x = q.x + 1.0
    return ps[0].x


def test_an_element_also_passed_on_its_own():
    p = P(1.0)
    g = scram(cache=False, fallback=False)(element_and_object)
    assert g([p], p) == 6.0 and p.x == 6.0 and g.signatures[0][1].name == 'element of #0'
    assert g([P(1.0)], P(1.0)) == 5.0 and len(g.signatures) == 2


def test_unchanged_values_keep_their_identity():
    big = 10 ** 15 + 7
    lst = [P(1.0, hits=big), P(2.0, hits=big + 1)]
    old = [p.hits for p in lst]
    scram(cache=False)(flags)(lst)
    assert all(p.hits is o for p, o in zip(lst, old))


def test_writes_before_an_exception_and_the_traceback():
    g = scram(cache=False)(method_raises)
    lst = ps(3)
    with pytest.raises(IndexError) as info:
        g(lst, np.arange(3.0))
    assert (lst[0].x, lst[0].hits, lst[1].x) == (7.0, 1, 1.5) and g.signatures
    names = [f.name for f in traceback.extract_tb(info.value.__traceback__) if f.name in ('method_raises', 'bump')]
    assert names == ['method_raises', 'bump']


def test_reasons():
    with pytest.raises(CompileError, match='only len'):
        scram(cache=False, fallback=False)(passes_list)(ps())


def test_list_and_dict_in_one_call():
    def f(d, xs, lst):
        xs[0] = d['a'] + d['b']
        lst[0].x = 3.0
    xs, lst = [1.0, 2.0], ps(2)
    g = scram(cache=False)(f)
    g({'a': 1.0, 'b': 2.0}, xs, lst)
    assert xs == [3.0, 2.0] and lst[0].x == 3.0 and g.signatures


def enumerate_in_comprehension(ps):
    return sum((i * p.x for i, p in enumerate(ps)), 0.0) + sum((i * p.x for i, p in enumerate(ps, 2) if i > 2), 0.0)


def test_enumerate_in_a_comprehension():
    same(enumerate_in_comprehension, lambda: [ps()], True)
