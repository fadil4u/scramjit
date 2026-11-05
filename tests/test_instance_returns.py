"""Returning an instance, an element of a list of instances, and q = p: the very object comes back, as in Python."""

import dataclasses
import warnings

import pytest

from scramjit import CompileError, FallbackWarning, scram

from test_classes import same


@dataclasses.dataclass
class P:
    x: float
    v: float = 0.0


def nearest(ps, x):
    best = ps[0]
    for p in ps:
        if abs(p.x - x) < abs(best.x - x):
            best = p
    best.v = 1.0
    return best


def by_index(ps, i):
    ps[i].v += 2.0
    return ps[i]


def pick(a, b, first):
    q = a
    q.v += 1.0
    if first:
        return q
    return b


def returns_other_thing(a, flag):
    if flag:
        return a
    return 1.0


def falls_off(a, flag):
    if flag:
        return a


def best_after_loop(ps):
    for p in ps:
        p.v = p.x * 2
    return p


def alias_late(a, b):
    a.v = 1.0
    q = a
    q.v += 1.0
    return q.v + b.v


def alias_used_before(a):
    s = 0.0
    if a.x > 0:
        s = 1.0
    q = a
    return q.x + s


def mk():
    return [P(1.0), P(5.0), P(3.0)]


@pytest.mark.parametrize('f,make,compiled', [
    (nearest, lambda: [mk(), 3.2], True),
    (nearest, lambda: [mk(), -9.0], True),
    (by_index, lambda: [mk(), 1], True),
    (by_index, lambda: [mk(), -1], True),
    (by_index, lambda: [mk(), 7], True),          # IndexError
    (pick, lambda: [P(1.0), P(2.0), True], True),
    (pick, lambda: [P(1.0), P(2.0), False], True),
    (returns_other_thing, lambda: [P(1.0), True], False),
    (falls_off, lambda: [P(1.0), True], False),
    (best_after_loop, lambda: [mk()], False),       # p may be unbound after the loop, as for any loop variable
    (best_after_loop, lambda: [[]], False),
    (alias_late, lambda: [P(1.0), P(2.0)], True),
    (alias_used_before, lambda: [P(1.0)], True),
])
def test_returns_and_aliases(f, make, compiled):
    same(f, make, compiled)


def test_identity():
    lst = mk()
    g = scram(cache=False)(nearest)
    assert g(lst, 3.2) is lst[2] and g(lst, 3.2) is lst[2] and g.signatures
    a, b = P(1.0), P(2.0)
    h = scram(cache=False)(pick)
    assert h(a, b, True) is a and h(a, b, False) is b and a.v == 2.0


def test_one_object_under_two_names():
    a = P(1.0)
    h = scram(cache=False)(pick)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert h(a, a, False) is a and a.v == 1.0


def test_a_callee_returning_an_instance_stays_python():
    inner = scram(cache=False)(by_index)

    @scram(cache=False)
    def outer(ps):
        q = inner(ps, 0)
        return 1.0

    with pytest.raises(CompileError):
        scram(cache=False, fallback=False)(outer.__wrapped__)(mk())
