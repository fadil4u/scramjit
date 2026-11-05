"""One object under several names: two parameters, two attributes, an attribute and a parameter, or one object passed
twice to a compiled callee. Which names share an object is part of the signature (the alias pattern), so the kernel has
one set of variables for all of them. Everything observable must match CPython."""

import dataclasses

import pytest

from scramjit import scram

from test_classes import same


@dataclasses.dataclass
class P:
    x: float
    m: float = 1.0

    def absorb(self, other):
        self.m += other.m
        other.m = 0.0

    def bump(self):
        self.x += 1.0


@dataclasses.dataclass
class Node:
    x: float
    next: object = None


@scram(cache=False)
def chase(a, b):
    a.x += 1.0
    return b.x


def chase_same(p):
    s = chase(p, p)
    return s + p.x


def self_absorb(p):
    p.absorb(p)
    return p.m


def absorb_element(ps, i):
    p = ps[i]
    p.absorb(p)
    return ps[i].m


def chase_element(ps, i):
    return chase(ps[i], ps[i]) + ps[i].x


def chase_made(a):
    v = P(a, 2.0)
    return chase(v, v) + v.x


def returns_second(a, b):
    a.x = 5.0
    return b


def through_next(b, a):
    a.next.x += 1.0
    return b.x


def through_self(n):
    n.next.x += 2.0
    return n.x


def both_ways(a, b):
    a.bump()
    b.bump()
    return a.x + b.x


def ring():
    n = Node(1.0)
    n.next = n
    return n


def linked():
    b = Node(1.0)
    return [b, Node(2.0, b)]


@pytest.mark.parametrize('f,make', [
    (chase_same, lambda: [P(1.0)]),
    (self_absorb, lambda: [P(1.0, 2.0)]),
    (absorb_element, lambda: [[P(1.0, 2.0), P(3.0, 4.0)], 1]),
    (chase_element, lambda: [[P(1.0), P(3.0)], 0]),
    (chase_made, lambda: [1.5]),
    (returns_second, lambda: (lambda p: [p, p])(P(1.0))),
    (through_next, linked),
    (both_ways, lambda: (lambda p: [p, p])(P(1.0))),
    (both_ways, lambda: [P(1.0), P(1.0)]),
])
def test_one_object_many_names(f, make):
    same(f, make, True)


def test_the_pattern_is_part_of_the_signature():
    g = scram(cache=False, fallback=False)(both_ways)
    p, q = P(1.0), P(1.0)
    assert g(p, p) == 6.0 and p.x == 3.0
    assert g(p, q) == 6.0 and (p.x, q.x) == (4.0, 2.0)
    assert len(g.signatures) == 2
    assert g(q, q) == 8.0 and q.x == 4.0 and len(g.signatures) == 2


def test_returns_the_object_itself():
    g = scram(cache=False, fallback=False)(returns_second)
    p = P(1.0)
    assert g(p, p) is p and p.x == 5.0


def test_an_object_holding_itself():
    g = scram(cache=False, fallback=False)(through_self)
    n = ring()
    assert g(n) == 3.0 and n.next is n and n.x == 3.0
    m = Node(1.0, Node(5.0))
    assert g(m) == 1.0 and m.next.x == 7.0 and len(g.signatures) == 2


@dataclasses.dataclass
class B:
    pos: P
    vel: P


@scram(cache=False)
def bump_pos(b):
    b.pos.x += 1.0
    return b.vel.x


def passes_on(a, b):
    a.pos.x = 10.0
    r = bump_pos(b)
    return r + a.pos.x


def shared_inner():
    v = P(1.0)
    return [B(P(5.0), P(6.0)), B(v, v)]


def shared_across():
    a = B(P(1.0), P(2.0))
    return [a, B(a.pos, P(3.0))]


def test_passing_on_an_object_that_shares_inside_itself():
    same(passes_on, shared_inner, True)


def test_passing_on_an_object_sharing_with_another_argument():
    same(passes_on, shared_across, True)      # the callee gets the other argument's object as its own fields

