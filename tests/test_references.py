"""References chosen by data: walking a linked structure (n = n.next), rebinding one (a.pos = b.pos, cur.next = prev),
and None. The objects references reach are one pool per call, a reference is an index in it. Everything observable must
match CPython, the AttributeError for reading through None included."""

import dataclasses

import pytest

from scramjit import CompileError, scram

from test_classes import same


@dataclasses.dataclass
class Node:
    x: float
    next: object = None


@dataclasses.dataclass
class Tree:
    key: int
    left: object = None
    right: object = None
    hits: int = 0


@dataclasses.dataclass
class V:
    x: float
    y: float


@dataclasses.dataclass
class Body:
    pos: V
    vel: V


def total(n):
    s = 0.0
    while n is not None:
        s += n.x
        n = n.next
    return s


def last(n):
    while n.next is not None:
        n = n.next
    return n


def reverse(head):
    prev = None
    cur = head
    while cur is not None:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


def scale_all(n, k):
    while n:
        n.x *= k
        n = n.next
    return k


def too_far(n, steps):
    s = 0.0
    for i in range(steps):
        s += n.x
        n = n.next
    return s


def find(root, key):
    node = root
    depth = 0
    while node is not None:
        node.hits += 1
        if key == node.key:
            return node
        node = node.left if key < node.key else node.right
        depth += 1
    return None


def rebind(a, b):
    a.pos = b.pos
    a.pos.x = 9.0
    return b.pos.x


def rebind_if(a, b, c):
    if c:
        a.vel = b.vel
    a.vel.y += 1.0
    return b.vel.y


def unlink(a):
    a.next = None
    return a.x


def chain(*xs):
    head = None
    for x in reversed(xs):
        head = Node(x, head)
    return head


def ring():
    n = Node(1.0)
    n.next = Node(2.0, n)
    return n


def bst():
    return Tree(5, Tree(3, Tree(1), Tree(4)), Tree(8, None, Tree(9)))


def bodies():
    return [Body(V(1.0, 2.0), V(3.0, 4.0)), Body(V(5.0, 6.0), V(7.0, 8.0))]


@pytest.mark.parametrize('f,make', [
    (total, lambda: [chain(1.0, 2.0, 3.0)]),
    (total, lambda: [None]),
    (last, lambda: [chain(1.0, 2.0, 3.0)]),
    (reverse, lambda: [chain(1.0, 2.0, 3.0, 4.0)]),
    (reverse, lambda: [None]),
    (scale_all, lambda: [chain(1.0, 2.0), 3.0]),
    (too_far, lambda: [chain(1.0, 2.0), 3]),          # AttributeError: 'NoneType' object has no attribute 'x'
    (find, lambda: [bst(), 4]),
    (find, lambda: [bst(), 7]),
    (rebind, bodies),
    (rebind_if, lambda: bodies() + [True]),
    (rebind_if, lambda: bodies() + [False]),
    (unlink, lambda: [chain(1.0, 2.0)]),
])
def test_references(f, make):
    same(f, make, True)


def test_returned_objects_are_the_objects():
    g = scram(cache=False, fallback=False)(last)
    head = chain(1.0, 2.0, 3.0)
    assert g(head) is head.next.next
    t = bst()
    h = scram(cache=False, fallback=False)(find)
    assert h(t, 4) is t.left.right and t.hits == 1 and t.left.hits == 1 and t.right.hits == 0
    assert h(t, 7) is None


def compares_equal(a, b):
    return a.next == b


def two_classes(n):
    while n is not None:
        n = n.next
    return 0.0


@pytest.mark.parametrize('f,make,why', [
    (compares_equal, lambda: [chain(1.0, 2.0), None], ''),
    (two_classes, lambda: [Node(1.0, Tree(3))], 'one pool, of one class'),
])
def test_stays_python(f, make, why):
    same(f, make, False)
    with pytest.raises(CompileError, match=why):
        scram(cache=False, fallback=False)(f)(*make())


def test_a_cycle_is_each_object_once():
    g = scram(cache=False, fallback=False)(too_far)
    r = ring()
    assert g(r, 7) == too_far(ring(), 7) and r.next.next is r
    assert g.signatures[0][-1].name.startswith('pool Node')
