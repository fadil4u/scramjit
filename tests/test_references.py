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
    (two_classes, lambda: [Node(1.0, Tree(3))], 'two classes'),     # one name, objects of two classes
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


@dataclasses.dataclass
class Item:
    weight: float
    count: int = 0


@dataclasses.dataclass
class Link:
    item: object
    next: object = None


def heaviest(head):
    best = None
    n = head
    while n is not None:
        n.item.count += 1
        if best is None or n.item.weight > best.weight:
            best = n.item
        n = n.next
    return best


def items():
    shared = Item(2.5)
    return [Link(Item(1.0), Link(shared, Link(Item(0.5), Link(shared))))]


def test_pools_of_two_classes():
    same(heaviest, items, True)
    g = scram(cache=False, fallback=False)(heaviest)
    head = items()[0]
    assert g(head) is head.next.item and head.next.item.count == 2


@dataclasses.dataclass
class Cell:
    v: float
    next: object = None

    def bump(self, k):
        self.v += k
        return self.v

    def ahead(self):
        return 0.0 if self.next is None else self.next.v


@scram(cache=False)
def weight(c):
    return c.v * 2.0


@scram(cache=False)
def length(c):
    k = 0
    while c is not None:
        k += 1
        c = c.next
    return k


def methods_on_references(head):
    s = 0.0
    n = head
    while n is not None:
        s += n.bump(1.0) + n.ahead()
        n = n.next
    return s


def references_given_on(head):
    s = 0.0
    n = head
    while n is not None:
        s += weight(n) + length(n)
        n = n.next
    return s + length(None)


def cells():
    return [Cell(1.0, Cell(2.0, Cell(3.0)))]


@pytest.mark.parametrize('f', [methods_on_references, references_given_on])
def test_references_given_to_compiled_functions(f):
    same(f, cells, True)


def insert(root, key):
    node = root
    while True:
        node.hits += 1
        if key < node.key:
            if node.left is None:
                node.left = Tree(key)
                return root
            node = node.left
        else:
            if node.right is None:
                node.right = Tree(key)
                return root
            node = node.right


def insert_many(root, keys):
    for k in keys:
        insert_into(root, k)
    return root


@scram(cache=False)
def insert_into(root, key):
    return 0


def prepend(head, xs):
    for x in xs:
        head = Node(x, head)
    return head


def grow_many(head, n):
    tail = head
    while tail.next is not None:
        tail = tail.next
    for i in range(n):
        tail.next = Node(float(i))
        tail = tail.next
    return total_of_nodes(head)


def total_of_nodes(n):
    return 0.0


def shape(t):
    return None if t is None else (t.key, t.hits, shape(t.left), shape(t.right))


@pytest.mark.parametrize('key', [0, 2, 6, 10])
def test_new_instances_kept_in_a_tree(key):
    g = scram(cache=False, fallback=False)(insert)
    a, b = bst(), bst()
    assert g(a, key) is a and insert(b, key) is b and shape(a) == shape(b)
    assert type(a.right.right if key == 10 else a) is Tree


def test_new_instances_kept_in_a_list():
    g = scram(cache=False, fallback=False)(prepend)
    head = chain(1.0, 2.0)
    r = g(head, [3.0, 4.0, 5.0])
    assert r == prepend(chain(1.0, 2.0), [3.0, 4.0, 5.0]) and r.next.next.next is head
    assert list(vars(r)) == ['x', 'next']                  # as __init__ assigned them


def grow(head, n):
    tail = head
    while tail.next is not None:
        tail = tail.next
    for i in range(n):
        tail.next = Node(i * 0.5)
        tail = tail.next
    return tail.x


def test_room_runs_out_and_it_runs_again():
    g = scram(cache=False, fallback=False)(grow)
    a, b = chain(1.0, 2.0), chain(1.0, 2.0)
    assert g(a, 200) == grow(b, 200) and a == b             # past the first room: made again with more


def test_made_from_nothing_stays_python():
    same(prepend, lambda: [None, [1.0, 2.0]], False)


def walk_all(heads):
    s = 0.0
    for n in heads:
        while n is not None:
            s += n.x
            n = n.next
    return s


def advance_all(heads):
    for i in range(len(heads)):
        if heads[i] is not None:
            heads[i] = heads[i].next
    return len(heads)


@pytest.mark.parametrize('f,make', [
    (walk_all, lambda: [[chain(1.0, 2.0), None, chain(3.0)]]),
    (advance_all, lambda: [[chain(1.0, 2.0), None, chain(3.0)]]),
    (walk_all, lambda: (lambda c: [[c, c.next, c]])(chain(1.0, 2.0, 3.0))),
])
def test_lists_of_references(f, make):
    same(f, make, True)
