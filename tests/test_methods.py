"""Calling a method of an instance from compiled code: a plain function on the class, or a @scram one, compiled as a
callee with the instance's attributes as self. Everything observable must match CPython, as in test_classes."""

import dataclasses
import math
import traceback
import warnings

import numpy as np
import pytest

from scramjit import CompileError, FallbackWarning, scram

from test_classes import same

X = np.arange(1.0, 6.0)
GAIN = 2.0


@dataclasses.dataclass
class V:
    x: float
    y: float
    n: int = 0

    def norm2(self):
        return self.x * self.x + self.y * self.y

    def scale(self, k):
        self.x *= k
        self.y *= k
        self.n += 1

    def unit(self):
        r = math.sqrt(self.norm2())
        self.scale(1.0 / r)
        return r

    def shifted(self, dx=1.0, *, unused=None):
        return self.x + dx

    def shifted_k(self, dx=1.0):
        return self.x + dx

    def bump_then_index(self, a, i):
        self.n += 1
        self.x = 99.0
        return a[i]

    def get_and_inc(self):
        self.n += 1
        return self.n

    def with_global(self):
        return self.x * GAIN

    def to_float(self):
        self.n = 0.5

    def again(self, k):
        return 0.0 if k == 0 else self.again(k - 1) + 1.0

    def positive(self):
        return self.x > 0.0

    def dec(self):
        self.x -= 1.0
        return self.x > 0.0

    @staticmethod
    def static_one():
        return 1.0

    @classmethod
    def class_one(cls):
        return 1.0

    @property
    def doubled(self):
        return self.x * 2

    @scram(cache=False)
    def compiled_norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)


@dataclasses.dataclass
class Holder:
    a: np.ndarray
    k: float = 2.0

    def scale_all(self):
        for i in range(len(self.a)):
            self.a[i] *= self.k

    def view(self):
        return self.a[1:]

    def total(self):
        s = 0.0
        for v in self.a:
            s += v
        return s


class Sub(V):
    def norm2(self):
        return 1.0


def run_unit(p, steps):
    s = 0.0
    for _ in range(steps):
        s += p.unit()
        p.x += 1.0
    return s


def calls_with_defaults(p):
    return p.shifted_k() + p.shifted_k(2.0) + p.shifted_k(dx=3.0)


def kw_only(p):
    return p.shifted(unused=None)


def raises_in_method(p, a):
    p.n = 5
    return p.bump_then_index(a, 10)


def order_left(p):
    return p.n * 10 + p.get_and_inc()


def order_right(p):
    return p.get_and_inc() * 10 + p.n


def reads_changed_in_loop(p, k):
    s = 0
    for _ in range(k):
        s += p.n
        p.get_and_inc()
    return s


def reads_global(p):
    return p.with_global()


def changes_type(p):
    p.to_float()
    return p.n


def recursive(p):
    return p.again(3)


def in_while_test(p):
    k = 0
    while p.positive() and k < 3:
        k += 1
    return k


def writes_in_while_test(p):
    k = 0
    while p.dec():
        k += 1
    return k


def writes_in_comprehension(p):
    return sum([p.get_and_inc() for _ in range(3)], 0)


def static(p):
    return p.static_one()


def classmethod_(p):
    return p.class_one()


def prop(p):
    return p.doubled


def bound(p):
    f = p.norm2
    return f()


def compiled_method(p):
    return p.compiled_norm() + 1.0


def array_methods(h):
    h.scale_all()
    return h.total() + h.view()[0]


def view_back(h):
    return h.view()


def two_objects(a, b):
    return a.norm2() + b.norm2()


def passes_other(a, b):
    return a.shifted_k(b.x)


@pytest.mark.parametrize('f,make,compiled', [
    (run_unit, lambda: [V(3.0, 4.0), 5], True),
    (run_unit, lambda: [V(0.0, 0.0), 1], True),            # ZeroDivisionError inside a method two calls down
    (calls_with_defaults, lambda: [V(1.0, 2.0)], True),
    (kw_only, lambda: [V(1.0, 2.0)], False),
    (raises_in_method, lambda: [V(1.0, 2.0), X.copy()], True),
    (order_left, lambda: [V(1.0, 2.0, 3)], True),
    (order_right, lambda: [V(1.0, 2.0, 3)], True),
    (reads_changed_in_loop, lambda: [V(1.0, 2.0, 3), 4], True),
    (reads_global, lambda: [V(1.0, 2.0)], False),
    (changes_type, lambda: [V(1.0, 2.0)], False),
    (recursive, lambda: [V(1.0, 2.0)], False),
    (in_while_test, lambda: [V(1.0, 2.0)], True),
    (writes_in_while_test, lambda: [V(3.0, 2.0)], False),
    (writes_in_comprehension, lambda: [V(3.0, 2.0)], False),
    (static, lambda: [V(1.0, 2.0)], False),
    (classmethod_, lambda: [V(1.0, 2.0)], False),
    (prop, lambda: [V(1.0, 2.0)], False),
    (bound, lambda: [V(1.0, 2.0)], False),
    (compiled_method, lambda: [V(3.0, 4.0)], True),
    (run_unit, lambda: [Sub(3.0, 4.0), 2], True),           # the subclass's norm2
    (array_methods, lambda: [Holder(X.copy())], True),
    (two_objects, lambda: [V(1.0, 2.0), V(3.0, 4.0)], True),
    (passes_other, lambda: [V(1.0, 2.0), V(3.0, 4.0)], True),
])
def test_method_calls(f, make, compiled):
    same(f, make, compiled)


def test_returned_view_is_of_the_attribute():
    h = Holder(X.copy())
    v = scram(cache=False)(view_back)(h)
    v[0] = -7.0
    assert h.a[1] == -7.0


def test_writes_before_an_exception_stay_and_the_traceback_has_the_method():
    g = scram(cache=False)(raises_in_method)
    p = V(1.0, 2.0)
    with pytest.raises(IndexError) as info:
        g(p, X)
    assert (p.n, p.x) == (6, 99.0) and g.signatures
    frames = [f for f in traceback.extract_tb(info.value.__traceback__) if f.name in ('raises_in_method', 'bump_then_index')]
    assert [f.name for f in frames] == ['raises_in_method', 'bump_then_index']
    assert 'return a[i]' in frames[-1].line and 'bump_then_index(a, 10)' in frames[0].line


def test_method_on_the_instance_or_replaced_later():
    g = scram(cache=False)(lambda_free_norm)
    p = V(3.0, 4.0)
    assert g(p) == 25.0 and g(p) == 25.0 and g.signatures      # second call through the call plan
    p.norm2 = lambda: 7.0                                         # the instance's own now
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(p) == 7.0
    del p.norm2
    assert g(p) == 25.0

    class W(V):
        pass
    w = W(3.0, 4.0)
    assert g(w) == 25.0 and g(w) == 25.0
    W.norm2 = lambda self: 1.5                                   # the class changes after a plan was made
    assert g(w) == 1.5 or not g.signatures
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(w) == 1.5


def lambda_free_norm(p):
    return p.norm2()


@dataclasses.dataclass
class Counter:
    count: int = 0

    def add(self, k):
        self.count += k

    @scram(cache=False)
    def add_many(self, n):
        for i in range(n):
            self.add(i)
        return self.count


COUNTER = Counter()


@scram(cache=False)
def bump_global(n):
    for _ in range(n):
        COUNTER.add(2)
    return COUNTER.count


def test_scram_method_calling_a_method_and_a_global_instance():
    c = Counter()
    assert c.add_many(10) == 45 and c.count == 45 and Counter.add_many.signatures
    assert c.add_many(1) == 45
    COUNTER.count = 1
    assert bump_global(3) == 7 and COUNTER.count == 7 and bump_global.signatures


def test_reasons():
    with pytest.raises(CompileError, match='not a plain method'):
        scram(cache=False, fallback=False)(static)(V(1.0, 2.0))
    with pytest.raises(CompileError, match='V.norm2 is a method'):
        scram(cache=False, fallback=False)(bound)(V(1.0, 2.0))
    with pytest.raises(CompileError, match='a test'):
        scram(cache=False, fallback=False)(writes_in_while_test)(V(3.0, 2.0))
    with pytest.raises(CompileError, match='an int at some times and a float'):
        scram(cache=False, fallback=False)(changes_type)(V(1.0, 2.0))


def loop_with_python_around(p, k):
    label = str(k)
    for _ in range(k):
        p.scale(0.5)
    return label, p.x


def test_lifted_loop_calls_methods():
    g = scram(cache=False)(loop_with_python_around)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        p = V(8.0, 4.0)
        assert g(p, 3) == ('3', 1.0) and (p.y, p.n) == (0.5, 3)
    assert g._native_loops()
