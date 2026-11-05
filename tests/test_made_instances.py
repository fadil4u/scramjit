"""Instances made in a compiled function: v = C(...) with a dataclass, or an __init__ that only assigns, becomes a
variable per attribute; methods, passing on, isinstance and returning work. Everything observable must match CPython,
the returned object's attributes, their order and its class included."""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from scramjit import CompileError, FallbackWarning, scram

from test_classes import same


@dataclasses.dataclass
class Vec:
    x: float
    y: float = 0.0

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def scale(self, k):
        self.x *= k
        self.y *= k


@dataclasses.dataclass(frozen=True)
class Frozen:
    a: float
    b: int = 3


@dataclasses.dataclass(slots=True)
class Slotted:
    a: float
    n: int = 0


@dataclasses.dataclass(kw_only=True)
class KwOnly:
    a: float
    b: float = 1.0


@dataclasses.dataclass
class WithPost:
    a: float
    b: float = 0.0

    def __post_init__(self):
        self.b = self.a * 2


@dataclasses.dataclass
class WithFactory:
    a: float
    xs: list = dataclasses.field(default_factory=list)


class Plain:
    def __init__(self, a, b=2.0, *, c=1):
        """Only assigns."""
        self.a = a
        self.b = a * b + c


class Busy:
    def __init__(self, a):
        self.a = a
        print('made one')


class Meta(type):
    def __call__(cls, *args):
        return super().__call__(*args)


class WithMeta(metaclass=Meta):
    def __init__(self, a):
        self.a = a


@scram(cache=False)
def length(v):
    return v.norm()


def walk(n):
    total = 0.0
    for i in range(n):
        v = Vec(math.cos(i), y=math.sin(i))
        v.scale(2.0)
        total += v.norm() + v.x + length(v)
    return total


def make_vec(a, b):
    v = Vec(a, b)
    v.scale(0.5)
    return v


def make_default(a):
    v = Vec(a)
    return v


def make_plain(a):
    p = Plain(a)
    p.b += 1.0
    q = Plain(a, 3.0, c=2)
    return p.b + q.b


def returns_plain(a):
    p = Plain(a, c=5)
    return p


def frozen_read(a):
    f = Frozen(a)
    return f.a + f.b


def frozen_return(a):
    f = Frozen(a, 4)
    return f


def frozen_write(a):
    f = Frozen(a)
    f.a = 2.0
    return f.a


def slotted(a):
    s = Slotted(a)
    s.n += 1
    return s


def kw_only(a):
    k = KwOnly(a=a)
    return k.a + k.b


def post_init(a):
    w = WithPost(a)
    return w


def factory(a):
    w = WithFactory(a)
    return w.a


def busy(a):
    b = Busy(a)
    return b.a


def meta(a):
    m = WithMeta(a)
    return m.a


def too_many(a):
    v = Vec(a, a, a)
    return v.x


def unknown_attr(a):
    v = Vec(a)
    return v.z


def kinds(a):
    v = Vec(a)
    return isinstance(v, Vec), isinstance(v, Plain)


def maybe_unbound(n):
    for i in range(n):
        v = Vec(float(i))
    return v.x


def swap_both(a, b):
    v = Vec(a)
    w = Vec(b)
    v.x, w.x = w.x, v.x
    return v.x - w.x


def other_use(a):
    v = Vec(a)
    vs = [v]
    return len(vs)


def int_then_float(n):
    v = Vec(n)
    v.x = v.x + 0.5
    return v


@pytest.mark.parametrize('f,make,compiled', [
    (walk, lambda: [20], True),
    (make_vec, lambda: [3.0, 4.0], True),
    (make_vec, lambda: [np.float64(3.0), 4.0], True),
    (make_default, lambda: [1.5], True),
    (make_default, lambda: [2], True),
    (make_plain, lambda: [1.5], True),
    (frozen_read, lambda: [1.5], True),
    (frozen_return, lambda: [1.5], True),
    (frozen_write, lambda: [1.5], False),
    (slotted, lambda: [1.5], True),
    (kw_only, lambda: [1.5], True),
    (post_init, lambda: [1.5], True),
    (factory, lambda: [1.5], False),
    (busy, lambda: [1.5], False),
    (meta, lambda: [1.5], False),
    (too_many, lambda: [1.5], False),
    (unknown_attr, lambda: [1.5], False),
    (kinds, lambda: [1.5], True),
    (maybe_unbound, lambda: [3], False),
    (swap_both, lambda: [1.0, 2.0], True),
    (other_use, lambda: [1.0], False),
])
def test_made_instances(f, make, compiled):
    same(f, make, compiled)


def test_returned_objects_are_what_python_makes():
    for f, args in ((make_vec, (3.0, 4.0)), (returns_plain, (1.5,)), (frozen_return, (2.0,)), (slotted, (1.0,)), (post_init, (3.0,))):
        g = scram(cache=False)(f)
        a, b = f(*args), g(*args)
        assert type(a) is type(b) and g.signatures, f.__name__
        held = getattr(a, '__dict__', None)
        if held is not None:
            assert list(vars(a).items()) == list(vars(b).items()), f.__name__
        assert a == b or vars(a) == vars(b), f.__name__


@dataclasses.dataclass
class Box:
    w: float

    def area(self):
        return self.w * self.w


def box_area(a):
    b = Box(a)
    return b.area()


def test_the_class_changing_later():
    g = scram(cache=False)(box_area)
    assert g(2.0) == 4.0 and g(2.0) == 4.0 and g.signatures
    old = Box.area
    Box.area = lambda self: -1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FallbackWarning)
            assert g(2.0) == -1.0
    finally:
        Box.area = old
    assert g(2.0) == 4.0


def test_reasons():
    with pytest.raises(CompileError, match='default_factory'):
        scram(cache=False, fallback=False)(factory)(1.0)
    with pytest.raises(CompileError, match='does more than assign'):
        scram(cache=False, fallback=False)(busy)(1.0)
