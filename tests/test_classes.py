"""Instances: p.a on an object whose attributes run no code compiles as a variable. Everything observable must match
CPython: results, the object afterwards, exceptions and how far they got, and any code of the class that should run."""

import copy
import dataclasses
import enum
import types
import typing
import warnings
from collections import namedtuple

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

X = np.arange(1.0, 6.0)


class Plain:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f'Plain({self.__dict__})'


class Slotted:
    __slots__ = ('a', 'b')

    def __init__(self, a=None, b=None):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b


@dataclasses.dataclass
class Body:
    x: float
    v: float
    n: int = 0


@dataclasses.dataclass(frozen=True)
class Frozen:
    a: float
    b: int = 1


@dataclasses.dataclass(slots=True)
class SlotBody:
    x: float
    n: int = 0


class Point(typing.NamedTuple):
    x: float
    y: float


Pair = namedtuple('Pair', 'a b')


class WithClassAttr:
    k = 3.0
    count = 0

    def __init__(self, a):
        self.a = a


class Child(WithClassAttr):
    pass


class Logged:
    def __init__(self):
        object.__setattr__(self, 'a', 1.0)
        object.__setattr__(self, 'log', [])

    def __setattr__(self, name, value):
        self.log.append(name)
        object.__setattr__(self, name, value)


class Counting:
    reads = 0

    def __init__(self):
        self.a = 2.0

    def __getattribute__(self, name):
        if name == 'a':            # isinstance asks for __class__, for any object passed to any function
            type(self).reads += 1
        return object.__getattribute__(self, name)


class Lazy:
    def __init__(self):
        self.a = 2.0

    def __getattr__(self, name):
        return 7.5


class Prop:
    def __init__(self):
        self._a = 2.0

    @property
    def a(self):
        return self._a * 10


class Method:
    def __init__(self):
        self.a = 2.0

    def a2(self):
        return self.a * 2


class Descriptor:
    def __get__(self, obj, cls):
        return 11.0

    def __set__(self, obj, v):
        obj.__dict__['seen'] = v


class HasDescriptor:
    a = Descriptor()

    def __init__(self):
        self.__dict__['a'] = 1.0


class Color(enum.Enum):
    RED = 1
    BLUE = 2


def snapshot(a):
    if isinstance(a, np.ndarray):
        return ('array', a.tobytes(), a.dtype.str)
    if isinstance(a, tuple):
        return (type(a).__name__, [snapshot(v) for v in a])
    if isinstance(a, (int, float, bool, np.generic)) or a is None:
        return _value(a)
    if isinstance(a, list):
        return [snapshot(v) for v in a]
    if isinstance(a, enum.Enum) or isinstance(a, str):
        return repr(a)
    state = {}
    if hasattr(a, '__dict__'):
        state.update(object.__getattribute__(a, '__dict__'))
    for k in type(a).__mro__:
        slots = k.__dict__.get('__slots__', ())
        for s in (slots,) if isinstance(slots, str) else slots:
            if hasattr(a, s):
                state[s] = getattr(a, s)
    return (type(a).__name__, [(k, snapshot(v)) for k, v in state.items()])    # in order: a new key shows


def comparable(o):
    if o[0] == 'returns' and o[1][0] == 'tuple':
        return ('returns', [_value(x) for x in o[1][1]])
    return o


def same(f, make, compiled=None, **options):
    """f on fresh arguments, plain and compiled, twice; make() returns the arguments as a list."""
    g = scram(cache=False, **options)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for _ in range(2):
            plain, native = make(), make()
            want, got = comparable(outcome(f, plain)), comparable(outcome(g, native))
            assert want[:4] == got[:4], (f.__name__, want, got)
            for p, c in zip(plain, native):
                assert snapshot(p) == snapshot(c), f.__name__
    if compiled is not None:
        assert bool(g.signatures) == compiled, (f.__name__, g._why)
    return g


def reads(p, x):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * p.k + p.b
    return s


def writes(p, x):
    for v in x:
        p.total += v
        p.count += 1
    return p.count


def overwrites(p):
    p.a = 2
    p.b = 2.5
    return p.a + p.b


def int_to_float(p):
    p.a = p.a + 0.5
    return p.a


def conditional_write(p):
    if p.a > 100.0:
        p.b = 1.0
    return p.a


def array_read(p):
    s = 0.0
    for i in range(p.x.shape[0]):
        s += p.x[i]
    return s


def array_write(p):
    for i in range(len(p.x)):
        p.x[i] = p.x[i] * p.k


def array_rebind(p):
    p.x = p.y
    return p.x[0]


def array_return(p):
    return p.x


def view_return(p):
    return p.x[1:3]


def tuple_read(p):
    a, b = p.t
    return a * b


def tuple_write(p):
    p.t = (1.0, 2.0)
    return p.a


def enum_read(p):
    return 1 if p.c == Color.RED else 2


def raises_after_store(p, x):
    p.a = 5.0
    p.b = x[100]
    return p.a


def no_attribute(p):
    return p.missing + 1.0


def adds_attribute(p):
    p.new = 1.0
    return p.new


def passes_on(p):
    return len(repr(p))


def rebinds(p):
    p = 3.0
    return p


def deletes(p):
    del p.a
    return 1


def has(p):
    return hasattr(p, 'a')


def two(p, q):
    p.a = 2.0
    q.a = q.a + 1.0
    return p.a


def in_comprehension(p, x):
    return sum((v * p.k for v in x), 0.0)


def only_a(p):
    return p.a * 2.0


def writes_a(p):
    p.a = p.a + 1.0
    return p.a


def calls_method(p):
    return p.a2() + 1.0


def nested(p):
    return p.inner.a + 1.0


def t_args(*a):
    return lambda: [copy.deepcopy(v) for v in a]


@pytest.mark.parametrize('f,make,compiled', [
    (reads, lambda: [Plain(k=2.0, b=1, other='anything'), X.copy()], True),
    (reads, lambda: [Plain(k=2, b=1.5), X.copy()], True),
    (reads, lambda: [Plain(k=np.float64(2.5), b=np.int64(1)), X.copy()], True),
    (reads, lambda: [Plain(k=True, b=False), X.copy()], True),
    (reads, lambda: [Plain(k=2.0, b=1 << 70), X.copy()], False),
    (reads, lambda: [Plain(k='s', b=1), X.copy()], False),
    (writes, lambda: [Plain(total=0.0, count=0, label='x'), X.copy()], True),
    (writes, lambda: [Body(0.0, 0.0), np.empty(0)], False),
    (overwrites, lambda: [Plain(a=1, b=1.0)], True),
    (int_to_float, lambda: [Plain(a=1)], None),
    (conditional_write, lambda: [Plain(a=1.0, b=7.0, c=0)], True),
    (conditional_write, lambda: [Plain(a=1.0, b=-0.0)], True),
    (conditional_write, lambda: [Plain(a=1.0, b=float('nan'))], True),
    (array_read, lambda: [Plain(x=X.copy())], True),
    (array_write, lambda: [Plain(x=X.copy(), k=3.0)], True),
    (array_rebind, lambda: [Plain(x=X.copy(), y=X * 2)], False),
    (array_return, lambda: [Plain(x=X.copy())], True),
    (view_return, lambda: [Plain(x=X.copy())], True),
    (tuple_read, lambda: [Plain(t=(2.0, 3.5))], True),
    (tuple_write, lambda: [Plain(t=(2.0, 3.5), a=1.0)], False),
    (enum_read, lambda: [Plain(c=Color.BLUE)], True),
    (raises_after_store, lambda: [Plain(a=1.0, b=2.0), X.copy()], True),
    (no_attribute, lambda: [Plain(a=1.0)], False),
    (adds_attribute, lambda: [Plain(a=1.0)], False),
    (passes_on, lambda: [Plain(a=1.0)], None),
    (rebinds, lambda: [Plain(a=1.0)], None),
    (deletes, lambda: [Plain(a=1.0)], False),
    (has, lambda: [Plain(a=1.0)], False),
    (in_comprehension, lambda: [Plain(k=2.0), X.copy()], True),
])
def test_plain_instances(f, make, compiled):
    same(f, make, compiled)


@pytest.mark.parametrize('f,make,compiled', [
    (only_a, lambda: [Slotted(a=2.0)], True),                       # __slots__
    (writes_a, lambda: [Slotted(a=2.0)], True),
    (only_a, lambda: [Slotted(b=2.0)], False),                      # an unset slot: Python's AttributeError
    (only_a, lambda: [types.SimpleNamespace(a=2.5)], True),
    (writes_a, lambda: [types.SimpleNamespace(a=2.5)], True),
    (only_a, lambda: [Pair(2.0, 'text')], True),                    # namedtuple fields
    (writes_a, lambda: [Pair(2.0, 'text')], False),                 # AttributeError, before anything else
    (only_a, lambda: [Frozen(3.0)], True),
    (writes_a, lambda: [Frozen(3.0)], False),                       # FrozenInstanceError
    (only_a, lambda: [WithClassAttr(4.0)], True),
    (only_a, lambda: [Child(4.0)], True),
    (only_a, lambda: [Logged()], True),                             # __setattr__ doesn't matter to a read
    (writes_a, lambda: [Logged()], False),                          # and must run for a write
    (only_a, lambda: [Lazy()], False),                              # __getattr__
    (only_a, lambda: [Prop()], False),
    (calls_method, lambda: [Method()], True),                      # a plain method (test_methods.py)
    (only_a, lambda: [HasDescriptor()], False),                     # a data descriptor beats the instance's dict
    (nested, lambda: [Plain(inner=Plain(a=1.0))], True),            # an object in an attribute (test_nested_instances.py)
    (only_a, lambda: [3.5 + 1j], False),
    (only_a, lambda: ['text'], False),
])
def test_kinds_of_object(f, make, compiled):
    same(f, make, compiled)


def test_getattribute_runs():
    Counting.reads = 0
    g = scram(cache=False)(only_a)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(Counting()) == 4.0
    assert not g.signatures and Counting.reads == 1


def k_attr(p):
    return p.k * 2.0


def count_write(p):
    p.count = p.count + 1
    return p.count


def test_class_attributes():
    assert same(k_attr, lambda: [WithClassAttr(1.0)], True)
    inst = WithClassAttr(1.0)
    inst.k = 5.0                                    # the instance's comes first
    assert scram(cache=False)(k_attr)(inst) == 10.0
    same(count_write, lambda: [WithClassAttr(1.0)], False)    # would add it to the instance, whichever way the code goes
    has_it = WithClassAttr(1.0)
    has_it.count = 4
    g = scram(cache=False)(count_write)
    assert g(has_it) == 5 and has_it.count == 5 and g.signatures


def test_order_of_a_new_attribute_kept():
    p = Plain(a=1.0, b=2.0)
    assert scram(cache=False)(only_a)(p) == 2.0
    assert list(vars(p)) == ['a', 'b']


def test_one_object_under_two_names():
    p = Plain(a=1.0)
    same(two, lambda: (lambda o: [o, o])(Plain(a=1.0)), False)
    g = scram(cache=False)(two)
    r = Plain(a=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(r, r) == 3.0
    assert r.a == 3.0
    other = Plain(a=10.0)
    assert g(p, other) == 2.0 and other.a == 11.0 and g.signatures


def dict_and_object(p, d):
    p.a = 2.0
    d['a'] = d['a'] + 1.0
    return p.a


def test_an_object_and_its_own_dict():
    p = Plain(a=1.0)
    g = scram(cache=False)(dict_and_object)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(p, vars(p)) == 3.0
    assert p.a == 3.0 and not g.signatures
    q = Plain(a=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g(q, vars(Plain(a=4.0)))
        assert q.a == 2.0 and g.signatures


def test_returned_array_is_the_attribute():
    p = Plain(x=X.copy())
    g = scram(cache=False)(array_return)
    assert g(p) is p.x and g.signatures
    v = scram(cache=False)(view_return)(p)
    v[0] = -1.0
    assert p.x[1] == -1.0


def test_values_and_classes_changing_between_calls():
    class Mutable:
        def __init__(self):
            self.a = 1.0

    g = scram(cache=False)(only_a)
    m = Mutable()
    assert g(m) == 2.0 and len(g.signatures) == 1
    assert g(m) == 2.0
    m.a = 4.0
    assert g(m) == 8.0
    m.a = 4                           # a new type is a new signature
    assert g(m) == 8.0 and len(g.signatures) == 2
    m.a = 'x'
    with warnings.catch_warnings(), pytest.raises(TypeError):
        warnings.simplefilter('ignore', FallbackWarning)
        g(m)
    m.a = 3.0
    assert g(m) == 6.0
    Mutable.a = property(lambda self: 100.0)        # the class changes: a property now wins over the instance's value
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(m) == 200.0
    del Mutable.a
    assert g(m) == 6.0
    Mutable.__getattribute__ = lambda self, name: 50.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(m) == 100.0


def test_class_rearranged_after_a_plan():
    class Base:
        pass

    class Other:
        a = property(lambda self: 100.0)

    class Mutable(Base):
        pass

    g = scram(cache=False)(only_a)
    m = Mutable()
    m.a = 1.5
    assert g(m) == 3.0 and g(m) == 3.0
    Mutable.__bases__ = (Other,)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(m) == 200.0
    Mutable.__bases__ = (Base,)
    assert g(m) == 3.0
    Base.a = 8.0                       # the instance's still comes first
    assert g(m) == 3.0
    del m.a                            # and now the class's
    assert g(m) == 16.0


def test_written_attribute_gone_from_the_instance_after_a_plan():
    class Mutable:
        a = 1.0

    def f(p, go):
        if go:
            p.a = 2.0
        return p.a

    g = scram(cache=False)(f)
    m = Mutable()
    m.a = 5.0
    assert g(m, False) == 5.0 and g(m, False) == 5.0 and g.signatures
    del m.a
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(m, False) == 1.0
    assert vars(m) == {}               # nothing assigned, so nothing added


def test_setattr_added_to_the_class_later():
    log = []

    class Mutable:
        def __init__(self):
            self.a = 1.0

    g = scram(cache=False)(writes_a)
    m = Mutable()
    assert g(m) == 2.0 and g.signatures
    Mutable.__setattr__ = lambda self, name, v: (log.append(name), object.__setattr__(self, name, v))[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(m) == 3.0
    assert log == ['a']


class Particle:
    def __init__(self, x, v):
        self.x, self.v = x, v
        self.steps = 0

    @scram(cache=False)
    def advance(self, dt, n):
        for _ in range(n):
            self.x += self.v * dt
            self.steps += 1
        return self.x

    @scram(cache=False)
    def tell(self, dt=2.0):
        return self.x * dt

    @scram(cache=False)
    def calls_python(self):
        return str(self.describe())

    def describe(self):
        return self.x


def test_methods():
    p = Particle(1.0, 2.0)
    assert p.advance(0.5, 4) == 5.0 and (p.x, p.steps) == (5.0, 4)
    assert Particle.advance.signatures
    assert p.tell() == 10.0 and p.tell(dt=1.0) == 5.0 and Particle.tell.signatures
    assert Particle.advance(p, 0.5, 1) == 6.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert p.calls_python() == '6.0'
    assert not Particle.calls_python.signatures


@dataclasses.dataclass
class Settings:
    gain: float = 2.0
    total: float = 0.0


SETTINGS = Settings()
LAZY = Lazy()


@scram(cache=False)
def reads_global(x):
    s = 0.0
    for v in x:
        s += v * SETTINGS.gain
    return s


@scram(cache=False)
def writes_global(x):
    for v in x:
        SETTINGS.total += v
    return SETTINGS.total


@scram(cache=False)
def reads_python_global():
    return LAZY.a + LAZY.anything


@scram(cache=False)
def callee_reads_global():
    return reads_global(X)


def test_instance_at_module_level():
    assert reads_global(X) == 30.0 and reads_global.signatures
    SETTINGS.gain = 3.0                # read at each call
    assert reads_global(X) == 45.0
    assert len(reads_global.signatures) == 1
    SETTINGS.gain = 3
    assert reads_global(X) == 45.0
    SETTINGS.gain = 2.0
    SETTINGS.total = 1.0
    assert writes_global(X) == 16.0 and SETTINGS.total == 16.0 and writes_global.signatures
    SETTINGS.total = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert reads_python_global() == 2.0 + 7.5
        assert callee_reads_global() == 30.0


def loop_with_python_around(p, x):
    label = str(len(x))
    for v in x:
        p.total += v * p.k
    return label, p.total


def test_lifted_loop():
    g = scram(cache=False)(loop_with_python_around)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        p = Plain(total=0.0, k=2.0)
        assert g(p, X) == ('5', 30.0) and p.total == 30.0
    assert g._native_loops()


def test_reason_says_what_is_wrong():
    from scramjit import CompileError
    for make, text in ((lambda: Prop(), 'descriptor'), (lambda: Plain(b=1.0), 'no attribute a'), (lambda: Lazy(), '__getattr__'),
                       (lambda: Plain(a='s'), 'a is a str')):
        with pytest.raises(CompileError, match=text):
            scram(cache=False, fallback=False)(only_a)(make())
    with pytest.raises(CompileError, match='__setattr__'):
        scram(cache=False, fallback=False)(writes_a)(Frozen(1.0))
    with pytest.raises(CompileError, match='only p.attribute compiles'):
        scram(cache=False, fallback=False)(mixed_use)(Plain(a=1.0))


def mixed_use(p):
    if p.a > 0:
        q = p                          # not at the top of the body: q might not be p
    return p.a


def test_private_names_stay_python():
    class Hidden:
        def __init__(self):
            self.__a = 2.0
            setattr(self, '__a', 'the wrong one')

        @scram(cache=False)
        def read(self):
            return self.__a * 2.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert Hidden().read() == 4.0


def test_boundscheck_off_and_parity_off_keep_attributes():
    def f(p, x):
        p.t = 0.0
        for i in range(len(x)):
            p.t += x[i]
        return p.t
    for options in ({'boundscheck': False}, {'parity': False}):
        p = Plain(t=9.0)
        assert scram(cache=False, **options)(f)(p, X) == 15.0 and p.t == 15.0


def swaps(p, q):
    p.a, q.a = q.a, p.a


def test_swap_between_two_objects():
    g = scram(cache=False)(swaps)
    a, b = Plain(a=1.0), Plain(a=7.0)
    g(a, b)
    assert (a.a, b.a) == (7.0, 1.0) and g.signatures
    g(a, b)                            # now through the call plan
    assert (a.a, b.a) == (1.0, 7.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g(a, a)                        # the plan turns down one object under two names
    assert a.a == 1.0


def stores_by_loop_and_unpacking(p):
    for p.i in range(3):
        pass
    p.a, p.b = 1.5, p.i
    return p.a


def test_other_ways_to_store():
    same(stores_by_loop_and_unpacking, lambda: [Plain(i=0, a=0.0, b=0)], True)


def shadowed(p):
    f = lambda p: p.a
    return f(p)


def shadowed_in_comprehension(p, x):
    return sum([p.a for p in x], 0.0)


def test_another_variable_with_the_same_name():
    same(shadowed, lambda: [Plain(a=2.0)], False)
    same(shadowed_in_comprehension, lambda: [Plain(a=2.0), [Plain(a=5.0)]], False)


@scram(cache=False)
def callee_of_shared(x):
    return x * SETTINGS.gain


@scram(cache=False)
def caller_of_shared(x):
    return callee_of_shared(x) + SETTINGS.gain


def test_caller_and_callee_read_one_instance():
    SETTINGS.gain = 2.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert caller_of_shared(3.0) == 8.0
