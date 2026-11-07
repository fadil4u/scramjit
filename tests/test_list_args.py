"""Lists passed in: copied into an array at the call and, if the function writes them, the changed elements copied
back, also when it raises; the same results as Python's, which works on the list itself."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import outcome


def total(xs):
    s = 0.0
    for v in xs:
        s += v
    return s


def scale(xs, k: float):
    for i in range(len(xs)):
        xs[i] = xs[i] * k
    return len(xs)


def raise_partway(xs, stop: int):
    for i in range(len(xs)):
        if i == stop:
            raise_now = xs[len(xs)]          # IndexError, after some writes
        xs[i] += 1
    return xs[0]


def counts(flags):
    n = 0
    for f in flags:
        if f:
            n += 1
    flags[0] = not flags[0]
    return n


def both(a, b):
    for i in range(len(a)):
        a[i] = b[i] * 2
    return a[0] + b[0]


def strict(f, *args):
    g = scram(cache=False)(f)
    plain = [list(a) if isinstance(a, list) else a for a in args]
    mine = [list(a) if isinstance(a, list) else a for a in args]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        got = outcome(g, mine)
    assert got[:3] == outcome(f, plain)[:3], (f.__name__, args)
    for p, m in zip(plain, mine):
        if isinstance(p, list):
            assert [type(x) for x in p] == [type(x) for x in m] and repr(p) == repr(m), (f.__name__, p, m)
    assert g.signatures, (f.__name__, [str(m.message) for m in w])
    return mine


@pytest.mark.parametrize('xs', [[1.0, 2.5, -3.0], [0.5], [float('nan'), -0.0, 1e308]])
def test_floats(xs):
    strict(total, xs)
    strict(scale, xs, 2.0)
    strict(scale, xs, -0.0)


@pytest.mark.parametrize('xs', [[1, 2, 3], [2**62, -5], [0]])
def test_ints(xs):
    strict(raise_partway, xs, 1)
    strict(raise_partway, xs, 9)


def test_bools():
    strict(counts, [True, False, True])


def test_untouched_elements_stay_the_same_objects():
    xs = [1.5, 2.5, 3.5]
    before = [id(x) for x in xs]
    g = scram(cache=False)(scale)
    g(xs, 1.0)                               # nothing changes value
    assert [id(x) for x in xs] == before


def test_two_lists_and_one_twice():
    strict(both, [0, 0, 0], [1, 2, 3])
    a, b = [1, 2, 3], [1, 2, 3]
    g = scram(cache=False)(both)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(a, a) == both(b, b)         # one list under two names runs as Python
    assert a == b


@pytest.mark.parametrize('xs', [[], [1, 2.5], [1, True], [2**70]])
def test_other_lists_stay_python(xs):
    g = scram(cache=False)(total)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert outcome(g, [list(xs)])[:3] == outcome(total, [list(xs)])[:3]
    assert not g.signatures


def test_numpy_scalars_in_a_list():
    """From 0.0.4 a list of numpy scalars passed in is held by reference (tests/test_glists.py)."""
    g = scram(cache=False)(total)
    xs = [np.float64(1.0), np.float64(2.5)]
    assert outcome(g, [list(xs)])[:3] == outcome(total, [list(xs)])[:3] and g.signatures


def test_storing_another_type_stays_python():
    def f(xs):
        xs[0] = 1.5
        return xs[0]
    g = scram(cache=False)(f)
    xs = [1, 2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(xs) == 1.5 and xs == [1.5, 2]
    assert not g.signatures


TABLE = [0.5, 1.5, 2.5]
COUNTS = [0, 0, 0]


@scram
def weigh(x: float):
    s = 0.0
    for i in range(len(TABLE)):
        s += TABLE[i] * x
        COUNTS[i] += 1
    return s


def test_module_level_lists():
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        assert weigh(2.0) == 9.0 and weigh(1.0) == 4.5
    assert COUNTS == [2, 2, 2]
    TABLE[0] = 10.0                           # read at each call, as the other globals
    assert weigh(1.0) == 14.0


def squares(n: int):
    out = [0.0] * n
    for i in range(n):
        out[i] = i * i * 0.5
    return out


def evens(a):
    return [v * 2 for v in a]


def flags(n: int):
    return [i % 3 == 0 for i in range(n)]


def same_list(xs):
    xs[0] += 1
    return xs


def ints_literal(k: int):
    return [k, k + 1, k * 2]


@pytest.mark.parametrize('n', [0, 1, 5])
def test_returning_lists(n):
    strict(squares, n)
    strict(flags, n)
    strict(ints_literal, n)
    if n:                                    # an empty list's element type isn't known: that call runs as Python
        strict(evens, [1.5, -2.0, 0.25][:n])


def test_returning_the_list_passed_in():
    xs = [1, 2, 3]
    g = scram(cache=False)(same_list)
    assert g(xs) is xs and xs == [2, 2, 3]


def test_a_returned_list_is_a_new_one_each_call():
    g = scram(cache=False)(squares)
    a, b = g(3), g(3)
    assert a == b and a is not b and type(a) is list
