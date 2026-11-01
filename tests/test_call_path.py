"""Calls after the first take a generated fast path. It must change only speed."""

import threading
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram


def add(a: int, b: int): return a + b
def addf(x: float, y: float): return x + y
def nothing(a): a[0] = 1.0
def first(a): return a[0]
def shape(a): return a.shape[0] * 10 + a.shape[1]
def pair(x: float): return x, x + 1
def flag(a, c: int):
    v = 1.0
    if c:
        v = a[0]
    return v
def div(a: int, b: int): return a // b
def sq(x: float): return __import__('math') and x
def root(a): return a[0] ** 0.5
def tup(t): return t[0] + t[1]
def boolean(x: float): return x > 1.0
def zero_args(): return 42
def many(a, b, c: float, d: int, e): return a[0] + b[0] + c + d + e[0]
def nptype(x): return x * 2


def twice(f, *args):
    """The first call takes the general path, the second the plan."""
    return f(*args), f(*args)


def test_scalars():
    g = scram(add, fallback=False)
    assert twice(g, 2, 3) == (5, 5)
    assert g(-7, 7) == 0
    assert type(g(1, 2)) is int
    h = scram(addf, fallback=False)
    assert twice(h, 1.5, 2.25) == (3.75, 3.75) and type(h(1.0, 2.0)) is float
    assert twice(scram(boolean, fallback=False), 2.0) == (True, True)
    assert scram(boolean, fallback=False)(0.5) is False
    assert twice(scram(zero_args, fallback=False)) == (42, 42)


def test_an_int_past_64_bits_goes_to_python():
    g = scram(add)
    assert g(1, 2) == 3 and g(1, 2) == 3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(2 ** 70, 1) == 2 ** 70 + 1
        assert g(1, -(2 ** 63) - 1) == -(2 ** 63)
    assert g((1 << 63) - 1, 0) == (1 << 63) - 1            # the edges still fit
    assert g(-(1 << 63), 0) == -(1 << 63)
    assert g(1, 2) == 3


def test_arrays_of_other_kinds_are_turned_down_correctly():
    g = scram(first)
    a = np.array([1.5, 2.5])
    assert twice(g, a) == (1.5, 1.5)
    assert type(g(a)) is np.float64
    i = np.array([7, 8], np.int64)
    assert g(i) == 7 and type(g(i)) is np.int64
    assert g(a) == 1.5                                     # and back
    assert g(np.array([9.0, 1.0])[::-1]) == 1.0            # strided: runs as Python, right answer
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(np.array([1.0, 2.0], dtype='>f8')) == 1.0     # byte-swapped
    two = np.array([[3.0, 4.0]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert np.array_equal(g(two), two[0])              # 2-D: a[0] is a row, in Python
    with pytest.raises(IndexError):
        g(np.zeros(0))                                     # empty: no buffer, but still Python's error
    assert g(a) == 1.5


def is_native(g, *args):
    """Whether this call would run compiled, judged by the plan it leaves behind."""
    g(*args)
    return g._plans[tuple(map(type, args))] is not g._fn


def test_an_unusual_first_call_doesnt_send_later_ones_to_python():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g = scram(add)
        assert g(2 ** 70, 1) == 2 ** 70 + 1
        assert g(1, 2) == 3 and is_native(g, 1, 2)
        h = scram(first)
        assert h(np.array([9.0, 1.0])[::-1]) == 1.0        # strided first
        a = np.array([1.5])
        assert h(a) == 1.5 and is_native(h, a)
        k = scram(uses_dict)
        k(1.0)
        assert not is_native(k, 1.0)                       # always Python: straight there


def test_read_only_arrays():
    a = np.array([1.5, 2.5])
    a.flags.writeable = False
    g = scram(first, fallback=False)
    assert twice(g, a) == (1.5, 1.5)
    b = np.array([4.5])
    assert g(b) == 4.5 and g(b) == 4.5 and g(a) == 1.5
    w = scram(nothing)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        c = np.zeros(2)
        w(c)
        w(c)
        assert c[0] == 1.0
        with pytest.raises(ValueError):
            w(a)                                           # writing a read-only array: Python's error


def test_writes_land_in_the_array():
    g = scram(nothing, fallback=False)
    a = np.zeros(3)
    g(a)
    b = np.zeros(3)
    assert g(b) is None and b[0] == 1.0
    assert twice(scram(shape, fallback=False), np.zeros((2, 3))) == (23, 23)
    assert scram(shape, fallback=False)(np.zeros((4, 5))) == 45


def test_tuples_numpy_scalars_and_flags():
    assert twice(scram(pair, fallback=False), 1.5) == ((1.5, 2.5), (1.5, 2.5))
    assert twice(scram(tup, fallback=False), (1, 2)) == (3, 3)
    g = scram(nptype, fallback=False)
    assert twice(g, np.float64(1.5)) == (3.0, 3.0) and type(g(np.float64(1.5))) is np.float64
    assert type(g(1.5)) is float and type(g(np.int64(2))) is np.int64 and g(np.int64(2)) == 4
    f = scram(flag, fallback=False)
    a = np.array([4.0])
    assert type(f(a, 0)) is float and type(f(a, 0)) is float
    assert type(f(a, 1)) is np.float64 and type(f(a, 0)) is float
    assert twice(scram(root, fallback=False), a) == (2.0, 2.0)                  # uses the libm table
    assert twice(scram(many, fallback=False), a, a, 1.0, 2, a) == (15.0, 15.0)


def test_errors_and_recovery():
    g = scram(div, fallback=False)
    assert g(7, 2) == 3 and g(7, 2) == 3
    for _ in range(3):
        with pytest.raises(ZeroDivisionError):
            g(1, 0)
        assert g(9, 3) == 3                                # a failed call leaves nothing behind


def test_contracts_still_apply():
    g = scram(add)
    g(1, 2)
    g(1, 2)
    with pytest.raises(TypeError):
        g(1.5, 2)
    with pytest.raises(TypeError):
        g(1.5, 2)                                          # no plan for a call that fails its contract


def test_keywords_and_wrong_counts():
    g = scram(add)
    g(1, 2)
    assert g(1, b=2) == 3 and g(a=1, b=2) == 3
    with pytest.raises(TypeError):
        g(1)
    with pytest.raises(TypeError):
        g(1, 2, 3)


def test_fallback_goes_straight_to_python():
    def _(): pass
    g = scram(uses_dict)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(1.0) == {1.0: 1} and g(2.0) == {2.0: 1}


def uses_dict(x: float): return {x: 1}


def test_threads_and_reentrancy_keep_results_apart():
    g = scram(add, fallback=False)
    g(0, 0)
    g(0, 0)
    bad = []

    def work(k):
        for i in range(20000):
            if g(i, k) != i + k:
                bad.append((i, k))
    ts = [threading.Thread(target=work, args=(k * 1000,)) for k in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not bad
