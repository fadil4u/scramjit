"""Whole-array arithmetic in compiled functions: a * 2 + b, np.sqrt(a), with broadcasting. Each result must be numpy's:
the same dtype, shape and bytes, or the same exception with the same message."""

import numpy as np
import pytest

from scramjit import scram
from scramjit._errors import CompileError


def same(f, *args):
    """f compiled (no fallback) gives what f gives in Python, for these arguments."""
    c = scram(f, fallback=False, cache=False)
    try:
        want = f(*args)
    except Exception as e:                                  # noqa: BLE001
        with pytest.raises(type(e)) as got:
            c(*args)
        assert str(got.value) == str(e)
        return
    got = c(*args)
    assert type(got) is type(want)
    if isinstance(want, np.ndarray):
        assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes(), (got, want)
        assert got.flags.c_contiguous
    else:
        assert got == want or (got != got and want != want)


def lin(a, b, k):
    return a * 2.0 + b * k


def mixed(a, b):
    return (a - b) / (a + 1.5) - -a


def funcs(a):
    return np.sqrt(np.abs(a)) + np.exp(a / 10) - np.maximum(a, 0.5) * np.sin(a)


def power(a):
    return a ** 2 + a ** 0.5


def ipower(a):
    return a ** 3 - 2 * a


def add(a, b):
    return a + b


def views(a):
    return a[1:] * 2 - a[:-1]


def rows(m, v):
    return m[0] * v + m[1]


def assign(a, b):
    y = a * b
    y = y + 1
    return y


def into(a, b):
    out = np.zeros(len(a) + 1)
    out[1:] = a * b - 1
    return out


def loop(a, b, n):
    y = a.copy()
    for _ in range(n):
        y = y * 0.5 + b
    return y


def element(a, b, i):
    return a * b[i]


FLOATS = np.array([0.0, -0.0, 1.5, -2.25, 1e300, -1e-310, np.inf, -np.inf, np.nan, 3.0, 7.5, -1.0])


@pytest.mark.parametrize('dtype', ['float64', 'int64', 'float32', 'int32', 'uint8'])
def test_operators_by_dtype(dtype):
    a = (FLOATS if dtype.startswith('float') else np.arange(-6, 6)).astype(dtype) if dtype != 'uint8' else np.arange(12, dtype='uint8')
    b = a[::-1].copy()
    same(lin, a, b, 3)
    same(lin, a, b, 0.5)
    same(add, a, b)
    same(views, a)
    same(assign, a, b)


def test_mixed_dtypes_and_scalars():
    f, i = FLOATS, np.arange(12)
    same(add, f, i)
    same(add, i, f)
    same(add, i, np.arange(12, dtype=np.int32))
    same(add, np.arange(12, dtype=np.float32), i)
    same(lin, i, i, 2.5)
    same(lin, i, f, np.float64(2.0))
    same(mixed, f, f[::-1].copy())
    same(mixed, i, i + 1)                                  # int / float: float, ints / ints: true division


def test_numpy_functions():
    same(funcs, FLOATS)
    same(funcs, np.linspace(-3, 3, 50))
    same(power, np.abs(FLOATS))
    same(ipower, np.arange(-5, 5))


def test_broadcasting_and_its_errors():
    m = np.arange(6.0).reshape(2, 3)
    same(add, m, np.array([1.0, 2.0, 3.0]))
    same(add, np.array([1.0, 2.0, 3.0]), m)
    same(add, m, np.array([[10.0], [20.0]]))
    same(add, np.array([[10.0], [20.0]]), np.array([[1.0, 2.0, 3.0]]))
    same(add, np.ones(1), np.arange(4.0))
    same(add, np.ones(3), np.ones(4))                       # ValueError, numpy's message
    same(add, m, np.ones(2))
    same(add, m, np.ones((3, 2)))
    same(rows, m, np.array([1.0, 0.5, 2.0]))


def test_empty_and_views():
    same(add, np.zeros(0), np.zeros(0))
    same(views, np.arange(1.0))
    same(add, np.arange(10.0)[::2], np.arange(5.0))
    same(add, np.arange(10.0)[::-3], 1.0)


def test_in_statements_and_loops():
    a, b = np.linspace(0, 1, 100), np.linspace(1, 2, 100)
    same(into, a, b)
    same(loop, a, b, 20)
    same(element, a, b, 7)
    same(element, a, b, 100)                               # IndexError, as Python


def test_compiles_without_falling_back():
    for f, args in ((lin, (FLOATS, FLOATS, 2.0)), (funcs, (FLOATS,)), (loop, (FLOATS, FLOATS, 3))):
        scram(f, fallback=False, cache=False)(*args)


def test_what_stays_python():
    m = np.arange(6.0).reshape(2, 3)
    for f, args in ((add, (m.T, m.T)), (add, (np.ones((2, 2, 2)), 1.0)), (ipower_k, (np.arange(4), 2))):
        with pytest.raises(CompileError):
            scram(f, fallback=False, cache=False)(*args)
        assert scram(f, cache=False)(*args).tobytes() == f(*args).tobytes()


def ipower_k(a, k):
    return a ** k                                           # an int array to a power that may be negative
