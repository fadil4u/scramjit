"""Default parameter values: a call leaving some out is bound as Python binds it, then compiled like any other."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import outcome


def scale(x, k=2.0, offset=0, flag=True):
    s = 0.0
    for v in x:
        s += v * k + offset
    return s if flag else -s


def fill(out, value=1.5, out2=np.zeros(3)):     # a default array: the same object every call, as in Python
    for i in range(len(out)):
        out[i] = value
    out2[0] += 1.0
    return out2[0]


def plain(a, b=3):
    return a * b


def strict(f):
    g = scram(fallback=False, cache=False)(f)
    return g


X = np.arange(5.0)


@pytest.mark.parametrize('args, kwargs', [((X,), {}), ((X, 3.0), {}), ((X,), {'offset': 1}), ((X, 1.0, 2), {}),
                                          ((X,), {'flag': False, 'k': 0.5}), ((), {'x': X}), ((X, 1.0, 2, False), {})])
def test_from_python(args, kwargs):
    g = strict(scale)
    assert g(*args, **kwargs) == scale(*args, **kwargs)
    assert g.signatures


@pytest.mark.parametrize('args, kwargs', [((), {}), ((X, 1.0, 2, False, 9), {}), ((X,), {'nope': 1}), ((X, 1.0), {'k': 2.0})])
def test_bad_calls_raise_what_python_raises(args, kwargs):
    g = scram(cache=False)(scale)
    assert outcome(lambda: g(*args, **kwargs), [])[:3] == outcome(lambda: scale(*args, **kwargs), [])[:3]


def test_a_default_array_is_the_same_object():
    g = scram(cache=False)(fill)
    a, b = np.zeros(4), np.zeros(4)
    assert g(a) == 1.0 and g(a) == 2.0          # the default's own memory, kept between calls
    assert (a == 1.5).all()
    assert fill.__defaults__[1][0] == 2.0
    g(b, 7.0)
    assert (b == 7.0).all()


def test_a_changed_default_is_seen():
    def f(x, k=2):
        return x * k
    g = scram(cache=False)(f)
    assert g(3) == 6
    f.__defaults__ = (10,)
    assert g(3) == 30


@scram
def callee(x, k=2.0, n=3):
    return x * k + n


@scram
def caller(x):
    return callee(x) + callee(x, 1.0) + callee(x, n=0) + callee(k=0.5, x=x)


@scram
def callee_with_array(x, a=np.ones(2)):
    return x + a[0]


@scram
def caller_leaving_an_array(x):
    return callee_with_array(x)


def test_from_compiled_code():
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        assert caller(2.0) == caller.__wrapped__(2.0)
    assert caller.signatures


def test_an_array_default_left_out_by_compiled_code_says_so():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert caller_leaving_an_array(1.0) == 2.0
    assert any('whose default is a ndarray' in str(m.message) for m in w), [str(m.message) for m in w]


def test_keyword_only_compiles():
    def f(x, *, k=2):
        return x * k
    g = scram(cache=False)(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert g(3) == 6 and g(3, k=4) == 12
    assert g.signatures
    with pytest.raises(TypeError):
        g(3, 4)
