"""numpy functions on single numbers: same bits and same types as numpy, whatever the argument, and anything
whose result depends on the platform or the numpy version stays Python."""

import math
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

np.seterr(all='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 2.5, -2.5, 3.5, 1e-300, -1e-300, 1e300, -1e300, 5e-324, -5e-324, 1.7976931348623157e308,
          math.inf, -math.inf, math.nan, -math.nan, math.pi, 1e22, 710.0, -745.2, 0.1, 1.5, -0.4, 100.0]
INTS = [0, 1, -1, 2, 7, -7, 100, 2**53 + 1, 2**62, -2**63, 2**63 - 1]
SAMPLE = np.random.default_rng(9).uniform(-20, 20, 1500)


def make(name, count=1):
    """A function calling np.<name>, written to a file's worth of source: scramjit reads source."""
    args = ', '.join('abc'[:count])
    src = f'def f({args}):\n    return np.{name}({args})\n'
    space = {'np': np}
    exec(src, space)
    from scramjit._dispatch import Function
    import ast
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys = {}
    g._first = 1
    return space['f'], g


def check(name, *args, compiled=None):
    f, g = make(name, len(args))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(2):
            want, got = outcome(f, list(args)), outcome(g, list(args))
            assert want[:3] == got[:3], (name, args, want, got)
    if compiled is not None:
        assert bool(g.signatures) == compiled, (name, args, g._why)


ONE = ['sqrt', 'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh',
       'arcsinh', 'arccosh', 'arctanh', 'cbrt', 'fabs', 'rint', 'floor', 'ceil', 'trunc', 'abs', 'absolute', 'sign', 'square', 'isnan', 'isinf',
       'isfinite', 'signbit']
TWO = ['arctan2', 'hypot', 'fmod', 'copysign', 'minimum', 'maximum', 'fmin', 'fmax']


@pytest.mark.parametrize('name', ONE)
def test_floats_keep_their_bits(name):
    f, g = make(name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for v in FLOATS:
            for x in (v, np.float64(v)):
                assert _value(f(x)) == _value(g(x)), (name, x)
        for v in SAMPLE[:300]:
            assert _value(f(float(v))) == _value(g(float(v))), (name, v)
    assert g.signatures


@pytest.mark.parametrize('name', ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'arcsin', 'log1p', 'cbrt', 'expm1', 'arctan', 'sinh', 'cosh'])
def test_a_lot_of_values_in_a_loop(name):
    src = f'def f(x, y):\n    for i in range(len(x)):\n        y[i] = np.{name}(x[i])\n'
    space = {'np': np}
    exec(src, space)
    import ast
    from scramjit._dispatch import Function
    g = Function(space['f'], fallback=False, cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys, g._first = {}, 1
    x = np.concatenate([SAMPLE, np.array(FLOATS), 10.0 ** np.random.default_rng(1).uniform(-300, 300, 1500)])
    a, b = np.zeros(len(x)), np.zeros(len(x))
    space['f'](x, a)
    g(x, b)
    same = (a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a) & np.isnan(b))
    assert same.all(), (name, x[~same][:4], a[~same][:4], b[~same][:4])


@pytest.mark.parametrize('name', TWO)
def test_two_arguments(name):
    f, g = make(name, 2)
    values = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -2.5, 3.0, math.inf, -math.inf, math.nan, -math.nan, 1e300, 5e-324]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for a in values:
            for b in values:
                for x, y in ((a, b), (np.float64(a), b), (a, np.float64(b))):
                    assert _value(f(x, y)) == _value(g(x, y)), (name, x, y)
        for a, b in ((3.0, 2), (3, 2.0), (np.int64(3), 2.0), (2.0, np.int64(5))):
            assert _value(f(a, b)) == _value(g(a, b)), (name, a, b)
    assert g.signatures


@pytest.mark.parametrize('name', [n for n in ONE if n not in ('floor', 'ceil', 'trunc', 'abs', 'absolute', 'sign', 'square')])
def test_ints_become_floats(name):
    for v in INTS:
        check(name, v, compiled=True)
        check(name, np.int64(v), compiled=True)


@pytest.mark.parametrize('name', ['abs', 'absolute', 'sign', 'square'])
def test_numpy_ints_stay_numpy_ints(name):
    for v in INTS:
        check(name, np.int64(v), compiled=True)


def test_numpy_int_pairs():
    for a in (np.int64(3), np.int64(-2**63)):
        for b in (np.int64(2), np.int64(2**63 - 1)):
            check('minimum', a, b, compiled=True)
            check('maximum', a, b, compiled=True)


# What depends on the platform or the numpy version


@pytest.mark.parametrize('name', ['abs', 'sign', 'square', 'floor', 'ceil', 'trunc'])
def test_a_python_int_stays_python(name):
    check(name, 4, compiled=False)


@pytest.mark.parametrize('name', ['floor', 'ceil', 'trunc'])
def test_rounding_a_numpy_int_stays_python(name):
    check(name, np.int64(4), compiled=False)


@pytest.mark.parametrize('name', ['minimum', 'maximum', 'fmod'])
def test_two_python_ints_stay_python(name):
    check(name, 3, 2, compiled=False)


@pytest.mark.parametrize('name', ['sqrt', 'exp', 'floor', 'abs', 'sign', 'isnan'])
def test_bools_stay_python(name):
    for v in (True, np.bool_(True)):                 # narrow types compile: tests/test_narrow.py
        check(name, v, compiled=False)


def test_tanh_stays_python():
    check('tanh', 0.5, compiled=False)


def test_power_stays_python():
    check('power', -0.0, 0.5, compiled=False)


def test_an_array_stays_python():
    check('sqrt', np.arange(4.0), compiled=False)
    check('minimum', np.arange(4.0), 1.0, compiled=False)


def test_other_shapes_of_call():
    def out_kw(x, y):
        return np.sqrt(x, out=y)

    def where_kw(x):
        return np.sqrt(x, where=True)

    def wrong_count(x):
        return np.sqrt(x, x, x)
    for f, args in ((where_kw, (4.0,)), (wrong_count, (4.0,))):
        g = scram(cache=False)(f)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert outcome(g, list(args))[:3] == outcome(f, list(args))[:3]


# As it's written in real code


def constants(x):
    s = 0.0
    for i in range(len(x)):
        s += np.sin(x[i] * np.pi) + np.e
    return s + np.inf * 0.0


def in_a_condition(x):
    n = 0
    for v in x:
        if np.isnan(v) or np.isinf(v):
            n += 1
    return n


def clamp(x, y):
    for i in range(len(x)):
        y[i] = np.minimum(np.maximum(x[i], -1.0), 1.0)


def constructors(x, k):
    a = np.float64(k)
    b = np.int64(k)
    c = np.float64(x[0])
    return a * 0.5, b + 1, c


def mixed_with_python_math(x):
    s = 0.0
    for v in x:
        s += np.sqrt(abs(v)) + math.sqrt(abs(v))
    return s


def result_is_numpy(y):
    z = np.sqrt(y)
    return z / 0.0


def compare(f, *args, compiled=True):
    copies = lambda: [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    g = scram(cache=False)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plain, native = copies(), copies()
        want, got = outcome(f, plain), outcome(g, native)
    assert repr(want[:3]) == repr(got[:3]), f.__name__
    for p, c in zip(plain, native):
        if isinstance(p, np.ndarray):
            assert p.tobytes() == c.tobytes()
    assert bool(g.signatures) == compiled, g._why
    return g


X = np.array([0.5, -2.5, 3.0, 0.0])


def test_in_real_code():
    compare(constants, X)
    compare(in_a_condition, np.array([1.0, np.nan, np.inf, -np.inf]))
    compare(clamp, X, np.zeros(4))
    compare(constructors, X, 7)
    compare(mixed_with_python_math, X)
    compare(result_is_numpy, 4.0)                  # numpy's division: inf, not ZeroDivisionError


from numpy import sqrt as root  # noqa: E402


def imported_by_name(x):
    return root(x)


def test_imported_by_name():
    compare(imported_by_name, 4.0)


def shadowed(np, x):
    return np.sqrt(x)


class Fake:
    @staticmethod
    def sqrt(x):
        return 'not numpy'


def test_a_local_named_np_is_not_numpy():
    g = scram(cache=False)(shadowed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert g(Fake, 4.0) == 'not numpy'


@pytest.mark.parametrize('options', [{'boundscheck': False}])
def test_options(options):
    g = scram(cache=False, **options)(clamp)
    a, b = np.zeros(4), np.zeros(4)
    clamp(X, a)
    g(X, b)
    assert a.tobytes() == b.tobytes()


def test_parity_off_compiles_and_agrees_on_ordinary_values():
    g = scram(cache=False, parity=False, fallback=False)(mixed_with_python_math)
    assert abs(g(X) - mixed_with_python_math(X)) < 1e-9


def lifted(x, name):
    tag = {name: 1}
    s = 0.0
    for v in x:
        s += np.sqrt(abs(v))
    tag['s'] = s
    return tag


def test_lifted():
    g = scram(cache=False)(lifted)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert g(X, 'a') == lifted(X, 'a')
    assert g._native_loops()


@pytest.mark.parametrize('name', ['minimum', 'maximum', 'fmin', 'fmax'])
def test_min_and_max_of_numpy_ints(name):
    """fmin of two ints gave the larger one."""
    for a, b in ((np.int64(0), np.int64(1)), (np.int64(1), np.int64(0)), (np.int64(-5), np.int64(5))):
        check(name, a, b, compiled=True)
