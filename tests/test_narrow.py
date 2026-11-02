"""Arithmetic on narrow numpy scalars: the same value, type and exception as numpy (NEP 50), for every pair of
dtypes, and compiled rather than falling back."""

import ast
import itertools
import math
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram
from scramjit._dispatch import Function

from util import outcome

np.seterr(all='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

NARROW = ['float32', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8']
ALL = NARROW + ['int64', 'float64']
EDGE = {
    'float32': [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, -2.5, 0.1, 3.0, 1e-45, 1e38, 3.4028235e38, -3.4028235e38, 16777217.0, math.inf, -math.inf, math.nan],
    'float64': [0.0, -0.0, 1.0, -2.5, 0.1, 1e300, 1e-300, 2.0**53, math.inf, math.nan],
    'int64': [0, 1, -1, 3, -7, 2**31, 2**53 + 1, 2**63 - 1, -2**63],
    'int32': [0, 1, -1, 3, -7, 100, 2**31 - 1, -2**31, 65537],
    'int16': [0, 1, -1, 3, -7, 100, 32767, -32768],
    'int8': [0, 1, -1, 3, -7, 100, 127, -128],
    'uint32': [0, 1, 3, 7, 100, 2**31, 2**32 - 1],
    'uint16': [0, 1, 3, 7, 100, 32768, 65535],
    'uint8': [0, 1, 3, 7, 100, 128, 255],
}
PY_NUMBERS = [0, 1, -1, 7, 100, 255, 256, -129, 2**31, 2**40, 2**53 + 1, True, False, 0.0, -0.0, 0.1, 1.5, -2.5, 1e40, -1e40, 1e300, math.inf, math.nan]
BINARY = ['+', '-', '*', '/', '//', '%', '&', '|', '^']
COMPARE = ['==', '!=', '<', '<=', '>', '>=']


def make(src, **space):
    """A compiled function from source text: scramjit reads source. Returns (plain, compiled)."""
    space['np'] = np
    exec(src, space)
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys = {}
    g._first = 1
    return space['f'], g


def agree(f, g, args, compiled=True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        want, got = outcome(f, list(args)), outcome(g, list(args))
    assert want[:3] == got[:3], (args, want, got)
    if compiled is not None:
        assert bool(g.signatures) == compiled, (args, g._why)


def narrow_pairs():
    return [(a, b) for a, b in itertools.product(ALL, ALL) if a in NARROW or b in NARROW]


@pytest.mark.parametrize('op', BINARY)
def test_two_numpy_scalars(op):
    f, g = make(f'def f(a, b):\n    return a[0] {op} b[0]\n')
    for da, db in narrow_pairs():
        if 'float' in da + db and op in '&|^':
            continue
        for x, y in itertools.product(EDGE[da], EDGE[db]):
            agree(f, g, [np.array([x], da), np.array([y], db)])


@pytest.mark.parametrize('op', COMPARE)
def test_comparing_two_numpy_scalars(op):
    f, g = make(f'def f(a, b):\n    return a[0] {op} b[0]\n')
    for da, db in itertools.product(ALL, ALL):
        for x, y in itertools.product(EDGE[da], EDGE[db]):
            agree(f, g, [np.array([x], da), np.array([y], db)])


@pytest.mark.parametrize('op', BINARY + COMPARE)
def test_with_a_python_number(op):
    f, g = make(f'def f(a, b):\n    return a[0] {op} b\n')
    r, h = make(f'def f(a, b):\n    return b {op} a[0]\n')
    for d in NARROW:
        for x, y in itertools.product(EDGE[d], PY_NUMBERS):
            if op in '&|^' and ('float' in d or isinstance(y, float)):
                continue
            agree(f, g, [np.array([x], d), y])
            agree(r, h, [np.array([x], d), y])


@pytest.mark.parametrize('op,lit', [('+', '1'), ('-', '1'), ('*', '2'), ('+', '300'), ('-', '-1'), ('//', '2'), ('%', '4'), ('//', '256'),
                                    ('%', '256'), ('/', '2'), ('+', '1.5'), ('*', '0.1'), ('&', '255'), ('^', '-1'), ('|', '256'), ('==', '0.1'), ('<', '300')])
def test_with_a_literal(op, lit):
    f, g = make(f'def f(a):\n    return a[0] {op} {lit}\n')
    for d in NARROW:
        if 'float' in d and op in '&|^':
            continue
        for x in EDGE[d]:
            agree(f, g, [np.array([x], d)])


@pytest.mark.parametrize('op', ['-', '+', '~', 'not '])
def test_unary(op):
    f, g = make(f'def f(a):\n    return {op}a[0]\n')
    for d in NARROW:
        if d == 'float32' and op == '~':
            continue
        for x in EDGE[d]:
            agree(f, g, [np.array([x], d)])


@pytest.mark.parametrize('d', NARROW)
def test_numpy_scalars_as_arguments_and_results(d):
    f, g = make('def f(x, y):\n    z = x + y\n    return z\n')
    for x, y in itertools.product(EDGE[d], EDGE[d]):
        agree(f, g, [np.dtype(d).type(x), np.dtype(d).type(y)])


@pytest.mark.parametrize('d', NARROW)
@pytest.mark.parametrize('init', ['0', '0.0', '1', '1.0', '-1.0', '0.5'])
def test_a_literal_that_becomes_narrow(d, init):
    """s starts as Python's and is numpy's from the first +=; with no iterations Python's comes back."""
    kind_matches = ('.' in init) == ('float' in d)
    f, g = make(f'def f(x):\n    s = {init}\n    for i in range(len(x)):\n        s += x[i] * x[i]\n    return s\n')
    for n in (0, 1, 2, 7):
        x = np.resize(np.array(EDGE[d][1:5], d), n)
        # 0.5 can't start an int's variable, and the int-then-float cases stay as Python
        agree(f, g, [x], compiled=None if not kind_matches else True)


def test_a_literal_narrow_cannot_hold_stays_python():
    f, g = make('def f(x):\n    s = 0.1\n    for i in range(len(x)):\n        s = x[i] + 1.0\n    return s\n')
    for x in (np.array([1.5], 'float32'), np.array([], 'float32')):
        agree(f, g, [x], compiled=None)
    f, g = make('def f(x):\n    s = 300\n    for v in x:\n        s = v + 1\n    return s\n')
    for x in (np.array([1], 'uint8'), np.array([], 'uint8')):
        agree(f, g, [x], compiled=None)


def test_an_exact_literal_that_narrow_can_hold_compiles():
    f, g = make('def f(x, c):\n    s = 0.0\n    for i in range(len(x)):\n        if c:\n            s = x[i]\n    return s\n')
    agree(f, g, [np.array([0.1], 'float32'), True])
    agree(f, g, [np.array([0.1], 'float32'), False])


@pytest.mark.parametrize('d', NARROW)
def test_stored_back_and_augmented(d):
    if 'float' in d:
        vals, lit = [0.1, 2.5, -1.0, 1e38], '1.5'
    else:
        vals, lit = [1, 7, 100], '3'
    f, g = make(f'def f(x, y):\n    for i in range(len(x)):\n        x[i] += y[i] * {lit}\n        y[i] = x[i] - {lit}\n    return x[0]\n')
    x = np.array(vals, d)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a1, b1, a2, b2 = x.copy(), x.copy(), x.copy(), x.copy()
        assert outcome(f, [a1, b1])[:3] == outcome(g, [a2, b2])[:3]
    assert a1.tobytes() == a2.tobytes() and b1.tobytes() == b2.tobytes()
    assert g.signatures, g._why


def test_python_int_out_of_range_raises_on_its_line():
    @scram
    def f(a, n):
        x = a[0]
        return x + n
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        with pytest.raises(OverflowError, match='Python integer 300 out of bounds for uint8'):
            f(np.array([1], np.uint8), 300)
        assert f(np.array([1], np.uint8), 255) == 0


def test_power_says_why_it_stays_python():
    f, g = make('def f(a):\n    return a[0] ** 2\n')
    agree(f, g, [np.array([3.0], 'float32')], compiled=False)
    assert 'numpy bools' not in str(g._why) and 'pow' in str(g._why), g._why


BUILTINS = ['abs(a[0])', 'min(a[0], b[0])', 'max(a[0], b[0])', 'min(a[0], 1)', 'max(1, a[0])', 'float(a[0])', 'int(a[0])', 'bool(a[0])',
            'round(a[0])', 'sum(a)', 'sum(a, 0.0)', 'sum(a, a[0])', 'sum(v * v for v in a)', 'min(a)', 'max(a)', 'pow(a[0], 2)', 'divmod(a[0], b[0])[0]',
            'math.floor(a[0])', 'math.ceil(a[0])', 'math.sqrt(a[0] * a[0])', 'math.fabs(a[0])', 'math.isnan(a[0])',
            'a[0] if a[0] > b[0] else b[0]', 'a[0] if a[0] > 1 else 1', 'a[0] or b[0]', 'a[0] and 1', 'not a[0]', 'a[int(b[0]) % len(a)]',
            'len(a) + a[0]', '[v + 1 for v in a][0]', '(a[0] + 1, b[0] * 2)[1]', 'a.sum()', 'a.max()', 'a.mean()', 'np.sqrt(a[0])', 'np.abs(a[0])',
            'np.int32(a[0])', 'np.float32(a[0])', 'sum(a[i] * b[i] for i in range(len(a)))',
            'math.prod(a)', 'math.prod(v + 1 for v in a)', 'math.fsum(a)', 'math.sumprod(a, b)', 'a.prod()', 'a.std()', 'np.sum(a)', 'np.dot(a, b)', 'a[0] * 2 == b[0] * 2',
            'math.gcd(a[0], b[0])', 'math.hypot(a[0], b[0])', 'a[0] + 1 < b[0] < a[0] * 2', 'range(a[0] % 5)[0] if a[0] % 5 else 0']


@pytest.mark.parametrize('expr', BUILTINS)
def test_builtins_and_expressions_match_or_stay_python(expr):
    f, g = make(f'def f(a, b):\n    return {expr}\n', math=math)
    for d in ('float32', 'int32', 'int8', 'uint8', 'uint32'):
        vals = [v for v in EDGE[d] if v == v][:6]
        a, b = np.array(vals, d), np.array(vals[::-1], d)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for x, y in ((a, b), (b, a), (a[:1], b[:1])):
                agree(f, g, [x, y], compiled=None)


@scram
def mix(a, b):
    return a * b + 1


@scram
def mix_total(x):
    s = 0.0
    for v in x:
        s += mix(v, v)
    return s


def test_calls_between_compiled_functions():
    for x in (np.array([0.1, 2.5, -1.0], 'float32'), np.array([], 'float32')):
        with warnings.catch_warnings():
            warnings.simplefilter('error', FallbackWarning)
            got = mix_total(x)
        want = mix_total.__wrapped__(x)
        assert type(got) is type(want) and got == want


@scram
def show(v):
    return v


@scram
def flagged_through_a_call(x):
    s = 0.0
    for v in x:
        s += v
    return show(s)


def test_a_flagged_variable_passed_to_a_call():
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        for x in (np.array([0.1, 0.2], 'float32'), np.array([], 'float32')):
            got, want = flagged_through_a_call(x), flagged_through_a_call.__wrapped__(x)
            assert type(got) is type(want) and got == want


@pytest.mark.parametrize('d', ['uint8', 'int16', 'int32', 'float32'])
def test_forwarded_loads_and_views(d):
    @scram
    def recur(y):
        for i in range(1, len(y)):
            y[i] = y[i - 1] * 3 + 1
        return y

    @scram
    def rows(m):
        s = m[0, 0]
        for row in m:
            for v in row[1:]:
                s = s + v
        return s
    y = np.arange(1, 40, dtype=d)
    a, b = recur(y.copy()), recur.__wrapped__(y.copy())
    assert a.tobytes() == b.tobytes()
    m = np.arange(12, dtype=d).reshape(3, 4)
    got, want = rows(m), rows.__wrapped__(m)
    assert type(got) is type(want) and got == want
    assert recur.signatures and rows.signatures


def test_without_exact_parity():
    @scram(parity=False)
    def f(x, y):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * y[i] - x[i] / 2.0
        return s
    x = np.array([0.1, 0.2, 0.5], 'float32')
    got, want = f(x, x), f.__wrapped__(x, x)
    assert type(got) is np.float32
    assert abs(got - want) <= 1e-6


def test_lifted_loop_in_a_function_that_stays_python():
    @scram
    def f(x):
        d = {}                       # keeps the function in Python
        total = np.float32(0)
        for i in range(len(x)):
            total = total + x[i] * 2
        d['t'] = total
        return d['t']
    x = np.array([0.1, 0.2, 0.3], 'float32')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        got, want = f(x), f.__wrapped__(x)
    assert type(got) is type(want) and got == want
