"""M1: scalars, 1-D arrays, range, if, while; Python and numpy semantics; fallback."""

import itertools
import math
import random
import warnings

import numpy as np
import pytest

from scramjit import CompileError, FallbackWarning, scram
from util import FLOATS, INTS, same

strict = scram(fallback=False)


# ---- arithmetic: Python semantics on ints and floats

@strict
def add(a, b):
    return a + b


@strict
def sub(a, b):
    return a - b


@strict
def mul(a, b):
    return a * b


@strict
def div(a, b):
    return a / b


@strict
def floordiv(a, b):
    return a // b


@strict
def mod(a, b):
    return a % b


@pytest.mark.parametrize('f', [add, sub, mul, div, floordiv, mod])
def test_int_int(f):
    small = [x for x in INTS if abs(x) < 1 << 40]
    for a, b in itertools.product(small, small):
        same(f, a, b)


@pytest.mark.parametrize('f', [add, sub, mul, div, floordiv, mod])
def test_float_float(f):
    for a, b in itertools.product(FLOATS, FLOATS):
        same(f, a, b)


@pytest.mark.parametrize('f', [add, sub, mul, div, floordiv, mod])
def test_int_float(f):
    for a, b in itertools.product(INTS, FLOATS):
        same(f, a, b)
        same(f, b, a)


def test_true_division_is_correctly_rounded_for_large_ints():
    rng = random.Random(1)
    for _ in range(3000):
        a = rng.randrange(-(1 << 63), 1 << 63)
        b = rng.randrange(-(1 << 63), 1 << 63) >> rng.randrange(0, 63) or 1
        same(div, a, b)


@strict
def lt(a, b):
    return a < b


@strict
def le(a, b):
    return a <= b


@strict
def eq(a, b):
    return a == b


@strict
def ne(a, b):
    return a != b


@pytest.mark.parametrize('f', [lt, le, eq, ne])
def test_compare_int_float_exactly(f):
    big = 1 << 53
    ints = INTS + [big, big + 1, big - 1, (1 << 62) + 1, -(1 << 63), (1 << 63) - 1]
    floats = FLOATS + [float(big), float(big + 2), 2.0 ** 62, -(2.0 ** 63), 2.0 ** 63]
    for a, b in itertools.product(ints, floats):
        same(f, a, b)
        same(f, b, a)


# ---- numpy scalars behave like numpy, not like Python numbers

@strict
def elem_div(x, i, d):
    return x[i] / d


@strict
def elem_floordiv(x, i, d):
    return x[i] // d


@strict
def elem_mod(x, i, d):
    return x[i] % d


def test_numpy_scalars_divide_by_zero_like_numpy():
    xf = np.array([1.0, -2.5, 0.0, np.nan])
    xi = np.array([7, -7, 0], dtype=np.int64)
    for f in (elem_div, elem_floordiv, elem_mod):
        for x in (xf, xi):
            for i in range(len(x)):
                for d in (0, 0.0, 2, -2, 0.5):
                    same(f, x, i, d)


@strict
def np_ops(a, b):
    return a * b + a


def test_numpy_promotion():
    for a in (np.int64(3), np.float64(2.5), 3, 2.5):
        for b in (np.int64(-2), np.float64(0.5), -2, 0.5, True):
            same(np_ops, a, b)


# ---- conversions and builtins

@strict
def to_int(x):
    return int(x)


@strict
def to_float(x):
    return float(x)


def test_int_float_conversions():
    for x in FLOATS + [2.7, -2.7, 1e18, -1e18]:
        if abs(x) < 2.0 ** 63 or not math.isfinite(x):
            same(to_int, x)
    for x in INTS + [(1 << 63) - 1, -(1 << 63)]:
        same(to_float, x)
        same(to_int, x)


@strict
def mx(a, b):
    return max(a, b)


@strict
def mn(a, b, c):
    return min(a, b, c)


@strict
def ab(a):
    return abs(a)


def test_min_max_abs():
    vals = FLOATS
    for a, b in itertools.product(vals, vals):
        same(mx, a, b)
        same(mn, a, b, 1.0)
    for a in INTS + FLOATS:
        same(ab, a)


# ---- control flow

@strict
def ranges(a, b, c):
    s = 0
    for i in range(a, b, c):
        s += i
    return s


@strict
def range_count(a, b, c):
    n = 0
    for i in range(a, b, c):
        n += 1
    return n


@strict
def range_last(n):
    i = -100
    for i in range(n):
        pass
    return i


def test_range():
    for a, b, c in itertools.product([-7, 0, 3, 10], [-7, 0, 3, 10], [-3, -1, 1, 2, 5, 0]):
        same(ranges, a, b, c)
    for n in (-1, 0, 1, 5):
        same(range_last, n)
    same(range_count, (1 << 63) - 10, (1 << 63) - 1, 3)      # near the int64 limits: no overflow
    same(range_count, -(1 << 63), -(1 << 63) + 10, 4)
    same(range_count, (1 << 63) - 1, -(1 << 63), -(1 << 62))


@strict
def loops(n):
    total = 0
    i = 0
    while True:
        i += 1
        if i > n:
            break
        if i % 3 == 0:
            continue
        for j in range(i):
            if j > 5:
                break
            total += j * i
    return total


def test_while_break_continue():
    for n in range(0, 30):
        same(loops, n)


@strict
def boolops(a, b, c):
    if a < b < c and not (a == c) or b != 0:
        return 1
    return 2


@strict
def orvalue(a, b):
    return a or b


@strict
def ifexp(a, b):
    return a if a > b else b * 2.0


def test_bool_and_conditional_expressions():
    for a, b, c in itertools.product([0, 1, 2, -1], [0, 1, 2], [0, 2, 5]):
        same(boolops, a, b, c)
        same(boolops, float(a), float(b), float(c))
    for a, b in itertools.product([0, 3, -1], [0, 5]):
        same(orvalue, a, b)
        same(orvalue, float(a), float(b))
    for a, b in itertools.product([0.0, 3.0, math.nan], [1.0, math.nan]):
        same(ifexp, a, b)


# ---- arrays

@strict
def get(x, i):
    return x[i]


@strict
def put(x, i, v):
    x[i] = v


@strict
def bump(x, i, v):
    x[i] += v


def test_indexing_python_and_numpy_semantics():
    for x in (np.arange(5, dtype=np.float64), np.arange(5, dtype=np.int64)):
        for i in range(-7, 8):
            same(get, x, i)
            same(put, x, i, 3)
            same(bump, x, i, 2)
    same(put, np.zeros(3), 1, 2.5)


@strict
def ema(x, y, a: float):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]


@strict
def edit_distance(a, b, row):
    for j in range(len(b) + 1):
        row[j] = j
    for i in range(1, len(a) + 1):
        prev = row[0]
        row[0] = i
        for j in range(1, len(b) + 1):
            cur = row[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            row[j] = min(row[j] + 1, row[j - 1] + 1, prev + cost)
            prev = cur
    return row[len(b)]


@strict
def prefix_max(x, out):
    best = -math.inf
    for i in range(len(x)):
        if x[i] > best:
            best = x[i]
        out[i] = best
    return best


def test_kernels():
    rng = np.random.default_rng(3)
    for n in (1, 2, 17, 500):
        x = rng.random(n)
        same(ema, x, np.empty(n), 0.3)
        same(prefix_max, x, np.empty(n))
    for s, t in (('kitten', 'sitting'), ('', 'abc'), ('flaw', 'lawn'), ('a' * 40, 'b' * 3)):
        a = np.frombuffer(s.encode(), np.uint8).astype(np.int64)
        b = np.frombuffer(t.encode(), np.uint8).astype(np.int64)
        same(edit_distance, a, b, np.zeros(len(b) + 1, np.int64))
    # errors, with the same writes before them
    same(ema, np.ones(5), np.empty(3), 0.5)
    same(edit_distance, np.ones(3, np.int64), np.ones(4, np.int64), np.zeros(2, np.int64))


def test_boundscheck_off_still_wraps_negative_indices():
    f = scram(boundscheck=False, fallback=False)(get.__wrapped__)
    x = np.arange(4.0)
    assert f(x, -1) == 3.0 and f(x, 2) == 2.0


# ---- fallback, contracts, errors

def uses_a_list(n):
    out = []
    for i in range(n):
        out.append(i)
    return len(out)


def test_fallback_warns_once_and_gives_python_results():
    f = scram(uses_a_list)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert f(5) == 5 and f(7) == 7
    falls = [x for x in w if issubclass(x.category, FallbackWarning)]
    assert len(falls) == 1 and 'runs as Python' in str(falls[0].message)


def test_fallback_false_raises():
    with pytest.raises(CompileError, match='is not compiled'):
        scram(fallback=False)(uses_a_list)(3)


def test_unsupported_argument_types_fall_back():
    f = scram(add.__wrapped__)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert f('a', 'b') == 'ab'
        assert f(1 << 70, 1) == (1 << 70) + 1


def test_annotations_are_contracts():
    with pytest.raises(TypeError, match="argument 'a' must be float, not str"):
        ema(np.ones(2), np.ones(2), 'x')


@strict
def raises_value_error(x):
    return int(x)


def test_exception_points_at_the_python_line():
    got = same(raises_value_error, math.nan)
    assert got[0] == 'raises' and got[1] == 'ValueError'


# ---- random expressions against CPython

OPS = ['+', '-', '*', '/', '//', '%', '<', '<=', '==', '!=']


def rand_expr(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(['a', 'b', 'c', 'd', str(rng.randint(-9, 9)), repr(rng.choice([0.5, -2.0, 0.0, 3.25]))])
    if rng.random() < 0.15:
        return f'({rand_expr(rng, depth - 1)} if {rand_expr(rng, depth - 1)} else {rand_expr(rng, depth - 1)})'
    return f'({rand_expr(rng, depth - 1)} {rng.choice(OPS)} {rand_expr(rng, depth - 1)})'


def test_random_expressions():
    rng = random.Random(7)
    checked = 0
    for k in range(120):
        src = f'def r{k}(a, b, c, d):\n    return {rand_expr(rng, 3)}\n'
        ns = {}
        exec(compile(src, f'<random {k}>', 'exec'), ns)
        fn = ns[f'r{k}']
        # exec'd code has no source file; build the wrapper with the source text supplied
        import linecache
        linecache.cache[f'<random {k}>'] = (len(src), None, src.splitlines(True), f'<random {k}>')
        f = scram(fn, fallback=False)
        for args in ((3, -2, 0.5, 7), (0, 5, -1.5, -3), (-7, 2, 0.0, 1)):
            try:
                f(*args)
            except CompileError:
                break       # e.g. a conditional expression whose branches differ in type
            except Exception:  # noqa: BLE001  (Python raises too; same() compares the two)
                pass
            same(f, *args)
            checked += 1
    assert checked > 150


# ---- parity=False trades parity for speed

def test_exact_false_gives_ieee_results():
    fast_div = scram(div.__wrapped__, parity=False, fallback=False)
    assert fast_div(1.0, 0.0) == math.inf and math.isnan(fast_div(0.0, 0.0))
    assert fast_div(1, 0) == math.inf
    fast_lt = scram(lt.__wrapped__, parity=False, fallback=False)
    assert fast_lt(1, 2.5) is True
    with pytest.raises(ZeroDivisionError):             # an int // 0 still raises (C would trap)
        scram(floordiv.__wrapped__, parity=False, fallback=False)(1, 0)


@strict
def pow2(a):
    return a // 4 + a % 8 * 3 - a // 1 + a % 1


def test_power_of_two_divisors():
    for a in range(-40, 41):
        same(pow2, a)
    same(pow2, -(1 << 63))
