"""64-bit hashes written for Python ints: a value masked to 64 bits, or a literal up to 2**64 - 1, is held unsigned,
so FNV-1a 64, xorshift64 and splitmix64 give exactly Python's numbers."""

import ast
import itertools
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram
from scramjit._dispatch import Function

from util import outcome, same

M64 = 0xFFFFFFFFFFFFFFFF


@scram
def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= int(b)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


@scram
def xorshift64(seed: int, n: int):
    x = seed
    for _ in range(n):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x


@scram
def splitmix64(state: int, n: int, out):
    for i in range(n):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        out[i] = z / 18446744073709551616.0
    return state


@scram
def bucket(data, size: int):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ int(b)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % size, h // 3, h > 2 ** 63, h == 0xFFFFFFFFFFFFFFFF, float(h), h >> 60, h & 255, h < 1.5e19


def strict(f, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        r = same(f, *args)
    assert f.signatures, f._why
    return r


def test_fnv1a64():
    for s in (b'', b'a', b'hello, world', bytes(range(256))):
        strict(fnv1a64, np.frombuffer(s, np.uint8))
    assert fnv1a64(np.frombuffer(b'a', np.uint8)) == 0xAF63DC4C8601EC8C


def test_xorshift64():
    for seed in (1, 88172645463325252, 2**63 - 1):
        strict(xorshift64, seed, 1000)


def test_a_negative_seed_runs_as_python():
    assert xorshift64(-5, 3) == xorshift64.__wrapped__(-5, 3)


def test_splitmix64():
    strict(splitmix64, 0, 100, np.zeros(100))
    strict(splitmix64, 12345, 100, np.zeros(100))


def test_unsigned_division_comparison_and_float():
    for s in (b'', b'x', b'scramjit', bytes(range(100))):
        for size in (1, 7, 1024, 2**62):
            strict(bucket, np.frombuffer(s, np.uint8), size)


def make(src):
    space = {'np': np}
    exec(src, space)
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys = {}
    g._first = 1
    return space['f'], g


U = [0, 1, 2**63 - 1, 2**63, 2**63 + 5, M64 - 1, M64, 0xCBF29CE484222325]
I = [0, 1, -1, 3, -7, 2**62, -2**63, 2**63 - 1, True]


def wrapped(v):
    return (v + 2**63) % 2**64 - 2**63


@pytest.mark.parametrize('op', ['+', '-', '*', '//', '%', '&', '|', '^', '<<', '>>', '==', '!=', '<', '<=', '>', '>=', '/', '**'])
def test_operators_against_python(op):
    """u is a masked value in [0, 2**64), the other a Python int. A result is Python's, or, where Python's is past
    int64 and the operator can leave [0, 2**64), the documented wrap."""
    f, g = make(f'def f(a, b):\n    u = a & 0xFFFFFFFFFFFFFFFF\n    return u {op} b, b {op} u\n')
    for a, b in itertools.product(U, I):
        if op in ('<<', '>>', '**') and not (0 <= b < 70 and a < 70) and op != '>>' or op == '>>' and not 0 <= b < 70 or op == '**' and a > 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            want, got = outcome(f, [a, b]), outcome(g, [a, b])
        if want[0] == 'returns' and got[0] == 'returns':
            exact_only = op in ('&', '|', '^', '//', '%', '>>') or op in ('==', '!=', '<', '<=', '>', '>=')
            for x, y in zip(f(a, b), g(a, b)):
                ok = type(x) is type(y) and (x == y or (not exact_only and not -2**63 <= x < 2**63 and y in (wrapped(x), x % 2**64)))
                if op in ('|', '^') and b < 0:         # with a negative int the result is an ordinary int, which wraps
                    ok = ok or y == wrapped(x)
                assert ok, (op, a, b, f(a, b), g(a, b))
        else:
            assert want[:3] == got[:3], (op, a, b, want, got)


def test_masks_written_as_expressions():
    """(1 << 64) - 1 and 2 ** 64 - 1 are what Python makes them, not wrapped."""
    f, g = make('def f(a):\n    return a & ((1 << 64) - 1), a & (2 ** 64 - 1), (1 << 63) + a, 2 ** 63 > a, -(1 << 63)\n')
    for a in (0, 5, -1, -2**63, 2**63 - 1):
        assert outcome(f, [a]) == outcome(g, [a])
    assert g.signatures, g._why


MASK = (1 << 64) - 1
PRIME = 0x100000001B3


@scram
def fnv_globals(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ int(b)) * PRIME) & MASK
    return h


def test_module_constants_past_int64():
    strict(fnv_globals, np.frombuffer(b'hello', np.uint8))


@scram
def fill(seed: int, out):
    x = seed
    for i in range(len(out)):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        out[i] = x
    return x, out[0], out[len(out) - 1]


@scram
def mix_array(a):
    h = 0
    for v in a:
        h = (h * 31 + int(v)) & 0xFFFFFFFFFFFFFFFF
    return h, float(a[0]), a[0]


def test_uint64_arrays():
    strict(fill, 1, np.zeros(50, np.uint64))
    strict(fill, 2**64 - 1, np.zeros(3, np.uint64))
    strict(fill, 7, np.zeros(5, np.int64))           # numpy can't convert these: OverflowError
    strict(fill, 7, np.zeros(5, np.uint8))
    strict(fill, 7, np.zeros(5, np.float64))
    strict(mix_array, np.array([2**64 - 1, 5, 2**63], np.uint64))


@pytest.mark.parametrize('dtype', ['uint64', 'int64', 'uint8', 'int32'])
def test_storing_python_ints(dtype):
    f, g = make('def f(a, k):\n    a[0] = k\n    return a[0]\n')
    for k in (0, 5, -1, 255, 300, 2**31, 2**63 - 1, 2**63, 2**64 - 1):
        a1, a2 = np.zeros(2, dtype), np.zeros(2, dtype)
        assert outcome(f, [a1, k])[:3] == outcome(g, [a2, k])[:3], (dtype, k)
        assert a1.tobytes() == a2.tobytes()


@pytest.mark.parametrize('expr', ['a[0] + 1', 'a[0] == a[1]', '-a[0]', 'abs(a[0])', 'a.sum()', 'math.floor(a[0])', 'a[0] * k'])
def test_uint64_elements_match_numpy(expr):
    """tests/test_uint64.py goes further; math.floor() of one is the case that stays Python."""
    import math
    space = {'np': np, 'math': math}
    src = f'def f(a, k):\n    return {expr}\n'
    exec(src, space)
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys = {}
    g._first = 1
    a = np.array([5, 2**64 - 1], np.uint64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert outcome(space['f'], [a, 3])[:3] == outcome(g, [a, 3])[:3]
    assert bool(g.signatures) == ('math' not in expr), g._why


def test_a_variable_that_may_be_negative_says_so():
    f, g = make('def f(a, k):\n    h = a & 0xFFFFFFFFFFFFFFFF\n    for _ in range(3):\n        h = h - k\n    return h\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert outcome(f, [5, 3]) == outcome(g, [5, 3])
    assert not g.signatures and 'may be negative' in str(g._why)


def test_parameters_and_lists():
    f, g = make('def f(n, seed):\n    xs = [0] * n\n    x = seed\n    for i in range(n):\n        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF\n'
                '        xs[i] = x >> 33\n    return x, xs[n - 1]\n')
    for n, seed in ((3, 1), (10, 2**63 + 1), (5, -1)):
        assert outcome(f, [n, seed]) == outcome(g, [n, seed]), (n, seed)
    assert g.signatures, g._why


def test_math_of_an_int_past_int64_stays_python():
    import math
    space = {'np': np, 'math': math}
    src = 'def f(a):\n    return math.gcd(int(a[0]), 6)\n'
    exec(src, space)
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys = {}
    g._first = 1
    a = np.array([2**64 - 2], np.uint64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert outcome(space['f'], [a]) == outcome(g, [a])
    assert not g.signatures and 'past int64' in str(g._why), g._why
