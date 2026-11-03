"""<< and >>: Python's results on Python ints (with << wrapping past 64 bits, as * does), numpy's on numpy scalars."""

import ast
import itertools
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram
from scramjit._dispatch import Function

from util import outcome, same

np.seterr(all='ignore')

INTS = [0, 1, -1, 3, -7, 255, 2**31, 2**62, -2**63, 2**63 - 1, True, False]
COUNTS = [0, 1, 3, 7, 8, 15, 16, 31, 32, 62, 63, 64, 65, 200, -1, -64, True]
DTYPES = ['int64', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8']
EDGE = {'int64': [0, 1, -1, 5, -2**63, 2**63 - 1], 'int32': [0, 1, -1, 5, -2**31, 2**31 - 1], 'int16': [0, 1, -1, 5, -32768, 32767],
        'int8': [0, 1, -1, 5, -128, 127], 'uint32': [0, 1, 5, 2**32 - 1], 'uint16': [0, 1, 5, 65535], 'uint8': [0, 1, 5, 200, 255]}
SMALL_COUNTS = [0, 1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 100, -1]


def make(src):
    space = {'np': np}
    exec(src, space)
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys = {}
    g._first = 1
    return space['f'], g


def agree(f, g, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        want, got = outcome(f, list(args)), outcome(g, list(args))
    assert want[:3] == got[:3], (args, want, got)
    assert g.signatures, g._why


def wrap(v):
    return (v + 2**63) % 2**64 - 2**63


@pytest.mark.parametrize('op', ['<<', '>>'])
def test_python_ints(op):
    f, g = make(f'def f(a, b):\n    return a {op} b\n')
    for a, b in itertools.product(INTS, COUNTS):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            want, got = outcome(f, [a, b]), outcome(g, [a, b])
        if op == '<<' and want[0] == 'returns':       # the documented difference: << wraps, as * does
            want = ('returns', ('int', wrap(want[1][1])))
        assert want[:3] == got[:3], (a, b, want, got)
    assert g.signatures


def test_negative_count_raises_on_its_line():
    @scram
    def f(a, k):
        x = a + 1
        return x >> k
    with pytest.raises(ValueError, match='negative shift count'):
        f(3, -1)
    assert f(-9, 2) == -2 and f.signatures


@pytest.mark.parametrize('op', ['<<', '>>'])
def test_numpy_scalars(op):
    f, g = make(f'def f(a, b):\n    return a[0] {op} b[0]\n')
    for da, db in itertools.product(DTYPES, DTYPES):
        for x, y in itertools.product(EDGE[da], [c for c in SMALL_COUNTS if c >= 0 or db[0] != 'u']):
            if not np.can_cast(np.min_scalar_type(y), db) and y > 0 and y > np.iinfo(db).max:
                continue
            agree(f, g, [np.array([x], da), np.array([y], db)])


@pytest.mark.parametrize('op', ['<<', '>>'])
def test_with_python_ints(op):
    f, g = make(f'def f(a, b):\n    return a[0] {op} b\n')
    r, h = make(f'def f(a, b):\n    return b {op} a[0]\n')
    for d in DTYPES:
        for x, y in itertools.product(EDGE[d], SMALL_COUNTS + [300, -300, True]):
            agree(f, g, [np.array([x], d), y])
            if 0 <= x < 64:
                agree(r, h, [np.array([x], d), y])


def test_a_variable_that_may_be_numpy():
    """t is the int 1 or np.int64: a negative count raises only while it's Python's."""
    f, g = make('def f(x, k):\n    t = 1\n    for v in x:\n        t = t + v\n    return t << k\n')
    for n, k in itertools.product((0, 2), (3, -1, 63, 70)):
        want, got = outcome(f, [np.arange(n, dtype=np.int64), k]), outcome(g, [np.arange(n, dtype=np.int64), k])
        if want[0] == 'returns' and want[1][0] == 'int':      # << wraps on Python ints
            want = ('returns', ('int', wrap(want[1][1])))
        assert want[:3] == got[:3], (n, k, want, got)
    assert g.signatures


def test_augmented_and_a_hash():
    @scram
    def xorshift(seed, n):
        x = seed
        total = 0
        for _ in range(n):
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= x >> 17
            x ^= (x << 5) & 0xFFFFFFFF
            total = (total + x) & 0xFFFF
        return x, total

    @scram
    def fnv(data):
        h = 2166136261
        for b in data:
            h = ((h ^ int(b)) * 16777619) & 0xFFFFFFFF
        return h >> 1
    for seed in (1, 2463534242, 0xFFFFFFFF):
        same(xorshift, seed, 1000)
    same(fnv, np.frombuffer(b'hello, world', np.uint8))
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        xorshift(1, 3), fnv(np.frombuffer(b'ab', np.uint8))


def test_floats_and_numpy_bools_stay_python():
    f, g = make('def f(a, b):\n    return a << b\n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for a, b in ((1.0, 1), (np.bool_(True), 1)):
            assert outcome(f, [a, b])[:3] == outcome(g, [a, b])[:3]
    assert not g.signatures
