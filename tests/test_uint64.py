"""np.uint64, as numpy has it: wrapping with Python ints and unsigned dtypes, float64 with signed ints and floats, exact
comparisons with ints, and reductions compared unsigned."""

import ast
import itertools
import math
import warnings

import numpy as np
import pytest

from scramjit._dispatch import Function

from util import outcome

np.seterr(all='ignore')

M = 2**64 - 1
U = [0, 1, 5, 2**63 - 1, 2**63, 2**63 + 7, M - 1, M]
OTHERS = {
    'uint64': U, 'int64': [0, 1, -1, 7, -2**63, 2**63 - 1], 'uint8': [0, 1, 3, 255], 'int8': [0, 1, -1, -128],
    'uint32': [0, 1, 7, 2**32 - 1], 'int32': [0, -1, 7], 'float64': [0.0, -0.0, 1.5, -2.5, 2.0**64, math.inf, math.nan],
    'float32': [0.0, 1.5, -2.0, math.nan],
}
PY_NUMBERS = [0, 1, 3, 64, 65, -1, -7, 2**32, 2**32 + 3, 2**63, M, True, False, 0.0, 2.5, -1.5, 2.0**64, math.nan]
BINARY = ['+', '-', '*', '/', '//', '%', '**', '&', '|', '^', '<<', '>>']
COMPARE = ['==', '!=', '<', '<=', '>', '>=']


def make(src):
    space = {'np': np, 'math': math}
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


@pytest.mark.parametrize('op', BINARY + COMPARE)
def test_with_numpy_scalars(op):
    pairs = [make(f'def f(a, b):\n    return a[0] {op} b[0]\n'), make(f'def f(a, b):\n    return b[0] {op} a[0]\n')]
    for d, vals in OTHERS.items():
        for x, y in itertools.product(U, vals):
            if op == '**' and x > 3:
                continue
            for f, g in pairs:
                agree(f, g, [np.array([x], np.uint64), np.array([y], d)])


@pytest.mark.parametrize('op', BINARY + COMPARE)
def test_with_python_numbers(op):
    pairs = [make(f'def f(a, b):\n    return a[0] {op} b\n'), make(f'def f(a, b):\n    return b {op} a[0]\n')]
    for x, y in itertools.product(U, PY_NUMBERS):
        if op == '**' and (x > 3 or isinstance(y, int) and y > 64):
            continue
        for f, g in pairs:
            agree(f, g, [np.array([x], np.uint64), y])


def test_compiles():
    for src in ('def f(a, b):\n    return a[0] * b[0] + 7\n', 'def f(a, b):\n    return a[0] < b[0]\n',
                'def f(a, b):\n    return a[0] + b[1]\n', 'def f(a, b):\n    return -a[0], ~a[0], abs(a[0]), max(a[0], a[1])\n'):
        f, g = make(src)
        agree(f, g, [np.array([M, 3], np.uint64), np.array([2, 1], np.int8)])
        assert g.signatures, (src, g._why)


@pytest.mark.parametrize('op', ['sum', 'prod', 'mean', 'max', 'min', 'argmax', 'argmin', 'any', 'all', 'std', 'var'])
def test_reductions(op):
    f, g = make(f'def f(a):\n    return a.{op}()\n')
    rng = np.random.default_rng(2)
    for a in (np.array(U, np.uint64), np.array(U[::-1], np.uint64), np.array([], np.uint64), np.zeros(3, np.uint64),
              rng.integers(0, M, 1000, dtype=np.uint64, endpoint=True), rng.integers(0, 2**62, 20000, dtype=np.uint64)):
        agree(f, g, [a])
        agree(f, g, [a[::3]])
    assert g.signatures, g._why


def test_a_hash_over_uint64_arrays():
    f, g = make('def f(keys, out):\n    for i in range(len(keys)):\n        h = keys[i] * np.uint64(0x9E3779B97F4A7C15)\n'
                '        out[i] = h ^ (h >> np.uint64(29))\n    return out.max()\n')
    keys = np.arange(1000, dtype=np.uint64) * np.uint64(2**40)
    o1, o2 = np.zeros(1000, np.uint64), np.zeros(1000, np.uint64)
    assert outcome(f, [keys, o1]) == outcome(g, [keys, o2])
    assert o1.tobytes() == o2.tobytes()
    assert g.signatures, g._why


def test_converting():
    f, g = make('def f(a, k):\n    return np.uint64(k), np.uint64(a[0]), np.uint64(0xFFFFFFFFFFFFFFFF)\n')
    for k in (0, 5, 2**63, M, -1):
        for a in (np.array([-1], np.int64), np.array([7], np.uint8)):
            assert outcome(f, [a, k]) == outcome(g, [a, k]), (k, a)
    assert g.signatures, g._why


@pytest.mark.parametrize('src', ['def f(a, k):\n    return a[k]\n', 'def f(a, k):\n    return a[int(k)]\n',
                                 'def f(a, k):\n    a[k] = 1\n    return a[0]\n', 'def f(a, k):\n    xs = [1, 2, 3, 4]\n    return xs[int(k)]\n'])
def test_indices(src):
    """An np.uint64 or a Python int past int64 as an index: numpy's OverflowError, or the list's IndexError; an ordinary
    one indexes as usual."""
    f, g = make(src)
    for k in (0, 3, 9, 2**63, M):
        agree(f, g, [np.arange(4.0), np.uint64(k)])
        agree(f, g, [np.arange(4.0), np.array([k], np.uint64)[0]])
    assert g.signatures, g._why
