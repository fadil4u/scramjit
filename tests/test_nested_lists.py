"""Lists of lists passed in: rows are references, as in Python, and the elements written are copied back."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import outcome


def total(m):
    s = 0.0
    for row in m:
        for v in row:
            s += v
    return s


def transpose_into(m, out):
    for i in range(len(m)):
        for j in range(len(m[i])):
            out[j][i] = m[i][j]
    return out[0][0]


def through_a_row(m):
    row = m[1]
    row[0] = row[0] * 10
    m[1][1] += 1
    return row[0] + m[1][1]


def dp(grid):
    """Cheapest path down and right, in place."""
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                continue
            up = grid[i - 1][j] if i > 0 else 10 ** 9
            left = grid[i][j - 1] if j > 0 else 10 ** 9
            grid[i][j] += min(up, left)
    return grid[h - 1][w - 1]


def out_of_range(m, i: int):
    m[0][0] = 5
    return m[i][0]


def strict(f, *args):
    g = scram(cache=False)(f)
    copy = lambda: [[list(r) for r in a] if isinstance(a, list) else a for a in args]
    plain, mine = copy(), copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        got = outcome(g, mine)
    assert got[:3] == outcome(f, plain)[:3], (f.__name__, args)
    assert repr(plain) == repr(mine), (f.__name__, plain, mine)
    assert g.signatures, (f.__name__, [str(m.message) for m in w])


M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_reading():
    strict(total, M)


def test_writing_and_rows_as_references():
    strict(transpose_into, M, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    strict(through_a_row, M)
    strict(through_a_row, [[1, 2], [3, 4]])


def test_a_dynamic_programming_table():
    rng = np.random.default_rng(0)
    strict(dp, rng.integers(1, 9, (6, 7)).tolist())


@pytest.mark.parametrize('i', [1, 2, -2, -3])
def test_index_errors_are_lists(i):
    strict(out_of_range, [[1, 2], [3, 4]], i)


@pytest.mark.parametrize('m', [[[1.0], [2.0, 3.0]], [[]], [[1, 2.0]]])
def test_other_shapes_stay_python(m):
    g = scram(cache=False)(total)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert outcome(g, [m])[:3] == outcome(total, [m])[:3]
    assert not g.signatures


def test_one_row_twice_stays_python():
    r = [1.0, 2.0]
    m = [r, r]                                   # writing m[0][0] changes m[1][0] in Python
    g = scram(cache=False)(through_a_row)
    m2 = [r2 := [1.0, 2.0], r2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(m) == through_a_row(m2)
    assert repr(m) == repr(m2) and not g.signatures


@pytest.mark.parametrize('src', ['def f(m):\n    return m[0, 1]\n', 'def f(m):\n    return m[0][1:]\n', 'def f(m):\n    m[0] = [1.0, 2.0]\n    return 0\n',
                                 'def f(m):\n    return len(m.shape)\n', 'def f(m):\n    return m\n'])
def test_what_python_does_differently_stays_python(src):
    space = {}
    exec(src, space)
    f = space['f']
    g = scram(cache=False)(f)
    import ast
    from scramjit._dispatch import Function
    g = Function(f, cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys = {}
    g._first = 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        a, b = [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]
        assert outcome(g, [a])[:3] == outcome(f, [b])[:3] and a == b
    assert not g.signatures
