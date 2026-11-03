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
    import copy as copying
    copy = lambda: copying.deepcopy(list(args))
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
                                 'def f(m):\n    return len(m.shape)\n', 'def f(m):\n    return m[0]\n'])
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


def edit_distance(a, b):
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]


def lcs(a, b):
    t = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        row, up = t[i], t[i - 1]
        for j in range(1, len(b) + 1):
            row[j] = up[j - 1] + 1 if a[i - 1] == b[j - 1] else max(up[j], row[j - 1])
    return t[len(a)][len(b)]


def floats_table(h: int, w: int):
    g = [[0.5] * w for _ in range(h)]
    s = 0.0
    for r in g:
        for v in r:
            s += v
    return s, len(g)


@pytest.mark.parametrize('a, b', [('kitten', 'sitting'), ('', 'abc'), ('flaw', 'lawn'), ('a', '')])
def test_tables_made_in_the_function(a, b):
    x, y = [ord(c) for c in a], [ord(c) for c in b]
    if x and y:
        strict(edit_distance, x, y)
        strict(lcs, x, y)


@pytest.mark.parametrize('h, w', [(3, 4), (0, 5), (2, 0), (-1, 3), (3, -2)])
def test_table_sizes(h, w):
    g = scram(cache=False)(floats_table)
    assert outcome(g, [h, w]) == outcome(floats_table, [h, w])
    assert g.signatures, g._why


def test_one_row_repeated_stays_python():
    def f(n: int):
        m = [[0] * 3] * n                      # every row is the same list
        m[0][0] = 7
        return m[n - 1][0]
    g = scram(cache=False)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(3) == f(3) == 7
    assert not g.signatures


def make_table(h: int, w: int):
    t = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            t[i][j] = i * w + j
    return t


def make_floats(h: int):
    return [[0.5] * 3 for _ in range(h)]


def returns_its_argument(m):
    m[0][0] += 1
    return m


def bool_table(h: int, w: int):
    t = [[False] * w for _ in range(h)]
    t[0][0] = True
    return t


@pytest.mark.parametrize('h, w', [(3, 4), (1, 1), (2, 0), (0, 3)])
def test_returning_tables(h, w):
    g = scram(cache=False)(make_table)
    got = g(h, w)
    want = make_table(h, w)
    assert got == want and all(type(r) is list for r in got), (got, want)
    if got and got[0]:
        assert got[0] is not got[-1] or len(got) == 1                # separate rows
    assert g.signatures, g._why


def test_returning_floats_bools_and_the_argument():
    strict(make_floats, 2)
    strict(bool_table, 2, 2)
    m = [[1, 2], [3, 4]]
    g = scram(cache=False)(returns_its_argument)
    assert g(m) is m and m == [[2, 2], [3, 4]]
