"""Views: m[i], x[a:b], x[::s], m[:, j] and friends, read and written, must behave as numpy's do. Everything is compared
with CPython running the same function on real numpy views."""

import random
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

X = np.arange(1.0, 11.0)
M = np.arange(1.0, 25.0).reshape(4, 6)
C = np.arange(1.0, 25.0).reshape(2, 3, 4)


def comparable(o):
    if o[0] == 'returns' and o[1][0] == 'tuple':
        return ('returns', [_value(x) for x in o[1][1]])
    return o


def same(f, *args, compiled=True, **options):
    fresh = lambda: [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    g = scram(cache=False, **options)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(2):
            plain, native = fresh(), fresh()
            want, got = comparable(outcome(f, plain)), comparable(outcome(g, native))
            assert want[:4] == got[:4], (f.__name__, args)
            for p, c in zip(plain, native):
                if isinstance(p, np.ndarray):
                    assert p.tobytes() == c.tobytes(), f.__name__
    if compiled is not None:
        assert bool(g.signatures) == compiled, (f.__name__, g._why)
    return g


def total(v):
    s = 0.0
    for a in v:
        s += a
    return s


def row_sum(m, i):
    row = m[i]
    s = 0.0
    for j in range(len(row)):
        s += row[j]
    return s, len(row)


def chain(m, i, j):
    return m[i][j]


def chain_store(m, i, j):
    m[i][j] = -1.0
    m[i][j] += 2.0


def slice_loop(x, a, b):
    s = 0.0
    for v in x[a:b]:
        s += v
    return s, len(x[a:b])


def step_loop(x, a, b, c):
    s = 0.0
    n = 0
    for v in x[a:b:c]:
        s = s * 2.0 + v
        n += 1
    return s, n


def view_index(x, a, b, c, k):
    v = x[a:b:c]
    return v[k], v[-1]


def writes_through(x, a, b, c):
    v = x[a:b:c]
    for i in range(len(v)):
        v[i] = v[i] + 100.0
    return len(v)


def write_seen(x):
    v = x[2:5]
    v[0] = 99.0
    w = x[::-1]
    w[0] = 77.0
    return x[2], x[len(x) - 1], v[0], w[0]


def overlap(x):
    a = x[1:]
    b = x[:-1]
    for i in range(len(a)):
        a[i] = b[i] * 2.0
    return x[0]


def column(m, j):
    c = m[:, j]
    s = 0.0
    for i in range(len(c)):
        s = s * 2.0 + c[i]
        c[i] = s
    return s, len(c)


def block(m, a, b, c, d):
    v = m[a:b, c:d]
    s = 0.0
    h, w = v.shape
    for i in range(h):
        for j in range(w):
            s = s * 1.5 + v[i, j]
    return s, h, w, v.size, v.ndim


def rows(m):
    s = 0.0
    for row in m:
        for v in row:
            s = s * 1.5 + v
    return s


def rows_written(m):
    k = 0.0
    for row in m:
        for j in range(len(row)):
            row[j] = k
            k += 1.0


def rows_of_view(m, a, b):
    s = 0.0
    for row in m[a:b]:
        s += row[0]
    return s


def rows_strided(m):
    s = 0.0
    for row in m[::2]:
        for v in row[::-1]:
            s = s * 1.5 + v
    return s


def cube(c, i, j):
    p = c[i]
    r = p[j]
    q = c[i, j]
    t = c[:, j]
    return r[0], q[1], t[0, 1], c[i][j][2], c[i, j, 3], t.shape[0], t.shape[1]


def rebinds(x):
    s = 0.0
    while len(x) > 0:
        s = s * 2.0 + x[0]
        x = x[1:]
    return s


def rebinds_in_loop(m):
    s = 0.0
    row = m[0]
    for i in range(len(m)):
        row = m[i]
        for j in range(len(row)):
            s = s * 1.5 + row[j]
    return s


def rebound_mid_loop(m, x):
    """Its length changes under a loop bound computed before: stale facts would read past the end, or skip a check."""
    v = x[:6]
    s = 0.0
    for i in range(6):
        s += v[i]
        if i == 2:
            v = x[:3]
    return s


def rebound_after_len(x):
    v = x[:6]
    n = len(v)
    v = x[:2]
    s = 0.0
    for i in range(n):
        s += v[i]
    return s


def versioned_rebound(x, b):
    v = x[:8]
    s = 0.0
    for j in range(3):
        s += v[b + j]
        v = x[:2]
    return s


def forward_through_view(x):
    v = x[2:]
    for i in range(1, len(v)):
        v[i] = v[i - 1] * 2.0
    return x[3]


def forward_with_other_view(x):
    a = x[1:]
    for i in range(1, len(x)):
        x[i] = x[i - 1] + a[0]
    return a[0]


def negative_index_on_view(x, k):
    return x[2:6][k]


def assign_one_to_another(x):
    a = x[1:4]
    b = a
    b[0] = 5.0
    return a[0], x[1]


def out_of_range_row(m, i):
    row = m[i]
    return row[0]


def zero_step(x, c):
    return len(x[::c])


def zero_step_after_error(m, i, c):
    return len(m[i, ::c])


def index_bound_order(m, i, j):
    return len(m[i, j:]), len(m[j:, i])


# --- the shapes of a slice


BOUNDS = [None, 0, 1, 3, 5, 9, 10, 11, 100, -1, -3, -9, -10, -11, -100]
STEPS = [None, 1, 2, 3, 11, -1, -2, -3, -11]


def test_every_bound_and_step():
    """Every combination, against numpy, through one function that takes each as a parameter."""
    g = scram(cache=False, fallback=False)(step_loop)
    for a in BOUNDS:
        for b in BOUNDS:
            for c in STEPS:
                if None in (a, b, c):
                    continue
                assert comparable(outcome(g, [X, a, b, c]))[:3] == comparable(outcome(step_loop, [X, a, b, c]))[:3], (a, b, c)


def open_forms(x, a, b, c):
    s = 0.0
    for v in x[a:]:
        s = s * 2.0 + v
    for v in x[:b]:
        s = s * 2.0 + v
    for v in x[::c]:
        s = s * 2.0 + v
    for v in x[a::c]:
        s = s * 2.0 + v
    for v in x[:b:c]:
        s = s * 2.0 + v
    for v in x[:]:
        s = s * 2.0 + v
    for v in x[::]:
        s = s * 2.0 + v
    return s


def test_open_ended_slices():
    for a in (0, 3, -2, 100, -100):
        for b in (0, 3, -2, 100, -100):
            for c in (1, 2, -1, -3):
                same(open_forms, X, a, b, c)


def test_extreme_bounds():
    big = 2**63 - 1
    for a, b, c in ((big, big, 1), (-big, big, 1), (big, -big, -1), (-2**63, 2**63 - 1, 1), (0, 5, big), (5, 0, -big), (9, -big, -2**63), (0, 10, -2**63)):
        same(step_loop, X, a, b, c)


def test_on_empty():
    e = X[:0]
    same(slice_loop, e, 0, 5)
    same(open_forms, e, 1, 1, -1)
    same(rows, M[:0])
    same(rows, M[:, :0])
    same(row_sum, M[:, :0], 0)


def test_contiguity_is_only_assumed_when_true():
    same(block, M, 1, 3, 2, 5)
    same(block, M, 0, 4, 0, 6)
    same(block, M, 0, 0, 0, 6)
    same(block, M, 3, 1, 0, 6)
    same(block, M, -2, 100, -3, 100)


# --- rows, columns and chains


@pytest.mark.parametrize('i', [0, 3, -1, -4, 4, -5, 100])
def test_rows(i):
    same(row_sum, M, i)
    same(out_of_range_row, M, i)
    for j in (0, 5, -1, -6, 6, -7):
        same(chain, M, i, j)
        same(chain_store, M, i, j)


@pytest.mark.parametrize('j', [0, 5, -1, -6])
def test_columns(j):
    same(column, M, j)


@pytest.mark.parametrize('j', [6, -7])
def test_column_out_of_range(j):
    same(column, M, j)


def test_iterating_rows():
    same(rows, M)
    same(rows, M[1:2])
    same(rows_written, M)
    for a, b in ((0, 4), (1, 3), (3, 1), (-2, 100)):
        same(rows_of_view, M, a, b)
    same(rows_strided, M)


def test_three_dimensions():
    for i in (0, 1, -1):
        for j in (0, 2, -1):
            same(cube, C, i, j)
    same(cube, C, 2, 0)
    same(cube, C, 0, 3)


# --- writes


@pytest.mark.parametrize('a, b, c', [(0, 10, 1), (2, 8, 2), (9, 0, -1), (9, 0, -3), (5, 5, 1), (0, 100, 3), (-3, -100, -1)])
def test_writes_reach_the_array(a, b, c):
    same(writes_through, X, a, b, c)


def test_writes_show_through_every_view():
    same(write_seen, X)
    same(assign_one_to_another, X)


def test_overlapping_views_read_memory():
    same(overlap, X)
    same(forward_through_view, X)
    same(forward_with_other_view, X)


# --- what assumed arrays never change


def test_rebinding():
    same(rebinds, X)
    same(rebinds_in_loop, M)


def test_a_length_that_changes_under_a_loop():
    same(rebound_mid_loop, M, X)
    same(rebound_after_len, X)
    for b in (0, 5, -1):
        same(versioned_rebound, X, b)


@pytest.mark.parametrize('k', [0, 3, -1, -4, 4, -5])
def test_negative_index_on_a_view(k):
    same(negative_index_on_view, X, k)


def test_errors():
    same(zero_step, X, 0)
    same(zero_step, X, 2)
    for i, c in ((99, 0), (1, 0), (99, 1)):
        same(zero_step_after_error, M, i, c)
    for i, j in ((0, 0), (9, 0), (0, 99), (-9, -9)):
        same(index_bound_order, M, i, j)


# --- what stays in Python


def returns_view(x):
    return x[1:]


def returns_parameter(x):
    return x


def view_arithmetic(x):
    return x[1:] * 2.0


def truth_of_view(x):
    if x[1:]:
        return 1
    return 0


def view_in_tuple(x):
    t = (x[1:], 1)
    return t[1]


def slice_assignment(x):
    x[1:3] = 0.0


def mask(x):
    return x[x > 2.0]


def fancy(x, y):
    return x[y]


def new_axis(x):
    return x[None]


def ellipsis(m):
    return m[...]


def float_bound(x):
    return x[1.5:]


def bool_index(x):
    return x[True]


def two_types(x, m, flag):
    v = x
    if flag:
        v = m
    return v[0]


def contiguous_or_not(x, flag):
    v = x[1:]
    if flag:
        v = x[::2]
    return v[0]


def too_many(x):
    return x[1, 2]


def test_stays_python_and_stays_right():
    for f, args in ((returns_view, (X,)), (returns_parameter, (X,)), (view_arithmetic, (X,)), (truth_of_view, (X[:1],)), (view_in_tuple, (X,)),
                    (slice_assignment, (X,)), (mask, (X,)), (fancy, (X, np.array([1, 2]))), (new_axis, (X,)), (ellipsis, (M,)), (float_bound, (X,)),
                    (bool_index, (X,)), (two_types, (X, M, True)), (two_types, (X, M, False)), (contiguous_or_not, (X, True)), (too_many, (X,))):
        g = scram(cache=False)(f)
        fresh = lambda: [a.copy() if isinstance(a, np.ndarray) else a for a in args]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, n = fresh(), fresh()
            want, got = outcome(f, p), outcome(g, n)
        assert want[:3] == got[:3] or (want[0] == 'returns' and repr(f(*fresh())) == repr(g(*fresh()))), f.__name__
        for a, b in zip(p, n):
            if isinstance(a, np.ndarray):
                assert a.tobytes() == b.tobytes()


def read_only_store(x):
    v = x[1:]
    v[0] = 1.0


def read_only_read(x):
    return x[1:][0]


def test_read_only_stays_read_only():
    x = X.copy()
    x.flags.writeable = False
    g = scram(cache=False)(read_only_store)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert outcome(g, [x])[:2] == outcome(read_only_store, [x])[:2]
    assert scram(cache=False, fallback=False)(read_only_read)(x) == read_only_read(x)


def narrow(x):
    v = x[1:]
    v[0] = 3
    return v[0], len(v)


def test_other_dtypes():
    for dt in ('i8', 'u1', 'i4', 'f8'):
        same(narrow, np.arange(5, dtype=dt))
    same(total, np.arange(5, dtype='f8')[::2])


def lifted(x, name):
    tag = {name: 1}
    s = 0.0
    for v in x[1:]:
        s += v
    tag['s'] = s
    return tag


def test_lifting():
    g = same(lifted, X, 'a', compiled=False)
    assert g._native_loops()


@pytest.mark.parametrize('options', [{'boundscheck': False}, {'parity': False}])
def test_options(options):
    same(writes_through, X, 2, 8, 2, **options)
    same(rows, M, **options)
    same(column, M, 2, **options)


def test_random_chains():
    rng = random.Random(4)

    def rnd():
        return rng.choice([None, rng.randint(-8, 8)])

    def piece():
        a, b = rnd(), rnd()
        c = rng.choice([None, None, 1, 2, -1, -2, 3])
        return ':'.join('' if v is None else str(v) for v in (a, b, c))
    for _ in range(120):
        s1, s2, s3 = piece(), piece(), piece()
        src = (f'def f(m):\n    v = m[{s1}, {s2}]\n    w = v[{s3}]\n    s = 0.0\n    for row in w:\n        for e in row[::-1]:\n'
               f'            s = s * 1.5 + e\n    h, k = w.shape\n    if h and k:\n        w[0, 0] = -1.0\n    return s, h, k\n')
        space = {}
        exec(src, space)
        import ast
        from scramjit._dispatch import Function
        g = Function(space['f'], cache=False, fallback=False)
        g._tree, g._why_not = ast.parse(src).body[0], None
        g._dict_keys, g._first = {}, 1
        a, b = M.copy(), M.copy()
        want, got = comparable(outcome(space['f'], [a])), comparable(outcome(g, [b]))
        assert want[:3] == got[:3], src
        assert a.tobytes() == b.tobytes(), src
