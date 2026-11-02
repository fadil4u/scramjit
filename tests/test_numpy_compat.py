"""Slice assignment, views as arguments, rows in enumerate and zip, .T and tuples of views, against numpy."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

X = np.arange(1.0, 11.0)
M = np.arange(1.0, 25.0).reshape(4, 6)
I = np.arange(10, dtype=np.int64)


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
                    assert p.tobytes() == c.tobytes(), (f.__name__, p, c)
    if compiled is not None:
        assert bool(g.signatures) == compiled, (f.__name__, g._why)
    return g


# --- slice assignment


def fill(x, a, b, c, v):
    x[a:b:c] = v


def fill_all(x, v):
    x[:] = v


def fill_row(m, i, v):
    m[i] = v


def fill_column(m, j, v):
    m[:, j] = v


def fill_block(m, a, b, c, d, v):
    m[a:b, c:d] = v


def fill_cell_view(m, i, a, b, v):
    m[i, a:b] = v


def copy_in(x, y, a, b):
    x[a:b] = y


def copy_view(x, y, a, b, c, d):
    x[a:b] = y[c:d]


def shift_right(x):
    x[1:] = x[:-1]


def shift_left(x):
    x[:-1] = x[1:]


def reverse_in_place(x):
    x[:] = x[::-1]


def every_other(x):
    x[::2] = x[1::2]


def same_view(x):
    x[2:5] = x[2:5]


def row_to_row(m, i, j):
    m[i] = m[j]


def row_to_rows(m, a, b, i):
    m[a:b] = m[i]


def overlap_2d(m):
    m[1:, 1:] = m[:-1, :-1]


def column_to_column(m, i, j):
    m[:, i] = m[:, j]


def transposed_into_itself(m):
    m[:] = m.T


def int_into_float(x, y):
    x[:] = y


def scalar_then_more(x, v):
    x[1:3] = v
    x[0] = x[1] + 1.0
    return x[2]


def nested_view(x):
    v = x[1:8]
    v[2:4] = -1.0
    w = v[::-1]
    w[0:2] = x[:2]


def through_view_target(m, i):
    m[i][1:3] = 9.0
    m[i, :2][0] = 8.0


@pytest.mark.parametrize('a, b, c', [(0, 10, 1), (2, 5, 1), (0, 10, 2), (9, 0, -1), (9, 0, -3), (5, 5, 1), (5, 2, 1), (-3, 100, 1), (-100, 3, 1), (100, 200, 1)])
def test_a_number_into_a_slice(a, b, c):
    same(fill, X, a, b, c, 7.5)
    same(fill, X, a, b, c, 7)
    same(fill, X, a, b, c, np.float64(7.5))
    same(fill, I, a, b, c, 7)
    same(fill, I, a, b, c, np.int64(-7))


def test_special_values_and_whole_arrays():
    for v in (float('nan'), -0.0, float('inf')):
        same(fill_all, X, v)
    same(fill, X, 0, 5, 0, 1.0)                 # ValueError: step zero


@pytest.mark.parametrize('i', [0, 3, -1, -4, 4, -5])
def test_rows(i):
    same(fill_row, M, i, 3.5)


@pytest.mark.parametrize('j', [0, 5, -1, -6, 6, -7])
def test_columns(j):
    same(fill_column, M, j, 3.5)


def test_blocks():
    for a, b, c, d in ((1, 3, 2, 5), (0, 100, 0, 100), (3, 1, 0, 6), (-2, 100, -3, 100)):
        same(fill_block, M, a, b, c, d, 1.5)
    same(fill_cell_view, M, 2, 1, 4, 1.5)
    same(fill_cell_view, M, 9, 1, 4, 1.5)


@pytest.mark.parametrize('n, a, b', [(3, 0, 3), (3, 2, 5), (1, 0, 10), (1, 5, 5), (0, 5, 5), (10, 0, 10), (4, 0, 3), (2, 0, 3), (3, 0, 0), (0, 0, 3), (10, 5, 100)])
def test_arrays_of_every_shape(n, a, b):
    same(copy_in, X, np.arange(n) * 1.5, a, b)             # ValueError, in numpy's words, if it doesn't broadcast


def test_views_in():
    for a, b, c, d in ((0, 3, 2, 5), (1, 4, 0, 1), (0, 4, 0, 3), (5, 5, 3, 3)):
        same(copy_view, X, X * 2.0, a, b, c, d)
    same(int_into_float, X, I)
    same(int_into_float, X, I[:3])


def test_overlap_acts_as_if_copied():
    for f in (shift_right, shift_left, reverse_in_place, every_other, same_view, nested_view):
        same(f, X)
    same(row_to_row, M, 0, 2)
    same(row_to_row, M, 1, 1)
    same(row_to_rows, M, 0, 3, 1)
    same(row_to_rows, M, 0, 3, 9)
    same(overlap_2d, M)
    same(column_to_column, M, 0, 3)
    same(through_view_target, M, 1)
    same(transposed_into_itself, np.arange(16.0).reshape(4, 4))
    same(transposed_into_itself, M)                        # ValueError: (6, 4) into (4, 6)


def test_broadcasting_a_row():
    same(row_to_rows, M, 1, 3, 0)
    same(row_to_rows, M, 3, 1, 0)


def broadcast_2d(m, y):
    m[1:3] = y


@pytest.mark.parametrize('shape', [(6,), (1, 6), (2, 1), (2, 6), (1,), (1, 1), (2, 5), (3, 6), (5,), (0,), (0, 6)])
def test_broadcasting(shape):
    y = np.arange(float(np.prod(shape))).reshape(shape)
    same(broadcast_2d, M, y)


def test_after_and_order():
    same(scalar_then_more, X, 2.5)


def store_float_in_int(x):
    x[1:] = 1.5


def narrow_fill(x, v):
    x[1:3] = v


def three_d(c, y):
    c[1:] = y


def into_read_only(x):
    x[1:] = 1.0


def test_stays_right_when_it_stays_python():
    same(store_float_in_int, I, compiled=False)
    same(three_d, np.zeros((2, 3, 4)), np.ones((1, 3, 4)), compiled=False)
    r = X.copy()
    r.flags.writeable = False
    g = scram(cache=False)(into_read_only)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert outcome(g, [r])[:2] == outcome(into_read_only, [r])[:2]


def test_narrow_elements_fit_or_raise():
    a = np.zeros(4, dtype=np.uint8)
    same(narrow_fill, a, 200)
    same(narrow_fill, a, 300)                              # OverflowError
    same(narrow_fill, a, -1)


def test_three_dimensions_with_a_number():
    def g(c):
        c[1] = 2.0
        c[0, 1] = 3.0
        c[0, 0, :2] = 4.0
    # local functions have no module globals to resolve; these need none
    same(g, np.zeros((2, 3, 4)))


# --- views coming in from Python


def walk(x):
    s = 0.0
    for i in range(len(x)):
        s = s * 1.5 + x[i]
    return s, len(x)


def walk_2d(m):
    s = 0.0
    h, w = m.shape
    for i in range(h):
        for j in range(w):
            s = s * 1.5 + m[i, j]
    return s, h, w, m.size, m.ndim


def writes(x):
    for i in range(len(x)):
        x[i] = i * 2.0


def writes_2d(m):
    k = 0.0
    for row in m:
        row[0] = k
        k += 1.0


def reduces(x):
    return x.sum(), x.max(), x.argmin(), x.any(), x.mean()


def through_view(f, base, how, compiled=True):
    """f on how(a copy of base), so what arrives really is a view, and the base must end up the same too."""
    g = scram(cache=False)(f)
    p, n = base.copy(), base.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        want, got = comparable(outcome(f, [how(p)])), comparable(outcome(g, [how(n)]))
    assert want[:4] == got[:4], (f.__name__, want, got)
    assert p.tobytes() == n.tobytes()
    assert bool(g.signatures) == compiled, g._why
    return g


ONE_D = [lambda a: a[::2], lambda a: a[::-1], lambda a: a[1:8:3], lambda a: a[::-3], lambda a: a[5:5:2], lambda a: a[1::4]]
TWO_D = [lambda a: a[::2], lambda a: a[:, ::2], lambda a: a[::-1], lambda a: a[:, ::-1], lambda a: a[1:3, ::2], lambda a: a.T, lambda a: a[::2, ::3].T, lambda a: a[:, 1:3]]


@pytest.mark.parametrize('how', ONE_D)
def test_views_as_arguments(how):
    g = through_view(walk, X, how)
    assert all(t.strided for key in g.signatures for t in key if t.kind == 'array') or how(X).flags.c_contiguous
    through_view(writes, X, how)


@pytest.mark.parametrize('how', TWO_D)
def test_two_dimensional_views_as_arguments(how):
    through_view(walk_2d, M, how)
    through_view(writes_2d, M, how)


def test_writes_reach_the_array_behind_a_view():
    a, b = np.arange(10.0), np.arange(10.0)
    v = scram(cache=False, fallback=False)(writes)
    v(a[::2])
    writes(b[::2])
    assert a.tobytes() == b.tobytes()
    c, d = M.copy(), M.copy()
    w = scram(cache=False, fallback=False)(writes_2d)
    w(c[::2, ::-1])
    writes_2d(d[::2, ::-1])
    assert c.tobytes() == d.tobytes()
    e, f = M.copy(), M.copy()
    w(e.T)
    writes_2d(f.T)
    assert e.tobytes() == f.tobytes()


def test_reductions_on_a_view_argument():
    for how in ONE_D[:2]:
        v = how(X)
        g = scram(cache=False, fallback=False)(reduces)
        assert repr(g(v)) == repr(reduces(v))


def test_what_cannot_be_passed_stays_python():
    rec = np.zeros(4, dtype=[('a', 'f8'), ('b', 'i4')])          # gaps that aren't a whole number of elements
    through_view(walk, np.zeros(1), lambda a: rec['a'], compiled=False)
    off = np.zeros(4, dtype=[('p', 'i4'), ('a', 'f8'), ('q', 'i4')])  # a whole number, from an address that isn't aligned
    through_view(walk, np.zeros(1), lambda a: off['a'], compiled=False)
    through_view(walk, np.zeros(1), lambda a: np.ones(3).astype('>f8'), compiled=False)


def test_a_broadcast_array_with_stride_zero():
    through_view(walk, np.zeros(1), lambda a: np.broadcast_to(np.array(2.5), (5,)))


def test_a_view_is_not_reused_for_a_contiguous_array():
    """The same types from Python, different layouts: each gets what it needs."""
    g = scram(cache=False, fallback=False)(walk)
    assert g(X)[0] == walk(X)[0]
    assert g(X[::2])[0] == walk(X[::2])[0]
    assert g(X)[0] == walk(X)[0]
    assert len(g.signatures) == 2


# --- rows, in enumerate and zip


def rows_enumerated(m):
    s = 0.0
    for i, row in enumerate(m):
        s += i * row[0]
        row[1] = -1.0
    return s


def rows_zipped(a, b):
    s = 0.0
    for p, q in zip(a, b):
        s = s * 1.5 + p[0] * q[1]
    return s


def rows_and_numbers(m, x):
    s = 0.0
    for row, v in zip(m, x):
        s += row[1] * v
    return s


def enumerate_view(m):
    s = 0.0
    for i, row in enumerate(m[1:3], 5):
        s += i + row[0]
    return s


def row_length_changes(m):
    s = 0.0
    for i, r in enumerate(m):
        s += r[len(r) - 1]
    return s


def builtin_sum_of_rows(m):
    return sum(m)


def rows_unpacked(m):
    s = 0.0
    for a, b in m:
        s += a * b
    return s


def rebinds_rows(m, n):
    s = 0.0
    for r in m:
        s += r[0]
        m = n
    return s


def test_rows_in_enumerate_and_zip():
    same(rows_enumerated, M)
    same(rows_zipped, M, M[:3] * 2.0)
    same(rows_and_numbers, M, X)
    same(enumerate_view, M)
    same(row_length_changes, M)
    same(rows_enumerated, M[:, :2])
    same(rows_zipped, M.T, M.T)


def comprehension_over_rows(m):
    return sum(r[0] for r in m)


def test_rows_stay_right_when_python():
    same(comprehension_over_rows, M, compiled=None)
    same(builtin_sum_of_rows, M, compiled=None)
    same(rows_unpacked, M[:, :2], compiled=False)
    same(rebinds_rows, M, M[:2], compiled=False)


# --- .T


def transposed(m):
    t = m.T
    return t[0, 1], t.shape[0], t.shape[1], m.T[1, 0], len(m.T)


def t_loop(m):
    s = 0.0
    for col in m.T:
        s = s * 1.5 + col[0]
    return s


def t_of_a_view(m):
    return m[1:3].T[0, 1], m[0].T[2], m.T.T[1, 2]


def t_store(m):
    m.T[0, 1] = -5.0
    m.T[1] = 1.0


def test_transpose():
    same(transposed, M)
    same(t_loop, M)
    same(t_of_a_view, M)
    same(t_store, M)
    same(transposed, np.arange(8.0).reshape(2, 2, 2), compiled=None)


# --- tuples holding views


def two_rows(m, i, j):
    a, b = m[i], m[j]
    return a[0] + b[1], len(a)


def swap_names(m):
    a, b = m[0], m[1]
    a, b = b, a
    return a[0], b[0]


def with_a_number(m, k):
    a, b, c = m[0], k * 2, m[1]
    return a[0] + b + c[0]


def rows_swap_aliased(m):
    m[0], m[1] = m[1], m[0]


def sliding(x):
    a, b = x[:3], x[3:]
    a[0], b[0] = b[0], a[0]
    return a[0], b[0], len(a), len(b)


def into_elements(m, x):
    x[0], r = m[0, 1], m[1]
    return r[0]


def too_few(m):
    a, b = m[0], m[1], m[2]
    return a[0]


def one_name(m):
    t = m[0], m[1]
    return t[0][0]


def test_tuples_of_views():
    same(two_rows, M, 0, 2)
    same(two_rows, M, 0, 9)
    same(swap_names, M)
    same(with_a_number, M, 3)
    same(rows_swap_aliased, M)                 # numpy's famous one: the rows don't really swap
    same(sliding, X)
    same(into_elements, M, X)


def test_tuples_stay_right_when_python():
    same(too_few, M, compiled=False)
    same(one_name, M, compiled=False)


@pytest.mark.parametrize('options', [{'boundscheck': False}, {'parity': False}])
def test_options(options):
    same(shift_right, X, **options)
    same(fill_row, M, 1, 2.0, **options)
    same(walk, X[::2], **options)
    same(rows_enumerated, M, **options)


# --- returning views


def returns_tail(x):
    return x[1:]


def returns_row(m, i):
    return m[i]


def returns_column(m, j):
    return m[:, j]


def returns_reversed(x):
    return x[::-1]


def returns_block(m):
    return m[1:3, ::2]


def returns_transpose(m):
    return m.T


def returns_itself(x):
    return x


def returns_through_a_name(x):
    y = x
    z = y
    return z


def returns_one_of_two(a, b, flag):
    v = a
    if flag:
        v = b
    return v[1:]


def returns_either(a, b, flag):
    if flag:
        return a
    return b[::2]


def writes_then_returns(x):
    v = x[2:]
    v[0] = -1.0
    return v


def returns_empty(x):
    return x[5:5]


def returns_rebound(x):
    while len(x) > 2:
        x = x[1:]
    return x


def returns_from_a_dict(p, q):
    return q['a'][1:]


def returns_view_of_view(m):
    return m[1:][0][::-1]


def views_equal(a, b):
    assert type(a) is type(b) is np.ndarray
    assert a.dtype == b.dtype and a.shape == b.shape and a.ndim == b.ndim and a.strides == b.strides
    assert a.tobytes() == b.tobytes()
    assert a.flags.writeable == b.flags.writeable


def test_returned_views_are_numpy_views():
    for f, args in ((returns_tail, (X,)), (returns_row, (M, 2)), (returns_row, (M, -1)), (returns_column, (M, 3)), (returns_reversed, (X,)), (returns_block, (M,)),
                    (returns_transpose, (M,)), (returns_empty, (X,)), (returns_rebound, (X,)), (returns_view_of_view, (M,)), (writes_then_returns, (X,))):
        g = scram(cache=False, fallback=False)(f)
        a, b = [v.copy() if isinstance(v, np.ndarray) else v for v in args], [v.copy() if isinstance(v, np.ndarray) else v for v in args]
        want, got = f(*a), g(*b)
        views_equal(want, got)
        if got.size:
            assert np.shares_memory(got, b[0]), f.__name__
            got.flat[0] = -77.0                    # a write through the result reaches the argument
            want.flat[0] = -77.0
            assert a[0].tobytes() == b[0].tobytes(), f.__name__


def test_returning_an_argument_returns_that_very_object():
    x = X.copy()
    assert scram(cache=False, fallback=False)(returns_itself)(x) is x
    assert scram(cache=False, fallback=False)(returns_through_a_name)(x) is x
    assert scram(cache=False, fallback=False)(returns_tail)(x) is not x


def test_which_argument():
    for flag in (True, False):
        a, b = X.copy(), X.copy() * 2.0
        r = scram(cache=False, fallback=False)(returns_one_of_two)(a, b, flag)
        assert np.shares_memory(r, b if flag else a) and not np.shares_memory(r, a if flag else b)
        views_equal(r, returns_one_of_two(a, b, flag))
        r = scram(cache=False, fallback=False)(returns_either)(a, b, flag)
        views_equal(r, returns_either(a, b, flag))
        assert (r is a) == flag


def test_returned_view_keeps_its_array_alive():
    import gc
    g = scram(cache=False, fallback=False)(returns_tail)
    r = g(np.arange(1000.0))
    gc.collect()
    junk = [np.zeros(1000) for _ in range(50)]
    assert r[0] == 1.0 and r[998] == 999.0 and junk


def test_read_only_stays_read_only_when_returned():
    x = X.copy()
    x.flags.writeable = False
    r = scram(cache=False, fallback=False)(returns_tail)(x)
    assert not r.flags.writeable
    views_equal(r, returns_tail(x))


def test_from_a_dict_and_a_view_argument():
    x = X.copy()
    r = scram(cache=False, fallback=False)(returns_from_a_dict)(1, {'a': x})
    assert np.shares_memory(r, x) and r[0] == x[1]
    v = X.copy()[::2]
    r = scram(cache=False, fallback=False)(returns_tail)(v)
    views_equal(r, returns_tail(v))
    assert np.shares_memory(r, v)


def returns_two(x):
    return x[1:], 3


def returns_kinds(x, flag):
    if flag:
        return x
    return 1.0


def test_returning_views_stays_right_when_python():
    g = scram(cache=False)(returns_two)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = g(X)
    assert r[1] == 3 and np.array_equal(r[0], X[1:])
    g = scram(cache=False)(returns_kinds)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert g(X, True) is X and g(X, False) == 1.0


def lifted_returns_view(x, name):
    tag = {name: 1}
    v = x
    for i in range(2):
        v = v[1:]
    tag['n'] = len(v)
    return v, tag


def test_lifted_view_comes_back():
    x = X.copy()
    g = scram(cache=False)(lifted_returns_view)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v, tag = g(x, 'a')
    w, t2 = lifted_returns_view(x, 'a')
    views_equal(v, w)
    assert tag == t2 and np.shares_memory(v, x)
    assert g._native_loops()


# --- std, var, integer mean, and the rest


RNG = np.random.default_rng(31)
LENGTHS = [0, 1, 2, 7, 8, 9, 64, 128, 129, 1000, 8191, 8192, 8193, 20000, 70000]


def stats_of(x):
    return x.std(), x.var(), np.std(x), np.var(x), x.mean(), np.mean(x), x.std(ddof=1), np.var(x, ddof=1), x.var(ddof=2), x.var(axis=None), x.std(0), x.var(axis=0, ddof=1)


@pytest.mark.parametrize('n', LENGTHS)
def test_stats_round_exactly(n):
    for x in (RNG.random(n) * 10.0 ** RNG.integers(-3, 3, n), RNG.integers(-10**15, 10**15, n)):
        same(stats_of, x)
        through_view(stats_of, x, lambda a: a[::2])
        through_view(stats_of, x, lambda a: a[::-1])


def shape_stats(x):
    return x.std(), x.var(), x.mean()


def test_stats_of_shapes_and_values():
    for shape in ((2, 3), (10, 10), (2, 3, 4), (200, 100)):
        same(shape_stats, RNG.random(shape))
    for v in ([np.nan], [1.0, np.nan], [np.inf, 1.0], [np.inf, -np.inf], [0.0, -0.0], [1e308, -1e308], [5.0, 5.0]):
        same(stats_of, np.array(v))


def ddof_var(x, d):
    return x.var(ddof=d), x.std(ddof=d)


@pytest.mark.parametrize('d', [0, 1, 2, 5, 9, 10, 11, 100, -1, -5, np.int64(2)])
def test_ddof(d):
    same(ddof_var, RNG.random(10), d)
    same(ddof_var, RNG.integers(0, 100, 10), d)


def test_ddof_that_is_not_an_integer_stays_python():
    same(ddof_var, RNG.random(10), 1.5, compiled=False)


@pytest.mark.parametrize('how', ['axis=1', 'keepdims=True', 'dtype=float', 'out=y', 'where=z', 'mean=0.0', 'correction=1'])
def test_other_std_arguments_stay_python(how):
    src = f'def f(x, y, z):\n    return x.std({how})\n'
    space = {}
    exec(src, space)
    import ast
    from scramjit._dispatch import Function
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._dict_keys, g._first = {}, 1
    a = RNG.random((3, 4))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert outcome(g, [a, np.zeros(4), a > 0.5])[:1] == outcome(space['f'], [a, np.zeros(4), a > 0.5])[:1]
    assert not g.signatures


def small_functions(x, lo, hi):
    return np.round(x), np.around(x), np.clip(x, lo, hi), np.degrees(x), np.radians(x), np.rad2deg(x), np.deg2rad(x)


VALUES = [0.0, -0.0, 0.5, -0.5, 1.5, 2.5, -2.5, 3.5, 0.4, -0.4, 1e300, np.inf, -np.inf, np.nan, 90.0, 180.0, 1.2345]


def test_small_functions():
    for x in VALUES:
        for lo, hi in ((0.0, 1.0), (-0.0, 0.0), (0.0, -0.0), (5.0, 1.0), (np.nan, 1.0), (0.0, np.nan), (-np.inf, np.inf)):
            same(small_functions, x, lo, hi)
            same(small_functions, np.float64(x), lo, hi)


def round_int(n):
    return np.round(n)


def round_places(x):
    return np.round(x, 2)


def clip_ints(n):
    return np.clip(n, 0, 1)


def test_small_functions_on_ints_stay_python():
    same(round_int, 3, compiled=False)
    same(round_places, 1.2345, compiled=False)
    same(clip_ints, 3, compiled=False)


def arg_ints(x):
    i = x.argmax()
    j = np.argmin(x)
    return x[i], x[j], i, j


def test_argmax_gives_what_numpy_does():
    same(arg_ints, np.array([3.0, 9.0, 1.0]))


def test_a_view_holds_the_array_it_is_of():
    """The address alone would look right with the wrong one: this is what keeps the memory alive."""
    for flag in (True, False):
        a, b = X.copy(), X.copy() * 2.0
        r = scram(cache=False, fallback=False)(returns_one_of_two)(a, b, flag)
        assert r.base.base is (b if flag else a)


def test_a_writable_and_a_read_only_one_in_one_variable_stay_python():
    a, b = X.copy(), X.copy()
    b.flags.writeable = False
    g = scram(cache=False)(returns_one_of_two)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = g(a, b, True)
    assert not r.flags.writeable and np.shares_memory(r, b) and not g.signatures


def second_of_two(a, b):
    v = b[1:]
    w = v[::2]
    for row in a:
        pass
    return w


def row_of_second(a, m):
    t = m[0]
    for row in m:
        t = row
    return t


def swapped_names(a, b):
    a, b = b, a
    return a[1:]


def test_every_way_of_making_a_view_keeps_its_array():
    a, b = X.copy(), X.copy() * 2.0
    r = scram(cache=False, fallback=False)(second_of_two)(a[:2].reshape(1, 2), b)
    assert r.base.base is b
    m = M.copy()
    r = scram(cache=False, fallback=False)(row_of_second)(a, m)
    assert r.base.base is m and np.array_equal(r, m[-1])
    r = scram(cache=False, fallback=False)(swapped_names)(a, b)
    assert r.base.base is b and np.array_equal(r, b[1:])

