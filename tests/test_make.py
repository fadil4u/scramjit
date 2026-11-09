"""Arrays made inside a function: np.zeros(n), x.copy() and friends compile, and what comes back is what numpy
would have made. Memory a loop stops using is freed as it goes, and memory still in use is not."""

import sys
import warnings

import numpy as np
import pytest

from scramjit import scram
from scramjit._dispatch import Compiled
from util import maxrss_kb

NEP = np.dtype(int).name == 'int64'


def outcome(fn, args):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return 'value', fn(*args)
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, str(e)


def same(p, q):
    if isinstance(p, tuple):
        return isinstance(q, tuple) and len(p) == len(q) and all(same(a, b) for a, b in zip(p, q))
    if isinstance(p, np.ndarray):
        return isinstance(q, np.ndarray) and p.dtype == q.dtype and p.shape == q.shape and p.tobytes() == q.tobytes() \
            and p.flags.writeable == q.flags.writeable
    return type(p) is type(q) and (p == q or p != p)


def compiled(f):
    return any(isinstance(c, Compiled) for c in f._cache.values())


def check(fn, *args, must_compile=True):
    def copies():
        return [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    want_args, got_args = copies(), copies()
    f = scram(fn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        want, got = outcome(fn, want_args), outcome(f, got_args)
    assert want[0] == got[0], (want, got)
    assert same(want[1], got[1]), (want, got)
    for a, b in zip(want_args, got_args):
        assert same(a, b), 'an argument changed differently'
    if must_compile:
        assert compiled(f), f'{fn.__name__} did not compile'
    return got[1]


def zeros(n):
    return np.zeros(n)


def ones(n):
    return np.ones(n)


def full(n, v):
    return np.full(n, v)


def tuple_shape(a, b):
    return np.zeros((a, b)), np.ones((b, a)), np.full((a, b), 2.5), np.zeros([a, b]), np.empty((0, a))


def dtypes(n):
    return (np.zeros(n, dtype=np.float32), np.ones(n, np.int32), np.zeros(n, dtype=np.uint8), np.ones(n, 'float64'),
            np.zeros(n, float), np.ones(n, dtype='i2'), np.full(n, 7, np.int16), np.full(n, 1.5, dtype=np.float32))


@pytest.mark.parametrize('n', [0, 1, 7, 1000])
def test_made_from_a_shape(n):
    check(zeros, n)
    check(ones, n)
    check(full, n, 2.5)
    check(full, n, 3, must_compile=NEP)      # on wasm32 the dtype depends on the number's size
    check(full, n, np.float64(1.25))
    check(tuple_shape, n, 2)
    check(dtypes, n, must_compile=not wraps_python_ints())


def test_zeros_are_zero_after_memory_is_reused():
    def dirty(n):
        total = 0.0
        for k in range(50):
            a = np.ones(n)
            a[0] = 5.0
            b = np.zeros(n)
            total += b.sum()
        return total
    assert check(dirty, 300) == 0.0
    assert check(dirty, 200_000) == 0.0


@pytest.mark.parametrize('bad', [-1, -5, 2 ** 62, 2 ** 45 * 2 ** 16])
def test_numpys_errors(bad):
    check(zeros, bad)
    check(ones, bad)
    check(full, bad, 1.0)


def test_numpys_errors_for_shapes():
    def f(a, b):
        return np.zeros((a, b))
    check(f, -1, 3)
    check(f, 3, -1)
    check(f, 2 ** 40, 2 ** 40)
    check(f, 0, 2 ** 62)


def test_an_error_has_the_line():
    def f(n):
        y = np.zeros(2)
        y = np.zeros(n)
        return y
    with pytest.raises(ValueError, match='negative dimensions') as e:
        scram(f)(-1)
    assert any(t.tb_lineno == f.__code__.co_firstlineno + 2 for t in iter_tb(e.value.__traceback__))


def iter_tb(tb):
    while tb:
        yield tb
        tb = tb.tb_next


def wraps_python_ints():
    try:
        np.full(1, 300, dtype=np.uint8)
    except OverflowError:
        return False
    return True


def test_a_python_int_too_big_for_the_dtype():
    def f(n, v):
        return np.full(n, v, dtype=np.uint8)

    def g(x, v):
        return np.full_like(x, v)
    must = not wraps_python_ints()      # numpy's versions differ, so an older one stays Python
    check(f, 2, 200, must_compile=must)
    check(f, 2, 300, must_compile=must)
    check(f, -1, 300, must_compile=must)
    check(g, np.zeros(2, dtype=np.uint8), 300, must_compile=must)


def test_likes():
    def f(x):
        return np.zeros_like(x), np.ones_like(x), np.full_like(x, 7), np.empty_like(x)[:0], np.zeros_like(x, dtype=np.float32)
    for x in (np.arange(5.0), np.arange(6.0).reshape(2, 3), np.arange(4, dtype=np.int32), np.arange(9.0)[::2], np.arange(0.0)):
        check(f, x, must_compile=x.dtype != np.int32 or not wraps_python_ints())


def test_likes_of_a_view_with_another_memory_layout_stay_python():
    def f(m):
        return np.zeros_like(m.T), np.copy(m.T), np.array(m.T)
    m = np.arange(6.0).reshape(2, 3)
    r = check(f, m, must_compile=False)
    assert r[0].flags.f_contiguous and not r[0].flags.c_contiguous


def test_copies():
    def f(x):
        y = x.copy()
        y[0] = 99.0
        return y, x[0]

    def g(x):
        return np.copy(x), np.array(x), np.asarray(x), x.copy()
    for x in (np.arange(4.0), np.arange(6.0).reshape(3, 2), np.arange(12.0).reshape(2, 3, 2), np.arange(9.0)[::-2]):
        check(f, x)
        check(g, x)


def test_a_copy_of_a_view_is_in_c_order():
    def f(m):
        return m.T.copy(), m[:, 1].copy(), m[::-1, ::2].copy(), m.flatten(), m.T.flatten()
    check(f, np.arange(12.0).reshape(3, 4))


def test_a_copy_is_writable_and_doesnt_write_through():
    def f(x):
        return x.copy()
    x = np.arange(3.0)
    x.flags.writeable = False
    r = scram(f)(x)
    assert r.flags.writeable and r.base is None and r.flags.owndata
    r[0] = 5.0
    assert x[0] == 0.0


def test_asarray_is_the_array_itself():
    def f(x):
        return np.asarray(x)
    x = np.arange(3.0)
    assert scram(f)(x) is x


def test_astype():
    def f(x):
        return x.astype(float), x.astype(np.float32), x.astype('float64')

    def g(x):
        return x.astype(np.int32), x.astype(np.uint8), x.astype(np.int64)
    for x in (np.arange(5), np.array([-1, 2 ** 40, 300]), np.arange(5.0) + 0.5, np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3)[:, ::2]):
        check(f, x)
    for x in (np.arange(5), np.array([-1, 2 ** 40, 300])):
        check(g, x)
    check(g, np.arange(5.0) + 0.5, must_compile=False)


@pytest.mark.parametrize('args', [(0,), (1,), (5,), (-3,), (2 ** 62,), (0.0,), (1.0,), (5.5,), (-3.5,), (float('nan'),), (float('inf'),)])
def test_arange_of_one(args):
    def f(n):
        return np.arange(n)
    check(f, *args, must_compile=NEP or isinstance(args[0], int))


@pytest.mark.parametrize('args', [(0, 10, 3), (10, 0, -3), (0, 10, -1), (0, 10, 0), (0.0, 1.0, 0.1), (0.0, 1.0, 0.0), (1, 2, 0.1), (0.1, 0.9, 0.05),
                                  (-1.3, 7.7, 0.33), (1e10, 1e10 + 5, 0.7), (0.0, float('inf'), 1.0), (float('nan'), 1.0, 1.0), (0, 1, float('inf')),
                                  (1, 0, float('inf')), (1e-320, 2e-320, 1e308), (np.float64(0.0), np.float64(1.0), np.float64(0.0)),
                                  (0, 2 ** 62, 2 ** 60), (2 ** 62, 2 ** 62 + 3, 1), (5, 5, 1), (0.3, 0.3, 0.1)])
def test_arange_of_three(args):
    def f(a, b, c):
        return np.arange(a, b, c)
    check(f, *args, must_compile=NEP or any(isinstance(a, float) for a in args))


@pytest.mark.parametrize('args', [(0.0, 1.0, 5), (0, 1, 5), (0.0, 1.0, 1), (0.0, 1.0, 0), (0.0, 1.0, -1), (1.0, 1.0, 4), (-3.3, 7.1, 17), (1e-320, 2e-320, 5),
                                  (0.0, 1e308, 5), (-1e308, 1e308, 5), (0.0, float('inf'), 3), (float('nan'), 1.0, 3), (0.1, 0.2, 1000), (np.float64(0.1), 0.7, 9)])
def test_linspace(args):
    def f(a, b, n):
        return np.linspace(a, b, n), np.linspace(a, b, n, endpoint=False), np.linspace(a, b, num=n, endpoint=True)
    check(f, *args)


@pytest.mark.parametrize('n', [pytest.param(2 ** 40, marks=pytest.mark.skipif(sys.platform == 'darwin', reason='macOS hands out 8 TB lazily and kills the process filling it, numpy or not')), 2 ** 62, 2 ** 63 - 1])
def test_linspace_of_too_many(n):
    def f(n):
        return np.linspace(0.0, 1.0, n)
    r = outcome(scram(f), [n])
    assert r == outcome(f, [n])


def test_linspace_default_number():
    def f(a, b):
        return np.linspace(a, b)
    check(f, 0.0, 1.0)
    check(f, 3, 9)


@pytest.mark.parametrize('a,b', [(1.5, 2), (1, 2.5), (np.float64(1.5), 2), (1, 2)])
def test_array_of_numbers(a, b):
    def f(a, b):
        return np.array([a, b]), np.array([[a, b], [b, a], [a, a]]), np.array([a, b], dtype=np.float64)
    check(f, a, b, must_compile=NEP or isinstance(a, float) or isinstance(b, float))


def test_array_of_nothing_and_of_lists():
    def nothing():
        return np.array([])

    def of_floats(n):
        v = [0.0] * n
        for i in range(n):
            v[i] = i * 0.5
        return np.array(v)

    def of_ints(n):
        v = [i for i in range(n)]
        return np.array(v, dtype=np.int64)

    def no_dtype(n):
        v = [i for i in range(n)]
        return np.array(v)

    def ragged(a):
        return np.array([[a, a], [a]])

    def too_big(a):
        return np.array([a, 300], dtype=np.uint8)
    check(nothing)
    check(of_floats, 3)
    check(of_floats, 0)
    check(of_ints, 3)
    check(of_ints, 0)
    check(no_dtype, 3, must_compile=False)
    assert check(no_dtype, 0, must_compile=False).dtype == np.float64
    check(ragged, 1, must_compile=False)
    check(too_big, 1)


def test_used_like_any_array():
    def f(n):
        y = np.zeros(n)
        for i in range(n):
            y[i] = i * 2.0
        row = y[1:]
        row[0] = 50.0
        z = y
        z[n - 1] = 7.0
        s = 0.0
        for v in y:
            s += v
        return y, s, y.sum(), len(y), y.max(), np.mean(y)
    check(f, 5)

    def g(h, w):
        m = np.ones((h, w))
        for i, row in enumerate(m):
            row[1:] = i
        m[0] = 9.0
        m[:, 0] = m[:, 1]
        return m, m.T, m[1], m[:, 1], m.shape[0], m.sum()
    check(g, 3, 4)


def test_made_inline():
    def f(x, n):
        x[1:] = np.zeros(len(x) - 1)
        s = np.ones(n).sum()
        t = np.ones((n, 2))[1:].sum()
        for v in np.arange(0.0, 3.0):
            s += v
        for i, row in enumerate(np.ones((2, 3))):
            s += row[0] * i
        return s, t, np.zeros(n).T, len(np.zeros(n)), x.copy()[1:]
    check(f, np.arange(4.0), 3)


def test_two_at_once():
    def f(n):
        a, b = np.zeros(n), np.ones(n)
        a[:] = b
        b[0] = 5.0
        a, b = b, a
        return a, b
    check(f, 3)


def test_swapping_buffers_between_iterations():
    def f(n, steps):
        prev, cur = np.zeros(n), np.ones(n)
        for k in range(steps):
            for i in range(1, n - 1):
                cur[i] = (prev[i - 1] + prev[i + 1]) / 2 + 1
            prev, cur = cur, prev
        return prev, cur
    check(f, 8, 5)


def test_a_new_buffer_each_iteration():
    def f(n, steps):
        cur = np.zeros(n)
        for k in range(steps):
            nxt = np.zeros(n)
            for i in range(n):
                nxt[i] = cur[i] + k
            cur = nxt
        return cur
    check(f, 6, 5)


# ---- what comes back
def test_it_is_what_numpy_would_have_made():
    def f(n):
        return np.zeros(n)
    r = scram(f)(3)
    assert type(r) is np.ndarray and r.base is None and r.flags.owndata and r.flags.writeable and r.flags.aligned
    r.resize(5, refcheck=False)


def test_one_array_returned_twice_is_one_object():
    def f(n):
        y = np.zeros(n)
        z = y
        return y, z, y.T
    a, b, c = scram(f)(3)
    assert a is b
    assert c.base is not None and np.shares_memory(a, c)


def test_a_view_of_what_was_made_is_a_view_of_the_result():
    def f(n):
        y = np.zeros(n)
        return y[1:], y, y[::2]
    v, y, w = scram(f)(4)
    assert y.base is None and y.flags.owndata
    assert v.base.base is y and w.base.base is y
    v[0] = 5.0
    assert y[1] == 5.0
    w[1] = 6.0
    assert y[2] == 6.0
    assert w.strides == (16,)


def test_a_view_outlives_its_name():
    def f(n):
        m = np.ones((n, n))
        m[1, 1] = 5.0
        return m[1]
    r = scram(f)(3)
    assert r.tolist() == [1.0, 5.0, 1.0]
    assert r.base.base.shape == (3, 3)


def test_the_argument_comes_back_as_itself():
    def f(x, flag):
        y = x
        if flag:
            y = np.zeros(3)
        return y
    x = np.arange(3.0)
    g = scram(f)
    for _ in range(3):
        assert g(x, False) is x
        r = g(x, True)
        assert r is not x and r.tolist() == [0.0, 0.0, 0.0] and r.base is None


def test_arrays_and_numbers_in_a_tuple():
    def f(x, n):
        y = np.zeros(n)
        return x, y, y[0], n * 2, x[1:], float(n)
    x = np.arange(3.0)
    g = scram(f)
    for _ in range(3):
        a, b, c, d, e, h = g(x, 2)
        assert a is x and b.tolist() == [0.0, 0.0] and type(c) is np.float64 and d == 4 and h == 2.0
        assert e.base.base is x
    check(f, x, 2)


def test_tuples_with_different_returns():
    def f(x, flag):
        if flag:
            return x[::2], 1
        return x.copy(), 2
    check(f, np.arange(4.0), True)
    check(f, np.arange(4.0), False)
    g = scram(f)
    assert g(np.arange(4.0), True)[0].strides == (16,)
    assert g(np.arange(4.0), False)[0].strides == (8,)


def test_an_error_after_making_frees_and_raises():
    def f(n):
        y = np.zeros(n)
        y[n] = 1.0
        return y
    check(f, 3)


def test_many_calls_keep_memory_flat():
    def f(n):
        return np.zeros(n)
    g = scram(f)
    g(1000)
    entry = next(iter(g._cache.values()))
    module_free = entry.free
    frees = []

    def counting(p):
        frees.append(p)
        module_free(p)
    entry.free = counting
    for _ in range(50):
        g(1000)
    assert len(frees) == 50


# ---- freeing
def test_a_loop_making_arrays_does_not_keep_them():
    resource = pytest.importorskip('resource', reason='not on this platform', exc_type=ImportError)

    def f(k, n):
        total = 0.0
        for i in range(k):
            a = np.ones(n)
            total += a[i % n]
        return total
    g = scram(f)
    assert g(2, 1000) == 2.0
    before = maxrss_kb()
    assert g(4000, 125_000) == 4000.0        # 4 GB if kept
    grew = (maxrss_kb() - before) / 1024
    assert grew < 200, f'{grew} MB'


def test_views_keep_their_memory_through_a_sweep():
    def f(k, n):
        keep = np.zeros(4)
        total = 0.0
        for i in range(k):
            t = np.full(n, float(i))
            if i == 0:
                keep = t[1:5]
                keep[0] = 7.0
            total += keep[1] + t[0]
        return total, keep
    assert check(f, 3000, 2000)[1].tolist() == [7.0, 0.0, 0.0, 0.0]
    check(f, 40, 200_000)


def test_rows_being_looped_over_keep_their_memory():
    def f(k, n):
        total = 0.0
        for row in np.full((3, n), 2.0):
            for i in range(k):
                row2 = np.ones(n)
                total += row[0] + row2[0]
        return total
    check(f, 2000, 2000)


def test_a_chain_of_buffers_in_a_while_loop():
    def f(n, size):
        i = 0
        while np.zeros(size).sum() + i < n:
            i += 1
        return i
    check(f, 300, 100_000)


def test_not_inside_a_comprehension():
    def f(n):
        return sum([np.ones(3).sum() for i in range(n)])
    check(f, 3, must_compile=False)


# ---- with what else there is
def test_made_in_a_lifted_loop():
    def f(n, obj):
        y = np.zeros(n)
        for i in range(n):
            t = np.ones(3)
            t[0] = i
            y[i] = t.sum()
        z = str(obj)                         # doesn't compile, so the loop is lifted
        return y, z
    want = f(5, 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = scram(f)(5, 1)
    assert same(want, got)


def test_an_array_made_in_a_lifted_loop_comes_out():
    def f(n, obj):
        a = None
        for k in range(3):
            a = np.zeros(n)
            b = np.full(n, 2.0)
            a[0] = k
        return str(obj), a, b
    want = f(3, 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = scram(f)(3, 1)
    assert same(want, got)


def test_global_arrays_and_counts():
    g = scram(uses_global)
    assert g(2).tolist() == uses_global(2).tolist()
    counted = scram(uses_global, counts=True)
    assert counted(2).tolist() == uses_global(2).tolist()
    assert counted.line_counts()[0]


GLOBAL = np.arange(3.0)


def uses_global(k):
    y = GLOBAL.copy()
    y[0] = k
    return y


def test_dict_of_arrays():
    def f(p):
        y = p['a'].copy()
        y[0] = 5.0
        return y
    d = {'a': np.arange(3.0)}
    assert scram(f)(d).tolist() == [5.0, 1.0, 2.0]
    assert d['a'][0] == 0.0


def test_loops_over_a_made_array_lose_their_checks():
    def f(n):
        y = np.zeros(n)
        for i in range(len(y)):
            y[i] = i
        return y
    g = scram(f)
    g(3)
    entry = next(iter(g._cache.values()))
    assert 'SJ_FAIL(1' not in entry.source, 'y[i] with i in range(len(y)) needs no bounds check'


def test_rebinding_keeps_the_checks():
    def f(n):
        y = np.zeros(n)
        m = len(y)
        z = 0
        for k in range(2):
            for i in range(m):
                y[i] = i
                z += 1
            y = np.zeros(1)
        return y, z
    check(f, 3)


def test_one_assignment_in_a_loop_can_still_change_the_extent():
    def f():
        total = 0.0
        m = 0
        for k in range(2):
            y = np.zeros(3 - k * 2)
            if k == 0:
                m = len(y)
            for i in range(m):
                y[i] = 1.0
                total += y[i]
        return total
    check(f)


def test_what_isnt_returned_is_freed():
    def f(n):
        y = np.ones(n)
        return y.sum()
    g = scram(f, cache=False)
    assert g(1000) == 1000.0
    resource = pytest.importorskip('resource', reason='not on this platform', exc_type=ImportError)
    before = maxrss_kb()
    for _ in range(3000):
        g(1_000_000 // 8 * 8)
    grew = (maxrss_kb() - before) / 1024
    assert grew < 200, f'{grew} MB'


def test_a_large_array_comes_back_intact():
    def f(n):
        y = np.zeros(n)
        y[n - 1] = 3.0
        y[0] = 2.0
        return y, y[n // 2:]
    a, b = scram(f)(2_000_000)
    assert a[0] == 2.0 and a[-1] == 3.0 and a.sum() == 5.0 and b[-1] == 3.0


def test_a_lifted_loop_compiled():
    def f(n, obj):
        y = np.zeros(n)
        for i in range(n):
            t = np.ones(3)
            y[i] = t.sum() + i
        return y, str(obj)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = scram(f)
        assert g(3, 1)[0].tolist() == [3.0, 4.0, 5.0]
    assert any(compiled(loop) for loop in g._all_loops())


def test_preflight_knows_a_shape_is_an_int():
    import ast
    from scramjit import _preflight as P
    tree = ast.parse('''
import numpy as np
from scramjit import scram

@scram
def f(n, h, w, steps):
    a = np.zeros(n)
    m = np.ones((h, w))
    for _ in range(steps):
        a[0] += m[0, 0]
    return a, m
''')
    fdef = tree.body[2]
    a = P.best_analysis(fdef, P.module_globals(tree), 'x.py', 0, {'boundscheck': True, 'parity': True})
    assert a.state == 'compiled', [p.reason for p in a.problems]
