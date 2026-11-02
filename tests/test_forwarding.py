"""A temporary instead of y[i - 1] must never be seen: same results, same arrays, same errors, whatever
else the loop does and whichever arrays share memory."""

import numpy as np
import pytest

from scramjit import scram

from util import outcome


def ema(x, y, a):
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]


def running(y):
    for i in range(1, len(y)):
        y[i] = y[i] + y[i - 1]


def from_zero(y):                   # y[-1] on the first iteration: Python's last element
    for i in range(len(y)):
        y[i] = y[i - 1] + 1


def later_start(y, lo):
    for i in range(lo, len(y)):
        y[i] = y[i - 1] * 2 + 1


def read_early(x, y):
    s = 0.0
    for i in range(1, len(x)):
        p = y[i - 1]
        s += p
        y[i] = p + x[i]
    return s


def read_in_if(x, y):
    for i in range(1, len(x)):
        v = x[i]
        if v > 0.5:
            v = v + y[i - 1]
        y[i] = v


def read_after(x, y):               # after the store: still memory, and still y[i - 1]
    s = 0.0
    for i in range(1, len(x)):
        y[i] = x[i] + y[i - 1]
        s += y[i - 1]
    return s


def two_stores(x, y, z):            # z can be a view of y
    for i in range(1, len(x)):
        y[i] = x[i] + y[i - 1]
        z[i] = x[i] * 2.0


def conditional_store(x, y):
    for i in range(1, len(x)):
        p = y[i - 1]
        if x[i] > 0.5:
            y[i] = p + 1.0


def skips(x, y):
    for i in range(1, len(x)):
        if x[i] > 0.5:
            continue
        y[i] = y[i - 1] + 1.0


def breaks(x, y):
    for i in range(1, len(x)):
        y[i] = y[i - 1] + x[i]
        if y[i] > 3.0:
            break
    return y[0]


def stores_previous(x, y):
    for i in range(1, len(x)):
        y[i - 1] = y[i - 1] + x[i]


def lag_two(x, y):
    for i in range(2, len(x)):
        y[i] = y[i - 1] + y[i - 2] + x[i]


def raises_in_loop(x, y, k):
    for i in range(1, len(x)):
        y[i] = y[i - 1] + 1.0 / (i - k)


def fib_mod(y, m):
    for i in range(1, len(y)):
        y[i] = (y[i - 1] * 31 + i) % m


def same(f, *args, **options):
    """f on copies, plain and compiled: same outcome, same arrays. Arrays that share memory stay
    shared, since copies of a view would hide the bug: pass a function building args instead."""
    build = args[0] if len(args) == 1 and callable(args[0]) else lambda: [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    plain, compiled = build(), build()
    want = outcome(f, plain)
    got = outcome(scram(fallback=False, cache=False, **options)(f), compiled)
    assert want[:3] == got[:3], f.__name__
    for p, c in zip(plain, compiled):
        if isinstance(p, np.ndarray):
            assert p.tobytes() == c.tobytes(), f.__name__


X = np.random.default_rng(3).random(40)
SIZES = [0, 1, 2, 3, 17, 40]


@pytest.mark.parametrize('n', SIZES)
def test_results(n):
    x, y = X[:n].copy(), np.zeros(n)
    for f, args in ((running, (x,)), (from_zero, (x,)), (read_early, (x, y)), (read_in_if, (x, y)),
                    (read_after, (x, y)), (conditional_store, (x, y)), (skips, (x, y)), (breaks, (x, y)),
                    (stores_previous, (x, y)), (lag_two, (x, y)), (two_stores, (x, y, np.zeros(n)))):
        same(f, *args)
    if n:
        same(ema, x, y, 0.3)
    for lo in (1, 2, 5, 39, 40, 41):
        same(later_start, x, lo)


@pytest.mark.parametrize('n', [0, 1, 2, 5])
def test_second_array_too_short(n):
    x, y = X[:10].copy(), np.zeros(n)
    same(read_early, x, y)          # IndexError where Python raises, the earlier stores kept
    same(ema, x, np.zeros(max(n, 1)), 0.3)
    same(read_in_if, x, y)


def test_bits_survive():
    y = np.array([np.nan, -0.0, np.inf, -np.inf, 5e-324, 1.0, -0.0, np.nan])
    def copies(y):
        for i in range(1, len(y)):
            y[i] = y[i - 1]
    same(copies, y)
    same(running, y)


def test_ints_wrap_and_stay_exact():
    y = np.array([2**62, 2**62, 2**63 - 1, -2**63, 1], dtype=np.int64)
    same(running, y)
    same(fib_mod, np.ones(50, dtype=np.int64), 1000003)


@pytest.mark.parametrize('dtype', ['f4', 'i4', 'i2', 'i1', 'u1', 'u8'])
def test_other_dtypes_read_memory(dtype):
    """A narrow dtype converts on store, so the C value isn't what a load gives."""
    def scale(y):
        for i in range(1, len(y)):
            y[i] = y[i - 1] * 3 + 1
    y = np.ones(30, dtype=dtype)
    try:
        scale(y.copy())
    except Exception:  # noqa: BLE001
        pytest.skip('overflows in Python')
    f = scram(cache=False)(scale)
    z = y.copy()
    f(z)
    w = y.copy()
    scale(w)
    assert z.tobytes() == w.tobytes()


def test_same_array_twice():
    def build():
        y = X[:20].copy()
        return [y, y]
    same(read_early, build)
    same(read_after, build)
    same(read_in_if, build)


@pytest.mark.parametrize('shift', [-3, -2, -1, 0, 1, 2, 3])
def test_views_overlap(shift):
    """x and y over one buffer, offset: x[i] sees what y stored, as it must."""
    def build():
        b = X[:30].copy()
        lo, hi = (3, 3 + shift) if shift >= 0 else (3 - shift, 3)
        return [b[lo:lo + 20], b[hi:hi + 20]]
    same(read_early, build)
    same(read_in_if, build)
    same(ema, lambda: build() + [0.25])


@pytest.mark.parametrize('shift', [-2, -1, 0, 1, 2])
def test_second_store_through_a_view(shift):
    """z[i] lands on y[i - 1] when shift is -1: a temporary would miss it, so no forwarding here."""
    def build():
        b = X[:30].copy()
        y = b[5:25]
        return [X[:20].copy(), y, b[5 + shift:25 + shift]]
    same(two_stores, build)


def test_raises_partway():
    x = X[:20].copy()
    same(raises_in_loop, x, np.zeros(20), 7)    # ZeroDivisionError at i == 7, with 6 stores kept
    same(raises_in_loop, x, np.zeros(20), 100)


def test_across_interrupt_chunks():
    y = np.ones(200_000)
    same(halving, y)


def halving(y):
    for i in range(1, len(y)):
        y[i] = y[i - 1] * 0.5 + 1.0


@pytest.mark.parametrize('options', [{'boundscheck': False}, {'parity': False}, {'boundscheck': False, 'parity': False}])
def test_options(options):
    x, y = X.copy(), np.zeros(40)
    same(ema, x, y, 0.3, **options)
    same(read_early, x, y, **options)
    same(from_zero, x, **options)


def reads_of(f, *args, **options):
    c = scram(fallback=False, cache=False, **options)(f)
    c(*args)
    return c.c()


def test_it_happens_where_it_should():
    x, y = X.copy(), np.zeros(40)
    assert ' - 1 < v_y__n ?' in reads_of(ema, x, y, 0.3)
    assert ' - 1 < v_y__n ?' in reads_of(ema, x, y, 0.3, boundscheck=False)


@pytest.mark.parametrize('f, args', [
    (two_stores, 3), (conditional_store, 2), (skips, 2), (read_after, 2), (stores_previous, 2), (from_zero, 1),
])
def test_it_stays_away_where_it_must(f, args):
    pool = (X.copy(), np.zeros(40), np.zeros(40))
    assert ' - 1 < v_' not in reads_of(f, *pool[-args:] if args == 1 else pool[:args])
