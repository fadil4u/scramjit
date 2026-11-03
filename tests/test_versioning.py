"""Loop versioning of var + base indexes: one test before the loop must never change a result, an
exception, or how far a loop got before raising."""

import random

import numpy as np
import pytest

from scramjit import scram

from util import outcome


def read(a, b, c, n):
    s = 0.0
    for j in range(n):
        s += a[b * c + j]
    return s


def read_swapped(a, b, n):
    s = 0.0
    for j in range(n):
        s += a[j + b]
    return s


def offsets(a, b, c, n):
    s = 0.0
    for j in range(n):
        s += a[b + j + 1] + a[b - c + j] + a[j + b + c - 1]
    return s


def write(a, b, n):
    for j in range(n):
        a[b + j] = j + 0.5


def base_changes(a, b, n):
    s = 0.0
    for j in range(n):
        s += a[b + j]
        b -= 1
    return s


def base_changes_late(a, b, n):
    s = 0.0
    for j in range(n):
        if j == 2:
            b = -100
        s += a[b + j]
    return s


def conditional(a, b, n):
    s = 0.0
    for j in range(n):
        if j < 2:
            s += a[b + j]
    return s


def start(a, b, lo, n):
    s = 0.0
    for j in range(lo, n):
        s += a[b + j]
    return s


def two_axes(m, b, c, n):
    s = 0.0
    for j in range(n):
        s += m[b, c + j] + m[b + j, c]
    return s


def mixed(a, x, b, c, n):
    for j in range(n):
        x[c + j] = a[b + j] * 2.0


def nested(a, dim, n):
    s = 0.0
    for i in range(n):
        for j in range(dim):
            s += a[i * dim + j]
    return s


VALUES = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, -1, -2, -3, -9, -10, -11, -12]
# beyond int64 once added to: compiled ints wrap where Python's don't, so Python can't judge these
HUGE = [2**62, -2**62, 2**63 - 1, 2**63 - 2, 2**63 - 5, -2**63, -2**63 + 3]


def same(f, *args, **options):
    """f on copies of args, plain and compiled: same outcome and same arrays afterwards."""
    copy = lambda: [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    plain, compiled = copy(), copy()
    want = outcome(f, plain)
    got = outcome(scram(fallback=False, cache=False, **options)(f), compiled)
    if want[0] == 'raises' and options.get('boundscheck') is False:
        return                      # out of bounds with checks off is the caller's problem
    assert want[:3] == got[:3], (f.__name__, args)
    for p, c in zip(plain, compiled):
        if isinstance(p, np.ndarray):
            assert p.tobytes() == c.tobytes(), (f.__name__, args)


@pytest.fixture
def a():
    return np.arange(10.0) + 1.0


@pytest.mark.parametrize('b', VALUES)
@pytest.mark.parametrize('n', [0, 1, 2, 5, 10, 11])
def test_base_and_stop(a, b, n):
    same(read_swapped, a, b, n)
    same(write, a, b, n)
    same(start, a, b, 1, n)
    same(start, a, b, 3, n)
    same(conditional, a, b, n)


def unversioned(a, b, n):
    s = 0.0
    for j in range(n):
        s += a[j + b]
        b = b + 0                   # stored in the loop: every access keeps its own check
    return s


@pytest.mark.parametrize('b', HUGE)
@pytest.mark.parametrize('n', [0, 1, 2, 3, 6, 10])
def test_wrapping_matches_the_checked_loop(a, b, n):
    versioned = scram(fallback=False, cache=False)(read_swapped)
    checked = scram(fallback=False, cache=False)(unversioned)
    assert outcome(versioned, [a, b, n])[:3] == outcome(checked, [a, b, n])[:3]


@pytest.mark.parametrize('b', [0, 1, 2, 3, -1, -3, -5])
@pytest.mark.parametrize('c', [0, 1, 2, -1, -2])
@pytest.mark.parametrize('n', [0, 1, 3, 4, 10])
def test_products_and_offsets(a, b, c, n):
    same(read, a, b, c, n)
    same(offsets, a, b, c, n)
    same(two_axes, a.reshape(2, 5), b, c, n)
    same(mixed, a, a[::-1].copy(), b, c, n)


@pytest.mark.parametrize('b', [0, 1, 2, 3, 5, 9, 12, -1, -8])
@pytest.mark.parametrize('n', [0, 1, 3, 5, 9, 12])
def test_changes_in_the_loop(a, b, n):
    same(base_changes, a, b, n)
    same(base_changes_late, a, b, n)


@pytest.mark.parametrize('dim, n', [(2, 5), (5, 2), (3, 3), (3, 4), (0, 3), (-2, 3)])
def test_nested(a, dim, n):
    same(nested, a, dim, n)


@pytest.mark.parametrize('b, n', [(1, 5), (0, 10), (-3, 3), (-10, 10), (-5, 2)])
def test_boundscheck_off_keeps_negative_indexes(a, b, n):
    same(read_swapped, a, b, n, boundscheck=False)      # all in range, some below zero
    same(write, a, b, n, boundscheck=False)


def fast_copy(c):
    return c[c.index('>= 0 &&'):c.index('} else {')]


def test_the_test_is_emitted_and_the_checks_gone(a):
    f = scram(fallback=False, cache=False)(read_swapped)
    assert f(a, 1, 3) == 2.0 + 3.0 + 4.0
    fast = fast_copy(f.c())
    assert 'sj_io[4]' not in fast and '< 0 ?' not in fast   # no check, no wrap


def test_boundscheck_off_drops_the_negative_index_test_too(a):
    f = scram(fallback=False, cache=False, boundscheck=False)(read_swapped)
    assert f(a, -3, 2) == 8.0 + 9.0
    assert '< 0 ?' not in fast_copy(f.c())


def test_changing_base_is_not_versioned(a):
    f = scram(fallback=False, cache=False)(base_changes)
    f(a, 3, 2)
    assert '>= 0 &&' not in f.c() and 'sj_io[4]' in f.c()


def previous(a):
    s = 0.0
    for j in range(1, len(a)):
        s += a[j - 1]
    return s


def test_constant_offsets_stay_with_the_loop_start(a):
    """base would be -1: a test that always fails would send every call to the checked copy."""
    f = scram(fallback=False, cache=False)(previous)
    assert f(a) == sum(a[:-1])
    assert '>= 0 &&' not in f.c() and 'sj_io[4]' not in f.c()


def test_nested_loops_get_one_copy_of_the_outer_body(a):
    f = scram(fallback=False, cache=False)(nested)
    f(a, 2, 5)
    assert f.c().count('>= 0 &&') == 1


def test_random(a):
    rng = random.Random(11)
    for _ in range(150):
        b, c, n = rng.randint(-14, 14), rng.randint(-4, 4), rng.randint(0, 14)
        same(read, a, b, c, n)
        same(offsets, a, b, c, n)
        same(write, a, b, n)


# A positive literal step, and a start that isn't known to be nonnegative: the test uses the last value the
# variable takes, and checks the start, so the outcome never depends on which copy ran.

def pairs(q, n):
    for i in range(0, n, 2):
        a, b = q[i], q[i + 1]
        q[i] = a - b
        q[i + 1] = a + b
    return q[0]


def strided_read(a, lo, n, k3):
    s = 0.0
    for i in range(lo, n, 3):
        s += a[i] * 2.0 + a[i + k3]
    return s


def from_start(a, lo, n):
    s = 0.0
    for i in range(lo, n):
        s += a[i + 1] - a[i]
    return s


@pytest.mark.parametrize('n', [0, 1, 2, 3, 8, 9, 10, 11, 12])
def test_stepped_loops(a, n):
    same(pairs, a, n)
    for lo in (-4, -1, 0, 1, 2, 7):
        for k3 in (0, 1, 2, 5):
            same(strided_read, a, lo, n, k3)


@pytest.mark.parametrize('lo', [-12, -10, -3, -1, 0, 1, 4, 9, 10, 15])
@pytest.mark.parametrize('n', [0, 3, 9, 10, 11])
def test_runtime_starts(a, lo, n):
    same(from_start, a, lo, n)


def test_stepped_and_runtime_start_loops_are_versioned(a):
    for f, args in ((pairs, (a.copy(), 10)), (from_start, (a, 2, 9)), (strided_read, (a, 1, 7, 1))):
        g = scram(fallback=False, cache=False)(f)
        g(*args)
        c = g.c()
        assert '} else {' in c, f.__name__
        assert 'sj_io[4]' not in c[:c.index('} else {')], f.__name__     # the fast copy has no index check
