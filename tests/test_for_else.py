"""for ... else and while ... else: the else runs unless a break left the loop."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

X = np.arange(1.0, 6.0)
E = X[:0]


def comparable(o):
    if o[0] == 'returns' and o[1][0] == 'tuple':
        return ('returns', tuple(_value(x) for x in o[1][1]))
    return o


def same(f, *args, compiled=None, **options):
    fresh = lambda: [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    g = scram(cache=False, **options)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for _ in range(2):
            plain, native = fresh(), fresh()
            want, got = comparable(outcome(f, plain)), comparable(outcome(g, native))
            assert want[:4] == got[:4], (f.__name__, args)
            for p, c in zip(plain, native):
                if isinstance(p, np.ndarray):
                    assert _value(p) == _value(c)
    if compiled is not None:
        assert bool(g.signatures) == compiled, (f.__name__, g._why)
    return g


def search(x, t):
    for i in range(len(x)):
        if x[i] == t:
            found = i
            break
    else:
        found = -1
    return found


def search_values(x, t):
    for v in x:
        if v > t:
            r = v
            break
    else:
        r = 0.0
    return r


def returns_from_else(x, t):
    for v in x:
        if v == t:
            break
    else:
        return -1
    return 1


def ends_in_else(x, t):
    for v in x:
        if v == t:
            return 1
    else:
        return 2


def else_counts(x, t):
    c = 0
    for v in x:
        if v == t:
            break
        c += 1
    else:
        c += 100
    return c


def while_else(n, t):
    i = 0
    r = 0
    while i < n:
        if i == t:
            r = 5
            break
        i += 1
    else:
        r = 7
    return i, r


def continue_is_no_break(x, t):
    r = 0
    for v in x:
        if v == t:
            continue
        r += 1
    else:
        r += 100
    return r


def inner_break_is_not_mine(x, t):
    r = 0
    for v in x:
        for w in x:
            if w == t:
                break
            r += 1
    else:
        r += 1000
    return r


def inner_else(x, t):
    r = 0
    for v in x:
        for w in x:
            if w == t:
                break
        else:
            r += 1
            if v > t:
                break                # this one belongs to the outer loop
    else:
        r += 1000
    return r


def else_reads_the_loop_variable(x):
    i = -1
    for i in range(len(x)):
        pass
    else:
        i += 10
    return i


def versioned(x, y, t):
    """A loop that gets two copies, one checked: a break in either must count."""
    r = -1
    for i in range(1, len(x)):
        y[i] = x[i - 1]
        if x[i] == t:
            break
    else:
        r = 99
    return r


def base_versioned(x, b, t):
    r = -1
    for j in range(2):
        if x[b + j] == t:
            break
    else:
        r = 99
    return r


def over_zip(x, y, t):
    r = 0
    for v, w in zip(x, y):
        if v + w == t:
            break
        r += 1
    else:
        r = -1
    return r


def over_enumerate(x, t):
    r = 0
    for i, v in enumerate(x):
        if v == t:
            r = i
            break
    else:
        r = -1
    return r


def one_path_assigns(x, t):
    for v in x:
        if v == t:
            found = v
            break
    else:
        pass
    return found


def only_break_assigns(x, t):
    for v in x:
        if v == t:
            found = v
            break
    return found


def type_differs_on_each_path(x, t):
    for v in x:
        if v == t:
            r = 1.5
            break
    else:
        r = 0
    return r


def long_loop(n, t):
    r = 0
    for i in range(n):
        if i == t:
            r = 1
            break
    else:
        r = 2
    return r


@pytest.mark.parametrize('t', [3.0, 1.0, 5.0, 99.0])
@pytest.mark.parametrize('x', [X, E, X[:1]], ids=['many', 'none', 'one'])
def test_every_way_out(x, t):
    for f in (search, search_values, returns_from_else, ends_in_else, else_counts, continue_is_no_break,
              inner_break_is_not_mine, inner_else, over_enumerate):
        same(f, x, t, compiled=True)
    same(over_zip, x, x, t, compiled=True)
    same(type_differs_on_each_path, x, t)
    same(one_path_assigns, x, t)
    same(only_break_assigns, x, t)
    same(else_reads_the_loop_variable, x, compiled=True)


@pytest.mark.parametrize('n, t', [(0, 0), (5, 2), (5, 9), (5, 0), (5, 4)])
def test_while_else(n, t):
    same(while_else, n, t, compiled=True)


@pytest.mark.parametrize('t', [2.0, 5.0, 99.0])
def test_both_copies_of_a_versioned_loop(t):
    same(versioned, X, np.zeros(5), t, compiled=True)
    same(versioned, X, np.zeros(2), t, compiled=True)        # the checked copy: IndexError partway
    for b in (0, 3, 4, -5, -1, -9):
        same(base_versioned, X, b, t, compiled=True)


@pytest.mark.parametrize('t', [3, 70000, 200000])
def test_across_interrupt_chunks(t):
    same(long_loop, 140000, t, compiled=True)


@pytest.mark.parametrize('options', [{'boundscheck': False}, {'parity': False}])
def test_options(options):
    same(search, X, 3.0, compiled=True, **options)
    same(search, X, 99.0, compiled=True, **options)


def lifted(x, t, name):
    d = {name: 1}
    r = -1
    for v in x:
        if v == t:
            r = 1
            break
    else:
        r = 2
    d['r'] = r
    return d


def test_lifted_loop_with_an_else():
    for t in (3.0, 99.0):
        g = same(lifted, X, t, 'a')
        assert g._native_loops()
