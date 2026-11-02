"""One name, two types. Results, exact types and errors must match CPython, compiled wherever that's possible."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

X = np.arange(1.0, 6.0)
E = X[:0]


def comparable(o):
    """outcome(), with the elements of a returned tuple by exact bits too: NaN never equals itself."""
    if o[0] == 'returns' and o[1][0] == 'tuple':
        return ('returns', tuple(_value(x) for x in o[1][1]))
    return o


def same(f, *args, compiled=None, **options):
    fresh = lambda: [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    g = scram(cache=False, **options)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for _ in range(3):
            plain, native = fresh(), fresh()
            want, got = comparable(outcome(f, plain)), comparable(outcome(g, native))
            assert want[:4] == got[:4], (f.__name__, args)
            for p, c in zip(plain, native):
                if isinstance(p, np.ndarray):
                    assert _value(p) == _value(c)
    if compiled is not None:
        assert bool(g.signatures) == compiled, (f.__name__, g._why)
    return g


# Two variables with one name


def two_lives(n):
    x = 1
    y = x + n
    x = 0.5
    return x * 2, y


def in_branches(flag):
    if flag:
        x = 1
        y = x + 1
    else:
        x = 0.5
        y = x * 2
    return y


def loop_variable_reused(a, b):
    s = 0.0
    for v in a:
        s += v
    k = 0
    for v in b:
        k += v
    return s, k


def counter_then_float(n):
    i = 0
    while i < n:
        i += 1
    r = i
    i = 1.5
    return r, i * 2


def param_reused(x, k):
    t = x[0]
    k = 2.5
    return t * k


def test_separate_lives_compile():
    same(two_lives, 3, compiled=True)
    same(in_branches, True)                 # y is one variable, int or float, and not a literal: stays Python
    same(in_branches, False)
    same(loop_variable_reused, X, np.arange(3), compiled=True)
    same(loop_variable_reused, E, np.arange(3)[:0], compiled=True)
    same(counter_then_float, 4, compiled=True)
    same(param_reused, X, 3, compiled=True)


def really_one_variable(flag):
    x = 1
    if flag:
        x = 2.5
    y = x
    return y


def one_variable_loop(n):
    x = 1
    r = n
    for i in range(n):
        r = x
        x = 0.5
    return r


def test_one_variable_stays_correct():
    same(really_one_variable, True)
    same(really_one_variable, False)
    for n in (0, 1, 2, 3):
        same(one_variable_loop, n)


# An int that becomes a float


def total(x):
    s = 0
    for v in x:
        s += v
    return s


def total_by_index(x):
    s = 0
    for i in range(len(x)):
        s = s + x[i]
    return s


def product(x):
    p = 1
    for v in x:
        p *= v
    return p


def python_floats(n):
    s = 0
    for i in range(n):
        s += 0.5
    return s


def largest(x):
    m = -1
    for v in x:
        if v > m:
            m = v
    return m


def in_a_tuple(x):
    s = 0
    c = 0
    for v in x:
        s += v
        c += 1
    return s, c


def both(x):
    a = 0
    b = 1
    for v in x:
        a += v
        b *= v
    return a, b


def assigned_later(x, flag):
    s = 1.5
    if flag:
        s = 0
    for v in x:
        if v > 3:
            s = v
    return s


def divides(x, k):
    s = 0
    if k:
        s = s + x[0]
    t = 1.0 / (s + 0.0)
    return t, s


def zero_over(x, y):
    s = 0
    if y > 100:
        s = 2.5 + y
    return s / y, s % y, s // y, s ** y


def negative_zero(x):
    s = 0
    for v in x:
        s = s + v
    return s


@pytest.mark.parametrize('x', [X, E, X[:1]], ids=['many', 'none', 'one'])
def test_int_then_float(x):
    same(total, x, compiled=True)
    same(total_by_index, x, compiled=True)
    same(product, x, compiled=True)
    same(largest, x, compiled=True)
    same(in_a_tuple, x, compiled=True)
    same(both, x, compiled=True)
    same(assigned_later, x, True, compiled=True)
    same(assigned_later, x, False, compiled=True)


def test_types_after_no_iterations_are_python_s():
    g = same(total, E, compiled=True)
    assert type(g(E)) is int and g(E) == 0
    assert type(g(X)) is np.float64
    h = same(python_floats, 0, compiled=True)
    assert type(h(0)) is int
    assert type(h(3)) is float


def test_python_floats_stay_python_floats():
    same(python_floats, 3, compiled=True)
    same(python_floats, 0, compiled=True)


@pytest.mark.parametrize('values', [[-0.0], [-0.0, -0.0], [0.0, -0.0], [np.nan], [np.inf, -np.inf], [1e308, 1e308], [5e-324]])
def test_bits(values):
    x = np.array(values)
    same(negative_zero, x, compiled=True)
    same(total, x, compiled=True)
    same(product, x, compiled=True)


@pytest.mark.parametrize('k', [True, False])
def test_errors_with_an_int_inside(k):
    same(divides, np.array([0.0]), k)
    same(divides, np.array([2.0]), k)


@pytest.mark.parametrize('y', [0.0, 2.0, -2.0, 0.5, np.float64(0.0), np.float64(2.0), np.float64(-2.0), 101.0])
def test_every_operator_on_an_int_that_is_still_an_int(y):
    same(zero_over, X, y)


# Where int and float differ, it must not compile as if they were the same


def negated(x):
    s = 0
    t = -s
    for v in x:
        s += v
    return s, t


def read_as_index(x):
    k = 1
    y = x[k]
    k = 2.5
    return y, k


def copied(x):
    s = 0
    for v in x:
        s += v
    t = s
    return t


def plus_an_int(x):
    s = 0
    for v in x:
        s += v
        s = s + 1
    return s


def int_literal_math(x):
    s = 0
    u = s + 1
    for v in x:
        s += v
    return s, u


def truth(x):
    s = 0
    for v in x:
        if s:
            return 1
        s += v
    return 2


def two_tagged(x):
    a = 0
    b = 0
    c = a + b
    for v in x:
        a += v
        b += v
    return a + b, c


def large_literal(x):
    s = 9007199254740993
    for v in x:
        s += v
    return s


def from_a_variable(x, n):
    s = n
    t = s + 1
    for v in x:
        s += v
    return s, t


def as_int_division(x):
    s = 7
    t = s // 2
    for v in x:
        s += v
    return s, t


def int_compare(x, n):
    s = 3
    for v in x:
        if s == n:
            return -1
        s = s * v
    return s


def add_to_other(x):
    s = 0
    y = 1.5
    y += s
    for v in x:
        s += v
    return y, s


def two_tagged_augmented(x, flag):
    a = 0
    b = 2
    if flag:
        a = 1.5
        b = b * x[0]
    b += a
    return b, a


def test_two_that_may_still_be_ints():
    for flag in (True, False):
        same(two_tagged_augmented, X, flag)


def test_uses_where_int_and_float_differ():
    for f in (negated, copied, plus_an_int, int_literal_math, truth, two_tagged, large_literal, as_int_division):
        same(f, X)
        same(f, E)
    same(read_as_index, X)
    for n in (3, 2**53 + 1, 2**60):
        same(from_a_variable, X, n)
        same(from_a_variable, E, n)
        same(int_compare, X, n)
    same(add_to_other, X)


def test_large_literal_is_not_held_in_a_double():
    g = same(large_literal, E)
    assert type(g(E)) is int and g(E) == 9007199254740993


# An int coming in


def from_param(x, s):
    for v in x:
        s += v
    return s


@pytest.mark.parametrize('s', [0, 1, -7, 2**53, -2**53, 2**53 + 1, -2**53 - 1, 2**62, -2**63, 2**70, True, np.int64(3)])
def test_int_parameter_that_becomes_float(s):
    same(from_param, X, s)
    same(from_param, E, s)


def numpy_then_python(s, x):
    t = s * 2.0
    for v in x:
        s = 1.5
    return t, s


def float_param_then_int_literal(s, x):
    for v in x:
        s = 3
    return s


@pytest.mark.parametrize('s', [2.0, np.float64(2.0), -0.0])
def test_a_parameter_starts_as_what_it_is(s):
    same(numpy_then_python, s, X, compiled=True)
    same(numpy_then_python, s, E, compiled=True)
    same(float_param_then_int_literal, s, X, compiled=True)
    same(float_param_then_int_literal, s, E, compiled=True)


def test_int_parameter_too_big_for_a_double_runs_as_python():
    g = scram(cache=False)(from_param)
    assert g.__wrapped__(E, 2**53 + 1) == g(E, 2**53 + 1) == 2**53 + 1
    assert type(g(E, 5)) is int
    assert g.signatures                    # the small one compiled, and is still there


# Lifting


def lifted_total(x, name):
    d = {name: 1}
    s = 0
    for v in x:
        s += v
    d['s'] = s
    return d


def lifted_loop_variable(x, name):
    tag = str(name)
    k = 0
    for i in range(len(x)):
        k = k + i
    r = 0
    for i in x:
        r += i
    return k, r, tag


def compiled_loops(g):
    return sum(any(type(v).__name__ == 'Compiled' for v in f._cache.values()) for f in g._all_loops())


def test_the_loop_lifting_could_not_reach():
    for x in (X, E, X[:1]):
        g = same(lifted_total, x, 'a')
        assert compiled_loops(g) == 1
        h = same(lifted_loop_variable, x, 'a')
        assert compiled_loops(h) == 2


def lifted_big(x, name, big):
    d = {name: 1}
    s = big
    for v in x:
        s += v
    d['s'] = s
    return d


@pytest.mark.parametrize('big', [2**53 + 1, 3])
def test_lifted_int_too_big_for_a_double(big):
    same(lifted_big, E, 'a', big)
    same(lifted_big, X, 'a', big)


def test_a_name_with_a_web_suffix_already():
    def f(n):
        x = 1
        x__v1 = x + n
        x = 0.5
        return x * 2, x__v1
    same(f, 3)


@pytest.mark.parametrize('options', [{'boundscheck': False}, {'parity': False}])
def test_options(options):
    for x in (X, E):
        same(total, x, compiled=True, **options)
        same(in_a_tuple, x, compiled=True, **options)


def test_messages_show_the_name_the_user_wrote():
    g = scram(cache=False)(really_one_variable)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g(True)
    assert g._why and not any('__v' in why for _, why in g._why.values())


NAMES = ['a', 'b', 'c']
LITERALS = ['0', '1', '-1', '2', '0.5', '-0.0', '1.5', '3']


def random_function(rng, depth=0, loops=0):
    """Assignments of ints and floats to a few names, in branches and loops; every name assigned first."""
    def value():
        k = rng.random()
        if k < 0.35:
            return rng.choice(LITERALS)
        if k < 0.55:
            return rng.choice(NAMES)
        op = rng.choice(['+', '-', '*'])
        other = rng.choice(NAMES + ['v'] * loops + ['x[0]', '0.5', '2'])
        return f'{rng.choice(NAMES)} {op} {other}'

    def block(depth, loops, n):
        out = []
        for _ in range(n):
            k = rng.random()
            if k < 0.5 or depth > 2:
                out.append(f'{rng.choice(NAMES)} = {value()}')
            elif k < 0.7:
                out.append(f'{rng.choice(NAMES)} += {rng.choice(NAMES + ["x[0]", "0.5", "1"] + ["v"] * loops)}')
            elif k < 0.85:
                out.append(f'if {rng.choice(NAMES)} > {rng.choice(LITERALS)}:')
                out += ['    ' + x for x in block(depth + 1, loops, 2)]
                if rng.random() < 0.5:
                    out.append('else:')
                    out += ['    ' + x for x in block(depth + 1, loops, 1)]
            else:
                out.append('for v in x:')
                body = block(depth + 1, loops + 1, 2)
                if rng.random() < 0.2:
                    body += ['if v > 2:', f'    {rng.choice(["break", "continue"])}', f'{rng.choice(NAMES)} = {value()}']
                out += ['    ' + b for b in body]
        return out

    start = [f'{n} = {rng.choice(LITERALS)}' for n in NAMES]
    lines = start + block(0, 0, rng.randint(2, 6)) + [f'return {", ".join(rng.sample(NAMES, rng.randint(1, 3)))}']
    return 'def f(x):\n' + ''.join('    ' + x + '\n' for x in lines)


def past_64_bits(f, x):
    """Did an int go past 64 bits in CPython? Compiled ints wrap there, as documented."""
    import sys
    seen = []

    def trace(frame, event, arg):
        if frame.f_code is f.__code__:
            values = list(frame.f_locals.values()) + ([arg] if event == 'return' else [])
            seen.extend(v for v in values if type(v) is int and not -2**63 <= v < 2**63)
            if event == 'exception' and arg[0] is OverflowError:
                seen.append(arg[0])
            return trace
    sys.settrace(trace)
    try:
        f(x)
    except Exception:  # noqa: BLE001
        pass
    finally:
        sys.settrace(None)
    return bool(seen)


def check_random(seed, count):
    import random
    rng = random.Random(seed)
    compiled = 0
    for _ in range(count):
        src = random_function(rng)
        space = {}
        exec(src, space)
        f = space['f']
        for x in (X, E, np.array([-0.0, 2.5, np.nan])):
            want = comparable(outcome(f, [x]))
            g = _from_source(f, src)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FallbackWarning)
                got = comparable(outcome(g, [x]))
            assert want[:3] == got[:3] or past_64_bits(f, x), src
        compiled += bool(g.signatures)
    return compiled


def _from_source(f, src):
    """scramjit reads a function's source from its file, and this one has no file: write it there."""
    import ast
    from scramjit._dispatch import Function
    g = Function(f, cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    g._first = 1
    return g


def test_random_programs():
    assert check_random(2026, 60) > 15
