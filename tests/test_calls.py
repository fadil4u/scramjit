"""One compiled function calling another. The result is what Python gives, whichever way it went; these check that, and that it did compile."""

import math
import sys
import threading
import traceback
import types
import warnings

import numpy as np
import pytest

import calls_other_module as other
from calls_other_module import from_table, scaled, set_scale
from scramjit import scram
from scramjit._dispatch import Compiled

K = 100.0


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


def check(f, *args, must_compile=True):
    def copies():
        return [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    want_args, got_args = copies(), copies()
    want, got = outcome(python(f), want_args), outcome(f, got_args)
    assert want[0] == got[0], (want, got)
    assert same(want[1], got[1]), (want, got)
    for a, b in zip(want_args, got_args):
        assert same(a, b), f'an argument changed differently: {a} {b}'
    assert compiled(f) == must_compile, f'{f.__name__}: {f._why}'
    return got[1]


def python(f):
    """The function as plain Python all the way down: its calls go to the undecorated functions too."""
    g = dict(f.__wrapped__.__globals__)
    for name, v in list(g.items()):
        if hasattr(v, '_callee'):
            w = v.__wrapped__
            g[name] = types.FunctionType(w.__code__, g, w.__name__, None, w.__closure__) if w.__globals__ is f.__wrapped__.__globals__ else w
    return types.FunctionType(f.__wrapped__.__code__, g, f.__name__, None, f.__wrapped__.__closure__)


# ---- numbers
@scram
def sq(x):
    return x * x


@scram
def of_numbers(n):
    s = 0.0
    for i in range(n):
        s += sq(i * 0.5)
    return s, sq(3), sq(2.5), sq(True), sq(np.float64(1.5)), sq(np.int64(4))


def test_numbers_and_their_types():
    r = check(of_numbers, 5)
    assert type(r[1]) is int and type(r[2]) is float and type(r[4]) is np.float64 and type(r[5]) is np.int64


@scram
def overflow_ok(a):
    return a * a


@scram
def wraps(a):
    return overflow_ok(a) + 1


@scram
def of_elements(x):
    t = 0.0
    for i in range(len(x)):
        t = t + sq(x[i])
    return t, type_of_flag(t)


@scram
def type_of_flag(v):
    return v + 1.0


def test_a_variable_python_or_numpy_by_the_path_taken():
    r = check(of_elements, np.arange(3.0))
    assert type(r[0]) is np.float64 and type(r[1]) is np.float64
    r = check(of_elements, np.arange(0.0))
    assert type(r[0]) is float and type(r[1]) is float


@scram
def nothing(x, i):
    x[i] = 5.0


@scram
def as_a_statement(x):
    nothing(x, 0)
    nothing(x[1:], 0)
    sq(2.0)
    return x


def test_as_a_statement():
    check(as_a_statement, np.arange(3.0))


@scram
def pair(a, b):
    return a + b, a - b


@scram
def of_tuples(a, b):
    s, d = pair(a, b)
    t = pair(s, d)
    u = (a, b)
    v, w = swap_pair(u)
    return s, d, t[0], v, w, t[1]


@scram
def swap_pair(t):
    return t[1], t[0]


def test_tuples():
    check(of_tuples, 1.5, 2.5)
    check(of_tuples, 1, 2)


@scram
def by_keyword(a, b, c):
    return a - 2 * b + 4 * c


@scram
def keywords(x):
    return by_keyword(1, c=x, b=2), by_keyword(a=x, b=1.0, c=2), by_keyword(x, 1, 2)


def test_keywords():
    check(keywords, 2.5)


# ---- arrays
@scram
def fill(y, v):
    for i in range(len(y)):
        y[i] = v


@scram
def total(x):
    s = 0.0
    for v in x:
        s += v
    return s


@scram
def on_arrays(m):
    fill(m[0], 1.0)
    fill(m[:, 1], 2.0)
    fill(m[1][::-1][:1], 3.0)
    return total(m[0]), total(m[:, 1]), total(m.T[0]), m[0, 0]


def test_arrays_and_views_go_in_and_writes_come_back():
    check(on_arrays, np.zeros((3, 4)))


@scram
def read_only(x):
    return total(x) + total(x[1:])


@scram
def tries_writing(x):
    fill(x, 1.0)
    return x[0]


def test_read_only():
    x = np.arange(3.0)
    x.flags.writeable = False
    check(read_only, x)
    check(tries_writing, x)


@scram
def shift(a, b):
    for i in range(len(a)):
        a[i] = b[i] + 1.0


@scram
def overlapping(x):
    shift(x[1:], x[:-1])
    shift(x, x)
    return x


def test_two_names_for_one_memory():
    check(overlapping, np.arange(6.0))


# ---- arrays coming back
@scram
def identity(x):
    return x


@scram
def tail(x):
    return x[1:]


@scram
def made(n, v):
    y = np.full(n, v)
    y[0] = 0.0
    return y


@scram
def made_view(n):
    y = np.arange(float(n))
    return y[::2]


@scram
def same_and_view(x):
    return identity(x), tail(x), tail(tail(x)), identity(x[1:])


@scram
def of_made(n):
    a = made(n, 2.0)
    b = made_view(n)
    b[0] = 9.0
    return a, b, a.sum(), made(2, 1.0)


def test_the_argument_comes_back_as_itself():
    x = np.arange(5.0)
    a, b, c, d = scram(same_and_view.__wrapped__)(x)
    assert a is x
    assert b.base.base is x if b.base is not x else True
    assert np.shares_memory(b, x) and np.shares_memory(c, x) and np.shares_memory(d, x)
    b[0] = 50.0
    assert x[1] == 50.0
    check(same_and_view, np.arange(5.0))


def test_what_a_call_makes_comes_back_as_numpy_would_make_it():
    a, b, s, c = check(of_made, 6)
    assert a.flags.owndata and a.base is None
    assert b.base is not None and b.strides == (16,)


@scram
def returns_what_it_called(n):
    return made(n, 1.0)


@scram
def returns_tuple_from_call(x):
    return with_array(x, 3)


@scram
def with_array(x, n):
    y = np.zeros(n)
    y[0] = x[0]
    return y, n * 2, x, y[1:], x[0]


@scram
def unpacks(x, n):
    a, b, c, d, e = with_array(x, n)
    a[1] = 5.0
    c[0] = 7.0
    return a, b, c, d, e


def test_arrays_in_tuples_through_calls():
    check(returns_what_it_called, 3)
    r = check(returns_tuple_from_call, np.arange(3.0))
    assert np.shares_memory(r[0], r[3])
    x = np.arange(3.0)
    r = unpacks(x, 3)
    assert r[2] is x and x[0] == 7.0 and r[3][0] == 5.0
    check(unpacks, np.arange(3.0), 3)


@scram
def returns_either(x, c):
    if c:
        return x
    return x.copy()


@scram
def through(x, c):
    y = returns_either(x, c)
    y[0] = 5.0
    return y, x[0]


def test_which_it_is_is_known_at_runtime():
    check(through, np.arange(3.0), True)
    check(through, np.arange(3.0), False)
    x = np.arange(3.0)
    assert through(x, True)[0] is x
    assert through(x, False)[0] is not x


@scram
def a_scratch_buffer(n):
    t = np.ones(n)
    return t.sum()


@scram
def many_calls(k, n):
    s = 0.0
    for i in range(k):
        s += a_scratch_buffer(n)
    return s


@scram
def many_blocks(k, n):
    s = 0.0
    for i in range(k):
        y = made(n, 1.0)
        s += y[n - 1]
    return s


@scram
def keeps_one(k, n):
    keep = made(4, 1.0)
    s = 0.0
    for i in range(k):
        y = made(n, float(i))
        if i == 1:
            keep = y[:4]
        s += keep[1]
    return s, keep


def test_memory_stays_flat_across_calls():
    resource = pytest.importorskip('resource', reason='not on this platform', exc_type=ImportError)
    assert many_calls(2, 100) == 200.0
    assert many_blocks(2, 100) == 2.0
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    many_calls(3000, 125_000)       # 3 GB each if kept
    many_blocks(3000, 125_000)
    grew = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert grew < 200, f'{grew} MB'


def test_a_block_a_call_returned_survives_a_sweep():
    check(keeps_one, 200, 20_000)


# ---- order
@scram
def writes_then_returns(x, v):
    x[0] = v
    return v


@scram
def order(x):
    a = x[0] + writes_then_returns(x, 5.0) + x[0]
    b = writes_then_returns(x, 6.0) * x[0]
    x[0] += writes_then_returns(x, 7.0)
    c0, c1, c2 = x[0], writes_then_returns(x, 8.0), x[0]
    d = x.sum() + writes_then_returns(x, 1.0) + x.sum()
    e = x[0] < writes_then_returns(x, 100.0) < x[0] + 1.0
    return a, b, c0, c1, c2, d, e, x[0]


def test_python_order():
    check(order, np.arange(3.0))


@scram
def side(x, i):
    x[i] += 1.0
    return i


@scram
def short_circuit(x, c):
    r = 0
    if c and side(x, 0) >= 0:
        r += 1
    if not c or side(x, 1) >= 0:
        r += 2
    v = side(x, 2) if c else 7
    while side(x, 0) < 0:
        r = 100
    return r, v, x[0], x[1], x[2]


def test_only_what_python_evaluates():
    check(short_circuit, np.zeros(3), True)
    check(short_circuit, np.zeros(3), False)


@scram
def next_previous(y, i):
    y[i - 1] = 100.0
    return 1.0


@scram
def not_forwarded(y):
    for i in range(1, len(y)):
        y[i] = y[i - 1] + next_previous(y, i)
    return y


def test_a_call_ends_store_to_load_forwarding():
    check(not_forwarded, np.arange(5.0))


@scram
def bound(n):
    return n - 1


@scram
def in_headers(x, n):
    s = 0
    for i in range(bound(n)):
        s += bound(i)
    for i in range(bound(len(x))):
        x[i] = bound(i)
    k = n
    while bound(k) > 0:
        k = bound(k)
    return s, k, x[bound(1)]


def test_in_headers():
    check(in_headers, np.zeros(4), 5)


@scram
def in_a_comprehension(n):
    v = [sq(i * 0.5) for i in range(n)]
    return sum(sq(i) for i in range(n)), v[n - 1]


def test_in_a_comprehension():
    check(in_a_comprehension, 4)


# ---- what's reached from what
@scram
def level3(x):
    return math.sqrt(x) + sq(x)


@scram
def level2(x):
    return level3(x) + level3(x + 1.0)


@scram
def level1(x):
    return level2(x) * sq(2.0) + level3(x)


def test_chains_with_libm_and_shared_callees():
    check(level1, 2.0)


@scram
def reads_k(x):
    return x + K


@scram
def reads_k_too(x):
    return reads_k(x) + K


def test_a_new_value_of_a_global_a_callee_reads():
    global K
    saved = K
    try:
        assert reads_k_too(1.0) == 201.0
        entries = len(reads_k_too._cache)
        K = 1.0
        assert reads_k_too(1.0) == 3.0
        assert len(reads_k_too._cache) == entries
    finally:
        K = saved


@scram
def uses_other_module(x):
    return scaled(x) + K + from_table(1)


def test_a_callee_in_another_module_reads_its_own_globals():
    other_scale = other.SCALE
    try:
        assert uses_other_module(1.0) == 3.0 + 7 + 100.0 + 3.0
        assert compiled(uses_other_module)
        set_scale(2.0)
        assert uses_other_module(1.0) == 2.0 + 7 + 100.0 + 2.0
    finally:
        set_scale(other_scale)


@scram
def inc(x):
    return x + 1.0


@scram
def dec(x):
    return x - 1.0


CHOSEN = inc


@scram
def calls_chosen(x):
    return CHOSEN(x)


def test_a_name_given_another_function():
    global CHOSEN
    try:
        CHOSEN = inc
        assert calls_chosen(1.0) == 2.0
        CHOSEN = dec
        assert calls_chosen(1.0) == 0.0
        CHOSEN = lambda x: x * 10.0  # noqa: E731
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert calls_chosen(1.0) == 10.0
        CHOSEN = inc
        assert calls_chosen(1.0) == 2.0
    finally:
        CHOSEN = inc


def plain_python(x):
    return x + 1.0


@scram
def calls_plain(x):
    return plain_python(x)


@scram
def calls_uncompilable(x):
    return uncompilable(x)


@scram
def uncompilable(x):
    return str(x)


@scram
def recursive(n):
    return 1 if n < 2 else n * recursive(n - 1)


@scram
def even(n):
    return True if n == 0 else odd(n - 1)


@scram
def odd(n):
    return False if n == 0 else even(n - 1)


def closure():
    k = 3.0

    @scram
    def inner(x):
        return x + k
    return inner


INNER = closure()


@scram
def calls_closure(x):
    return INNER(x)


def test_what_cant_be_compiled_stays_python_and_says_why():
    for f, args, why in ((calls_plain, (1.0,), 'calls plain_python'), (calls_uncompilable, (1.0,), 'calls uncompilable'), (recursive, (5,), 'recursion'),
                         (even, (5,), 'recursion'), (calls_closure, (1.0,), 'closures')):
        check(f, *args, must_compile=False)
        assert why in str(next(iter(f._why.values()))), f._why


@scram
def takes_dict(p):
    return p['a'] + 1.0


@scram
def passes_dict(p):
    return takes_dict(p)


def test_a_dict_stays_python():
    assert passes_dict({'a': 1.0}) == 2.0
    assert not compiled(passes_dict)


# ---- errors
@scram
def oob(x, i):
    return x[i]


@scram
def middle(x, i):
    return oob(x, i) + 1.0


@scram
def outer(x, i):
    y = np.zeros(3)
    return middle(x, i) + y[0]


@scram
def divides(a, b):
    return a / b


@scram
def error_in_expression(a):
    return 1.0 + divides(a, 0.0)


def test_errors_are_pythons():
    check(outer, np.arange(3.0), 1)
    check(outer, np.arange(3.0), 9)
    check(error_in_expression, 1.0)


def test_the_traceback_has_every_frame():
    with pytest.raises(IndexError) as e:
        outer(np.arange(3.0), 9)
    frames = [(f.name, f.lineno) for f in traceback.extract_tb(e.value.__traceback__) if f.filename == __file__]
    names = [n for n, _ in frames]
    assert names[-3:] == ['outer', 'middle', 'oob'], frames
    lines = dict(frames)
    assert lines['oob'] == oob.__wrapped__.__code__.co_firstlineno + 2
    assert lines['middle'] == middle.__wrapped__.__code__.co_firstlineno + 2
    assert lines['outer'] == outer.__wrapped__.__code__.co_firstlineno + 3


def test_an_error_after_many_good_calls_with_a_reused_buffer():
    a = np.arange(3.0)
    for _ in range(5):
        assert outer(a, 1) == 2.0
        with pytest.raises(IndexError):
            outer(a, 9)


@scram
def rounds_int(x):
    return int(x)


@scram
def int_first(n, x):
    s = n
    for i in range(len(x)):
        s += x[i]
    return s


@scram
def calls_int_first(n, x):
    return int_first(n, x)


def test_a_param_that_becomes_a_float():
    check(calls_int_first, 3, np.arange(2.0), must_compile=False)
    check(calls_int_first, 3, np.arange(0.0), must_compile=False)
    assert 'int' in str(next(iter(calls_int_first._why.values())))


# ---- with what else there is
@scram
def lifted(x, obj):
    s = 0.0
    for i in range(len(x)):
        s += sq(x[i]) + inc(1.0)
    return s, str(obj)


def test_in_a_lifted_loop():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert lifted(np.arange(3.0), 1) == (5.0 + 3 * 2.0, '1')
    assert any(compiled(loop) for loop in lifted._all_loops())


def test_threads():
    if sys.platform == 'emscripten':
        return
    results = []

    def run():
        for _ in range(200):
            results.append(many_calls(5, 100))
    ts = [threading.Thread(target=run) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert set(results) == {500.0}


@scram(counts=True)
def counted(n):
    s = 0
    for i in range(n):
        bound(i)
        s += bound(i)
    return s


def test_counts_in_the_caller():
    assert counted(4) == counted.__wrapped__(4)
    runs, heads = counted.line_counts()
    first = counted.__wrapped__.__code__.co_firstlineno
    assert runs[first + 4] == 4 and runs[first + 5] == 4


def test_a_callee_called_from_python_too():
    assert sq(3.0) == 9.0
    assert compiled(sq)
    assert of_numbers(2)[1] == 9


@scram
def one_signature_many_callers(x):
    return sq(x) + sq(x) + sq(x + 1.0)


def test_one_c_function_per_signature():
    one_signature_many_callers(1.5)
    assert one_signature_many_callers.c().count('static int sj_call_') == 1
    check(one_signature_many_callers, 1.5)


def test_preflight_follows_calls():
    import ast
    from scramjit import _preflight as P
    tree = ast.parse("""
import numpy as np
from scramjit import scram

SCALE = 2.0

@scram
def step(state, a):
    state[0] += a * SCALE
    return state[0] * state[0]

@scram
def run(state, n):
    r = 0.0
    for i in range(n):
        r += step(state, 0.5)
    return r

def plain(x):
    return x

@scram
def calls_plain(x):
    return plain(x)

@scram
def rec(n):
    return 1 if n < 2 else n * rec(n - 1)
""")
    g = P.module_globals(tree)
    fs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    opts = {'boundscheck': True, 'parity': True}
    a = P.best_analysis(fs['run'], g, 'x.py', 0, opts)
    assert a.state == 'compiled', [p.reason for p in a.problems]
    assert a.signature[0].kind == 'array'
    a = P.best_analysis(fs['calls_plain'], g, 'x.py', 0, opts)
    assert a.state == 'python' and 'calls plain()' in a.problems[0].reason and 'decorate' in P.hint_for(a.problems[0].reason)
    a = P.best_analysis(fs['rec'], g, 'x.py', 0, opts)
    assert a.state == 'python' and 'recursion' in a.problems[0].reason
