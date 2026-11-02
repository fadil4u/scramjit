"""Loop lifting: a function that doesn't compile whole must behave exactly as in CPython, however its loops
run: same results and types, same arrays, same output in the same order, same exception at the same line."""

import contextlib
import io
import warnings

import numpy as np
import pytest

from scramjit import CompileError, FallbackWarning, scram

from util import outcome, _value


def run(f, args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = outcome(f, args)
    return r, out.getvalue()


def compiled_loops(g):
    return sum(any(type(v).__name__ == 'Compiled' for v in loop._cache.values()) + compiled_loops(loop)
               for loop in g._lifted or [])


def same(f, *args, loops=None, repeat=3):
    """f on copies of args, plain and decorated, called more than once so that lifted loops are past their
    first call. loops: how many loops must have compiled."""
    def fresh():
        return [a.copy() if isinstance(a, (np.ndarray, dict, list)) else a for a in args]
    g = scram(cache=False)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for _ in range(repeat):
            plain, lifted = fresh(), fresh()
            want, want_out = run(f, plain)
            got, got_out = run(g, lifted)
            assert want[:4] == got[:4], (f.__name__, args)
            assert want_out == got_out, f.__name__
            for p, c in zip(plain, lifted):
                if isinstance(p, np.ndarray):
                    assert _value(p) == _value(c), f.__name__
                else:
                    assert repr(p) == repr(c), f.__name__
    if loops is not None:
        assert compiled_loops(g) == loops, (f.__name__, g._lifted)
    return g


X = np.arange(1.0, 9.0)


def setup_then_loop(x, name):
    label = {'name': name}
    s = 0.0
    for v in x:
        s += v
    label['total'] = s
    return label


def test_setup_with_a_dict_and_a_string():
    same(setup_then_loop, X, 'a', loops=1)


def str_argument(x, name):
    s = 0.0
    for i in range(len(x)):
        s += x[i]
    return f'{name}: {s}'


def test_an_argument_that_cannot_be_passed():
    same(str_argument, X, 'total', loops=1)


def printing(x):
    print('start')
    s = 0.0
    for v in x:
        s += v
    print('middle', s)
    t = 0.0
    for v in x:
        t += v * v
    print('end', t)
    return s, t


def test_output_stays_in_order():
    same(printing, X, loops=2)          # both, though they reuse v


def live_loop_variable(x):
    print('a')
    for v in x:
        pass
    print(v)
    for v in x:
        pass
    return v


def reused_in_an_outer_loop(x):
    log = []
    s = -1.0
    for k in range(2):
        log.append(s)
        for v in x:
            s = v
    return log


def assigned_on_one_path(x, flag):
    print('a')
    s = 0.0
    for v in x:
        s = v
    if flag:
        s = 9.0
    return s


def test_what_is_read_later_comes_back():
    same(live_loop_variable, X)
    same(reused_in_an_outer_loop, X)
    same(assigned_on_one_path, X, True, loops=1)
    same(assigned_on_one_path, X, False, loops=1)


def loop_prints(x):
    s = 0.0
    for v in x:
        s += v
        print(s)
    return s


def test_a_loop_that_cannot_compile_stays_in_python():
    same(loop_prints, X, loops=0)


def inner_fits(x, y):
    log = []
    for k in range(3):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * k
        y[k] = s
        log.append(s)
    return log


def test_inner_loop_of_a_python_loop():
    same(inner_fits, X, np.zeros(3), loops=1)


def three_levels(x):
    log = []
    for a in range(2):
        for b in range(2):
            s = 0.0
            for i in range(len(x)):
                s += x[i]
            log.append((a, b, s))
    return log


def test_three_levels():
    same(three_levels, X, loops=1)


# Variables going in and out


def zero_trip(n, tag):
    tag = str(tag)
    for i in range(n):
        last = i
    return last, tag


def loop_variable_after(n, tag):
    tag = str(tag)
    for i in range(n):
        pass
    return i, tag


def defined_before(n, tag):
    tag = str(tag)
    i = -1
    last = -2
    for i in range(n):
        last = i * 2
    return i, last, tag


@pytest.mark.parametrize('n', [0, 1, 5])
def test_unbound_after_the_loop(n):
    same(zero_trip, n, 't')
    same(loop_variable_after, n, 't')
    same(defined_before, n, 't', loops=1)


def maybe_defined(x, flag):
    label = str(flag)
    if flag:
        s = 1.0
    for v in x:
        s = s + v
    return s, label


def read_first_iteration_only(x, flag):
    label = str(flag)
    if flag:
        p = 2.0
    t = 0.0
    for i in range(len(x)):
        if i == 3:
            t += p
        p = 7.0
    return t, label


@pytest.mark.parametrize('flag', [True, False])
def test_read_before_assigned(flag):
    same(maybe_defined, X, flag)
    same(maybe_defined, X[:0], flag)
    same(read_first_iteration_only, X, flag)


def local_only(x):
    log = {}
    s = 0.0
    for i in range(len(x)):
        t = x[i] * 2
        s += t
    log['s'] = s
    return log


def test_variable_used_only_inside():
    same(local_only, X, loops=1)


def type_changes(x, tag):
    tag = str(tag)
    s = 0
    for v in x:
        s += v
    return s, tag


def types_kept(x, tag):
    tag = str(tag)
    a = 1
    b = 1.5
    c = True
    d = x[0]
    for i in range(len(x)):
        a = a + i
        b = b * 1.5
        c = i > 2
        d = d + x[i]
    return a, b, c, d, tag


def test_types_of_what_comes_back():
    same(type_changes, X, 't')
    g = same(types_kept, X, 't', loops=1)
    assert g(X, 't')[3] == types_kept(X, 't')[3]


def ints_past_64_bits(n, tag):
    tag = str(tag)
    s = n
    for i in range(3):
        s = s + 1
    return s, tag


def test_big_ints_stay_python():
    same(ints_past_64_bits, 2**70, 't')
    same(ints_past_64_bits, 5, 't', loops=1)


def none_going_in(x):
    best = None
    for v in x:
        if best is None or v > best:
            best = v
    return best, str(best)


def test_none():
    same(none_going_in, X)


def several_signatures(x):
    tag = {}
    s = x[0] * 0
    for i in range(len(x)):
        s = s + x[i]
    tag['s'] = s
    return tag


def test_each_signature_on_its_own():
    g = scram(cache=False)(several_signatures)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for x in (X, X.astype(np.int64), [1, 2, 3], (1.5, 2.5)):
            for _ in range(2):
                assert repr(g(x)) == repr(several_signatures(x))


# Errors, in order


def raises_in_loop(x, y, k):
    tag = str(k)
    for i in range(len(x)):
        y[i] = x[i] * 2
        y[k + i] = 1.0
    return tag


def divides(x, y, k):
    tag = str(k)
    for i in range(len(x)):
        y[i] = 1 / (i - k)
    return tag


def raises_after(x, k):
    tag = {}
    s = 0.0
    for v in x:
        s += v
    tag[s] = 1
    return tag[k]


@pytest.mark.parametrize('k', [0, 1, 3, 8, 100, -3, -100])
def test_exceptions_and_what_they_leave_behind(k):
    same(raises_in_loop, X, np.zeros(8), k)
    same(divides, X, np.zeros(8), k)
    same(raises_after, X, k)


def caught(x, y, k):
    s = 0.0
    try:
        for i in range(len(x)):
            s += x[i]
            y[k + i] = s
    except IndexError:
        pass
    return s, str(k)


def caught_in_an_outer_loop(x, y, k):
    log = []
    for j in range(2):
        s = 0.0
        try:
            for i in range(len(x)):
                s += x[i]
                y[k + i] = s
        except IndexError as e:
            log.append((j, s, str(e)))
    return log


def caught_with_with(x, k):
    s = 0.0
    with contextlib.suppress(IndexError):
        for i in range(len(x)):
            s += x[k + i]
    return s, str(k)


@pytest.mark.parametrize('k', [0, 3, 8, 100])
def test_a_handler_sees_the_variables_as_they_were(k):
    same(caught, X, np.zeros(8), k, loops=0)
    same(caught_in_an_outer_loop, X, np.zeros(8), k, loops=0)
    same(caught_with_with, X, k, loops=0)


def returns_from_loop(x, tag):
    tag = str(tag)
    for i in range(len(x)):
        if x[i] > 4:
            return i, tag
    return -1, tag


def inner_of_a_returning_loop(x, tag):
    tag = str(tag)
    for k in range(3):
        s = 0.0
        for v in x:
            s += v
        if s > 100 * k:
            return k, s, tag
    return tag


def breaks(x, tag):
    tag = str(tag)
    s = 0.0
    last = -1
    for i in range(len(x)):
        if x[i] > 3:
            break
        s += x[i]
        last = i
    return s, last, tag


def continues(x, tag):
    tag = str(tag)
    s = 0.0
    for i in range(len(x)):
        if i % 2:
            continue
        s += x[i]
    return s, tag


def whiles(x, tag):
    tag = str(tag)
    i = 0
    s = 0.0
    while i < len(x) and s < 10:
        s += x[i]
        i += 1
    return i, s, tag


def for_else(x, tag):
    tag = str(tag)
    found = -1
    for i in range(len(x)):
        if x[i] > tag.count('z'):
            found = i
            break
    else:
        found = -2
    return found, tag


def test_control_flow():
    same(returns_from_loop, X, 't', loops=0)
    same(inner_of_a_returning_loop, X, 't', loops=1)
    same(breaks, X, 't', loops=1)
    same(continues, X, 't', loops=1)
    same(whiles, X, 't', loops=1)
    same(for_else, X, 'zzz')


# Functions that share or inspect their frame


def closure_reads(x):
    s = 0.0
    def get():
        return s
    for v in x:
        s += v
    return get(), s


def calls_locals(x):
    s = 0.0
    for v in x:
        s += v
    return dict(locals())['s'], str(s)


def with_del(x):
    s = 0.0
    for v in x:
        s += v
    t = s
    del s
    return t, str(t)


G = 0.0


def with_global(x):
    global G
    G = 0.0
    for v in x:
        G += v
    return G, str(G)


def generator(x):
    s = 0.0
    for v in x:
        s += v
        yield s


class Holder:
    __secret = 5.0

    def __init__(self):
        self.k = 2.0
        self.__other = 3.0

    def method(self, x):
        s = 0.0
        for v in x:
            s += v
        return s * self.k + self.__secret + self.__other

    def attribute_in_loop(self, x):
        s = 0.0
        for v in x:
            s += v * self.k
        return s


def test_frames_stay_as_they_are():
    same(closure_reads, X, loops=0)
    same(calls_locals, X, loops=0)
    same(with_del, X, loops=0)
    same(with_global, X, loops=0)
    g = scram(cache=False)(generator)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert list(g(X)) == list(generator(X))


def test_methods():
    h = Holder()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for f in (Holder.method, Holder.attribute_in_loop):
            g = scram(cache=False)(f)
            assert g(h, X) == f(h, X)
            assert g(h, X) == f(h, X)


def defaults(x, k=2.0, *rest, scale=3.0, **more):
    tag = str(rest) + str(more)
    s = 0.0
    for v in x:
        s += v * k * scale
    return s, tag


def test_every_kind_of_argument():
    g = scram(cache=False)(defaults)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for a, kw in (((X,), {}), ((X, 1.0), {}), ((X, 1.0, 9), {}), ((X,), {'scale': 1.0}), ((X,), {'z': 1})):
            for _ in range(2):
                assert g(*a, **kw) == defaults(*a, **kw)


def fact(n, tag):
    tag = str(tag)
    s = 1
    for i in range(1, n + 1):
        s *= i
    if n > 0:
        fact(n - 1, tag)
    return s, tag


def test_recursion():
    same(fact, 5, 't', loops=1)


# How it's reported


def test_fallback_false_still_raises():
    g = scram(fallback=False, cache=False)(setup_then_loop)
    with pytest.raises(CompileError, match='not compiled'):
        g(X, 'a')


def test_the_warning_says_so():
    g = scram(cache=False)(setup_then_loop)
    with pytest.warns(FallbackWarning, match='apart from the loops that compile'):
        g(X, 'a')
    h = scram(cache=False)(no_loops)
    with pytest.warns(FallbackWarning, match=r'runs as Python: '):
        h('a')


def no_loops(tag):
    return {tag: 1}


def test_one_warning_however_many_loops():
    g = scram(cache=False)(printing)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with contextlib.redirect_stdout(io.StringIO()):
            g(X)
            g(X)
    assert len([x for x in w if issubclass(x.category, FallbackWarning)]) == 1


def test_a_lifted_loop_does_not_appear_in_globals():
    same(setup_then_loop, X, 'a')
    assert not [k for k in globals() if k.startswith('__sj')]


def tiny_inner(x, n):
    log = []
    s = 0.0
    for k in range(n):
        for i in range(3):
            s += x[i]
        log.append(s)
    return log[-1]


def big_inner(x, n):
    log = []
    s = 0.0
    for k in range(n):
        for i in range(len(x)):
            s += x[i]
        log.append(s)
    return log[-1]


def test_a_tiny_inner_loop_stays_in_python_and_a_big_one_goes_native():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        tiny = scram(cache=False)(tiny_inner)
        assert tiny(X, 200) == tiny_inner(X, 200)
        tier = tiny._lifted[0]._py.__closure__[0].cell_contents      # the k loop prints nothing but appends
        assert type(tier).__name__ == 'Tier' and tier.left > 0
        big = scram(cache=False)(big_inner)
        data = np.ones(20000)
        assert big(data, 50) == big_inner(data, 50)
        assert type(big._lifted[0]._py.__closure__[0].cell_contents).__name__ == 'Function'


def grows(x, n):
    log = []
    s = 0.0
    for k in range(n):
        for i in range(min(k // 2, len(x))):
            s += x[i]
        log.append(s)
    return log[-1]


def test_a_loop_that_grows_goes_native_after_all():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g = scram(cache=False)(grows)
        data = np.ones(50000)
        assert g(data, 4000) == grows(data, 4000)
        assert type(g._lifted[0]._py.__closure__[0].cell_contents).__name__ == 'Function'
