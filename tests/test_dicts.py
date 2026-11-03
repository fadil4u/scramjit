"""Fixed dicts: p['a'] on a dict whose keys don't change compiles as a variable. Everything observable must match
CPython: results, the dict afterwards, exceptions and how far they got."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

X = np.arange(1.0, 6.0)


def deep(a):
    if isinstance(a, np.ndarray):
        return a.copy()
    if isinstance(a, dict):
        return type(a)((k, deep(v)) for k, v in a.items())
    if isinstance(a, list):
        return [deep(v) for v in a]
    return a


def state(a):
    if isinstance(a, dict):
        return ('dict', [(repr(k), state(v)) for k, v in a.items()])
    if isinstance(a, tuple):
        return ('tuple', [state(v) for v in a])
    return _value(a)


def comparable(o):
    if o[0] == 'returns' and o[1][0] == 'tuple':
        return ('returns', [_value(x) for x in o[1][1]])
    return o


def same(f, *args, compiled=None, **options):
    g = scram(cache=False, **options)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        for _ in range(2):
            plain, native = deep(list(args)), deep(list(args))
            want, got = comparable(outcome(f, plain)), comparable(outcome(g, native))
            assert want[:4] == got[:4], (f.__name__, args)
            for p, c in zip(plain, native):
                assert state(p) == state(c), f.__name__
    if compiled is not None:
        assert bool(g.signatures) == compiled, (f.__name__, g._why)
    return g


def reads(p, x):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * p['k'] + p['b']
    return s


def loop_bound(p, x):
    s = 0.0
    for i in range(p['n']):
        s += x[i]
    return s


def writes(p, x):
    for v in x:
        p['total'] += v
        p['count'] += 1
    return p['count']


def overwrites(p):
    p['a'] = 2
    p['b'] = 2.5
    return p['a'] + p['b']


def conditional_write(p, x):
    for v in x:
        if v > p['limit']:
            p['over'] = v
    return 0


def branch_on(p, x):
    if p['flag']:
        return x[0]
    return x[1]


def int_keys(p, x):
    return p[1] * x[0] + p[2]


def odd_keys(p):
    p['not an identifier'] += 1
    p['x-y'] = p['x-y'] * 2
    return p['not an identifier']


def array_field(p):
    s = 0.0
    for i in range(len(p['a'])):
        s += p['a'][i] * p['k']
        p['out'][i] = s
    return s


def tuple_field(p):
    a, b = p['pair']
    return a + b


def two_dicts(p, q, x):
    p['s'] = q['k'] * x[0]
    q['k'] = p['s'] + 1.0
    return p['s'] + q['k']


@pytest.mark.parametrize('p', [{'k': 2.0, 'b': 1.0}, {'k': 2, 'b': 1}, {'k': np.float64(2.0), 'b': 1.5}, {'k': True, 'b': 1},
                               {'k': 2.0, 'b': 1.0, 'name': 'ignored', 'other': [1, 2], 'nested': {'a': 1}}])
def test_reads(p):
    same(reads, p, X, compiled=True)


def test_reads_as_bound_and_test():
    same(loop_bound, {'n': 3}, X, compiled=True)
    same(loop_bound, {'n': 99}, X, compiled=True)      # IndexError
    same(branch_on, {'flag': True}, X, compiled=True)
    same(branch_on, {'flag': False}, X, compiled=True)
    same(int_keys, {1: 2.0, 2: 3.0}, X, compiled=True)


@pytest.mark.parametrize('p', [{'total': 0.0, 'count': 0}, {'total': 1.5, 'count': 2, 'tag': 'x'}, {'count': 5, 'total': np.float64(1.0)},
                               {'total': 0, 'count': 0}])
def test_writes_come_back(p):
    same(writes, p, X, compiled=True)
    same(writes, p, X[:0], compiled=True)


def test_stored_values_and_order():
    d = {'b': 1.0, 'a': 1, 'c': 'keep'}
    g = same(overwrites, d, compiled=True)
    g(d)
    assert list(d) == ['b', 'a', 'c'] and d == {'b': 2.5, 'a': 2, 'c': 'keep'}
    same(conditional_write, {'limit': 2.0, 'over': -1.0}, X, compiled=True)
    same(conditional_write, {'limit': 99.0, 'over': -1.0}, X, compiled=True)
    same(odd_keys, {'not an identifier': 1, 'x-y': 3.5}, compiled=True)


def test_nothing_written_is_left_alone():
    v = 7.25
    d = {'k': v, 'b': 1.0}
    g = same(reads, d, X, compiled=True)
    g(d, X)
    assert d['k'] is v


def test_array_values():
    d = {'a': X.copy(), 'k': 2.0, 'out': np.zeros(5)}
    g = same(array_field, d, compiled=True)
    d2 = deep(d)
    g(d2)
    assert d2['out'][4] == array_field(deep(d))


def test_tuple_values():
    same(tuple_field, {'pair': (1.0, 2)}, compiled=True)


def test_two_dicts():
    same(two_dicts, {'s': 0.0}, {'k': 2.0}, X, compiled=True)


# Exceptions: Python has stored what it stored


def raises_midway(p, x, k):
    for i in range(len(x)):
        p['count'] += 1
        p['s'] = p['s'] + x[k + i]
    return p['s']


def divides(p, x):
    p['before'] = 1.0
    p['r'] = 1.0 / p['d']
    p['after'] = 2.0
    return p['r']


@pytest.mark.parametrize('k', [0, 1, 3, 4, 5, 100])
def test_an_error_keeps_what_was_stored(k):
    same(raises_midway, {'count': 0, 's': 0.0}, X, k, compiled=True)


@pytest.mark.parametrize('d', [0.0, 2.0, 0])
def test_division(d):
    same(divides, {'before': 0.0, 'r': 0.0, 'after': 0.0, 'd': d}, X)


# What can't be a fixed dict runs as Python, and stays right


def missing_key_read(p):
    return p['nope']


def adds_a_key(p, x):
    p['new'] = x[0]
    return x[0]


def uses_len(p):
    return len(p) + p['a']


def uses_in(p):
    return p['a'] if 'a' in p else 0


def uses_get(p):
    return p.get('a', 0)


def variable_key(p, k):
    return p[k]


def iterates(p):
    s = 0
    for k in p:
        s += p[k]
    return s


def returns_it(p):
    p['a'] = 2
    return p


def deletes(p):
    del p['a']
    return p['b']


def rebinds(p, q):
    t = p['a']
    p = q
    return t + p['a']


def nested(p):
    return p['in']['x']


def stores_a_tuple(p):
    p['a'] = (1, 2)
    return 1


def test_other_uses():
    d = {'a': 1, 'b': 2.5}
    for f, args in ((missing_key_read, (d,)), (adds_a_key, (d, X)), (uses_len, (d,)), (uses_in, (d,)), (uses_get, (d,)),
                    (variable_key, (d, 'a')), (variable_key, (d, 'zz')), (iterates, ({'a': 1, 'b': 2},)), (returns_it, (d,)),
                    (deletes, (d,)), (rebinds, (d, {'a': 5})), (nested, ({'in': {'x': 1}},)), (stores_a_tuple, (d,))):
        same(f, *args)


def test_value_types_that_cannot_be_passed():
    same(reads, {'k': 'text', 'b': 1.0}, X)
    same(reads, {'k': None, 'b': 1.0}, X)
    same(reads, {'k': 2**70, 'b': 1.0}, X)
    same(reads, {'b': 1.0}, X)                          # KeyError


def test_same_dict_twice():
    def build():
        d = {'s': 1.0, 'k': 2.0}
        return [d, d, X]
    plain, native = build(), build()
    g = scram(cache=False)(two_dicts)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        want, got = outcome(two_dicts, plain), outcome(g, native)
    assert want[:2] == got[:2] and plain[0] == native[0]
    assert native[0] is native[1]


def test_subclasses_run_as_python():
    import collections
    same(reads, collections.OrderedDict(k=2.0, b=1.0), X, compiled=False)

    class Loud(dict):
        def __getitem__(self, k):
            return 100.0
    d = Loud(k=2.0, b=1.0)
    g = scram(cache=False)(reads)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(d, X) == reads(d, X)
    dd = collections.defaultdict(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(dd, X) == 0.0 and set(dd) == {'k', 'b'}


# Between calls


def test_a_changed_type_is_a_new_signature():
    g = scram(cache=False)(reads)
    assert g({'k': 2.0, 'b': 1.0}, X) == reads({'k': 2.0, 'b': 1.0}, X)
    assert g({'k': 2, 'b': 1}, X) == reads({'k': 2, 'b': 1}, X)
    assert len(g.signatures) == 2


def test_a_call_that_ran_as_python_does_not_stick():
    g = scram(cache=False)(reads)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        with pytest.raises(KeyError):
            g({'b': 1.0}, X)
        assert g({'k': 2.0, 'b': 1.0}, X) == reads({'k': 2.0, 'b': 1.0}, X)
    assert g.signatures


def test_values_changed_between_calls():
    g = scram(cache=False)(reads)
    d = {'k': 2.0, 'b': 1.0}
    for k in (2.0, 3.0, -1.5):
        d['k'] = k
        assert g(d, X) == reads(d, X)
    assert len(g.signatures) == 1


# With the rest of the compiler


def int_becomes_float(p, x):
    for v in x:
        p['s'] += v
    return 0


def int_literal_slot(p, x):
    p['s'] = 0
    for v in x:
        p['s'] += v
    return 0


def float_into_int_uses(p, x):
    p['n'] = 1.5
    t = p['n'] + 1
    return t


def versioned(p, x):
    s = 0.0
    for j in range(2):
        s += x[p['base'] + j]
    return s


def lifted(p, x, name):
    tag = {name: 1}
    for v in x:
        p['s'] += v * p['k']
    tag['s'] = p['s']
    return tag


def lifted_inner(p, x):
    log = []
    for k in range(3):
        for v in x:
            p['s'] += v
        log.append(p['s'])
    return log


@pytest.mark.parametrize('x', [X, X[:0]], ids=['many', 'none'])
@pytest.mark.parametrize('s', [0, 3, 0.5, 2**53 + 1, np.float64(1.0)])
def test_an_int_slot_that_becomes_float(x, s):
    same(int_becomes_float, {'s': s}, x)
    same(int_literal_slot, {'s': s}, x)


def test_a_slot_that_changes_badly():
    same(float_into_int_uses, {'n': 1}, X)


@pytest.mark.parametrize('base', [0, 3, 4, -5, -1, -9])
def test_versioned_loops(base):
    same(versioned, {'base': base}, X, compiled=True)


def test_lifting():
    g = same(lifted, {'s': 0.0, 'k': 2.0}, X, 'a')
    assert g._native_loops()
    h = same(lifted_inner, {'s': 0.0}, X)
    assert h._native_loops()


@pytest.mark.parametrize('options', [{'boundscheck': False}, {'parity': False}])
def test_options(options):
    same(reads, {'k': 2.0, 'b': 1.0}, X, compiled=True, **options)
    same(writes, {'total': 0.0, 'count': 0}, X, compiled=True, **options)


def test_the_loop_is_the_same_as_with_locals():
    def with_dict(p, x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * p['k']
        return s

    def with_local(k, x):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * k
        return s
    a = scram(cache=False)(with_dict)
    b = scram(cache=False)(with_local)
    a({'k': 2.0}, X)
    b(2.0, X)
    strip = lambda c: [x.strip() for x in c.splitlines() if not x.lstrip().startswith('#line')][1:]
    assert [x.replace('v_p__k', 'v_k') for x in strip(a.c())] == strip(b.c())


def test_messages_use_the_written_form():
    g = scram(cache=False)(float_into_int_uses)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g({'n': 1}, X)
    why = ' '.join(r for _, r in g._why.values())
    assert 'p__n' not in why


# The fast call path, and what it must turn down


def test_the_fast_path_is_taken():
    g = scram(cache=False)(reads)
    d = {'k': 2.0, 'b': 1.0}
    g(d, X)
    plan = next(iter(g._plans.values()))
    assert plan.__name__ == 'plan'
    assert g(d, X) == reads(d, X)


def test_the_fast_path_turns_down_what_it_cannot_vouch_for():
    g = scram(cache=False)(reads)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g({'k': 2.0, 'b': 1.0}, X)
        for d in ({'k': 2, 'b': 1.0}, {'k': 'x', 'b': 1.0}, {'b': 1.0}, {'k': 2**70, 'b': 1.0}, {'k': True, 'b': 1.0},
                  {'k': np.float64(3.0), 'b': 1.0}, {'k': 2.0, 'b': 1.0, 'extra': 'x'}):
            assert outcome(g, [d, X])[:3] == outcome(reads, [d, X])[:3], d
        assert g({'k': 3.0, 'b': 1.0}, X) == reads({'k': 3.0, 'b': 1.0}, X)


def test_the_fast_path_writes_back_and_keeps_going_after_errors():
    g = scram(cache=False)(raises_midway)
    for k in (0, 3, 100, 1):
        a, b = {'count': 0, 's': 0.0}, {'count': 0, 's': 0.0}
        assert outcome(g, [a, X, k])[:3] == outcome(raises_midway, [b, X, k])[:3]
        assert a == b


def test_the_fast_path_with_one_dict_twice():
    g = scram(cache=False)(two_dicts)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        g({'s': 0.0}, {'k': 2.0}, X)
        g({'s': 0.0}, {'k': 2.0}, X)
        a = {'s': 1.0, 'k': 2.0}
        b = {'s': 1.0, 'k': 2.0}
        assert g(a, a, X) == two_dicts(b, b, X) and a == b


def test_the_fast_path_with_big_ints_and_numpy_values():
    g = scram(cache=False)(writes)
    for d in ({'total': np.float64(1.0), 'count': 1}, {'total': np.float64(2.0), 'count': 2}, {'total': 1.5, 'count': 2**70}):
        a, b = deep(d), deep(d)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FallbackWarning)
            assert outcome(g, [a, X])[:2] == outcome(writes, [b, X])[:2]
        assert state(a) == state(b)


def test_a_list_in_a_dict_runs_as_python():
    """A list value can't be copied back through the dict yet: that call runs as Python, with Python's result."""
    def f(p):
        p['xs'][0] += 1.0
        return p['xs'][0]
    g = scram(cache=False)(f)
    d1, d2 = {'xs': [1.0, 2.0]}, {'xs': [1.0, 2.0]}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert g(d1) == f(d2) and d1 == d2
