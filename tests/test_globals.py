"""Numbers, tuples, arrays and dicts at module level: read at each call, like arguments. Everything must match CPython."""

import math
import threading
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import _value, outcome

K = 3
RATE = 0.25
FLAG = True
TABLE = (1.0, 2.5, 4.0)
MIXED = (2, 0.5)
WEIGHTS = np.array([0.5, 1.5, 2.5, 3.5])
GRID = np.arange(12.0).reshape(3, 4)
SETTINGS = {'dt': 0.5, 'steps': 4, 'name': 'unused'}
STATS = {'count': 0, 'total': 0.0}
TEXT = 'text'
NOTHING = None
LIST = [1.0, 2.0]
NP_VALUE = np.float64(1.5)
LATE_NAME = 'see the last test'


def fresh(args):
    return [a.copy() if isinstance(a, np.ndarray) else a for a in args]


def comparable(o):
    if o[0] == 'returns' and o[1][0] == 'tuple':
        return ('returns', [_value(x) for x in o[1][1]])
    return o


def same(f, *args, compiled=True, **options):
    g = scram(cache=False, **options)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(3):
            plain, native = fresh(args), fresh(args)
            want, got = comparable(outcome(f, plain)), comparable(outcome(g, native))
            assert want[:4] == got[:4], (f.__name__, want, got)
            for p, c in zip(plain, native):
                if isinstance(p, np.ndarray):
                    assert p.tobytes() == c.tobytes()
    assert bool(g.signatures) == compiled, (f.__name__, g._why)
    return g


X = np.arange(1.0, 9.0)


def number_in_loop(x):
    s = 0.0
    for i in range(len(x)):
        s += x[i % K] * RATE
    return s


def as_a_range(x):
    s = 0.0
    for i in range(K):
        s += x[i]
    return s


def in_a_test(x):
    return x[0] if FLAG else x[1]


def tuple_constant(i):
    a, b, c = TABLE
    return a + b * c + TABLE[0] + i


def mixed_tuple():
    return MIXED[0] + MIXED[1]


def global_array(x):
    s = 0.0
    for i in range(len(WEIGHTS)):
        s += WEIGHTS[i] * x[i]
    return s


def global_view(i):
    row = GRID[i]
    return row.sum() + GRID[1, 2] + GRID[:, 1].max()


def global_reduction():
    return WEIGHTS.mean() + np.std(WEIGHTS)


def global_dict(x):
    s = 0.0
    for k in range(SETTINGS['steps']):
        s += x[k] * SETTINGS['dt']
    return s


def numpy_scalar_constant(x):
    return x[0] * NP_VALUE


def only_a_math_function(x):
    return math.sqrt(x[0]) + np.sqrt(x[1])


def test_each_kind():
    same(number_in_loop, X)
    same(as_a_range, X)
    same(in_a_test, X)
    same(tuple_constant, 2)
    same(mixed_tuple)
    same(global_array, X)
    same(global_view, 1)
    same(global_view, 99)
    same(global_reduction)
    same(global_dict, X)
    same(numpy_scalar_constant, X)
    same(only_a_math_function, X)


def takes_no_arguments():
    return K * 2


def test_no_arguments_at_all():
    same(takes_no_arguments)


def writes_global_array(i):
    WEIGHTS[i] = WEIGHTS[i] + 1.0


def writes_global_dict(x):
    for v in x:
        STATS['count'] += 1
        STATS['total'] += v


def test_writes_reach_the_global():
    before = WEIGHTS.copy()
    g = scram(cache=False, fallback=False)(writes_global_array)
    g(1)
    assert WEIGHTS[1] == before[1] + 1.0
    WEIGHTS[:] = before
    STATS.update(count=0, total=0.0)
    h = scram(cache=False, fallback=False)(writes_global_dict)
    h(X)
    assert STATS['count'] == 8 and STATS['total'] == X.sum()
    assert type(STATS['total']) is np.float64


def returns_global_array():
    return WEIGHTS


def returns_view_of_global():
    return WEIGHTS[1:]


def test_returning_one():
    assert scram(cache=False, fallback=False)(returns_global_array)() is WEIGHTS
    r = scram(cache=False, fallback=False)(returns_view_of_global)()
    assert np.shares_memory(r, WEIGHTS) and np.array_equal(r, WEIGHTS[1:])


def test_a_new_value_needs_no_recompile():
    global K
    g = scram(cache=False, fallback=False)(number_in_loop)
    saved = K
    try:
        for K in (3, 5, 1, 2):
            assert g(X) == number_in_loop(X)
    finally:
        K = saved
    assert len(g.signatures) == 1


def test_a_new_type_is_a_new_signature():
    global RATE
    g = scram(cache=False, fallback=False)(number_in_loop)
    saved = RATE
    try:
        assert g(X) == number_in_loop(X)
        RATE = 2
        assert g(X) == number_in_loop(X) and type(g(X)) is type(number_in_loop(X))
        RATE = np.float64(0.5)
        assert g(X) == number_in_loop(X) and type(g(X)) is type(number_in_loop(X))
    finally:
        RATE = saved
    assert len(g.signatures) == 3


def test_a_global_that_stops_being_a_number_runs_as_python():
    global K
    g = scram(cache=False)(number_in_loop)
    saved = K
    try:
        assert g(X) == number_in_loop(X)
        K = 'text'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert outcome(g, [X])[:3] == outcome(number_in_loop, [X])[:3]
        K = 2**70
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert outcome(g, [X])[:3] == outcome(number_in_loop, [X])[:3]
        K = 3
        assert g(X) == number_in_loop(X)
    finally:
        K = saved


def test_a_global_that_goes_away_is_a_name_error():
    global K
    g = scram(cache=False)(number_in_loop)
    saved = K
    try:
        g(X)
        del K
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(NameError):
                g(X)
    finally:
        K = saved
    assert g(X) == number_in_loop(X)


def test_an_array_swapped_for_another():
    global WEIGHTS
    g = scram(cache=False, fallback=False)(global_array)
    saved = WEIGHTS
    try:
        WEIGHTS = np.ones(8)
        assert g(X) == global_array(X)
        WEIGHTS = saved[::-1]
        assert g(X) == global_array(X)
        WEIGHTS = np.ones((2, 2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert outcome(scram(cache=False)(global_array), [X])[:2] == outcome(global_array, [X])[:2]
    finally:
        WEIGHTS = saved


def shadowed_by_local(x):
    K = 2
    return x[K]


def shadowed_by_loop(x):
    s = 0.0
    for K in range(2):
        s += x[K]
    return s


def shadowed_by_comprehension(x):
    return sum((x[K] for K in range(2)), 0.0)


def shadowed_by_parameter(K):
    return K + 1


def global_and_comprehension(x):
    return sum((x[i] * RATE for i in range(K)), 0.0)


def test_names_mean_what_they_mean_in_python():
    same(shadowed_by_local, X)
    same(shadowed_by_loop, X)
    same(shadowed_by_comprehension, X)
    same(shadowed_by_parameter, 5)
    same(global_and_comprehension, X)
    assert scram(cache=False)(shadowed_by_local)._hidden is None
    g = scram(cache=False, fallback=False)(shadowed_by_local)
    g(X)
    assert g._hidden == []


def uses_text(x):
    return x[0] if TEXT == 'text' else x[1]


def uses_none(x):
    return x[0] if NOTHING is None else x[1]


def uses_list(i):
    return LIST[i]


def calls_a_global(x):
    return helper(x)


def helper(x):
    return x[0]


def undefined(x):
    return x[0] + NOT_DEFINED_ANYWHERE  # noqa: F821


def conditionally_undefined(x, flag):
    if flag:
        return x[0] + NOT_DEFINED_ANYWHERE  # noqa: F821
    return x[1]


def assigns_global(x):
    global STATE
    STATE = x[0]
    return STATE


def test_what_stays_python_stays_right():
    same(uses_text, X, compiled=False)
    same(uses_none, X, compiled=False)
    same(uses_list, 1, compiled=True)          # a list of numbers at module level compiles, from 0.0.4
    same(calls_a_global, X, compiled=False)
    same(undefined, X, compiled=False)
    same(conditionally_undefined, X, True, compiled=False)
    same(conditionally_undefined, X, False, compiled=False)
    same(assigns_global, X, compiled=False)


def lifted(x, name):
    tag = {name: K}
    s = 0.0
    for i in range(len(x)):
        s += x[i] * RATE + TABLE[0]
    tag['s'] = s
    return tag


def test_lifted_loops_read_them_too():
    g = same(lifted, X, 'a', compiled=False)
    assert g._native_loops()


def test_a_constant_defined_after_the_function():
    def late(x):
        return x[0] * LATE_VALUE
    g = scram(cache=False)(late)
    globals()['LATE_VALUE'] = 2.5
    try:
        assert scram(cache=False, fallback=False)(late)(X) == late(X)
    finally:
        del globals()['LATE_VALUE']


def test_options():
    for options in ({'boundscheck': False}, {'parity': False}):
        same(number_in_loop, X, **options)
        same(global_array, X, **options)


def test_no_globals_means_no_change():
    def plain(x):
        return x[0]
    g = scram(cache=False)(plain)
    g(X)
    assert g._hidden == [] and g._ctree is None


def test_signatures_in_reports_show_only_real_arguments():
    import contextlib
    import io
    g = scram(cache=False, fallback=False)(number_in_loop)
    g(X)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g.preflight()
    head = out.getvalue().splitlines()[0]
    assert head.startswith('number_in_loop(float64[:])'), head


def wide_fast_path(x, y, z):
    return x[0] + y + z + K + RATE


def test_the_fast_path_with_globals():
    g = scram(cache=False, fallback=False)(wide_fast_path)
    assert g(X, 1.0, 2) == wide_fast_path(X, 1.0, 2)
    plan = next(iter(g._plans.values()))
    assert plan.__name__ == 'plan'
    assert g(X, 1.0, 2) == wide_fast_path(X, 1.0, 2)
