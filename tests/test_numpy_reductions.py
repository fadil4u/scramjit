"""np.sum(x), x.max() and the rest: the bits and types numpy gives, for every shape of array, with anything that doesn't give one
number that can be matched staying Python. Two NaNs of different signs are the one documented exception."""

import ast
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram
from scramjit._dispatch import Function

from util import _value, outcome

np.seterr(all='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

OPS = ['sum', 'prod', 'mean', 'max', 'min', 'any', 'all', 'argmax', 'argmin']
INT_OPS = [o for o in OPS if o != 'mean']
RNG = np.random.default_rng(12)


def make(body, *params):
    src = f'def f({", ".join(params)}):\n    return {body}\n'
    space = {'np': np}
    exec(src, space)
    g = Function(space['f'], cache=False)
    g._tree, g._why_not = ast.parse(src).body[0], None
    from scramjit._dispatch import _dict_keys
    g._dict_keys, g._first = _dict_keys(g._tree), 1
    return space['f'], g


def check(body, args, params=('x',), compiled=True):
    f, g = make(body, *params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(2):
            want, got = outcome(f, list(args)), outcome(g, list(args))
            assert want[:3] == got[:3], (body, want, got)
    if compiled is not None:
        assert bool(g.signatures) == compiled, (body, g._why)


def floats(n):
    return RNG.random(n) * 10.0 ** RNG.integers(-8, 8, n)


LENGTHS = [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 64, 127, 128, 129, 130, 257, 1000, 8191, 8192, 8193, 8200, 16385, 40000, 100000]


@pytest.mark.parametrize('n', LENGTHS)
def test_sum_rounds_exactly(n):
    x = floats(n)
    check('np.sum(x)', [x])
    check('x.sum()', [x])
    check('np.mean(x)', [x])
    check('x.mean()', [x])
    check('np.prod(x)', [x])


@pytest.mark.parametrize('n', [64, 300, 1000, 20000])
@pytest.mark.parametrize('step', [2, 3, -1, -2, 5])
def test_views_with_gaps(n, step):
    x = floats(n)
    v = 'x[::%d]' % step
    for op in ('sum', 'mean', 'prod', 'max', 'min', 'argmax', 'argmin', 'any', 'all'):
        check(f'{v}.{op}()', [x])
        check(f'np.{op}({v})', [x])


@pytest.mark.parametrize('shape', [(2, 3), (10, 10), (7, 40), (100, 30), (2, 3, 4), (5, 5, 5), (10, 10, 10), (1, 300), (300, 1), (0, 3), (3, 0), (200, 200)])
def test_contiguous_arrays_of_any_shape(shape):
    x = floats(int(np.prod(shape))).reshape(shape)
    for op in OPS:
        check(f'x.{op}()', [x])
        check(f'np.{op}(x)', [x])


def test_rows_and_slices():
    m = floats(600).reshape(20, 30)
    for op in OPS:
        check(f'm[3].{op}()', [m], ('m',))
        check(f'm[i].{op}()', [m, 7], ('m', 'i'))
        check(f'np.{op}(m[1:5])', [m], ('m',))
        check(f'm[:, 2].{op}()', [m], ('m',))
    check('m[99].sum()', [m], ('m',))                 # IndexError, as numpy
    check('m[:, 99].max()', [m], ('m',))


@pytest.mark.parametrize('op', INT_OPS)
def test_integers(op):
    for x in ([3, 1, 2], [2**63 - 1, 1], [-2**63, -1], [0, 0], [0, 5], [5], [], [2**40, 2**40, 2**40], [-3, -3, 7, 7, -9]):
        a = np.array(x, dtype=np.int64)
        if op in ('argmax', 'argmin') and np.intp is not np.int64:
            check(f'x.{op}()', [a], compiled=False)
        else:
            check(f'x.{op}()', [a])
            check(f'np.{op}(x)', [a])


SPECIAL = [[0.0, -0.0], [-0.0, 0.0], [-0.0], [0.0], [-0.0, -0.0], [0.0, -0.0, 0.0], [-0.0, 0.0, -0.0], [np.inf, -np.inf], [np.inf, 1.0], [-np.inf, 1.0],
           [1e308, 1e308], [5e-324, -5e-324], [np.nan], [1.0, np.nan], [np.nan, 1.0], [1.0, np.nan, 2.0], [0.0, np.nan], [-np.nan], [1.0, -np.nan, 3.0],
           [2.0, 2.0], [1.0, 2.0, 2.0, 1.0]]


@pytest.mark.parametrize('values', SPECIAL, ids=str)
@pytest.mark.parametrize('op', OPS)
def test_special_values(values, op):
    check(f'x.{op}()', [np.array(values)])


@pytest.mark.parametrize('op', ['max', 'min'])
@pytest.mark.parametrize('n', [2, 3, 5, 8, 9, 16, 17, 33, 64, 100, 1000])
def test_signed_zeros_at_every_length(op, n):
    f, g = make(f'x.{op}()', 'x')
    for _ in range(25):
        a = RNG.choice([0.0, -0.0, 0.0, 1.0, -1.0] if _ % 2 else [0.0, -0.0], n)
        assert _value(f(a)) == _value(g(a)), a


@pytest.mark.parametrize('op', ['max', 'min'])
def test_two_nans_still_give_a_nan(op):
    f, g = make(f'x.{op}()', 'x')
    a = RNG.random(50)
    a[3], a[40] = np.nan, -np.nan
    assert np.isnan(f(a)) and np.isnan(g(a))


@pytest.mark.parametrize('op', ['max', 'min', 'argmax', 'argmin'])
def test_empty_raises(op):
    check(f'x.{op}()', [np.zeros(0)])
    check(f'np.{op}(x)', [np.zeros(0)])
    check(f'x[5:].{op}()', [np.ones(3)])


# axis and keepdims


ONE = np.array([3.0, 1.0, 2.0])
TWO = floats(12).reshape(3, 4)


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('how', ['axis=None', 'axis=0', 'axis=-1', '0', '-1', 'None', 'keepdims=False', 'axis=None, keepdims=False', 'axis=0, keepdims=False'])
def test_one_number_from_one_dimension_compiles(op, how):
    check(f'x.{op}({how})', [ONE])
    check(f'np.{op}(x, {how})', [ONE])


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('how', ['axis=1', 'axis=-2', 'axis=2', '1', 'keepdims=True', 'axis=0, keepdims=True', 'axis=None, keepdims=True', 'axis=(0,)', 'axis=[0]', 'axis=0.0'])
def test_one_dimension_other_forms_stay_python(op, how):
    check(f'x.{op}({how})', [ONE], compiled=False)
    check(f'np.{op}(x, {how})', [ONE], compiled=False)


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('how', ['axis=0', 'axis=1', 'axis=-1', '0', 'keepdims=True', 'axis=0, keepdims=True', 'axis=(0, 1)', 'axis=(0,)', 'axis=2'])
def test_two_dimensions_with_an_axis_stay_python(op, how):
    f, g = make(f'x.{op}({how})', 'x')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            want = f(TWO)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                g(TWO)
        else:
            got = g(TWO)
            assert type(got) is type(want) and np.shape(got) == np.shape(want) and np.array_equal(got, want)
    assert not g.signatures, how


@pytest.mark.parametrize('how', ['axis=None', 'axis=None, keepdims=False', 'None', 'keepdims=False'])
def test_two_dimensions_flattened_compile(how):
    for op in OPS:
        check(f'x.{op}({how})', [TWO])


@pytest.mark.parametrize('how', ['dtype=np.int64', 'dtype=float', 'out=y', 'initial=5.0', 'where=z', 'ddof=1', 'order="C"', 'bogus=1'])
def test_other_arguments_stay_python(how):
    z = ONE > 1.5
    f, g = make(f'x.sum({how})', 'x', 'y', 'z')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a, b = np.zeros(()), np.zeros(())
        assert outcome(f, [ONE, a, z])[:2] == outcome(g, [ONE, b, z])[:2] or repr(f(ONE, a, z)) == repr(g(ONE, b, z))
    assert not g.signatures


def test_an_axis_in_a_variable_stays_python():
    check('x.sum(axis=k)', [ONE, 0], ('x', 'k'), compiled=False)
    check('np.sum(x, k)', [TWO[0], -1], ('x', 'k'), compiled=False)


# Everything else that must stay Python


@pytest.mark.parametrize('dtype', ['f4', 'i4', 'u1', 'i1', 'u8', '?', 'f2'])
@pytest.mark.parametrize('op', OPS)
def test_other_dtypes_stay_python(dtype, op):
    a = np.arange(1, 5).astype(dtype)
    check(f'x.{op}()', [a], compiled=False)


def test_views_with_gaps_in_many_dimensions_stay_python():
    m = floats(600).reshape(20, 30)
    for v in ('m[::2]', 'm[:, ::2]', 'm[::-1]'):
        check(f'{v}.sum()', [m], ('m',), compiled=False)
        check(f'{v}.max()', [m], ('m',), compiled=False)


def test_not_arrays():
    check('np.sum(x)', [3.0], compiled=False)
    check('np.max(x)', [(1.0, 2.0)], compiled=False)
    check('np.sum(x)', [[1.0, 2.0]], compiled=False)
    check('x.sum()', [ONE.tolist()], compiled=False)        # AttributeError


def test_integer_mean_compiles():
    check('x.mean()', [np.arange(5, dtype=np.int64)])


# In real code


def normalize_rows(m):
    for i in range(len(m)):
        row = m[i]
        total = row.sum()
        for j in range(len(row)):
            row[j] = row[j] / total
    return m[0].sum()


def in_conditions(x):
    if x.any() and not np.all(x):
        return x.max() - x.min()
    return x.mean() * 2.0


def result_keeps_numpy_rules(x):
    return x.sum() / 0.0


def nested(m):
    best = -1.0
    at = -1
    for row in m:
        for k in range(1, len(row)):
            s = row[:k].sum() + row[k:].max()
            if s > best:
                best = s
                at = k
    return best, at


def full_of_side_effects(x, y):
    y[0] = x.sum()
    x[0] = -1.0
    y[1] = x.sum()
    y[2] = x[1:].sum() + x[:1].sum()


def lifted(x, name):
    tag = {name: 1}
    s = 0.0
    for i in range(1, len(x)):
        s += x[:i].sum()
    tag['s'] = s
    return tag


def dict_values(p):
    return p['a'].sum() * p['k']


def compare(f, *args, compiled=True):
    fresh = lambda: [a.copy() if isinstance(a, (np.ndarray, dict)) else a for a in args]
    g = scram(cache=False)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p, n = fresh(), fresh()
        want, got = outcome(f, p), outcome(g, n)
    assert repr(want[:3]) == repr(got[:3]), f.__name__
    for a, b in zip(p, n):
        if isinstance(a, np.ndarray):
            assert a.tobytes() == b.tobytes(), f.__name__
    assert bool(g.signatures) == compiled, g._why
    return g


def test_real_code():
    m = floats(600).reshape(20, 30)
    compare(normalize_rows, m)
    for x in (floats(30), np.zeros(4), np.array([0.0, 1.0])):
        compare(in_conditions, x)
    compare(result_keeps_numpy_rules, floats(8))
    compare(nested, floats(120).reshape(6, 20))
    compare(full_of_side_effects, floats(20), np.zeros(3))
    compare(dict_values, {'a': floats(5), 'k': 2.0})


def test_lifted_loops_use_them():
    g = compare(lifted, floats(30), 'a', compiled=False)
    assert g._native_loops()


@pytest.mark.parametrize('options', [{'boundscheck': False}, {'parity': False}])
def test_options(options):
    for op in OPS:
        src = f'def f(x):\n    return x.{op}()\n'
        space = {}
        exec(src, space)
        g = Function(space['f'], cache=False, **options)
        g._tree, g._why_not = ast.parse(src).body[0], None
        g._dict_keys, g._first = {}, 1
        x = floats(300)
        assert _value(space['f'](x)) == _value(g(x)), op


def test_a_local_with_a_method_of_that_name_is_not_a_reduction():
    class Thing:
        def sum(self):
            return 'mine'

    def call_it(t):
        return t.sum()
    g = scram(cache=False)(call_it)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert g(Thing()) == 'mine'


def test_the_probe_can_tell_the_two_apart():
    from scramjit._compiler import sum_chunk
    assert sum_chunk() in (0, 8192)


def one_off(x):
    return x.max()


def in_a_loop(m):
    t = 0.0
    for i in range(len(m)):
        t += m[i].max()
    return t


def test_the_note_about_speed_is_only_outside_loops():
    a = scram(cache=False, fallback=False)(one_off)
    a(np.ones(3))
    assert any('SIMD' in v.gotchas.get(k, '') for v in a._cache.values() for k in v.gotchas)
    b = scram(cache=False, fallback=False)(in_a_loop)
    b(np.ones((2, 2)))
    assert not any(v.gotchas for v in b._cache.values())


def test_division_messages_are_this_python_s():
    f, g = make('a % b', 'a', 'b')
    assert outcome(f, [1.0, 0.0])[:3] == outcome(g, [1.0, 0.0])[:3]
    assert g.signatures
