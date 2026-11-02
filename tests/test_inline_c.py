"""A compiled function calling C: k = cimo.compile('...'), then k.f(x). What comes out is what Python's call to k.f gives."""

import math
import warnings

import cimo
import numpy as np
import pytest

from scramjit import scram
from scramjit._dispatch import Compiled

k = cimo.compile(r"""
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#define TWICE(x) ((x) * 2)
const double WEIGHTS[3] = {0.5, 0.25, 0.25};
static double helper(double x) { return TWICE(x) + 1.0; }
double add(double a, double b) { return a + b; }
double fused(double a, double b, double c) { return a * b + c; }
double uses_helper(double x) { return helper(x) * WEIGHTS[1]; }
int32_t narrow(int32_t x) { return x; }
int8_t tiny(int8_t x) { return x; }
uint8_t utiny(uint8_t x) { return x; }
uint32_t u32(uint32_t x) { return x; }
int64_t wide(int64_t x) { return x + 1; }
uint64_t uwide(uint64_t x) { return x; }
float single(float x) { return x + 0.1f; }
_Bool truth(_Bool x) { return x; }
int64_t mix(int64_t x) { uint64_t h = (uint64_t)x; h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; return (int64_t)h; }
double total(const double *a, int64_t n) { double s = 0; for (int64_t i = 0; i < n; i++) s += a[i]; return s; }
double grid(int h, int w, const double m[h][w]) { double s = 0; for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) s += m[i][j] * (i + 1); return s; }
void scale(double *a, int64_t n, double f) { for (int64_t i = 0; i < n; i++) a[i] *= f; }
int64_t count_bytes(const uint8_t *a, int64_t n) { int64_t c = 0; for (int64_t i = 0; i < n; i++) c += a[i]; return c; }
double any_memory(void *p) { return *(double *)p; }
double uses_heap(double x) { double *p = malloc(8); *p = x; double r = *p * 2; free(p); return r; }
double checks(double x) { assert(x > 0); return x; }
double sum(double a) { return a; }
double *pointer(double *a) { return a; }
int varargs(int a, ...) { return a; }
struct pt { double x; };
double by_value(struct pt p) { return p.x; }
""")

stateful = cimo.compile("""
int total_so_far = 0;
int add_up(int x) { total_so_far += x; return total_so_far; }
""")

add = k.add
one = cimo.compile('double one(double x) { return x + 1.0; }')
clash = cimo.compile('double add(double a, double b) { return a - b; }')
contracted = cimo.compile('double mulsum(double a, double b, double c) { return a * b + c; }')
uncontracted = cimo.compile('double mulsum2(double a, double b, double c) { return a * b + c; }', fma=False)
defined = cimo.compile('double scaled(double x) { return x * FACTOR; }', defines={'FACTOR': '3.0'})


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
        return isinstance(q, np.ndarray) and p.dtype == q.dtype and p.tobytes() == q.tobytes()
    return type(p) is type(q) and (p == q or p != p)


def compiled(f):
    return any(isinstance(c, Compiled) for c in f._cache.values())


def check(fn, *args, must_compile=True, why=None):
    def copies():
        return [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    want_args, got_args = copies(), copies()
    f = scram(fn)
    want, got = outcome(fn, want_args), outcome(f, got_args)
    assert want[0] == got[0], (want, got)
    if want[0] == 'value':
        assert same(want[1], got[1]), (want, got)
    for a, b in zip(want_args, got_args):
        assert same(a, b), 'an argument changed differently'
    assert compiled(f) == must_compile, f._why
    if why:
        assert why in str(next(iter(f._why.values()))), f._why
    return got[1]


def test_numbers():
    def f(a, b):
        s = 0.0
        for i in range(3):
            s = k.add(s, a * b)
        return s, add(a, b), k.uses_helper(a), k.fused(a, b, 0.1)
    r = check(f, 1.5, 2.5)
    assert type(r[0]) is float
    check(f, 1, 2)
    check(f, True, np.float64(0.5))


@pytest.mark.parametrize('v', [0, 1, -1, 127, 128, 255, 256, 300, -129, 2 ** 31, 2 ** 32 + 5, -(2 ** 31) - 1, 2 ** 40, 2 ** 62, True, False])
def test_integers_are_cut_as_python_cuts_them(v):
    def f(x):
        return k.narrow(x), k.tiny(x), k.utiny(x), k.u32(x), k.wide(x), k.mix(x), k.add(x, 0.5), k.single(x)
    check(f, v)


@pytest.mark.parametrize('v', [0.1, 1e300, -1e300, 1e-50, float('inf'), float('nan'), 2 ** 53 + 1, 16777217, 2 ** 62 + 1])
def test_floats_are_rounded_as_python_rounds_them(v):
    def f(x):
        return k.single(x), k.add(x, 0.0)
    check(f, v)


def test_what_python_takes_from_numpy():
    def narrow_of_np(x):
        return k.narrow(x), k.tiny(x)

    def double_of_any(x):
        return k.add(x, 1.0), k.single(x)
    check(narrow_of_np, np.int64(300))
    check(double_of_any, np.int64(300))
    check(double_of_any, np.float64(0.25))
    check(double_of_any, np.True_)


def test_what_python_raises_for_stays_python():
    def i(x):
        return k.narrow(x)

    def too_few(x):
        return k.add(x)

    def named(x):
        return k.add(a=x, b=x)
    check(i, 1.5, must_compile=False, why='narrow')
    check(i, np.float64(1.5), must_compile=False)
    check(i, np.True_, must_compile=False)
    check(too_few, 1.0, must_compile=False, why='with these arguments')
    check(named, 1.0, must_compile=False)


def test_a_numpy_integer_as_64_bits_stays_as_python_has_it():
    def f(x):
        return k.wide(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = scram(f)
        g(np.int64(3))
    assert not compiled(g) and 'int(...)' in str(next(iter(g._why.values())))

    def h(x):
        return k.wide(int(x[0]))
    check(h, np.array([3]))


def test_bool():
    def f(x):
        return k.truth(x)
    check(f, True)
    check(f, 300, must_compile=False)


def test_arrays():
    def f(x):
        s = k.total(x, len(x))
        k.scale(x, len(x), 2.0)
        return s, k.total(x, len(x)), k.total(x[1:], len(x) - 1), x, k.any_memory(x), k.total(np.ones(3), 3)
    check(f, np.arange(4.0))

    def g(m):
        return k.grid(m.shape[0], m.shape[1], m), k.grid(1, m.shape[1], m[1])
    check(g, np.arange(6.0).reshape(2, 3))

    def b(a):
        return k.count_bytes(a, len(a))
    check(b, np.arange(5, dtype=np.uint8))


def test_arrays_that_stay_python():
    def f(x):
        return k.total(x, 2)

    def s(x):
        k.scale(x, 2, 2.0)
        return x

    def v(x):
        return k.total(x[::2], 2)

    def n(x):
        return k.total(x, 2)
    check(f, np.arange(2, dtype=np.int64), must_compile=False)    # another element type: Python reinterprets it
    check(f, np.arange(2, dtype=np.float32), must_compile=False)
    check(v, np.arange(4.0), must_compile=False, why='gaps')
    check(n, 1.5, must_compile=False)
    ro = np.arange(2.0)
    ro.flags.writeable = False
    check(f, ro)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = scram(s)
        x = np.arange(2.0)
        x.flags.writeable = False
        g(x)
    assert not compiled(g)


def test_python_order_when_c_writes():
    def f(x):
        a = x[0] + scales_then_reads(x)
        return a, x[0]

    def forwarded(y):
        for i in range(1, len(y)):
            k.scale(y, i, 2.0)
            y[i] = y[i - 1] + 1.0
        return y
    check(forwarded, np.arange(1.0, 5.0))
    check(f, np.arange(1.0, 4.0))


@scram
def scales_then_reads(x):
    k.scale(x, 1, 2.0)
    return x[0]


def test_comments_defines_and_static_helpers_work():
    def f(x):
        return k.uses_helper(x), defined.scaled(x)
    check(f, 2.0)


def test_a_new_function_under_the_name():
    global add
    def f(a):
        return add(a, 1.0)
    g = scram(f)
    saved = add
    try:
        assert g(1.0) == 2.0
        add = clash.add
        assert g(1.0) == 0.0
        add = lambda a, b: a * 10.0  # noqa: E731
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert g(1.0) == 10.0
    finally:
        add = saved
    assert g(1.0) == 2.0


def test_two_modules():
    def f(x):
        return k.total(np.ones(2), 2) + one.one(x)
    check(f, 1.0)


def test_two_modules_that_clash_stay_python():
    def f(x):
        return k.add(x, 1.0) + clash.add(x, 1.0)
    check(f, 1.0, must_compile=False, why="one file")


def test_data_it_can_change_stays_python():
    def f(x):
        return stateful.add_up(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = scram(f)
        assert g(1) == 1 and g(2) == 3 and stateful.add_up(0) == 3
    assert not compiled(g) and 'const' in str(next(iter(g._why.values())))


def test_contraction_as_the_module_was_compiled():
    def f(a, b, c):
        return contracted.mulsum(a, b, c), uncontracted.mulsum2(a, b, c)
    check(f, 1.0, 2.0, 3.0, must_compile=False, why='different fma=')
    a, b, c = 1.0 + 2.0 ** -30, 1.0 - 2.0 ** -30, -1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert scram(lambda_contracted)(a, b, c) == contracted.mulsum(a, b, c)
        assert scram(lambda_uncontracted)(a, b, c) == uncontracted.mulsum2(a, b, c)


def lambda_contracted(a, b, c):
    return contracted.mulsum(a, b, c)


def lambda_uncontracted(a, b, c):
    return uncontracted.mulsum2(a, b, c)


def test_math_that_must_not_contract_stays_exact():
    def f(x):
        return math.gamma(x) + contracted.mulsum(x, x, x)
    check(f, 1.5, must_compile=False, why='fma=False')

    def g(x):
        return math.gamma(x) + uncontracted.mulsum2(x, x, x)
    check(g, 1.5)


def test_returns_that_stay_python():
    def f(x):
        return k.uwide(x)

    def p(x):
        return k.pointer(x)

    def v(x):
        return k.varargs(x, 1)

    def s(x):
        return k.by_value(x)

    def missing(x):
        return k.nope(x)
    check(f, 3, must_compile=False, why='uint64_t')
    check(v, 1, must_compile=False)
    check(missing, 1, must_compile=False, why='no such function')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = scram(p)
        g(np.arange(2.0))
    assert not compiled(g)


def test_a_c_function_named_like_a_method():
    def f(x):
        return k.sum(x)
    check(f, 1.5)


def test_heap_and_errors_in_c():
    def f(x):
        return k.uses_heap(x)
    check(f, 2.0)

    def g(x):
        y = np.zeros(3)
        return k.checks(x) + y[0]
    h = scram(g)
    assert h(1.0) == 1.0
    with pytest.raises(cimo.KernelError, match='assertion failed'):
        h(-1.0)
    assert h(2.0) == 2.0


@scram
def calls_c(x):
    return k.add(x, 1.0)


@scram
def calls_a_function_that_calls_c(x):
    s = 0.0
    for i in range(3):
        s += calls_c(x)
    return s


def test_through_a_compiled_call():
    assert calls_a_function_that_calls_c(1.0) == 6.0
    assert compiled(calls_a_function_that_calls_c) and 'static int sj_call' in calls_a_function_that_calls_c.c()


def test_it_is_inlined_where_it_can_be():
    def f(x):
        s = 0.0
        for i in range(len(x)):
            s = k.add(s, x[i])
        return s
    g = scram(f)
    assert g(np.arange(3.0)) == 3.0
    assert 'double add(double a, double b)' in g.c()


def test_preflight_reads_a_literal_module():
    import ast
    from scramjit import _preflight as P
    tree = ast.parse('''
import cimo
from scramjit import scram

k = cimo.compile("""
#include <stdint.h>
double total(const double *a, int64_t n) { double s = 0; for (int64_t i = 0; i < n; i++) s += a[i]; return s; }
""")
state = cimo.compile("int bump(int x) { static int n; n += x; return n; }")

@scram
def f(x, n):
    return k.total(x, n)

@scram
def g(x):
    return state.bump(x)
''')
    g = P.module_globals(tree)
    fs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    opts = {'boundscheck': True, 'parity': True}
    a = P.best_analysis(fs['f'], g, 'x.py', 0, opts)
    assert a.state == 'compiled', [p.reason for p in a.problems]
    a = P.best_analysis(fs['g'], g, 'x.py', 0, opts)
    assert a.state == 'python' and 'const' in a.problems[0].reason
