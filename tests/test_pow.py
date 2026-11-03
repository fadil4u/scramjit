"""** and pow(a, b) against CPython and numpy, bit for bit. Plain Python, so it also runs in Pyodide:
node tools/run_pyodide.mjs tests/test_pow.py"""

import math
import struct
import sys
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram


def f_square(x: float): return x ** 2
def f_cube(x: float): return x ** 3
def f_inv(x: float): return x ** -1
def f_zero(x: float): return x ** 0
def f_two_f(x: float): return x ** 2.0
def f_neg_f(x: float): return x ** -3.0
def f_big(x: float): return x ** 1000
def f_var(x: float, n: int): return x ** n
def f_base(x: float): return 2.0 ** x
def f_base_int(x: float): return 2 ** x
def f_base_zero(x: float): return 0.0 ** x
def f_inf(x: float): return x ** 1e308
def f_nested(x: float): return (x ** 2) ** 2 - x ** 2 ** 2
def f_builtin(x: float): return pow(x, 2)
def f_aug(x: float):
    x **= 2
    return x
def i_square(n: int): return n ** 2
def i_zero(n: int): return n ** 0
def i_big(n: int): return n ** 63
def i_neg(n: int): return n ** -1
def i_neg2(n: int): return n ** -2
def i_to_f(n: int): return n ** 2.0
def i_builtin(n: int): return pow(n, 3)
def i_aug(n: int):
    n **= 3
    return n
def i_bool(b: bool): return b ** 2
def a_half(a): return a[0] ** 0.5
def a_two(a): return a[0] ** 2
def a_frac_var(a, y: float): return a[0] ** y
def a_int_var(a, n: int): return a[0] ** n
def a_rpow(a, x: float): return x ** a[0]
def a_rint(a, n: int): return n ** a[0]
def a_elems(a): return a[0] ** a[1]
def a_loop(a):
    s = 0.0
    for v in a:
        s += v ** 0.5
    return s
def a_store(a):
    for i in range(len(a)):
        a[i] **= 2
    return a[0]
def n_two(a): return a[0] ** 2
def n_neg(a): return a[0] ** -1
def n_var(a, n: int): return a[0] ** n
def n_float(a): return a[0] ** 0.5
def n_elems(a): return a[0] ** a[1]
def n_rpow(a, n: int): return n ** a[0]
def n_big(a): return a[0] ** 70
def d_flag(a, x: float, c: int):
    v = x
    if c:
        v = a[0]
    return v ** 2
def d_flag_int(a, n: int, c: int):
    v = n
    if c:
        v = a[0]
    return v ** 2


# Python's result is complex for a negative base: compiled, these raise (a documented difference)
def c_half(x: float): return x ** 0.5
def c_neg_half(x: float): return x ** -0.5
def c_var(x: float, y: float): return x ** y
def c_int_base(n: int): return n ** 0.5
def c_int_base_var(n: int, y: float): return n ** y
def c_builtin(x: float): return pow(x, 0.5)
def c_aug(x: float):
    x **= 0.5
    return x
def c_flag(a, x: float, c: int):
    v = x
    if c:
        v = a[0]
    return v ** 0.5
def c_const_base(y: float): return 2.0 ** y


# must fall back
def i_var(n: int, k: int): return n ** k
def i_var_aug(n: int, k: int):
    n **= k
    return n
def i_builtin_var(n: int, k: int): return pow(n, k)
def i_true_base(b: bool, k: int): return b ** k
def fb_three(x: float): return pow(x, 2, 3)


X = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 10.0, 1e-5, 1e-160, -1e-160, 1e-200, 1e154, 1e155, 1e300, -1e300, 5e-324, -5e-324,
     1.7976931348623157e308, math.pi, -1074.0, -1075.0, 1024.0, 1023.5, math.inf, -math.inf, math.nan]
I = [0, 1, -1, 2, -2, 3, 7, -7, 10, 1 << 31, 1 << 53, -(1 << 53) - 1, (1 << 63) - 1, -(1 << 63)]
AF = [np.array([x, y]) for x in X for y in (2.0, 0.5, -1.0)]
AI = [np.array([i, j], np.int64) for i in I for j in (0, 1, 2, 3, -1, 70)]
wrap = lambda v: ((v + (1 << 63)) % (1 << 64)) - (1 << 63)  # noqa: E731


def norm(r):
    if isinstance(r, complex) and not isinstance(r, np.generic):
        return 'complex'
    if isinstance(r, (float, np.floating)):
        return (type(r).__name__, 'nan' if r != r else struct.pack('<d', float(r)).hex())
    if isinstance(r, (bool, np.bool_)):
        return (type(r).__name__, bool(r))
    return (type(r).__name__, wrap(int(r)))


def outcome(f, args):
    args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    try:
        r = norm(f(*args))
    except Exception as e:  # noqa: BLE001
        return (type(e).__name__, str(e))
    return (r, [a.tobytes() for a in args if isinstance(a, np.ndarray)])


CASES = [(f, [(x,) for x in X]) for f in (f_square, f_cube, f_inv, f_zero, f_two_f, f_neg_f, f_big, f_base, f_base_int, f_base_zero, f_inf, f_nested, f_builtin, f_aug)]
CASES += [(f_var, [(x, n) for x in X for n in I])]
CASES += [(f, [(i,) for i in I]) for f in (i_square, i_zero, i_big, i_neg, i_neg2, i_to_f, i_builtin, i_aug)]
CASES += [(i_bool, [(True,), (False,)])]
NONNEG = [0, 1, 2, 3, 7, 10, 63, 64, 70, 200]             # (Python computes these exactly, so not huge)
CASES += [(f, [(n, k) for n in I for k in NONNEG]) for f in (i_var, i_var_aug, i_builtin_var)]
CASES += [(i_true_base, [(b, k) for b in (True, False) for k in NONNEG])]
CASES += [(f, [(a,) for a in AF]) for f in (a_half, a_two, a_elems, a_loop, a_store)]
CASES += [(a_frac_var, [(a, y) for a in AF[::3] for y in X]), (a_int_var, [(a, n) for a in AF[::3] for n in I]), (a_rpow, [(a, x) for a in AF for x in X[::4]]), (a_rint, [(a, n) for a in AF for n in I[::3]])]
CASES += [(f, [(a,) for a in AI]) for f in (n_two, n_neg, n_float, n_elems, n_big)]
CASES += [(n_var, [(a, n) for a in AI[::6] for n in I]), (n_rpow, [(a, n) for a in AI for n in I[::3]])]
CASES += [(d_flag, [(a, x, c) for a in AF[::9] for x in X for c in (0, 1)]), (d_flag_int, [(a, n, c) for a in AI[::6] for n in I for c in (0, 1)])]
FALLS_BACK = [(fb_three, (2.0,))]
Y = X + [0.25, -0.25, 1.5, -1.5, 1e-300, 0.1]
CASES += [(f, [(x,) for x in X]) for f in (c_half, c_neg_half, c_builtin, c_aug, c_const_base)]
CASES += [(c_var, [(x, y) for x in X for y in Y]), (c_int_base, [(i,) for i in I]), (c_int_base_var, [(i, y) for i in I for y in Y])]
CASES += [(c_flag, [(a, x, c) for a in AF[::9] for x in X for c in (0, 1)])]


def main():
    np.seterr(all='ignore')
    warnings.simplefilter('ignore', RuntimeWarning)
    bad, checked, raised = [], 0, 0
    for f, cases in CASES:
        g = scram(f, fallback=False)
        for args in cases:
            checked += 1
            a, b = outcome(f, args), outcome(g, args)
            # the documented difference: an error, never a wrong answer (including where Python's complex overflows),
            # unless the function writes nothing and the call ran as Python
            if a == b:
                continue
            if a[0] == 'complex' or a == ('OverflowError', 'complex exponentiation'):
                if b[0] == 'ValueError' and 'gives a complex' in b[1]:
                    raised += 1
                else:
                    bad.append((f.__name__, args, a, b))
            elif a != b:
                bad.append((f.__name__, args, a, b))
    wrong = []
    for f, args in FALLS_BACK:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', FallbackWarning)
            if outcome(scram(f), args) != outcome(f, args) or not w:
                wrong.append(f.__name__)
    print(f'{sys.platform} {sys.version.split()[0]}: {len(CASES)} functions, {checked} cases, {len(bad)} differ, {raised} complex results raised; fall back wrongly: {wrong}')
    seen = {}
    for b in bad:
        seen.setdefault(b[0], []).append(b)
    for name, v in seen.items():
        print(f'  {name}: {len(v)}', str(v[0][1:])[:200])
    return bad + wrong


def test_pow_matches_cpython():
    assert main() == []


def test_negative_exponent_reruns_as_python():
    """A function that writes nothing runs as Python where ** would differ: Python's float, or its error."""
    for f in (i_var, i_var_aug, i_builtin_var):
        g = scram(f, fallback=False)
        for n, k in ((2, -1), (0, -1), (-3, -2), (1, -(1 << 62)), (2, 3)):
            assert outcome(g, [n, k]) == outcome(f, [n, k]), (f.__name__, n, k)


def i_var_writes(n: int, k: int, out):
    out[0] = 1.0
    return n ** k


def f_root_writes(x: float, out):
    out[0] = 1.0
    return x ** 0.5


def f_root(x: float):
    return x ** 0.5


def test_negative_exponent_is_the_documented_difference():
    """A function that has written something can't start again: an error, never a wrong answer."""
    g = scram(i_var_writes, fallback=False)
    for n, k in ((2, -1), (0, -1), (-3, -2)):
        try:
            g(n, k, np.zeros(1))
        except ValueError as e:
            assert 'negative k' in str(e) and 'float(n) ** k' in str(e)
        else:
            raise AssertionError((n, k))
    with pytest.raises(ValueError, match='complex'):
        scram(f_root_writes, fallback=False)(-4.0, np.zeros(1))
    for x in (-4.0, 4.0, -0.0):
        assert outcome(scram(f_root, fallback=False), [x]) == outcome(f_root, [x])


def test_the_rewrite_in_the_message_works():
    g = scram(_float_base, fallback=False)
    for n, k in ((2, -1), (2, 3), (-3, -2)):
        assert g(n, k) == _float_base(n, k)


def _float_base(n: int, k: int):
    return float(n) ** k


if __name__ == '__main__':
    main()
