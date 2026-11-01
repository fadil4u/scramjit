"""Every math function scramjit compiles against CPython's, bit for bit, on awkward inputs. Plain
Python, so it also runs in Pyodide (tools/run_pyodide.mjs tests/test_math_cases.py), where the libm differs."""

import math
import struct
import sys

from scramjit import scram


def f_sin(x: float): return math.sin(x)
def f_cos(x: float): return math.cos(x)
def f_tan(x: float): return math.tan(x)
def f_asin(x: float): return math.asin(x)
def f_acos(x: float): return math.acos(x)
def f_atan(x: float): return math.atan(x)
def f_asinh(x: float): return math.asinh(x)
def f_acosh(x: float): return math.acosh(x)
def f_atanh(x: float): return math.atanh(x)
def f_sinh(x: float): return math.sinh(x)
def f_cosh(x: float): return math.cosh(x)
def f_tanh(x: float): return math.tanh(x)
def f_exp(x: float): return math.exp(x)
def f_expm1(x: float): return math.expm1(x)
def f_log(x: float): return math.log(x)
def f_log2(x: float): return math.log2(x)
def f_log10(x: float): return math.log10(x)
def f_log1p(x: float): return math.log1p(x)
def f_erf(x: float): return math.erf(x)
def f_erfc(x: float): return math.erfc(x)
def f_sqrt(x: float): return math.sqrt(x)
def f_fabs(x: float): return math.fabs(x)
def f_floor(x: float): return math.floor(x)
def f_ceil(x: float): return math.ceil(x)
def f_trunc(x: float): return math.trunc(x)
def f_degrees(x: float): return math.degrees(x)
def f_radians(x: float): return math.radians(x)
def f_isnan(x: float): return math.isnan(x)
def f_isinf(x: float): return math.isinf(x)
def f_isfinite(x: float): return math.isfinite(x)
def f_log_i(n: int): return math.log(n)
def f_sqrt_i(n: int): return math.sqrt(n)
def f_atan2(x: float, y: float): return math.atan2(x, y)
def f_pow(x: float, y: float): return math.pow(x, y)
def f_fmod(x: float, y: float): return math.fmod(x, y)
def f_copysign(x: float, y: float): return math.copysign(x, y)
def f_hypot(x: float, y: float): return math.hypot(x, y)


ONE = [f for n, f in sorted(globals().items()) if n.startswith('f_') and f.__code__.co_argcount == 1 and not n.endswith('_i')]
TWO = [f for n, f in sorted(globals().items()) if n.startswith('f_') and f.__code__.co_argcount == 2]
INTS = [f_log_i, f_sqrt_i]
if sys.version_info >= (3, 11):                     # added to math in 3.11
    def f_exp2(x: float): return math.exp2(x)
    def f_cbrt(x: float): return math.cbrt(x)
    ONE += [f_exp2, f_cbrt]

X = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 2.5, -2.5, 3.0, 10.0, 0.1, -0.1, 0.9999999999999999, 1.0000000000000002,
     math.pi, -math.pi, math.pi / 2, math.e, 1e-5, -1e-5, 1e-16, 1e-300, -1e-300, 5e-324, -5e-324, 2.2250738585072014e-308,
     21.0, 30.0, 88.0, 709.0, 709.8, 710.0, -708.0, -745.0, -746.0, 1024.0, -1075.0, 1e15, 1e22, 1e300, -1e300,
     1.7976931348623157e308, -1.7976931348623157e308, math.inf, -math.inf, math.nan]
Y = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 3.0, -3.0, 2.5, 1e-300, 5e-324, 1e300, 1024.0, -1075.0, 1.7976931348623157e308, math.inf, -math.inf, math.nan]
I = [0, 1, -1, 2, 3, 10, 1 << 53, (1 << 53) + 1, (1 << 62), -(1 << 62), (1 << 63) - 1, -(1 << 63)]


def outcome(f, args):
    try:
        r = f(*args)
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, str(e)
    if isinstance(r, float):
        return 'float', struct.pack('<d', r).hex() if r == r else 'nan'  # NaN sign and payload aren't Python-visible rules
    return type(r).__name__, r


def main():
    bad, checked, compiled = [], 0, 0
    for fns, cases in ((ONE, [(x,) for x in X]), (INTS, [(i,) for i in I]), (TWO, [(x, y) for x in X for y in Y])):
        for f in fns:
            g = scram(f, fallback=False)
            compiled += 1
            for args in cases:
                checked += 1
                if f in (f_floor, f_ceil, f_trunc) and math.isfinite(args[0]) and abs(args[0]) >= 2.0 ** 63:
                    continue                        # integers are 64-bit: the documented difference
                a, b = outcome(f, args), outcome(g, args)
                if a != b:
                    bad.append((f.__name__, args, a, b))
    print(f'{sys.platform}: {compiled} functions, {checked} cases, {len(bad)} differ')
    for b in bad[:40]:
        print('  ', b)
    return bad


def test_math_matches_cpython():
    assert main() == []


if __name__ == '__main__':
    main()
