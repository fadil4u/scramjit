"""math's functions CPython implements itself, against CPython, bit for bit. Plain Python, so it
also runs in Pyodide: node tools/run_pyodide.mjs tests/test_math_ported.py (ints wrap at 64 bits,
the documented difference)."""

import math
import struct
import sys

import numpy as np

from scramjit import scram


def gamma(x: float): return math.gamma(x)
def lgamma(x: float): return math.lgamma(x)
def ulp(x: float): return math.ulp(x)
def frexp(x: float): return math.frexp(x)
def frexp2(x: float):
    m, e = math.frexp(x)
    return m + e
def modf(x: float): return math.modf(x)
def remainder(x: float, y: float): return math.remainder(x, y)
def nextafter(x: float, y: float): return math.nextafter(x, y)
def isclose(x: float, y: float): return math.isclose(x, y)
def isclose2(x: float, y: float): return math.isclose(x, y, rel_tol=0.25, abs_tol=0.5)
def isclose3(x: float, y: float): return math.isclose(x, y, rel_tol=y, abs_tol=x)
def ldexp(x: float, i: int): return math.ldexp(x, i)
def isqrt(n: int): return math.isqrt(n)
def factorial(n: int): return math.factorial(n)
def gcd2(a: int, b: int): return math.gcd(a, b)
def gcd3(a: int, b: int): return math.gcd(a, b, 12)
def lcm2(a: int, b: int): return math.lcm(a, b)
def lcm3(a: int, b: int): return math.lcm(a, b, 3)
def comb(n: int, k: int): return math.comb(n, k)
def perm(n: int, k: int): return math.perm(n, k)
def perm1(n: int): return math.perm(n)
def fsum(a): return math.fsum(a)
def fsum_gen(a): return math.fsum(x * x for x in a)
def prod_f(a): return math.prod(a, start=1.0)
def prod_i(a): return math.prod(a)
def prod_gen(a): return math.prod((x + 1 for x in a), start=2.0)
def dist(a, b): return math.dist(a, b)
def fsum_lit(x: float, y: float): return math.fsum([x, y, 1.0])
def fsum_tup(x: float, y: float): return math.fsum((x, y, -x))
def prod_tup(x: float, y: float): return math.prod((x, y), start=2.0)
def dist_tup(x: float, y: float): return math.dist((x, y), (y, 1.0))
def dist_var(x: float, y: float):
    p = (x, y)
    q = (1, y)
    return math.dist(p, q)
def dist_len(x: float): return math.dist((x, 1.0), (x,))


X = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.0, 2.5, -2.5, 3.0, -3.0, 3.5, 5.0, 10.0, 0.1, -0.1, math.pi, 20.5, 22.0, 23.0, 25.5, 100.5, 140.5, 170.0, 171.0,
     171.5, 172.0, 200.5, -170.5, -180.5, -200.5, 1e-5, 1e-16, 1e-21, -1e-21, 1e-300, 1e-310, 5e-324, -5e-324, 1e100, 1e300, -1e300, 2.0 ** 52 + 1, 2.0 ** 53,
     1.7976931348623157e308, -1.7976931348623157e308, 2.2250738585072014e-308, math.inf, -math.inf, math.nan]
XY = [(x, y) for x in X for y in [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 3.0, 1e-300, 5e-324, 1e300, 1.7976931348623157e308, math.inf, -math.inf, math.nan]]
I = [0, 1, 2, 3, 4, 7, 10, 12, 20, 21, 25, 60, 66, 67, 100, -1, -2, -12, 1 << 31, (1 << 31) - 1, 1 << 53, (1 << 62) + 3, (1 << 63) - 1, -(1 << 62)]
wrap = lambda v: ((v + (1 << 63)) % (1 << 64)) - (1 << 63)  # noqa: E731


def norm(r):
    if isinstance(r, tuple):
        return tuple(norm(v) for v in r)
    if isinstance(r, (float, np.floating)):
        return (type(r).__name__, 'nan' if r != r else struct.pack('<d', float(r)).hex())
    if isinstance(r, (bool, np.bool_)):
        return (type(r).__name__, bool(r))
    return (type(r).__name__, wrap(int(r)))


def outcome(f, args):
    try:
        return norm(f(*args))
    except Exception as e:  # noqa: BLE001
        return (type(e).__name__, str(e))


CASES = [(f, [(x,) for x in X]) for f in (gamma, lgamma, ulp, frexp, frexp2, modf)]
CASES += [(f, XY) for f in (remainder, nextafter, isclose, isclose2, isclose3)]
CASES += [(ldexp, [(x, i) for x in X for i in (0, 1, -1, 52, 1023, 1024, -1074, -1075, 2000, -2000, 1 << 40, -(1 << 40))])]
CASES += [(isqrt, [(i,) for i in I]), (perm1, [(i,) for i in I if i < 1 << 20])]
CASES += [(factorial, [(i,) for i in I if i < 1 << 16])]
CASES += [(f, [(a, b) for a in I for b in I]) for f in (gcd2, gcd3, lcm2, lcm3)]
SMALL = [i for i in I if abs(i) < 1 << 16] + [1000, 9999]
CASES += [(f, [(a, b) for a in SMALL for b in SMALL]) for f in (comb, perm)]
ARR = [np.array(v, float) for v in ([], [1.0], [1e100, 1.0, -1e100, 1e-30], [0.1] * 10, [1e308, 1e308, -1e308], [math.inf, 1.0], [math.inf, -math.inf], [math.nan, 1.0], [-0.0], [-0.0, 0.0],
                                    [1e-16, 1.0, 1e16], [2.0, 3.5, -1.25])]
CASES += [(f, [(a,) for a in ARR]) for f in (fsum, fsum_gen, prod_f, prod_gen)]
CASES += [(prod_i, [(np.array(v, np.int64),) for v in ([], [3], [2, 3, 4], [1 << 40, 1 << 40])])]
CASES += [(f, XY[::3]) for f in (fsum_lit, fsum_tup, prod_tup, dist_tup, dist_var)] + [(dist_len, [(1.0,)])]
CASES += [(dist, [(a, b) for a in ARR for b in ARR] + [((1.0, 2.0), (4.0, 6.0)), ((1, 2), (3, 7)), ((1.0,), (2.0, 3.0))])]
if sys.version_info >= (3, 12):
    def nextafter3(x: float, y: float, s: int): return math.nextafter(x, y, steps=s)
    def sumprod(a, b): return math.sumprod(a, b)
    CASES += [(nextafter3, [(x, y, s) for x, y in XY for s in (0, 1, 2, 1 << 40, -1)])]
    CASES += [(sumprod, [((1, 2), (3, 4)), (np.array([1, 2], np.int64), np.array([3, 4], np.int64)), (np.array([1], np.int64), np.array([1, 2], np.int64)), (np.array([], np.int64), np.array([], np.int64))])]
if sys.version_info >= (3, 13):
    def fma(x: float, y: float, z: float): return math.fma(x, y, z)
    CASES += [(fma, [(x, y, z) for x, y in XY[::5] for z in (0.0, -1.0, 1e300, math.inf, -math.inf, math.nan)])]


def main():
    np.seterr(all='ignore')                             # numpy warns; compiled code doesn't, and neither result changes
    bad, checked = [], 0
    for f, cases in CASES:
        g = scram(f, fallback=False)
        for args in cases:
            checked += 1
            if f in (lcm2, lcm3):                       # past 64 bits, wrapped steps don't compose
                try:
                    if abs(f(*args)) >= 1 << 63:
                        continue
                except Exception:  # noqa: BLE001, S110
                    pass
            a, b = outcome(f, args), outcome(g, args)
            if a != b:
                bad.append((f.__name__, args, a, b))
    print(f'{sys.platform} {sys.version.split()[0]}: {len(CASES)} functions, {checked} cases, {len(bad)} differ')
    seen = {}
    for b in bad:
        seen.setdefault(b[0], []).append(b)
    for name, v in seen.items():
        print(f'  {name}: {len(v)}', str(v[0][1:])[:170])
    return bad


def test_ported_math_matches_cpython():
    assert main() == []


if __name__ == '__main__':
    main()
