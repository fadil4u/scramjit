"""Check the hand-written C in kernels/c/ against the Python kernels, and time both.

For each kernel: run the Python and the C (compiled with cimo) on the same random inputs and
require identical results, bit for bit; provoke an error in both and require the same
exception, at the same Python line, after the same writes; then time pure CPython against the
C. Run from the repository root:

    python3 kernels/check.py            # with cimo's defaults
    python3 kernels/check.py --fma both # also with fused multiply-add contraction off

cimo is imported from the environment, or from ../cimo/python next to this repository.
Needs numpy (a development dependency only; scramjit itself doesn't).
"""

import argparse
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import flood_fill as ff  # noqa: E402
import gray_scott as gs  # noqa: E402
import viewshed as vs  # noqa: E402

ERRORS = {1: IndexError, 2: ZeroDivisionError, 3: OverflowError}


def import_cimo():
    try:
        import cimo
    except ImportError:
        sys.path.insert(0, os.path.join(HERE, '..', '..', 'cimo', 'python'))
        import cimo
    return cimo


def compile_c(cimo, name):
    t = time.perf_counter()
    m = cimo.load(os.path.join(HERE, 'c', name + '.c'))
    return getattr(m, name), time.perf_counter() - t


def best(f, repeat=5):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def python_error(f):
    """Run f; return (exception type, line in the kernel's file) or None."""
    try:
        f()
    except Exception as e:  # noqa: BLE001
        frames = [fr for fr in traceback.extract_tb(e.__traceback__)
                  if os.path.dirname(os.path.abspath(fr.filename)) == HERE and fr.name != '<lambda>']
        return type(e), frames[-1].lineno
    return None


def c_error(status, line):
    return None if status == 0 else (ERRORS[status], int(line[0]))


def report(name, same, err_same, t_py, t_c, t_compile):
    ok = 'identical' if same else 'DIFFERENT'
    eok = 'same error' if err_same else 'ERROR MISMATCH'
    print(f'{name:11s} {ok:9s}  {eok:14s}  python {t_py * 1e3:9.1f} ms   c {t_c * 1e3:7.3f} ms'
          f'   {t_py / t_c:7.0f}x   (compile {t_compile:.2f} s)')
    return same and err_same


def check_gray_scott(cimo, rng):
    k, t_compile = compile_c(cimo, 'gray_scott')
    n = 96
    u = rng.random((n, n))
    v = rng.random((n, n))
    params = (0.16, 0.08, 0.035, 0.065, 1.0)

    def run_py(u2, v2):
        gs.gray_scott(u, v, u2, v2, *params)

    def run_c(u2, v2, line):
        return k(u, *u.shape, v, *v.shape, u2, *u2.shape, v2, *v2.shape, *params, line)

    pu, pv = np.zeros_like(u), np.zeros_like(v)
    cu, cv = np.zeros_like(u), np.zeros_like(v)
    line = np.zeros(1, np.int64)
    t_py = best(lambda: run_py(pu, pv), 1)
    st = run_c(cu, cv, line)
    same = st == 0 and np.array_equal(pu, cu) and np.array_equal(pv, cv)
    t_c = best(lambda: run_c(cu, cv, line))

    # error: v2 one row short; Python raises at line 20 after writing everything before it
    pu, pv = np.zeros_like(u), np.zeros((n - 1, n))
    cu, cv = np.zeros_like(u), np.zeros((n - 1, n))
    pe = python_error(lambda: run_py(pu, pv))
    ce = c_error(run_c(cu, cv, line), line)
    err_same = pe == ce and np.array_equal(pu, cu) and np.array_equal(pv, cv)
    return report('gray_scott', same, err_same, t_py, t_c, t_compile)


def check_flood_fill(cimo, rng):
    k, t_compile = compile_c(cimo, 'flood_fill')
    n = 256
    grid = (rng.random((n, n)) < 0.62).astype(np.uint8)   # large connected regions of 1
    y0, x0 = int(np.argmax(grid) // n), int(np.argmax(grid) % n)

    def run_c(g, stack, y, x, new, line, ret):
        return k(g, *g.shape, stack, *stack.shape, y, x, new, ret, line)

    pg, cg = grid.copy(), grid.copy()
    line, ret = np.zeros(1, np.int64), np.zeros(1, np.int64)
    pn = None

    def run_py():
        nonlocal pn
        pn = ff.flood_fill(pg, np.zeros(n * n, np.int64), y0, x0, 7)
    t_py = best(run_py, 1)
    st = run_c(cg, np.zeros(n * n, np.int64), y0, x0, 7, line, ret)
    same = st == 0 and pn == ret[0] and np.array_equal(pg, cg)
    lg = grid.copy()
    assert ff.flood_fill_list(lg, y0, x0, 7) == pn and np.array_equal(lg, pg)

    def timed():
        g = grid.copy()
        run_c(g, np.zeros(n * n, np.int64), y0, x0, 7, line, ret)
    t_c = best(timed)

    # errors: a stack too small (IndexError mid-fill), and a value that doesn't fit uint8
    pg, cg = grid.copy(), grid.copy()
    pe = python_error(lambda: ff.flood_fill(pg, np.zeros(50, np.int64), y0, x0, 7))
    ce = c_error(run_c(cg, np.zeros(50, np.int64), y0, x0, 7, line, ret), line)
    err_same = pe == ce and np.array_equal(pg, cg)
    pe = python_error(lambda: ff.flood_fill(grid.copy(), np.zeros(9, np.int64), y0, x0, 300))
    ce = c_error(run_c(grid.copy(), np.zeros(9, np.int64), y0, x0, 300, line, ret), line)
    err_same = err_same and pe == ce
    return report('flood_fill', same, err_same, t_py, t_c, t_compile)


def check_viewshed(cimo, rng):
    k, t_compile = compile_c(cimo, 'viewshed')
    n = 64
    yy, xx = np.mgrid[0:n, 0:n] / n
    dem = np.sin(6 * xx) * np.cos(5 * yy) * 40 + rng.random((n, n)) * 3
    oy, ox, oh = n // 3, n // 2, 2.0

    pv, cv = np.zeros((n, n), np.uint8), np.zeros((n, n), np.uint8)
    line = np.zeros(1, np.int64)
    t_py = best(lambda: vs.viewshed(dem, pv, oy, ox, oh), 1)
    st = k(dem, *dem.shape, cv, *cv.shape, oy, ox, oh, line)
    same = st == 0 and np.array_equal(pv, cv)
    if st == 0 and not same:
        print(f'  viewshed: {int((pv != cv).sum())} of {pv.size} cells differ')
    t_c = best(lambda: k(dem, *dem.shape, cv, *cv.shape, oy, ox, oh, line))

    # error: vis one column short; Python raises at line 30 (or 20) partway through
    pv, cv = np.zeros((n, n - 1), np.uint8), np.zeros((n, n - 1), np.uint8)
    pe = python_error(lambda: vs.viewshed(dem, pv, oy, ox, oh))
    ce = c_error(k(dem, *dem.shape, cv, *cv.shape, oy, ox, oh, line), line)
    err_same = pe == ce and np.array_equal(pv, cv)
    return report('viewshed', same, err_same, t_py, t_c, t_compile)


MATH_C = r"""
#include <stdint.h>
#include <math.h>
void run(int64_t which, const double *x, const double *y, double *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        double a = x[i], b = y[i], r;
        switch (which) {
        case 0: r = hypot(a, b); break;
        case 1: r = sqrt(fabs(a)); break;
        case 2: r = exp(a / 20); break;
        case 3: r = log(fabs(a) + 1e-3); break;
        case 4: r = sin(a); break;
        case 5: r = cos(a); break;
        case 6: r = atan2(a, b); break;
        default: r = pow(fabs(a), b / 50); break;
        }
        out[i] = r;
    }
}
"""


def math_agreement(cimo, rng, count=100_000):
    """How often CPython's math module differs from the C library compiled code calls (cimo's)."""
    import math
    run = cimo.compile(MATH_C).run
    xs = rng.standard_normal(count) * 50
    ys = rng.standard_normal(count) * 50
    pyfs = [math.hypot, lambda a, b: math.sqrt(abs(a)), lambda a, b: math.exp(a / 20),
            lambda a, b: math.log(abs(a) + 1e-3), lambda a, b: math.sin(a), lambda a, b: math.cos(a),
            math.atan2, lambda a, b: math.pow(abs(a), b / 50)]
    names = ['hypot', 'sqrt', 'exp', 'log', 'sin', 'cos', 'atan2', 'pow']
    out = np.empty(count)
    xl, yl = xs.tolist(), ys.tolist()
    parts = []
    for which, (name, f) in enumerate(zip(names, pyfs)):
        run(which, xs, ys, out, count)
        diff = sum(f(a, b) != c for a, b, c in zip(xl, yl, out.tolist()))
        parts.append(f'{name} {diff}')
    print(f'CPython math vs cimo libm, differing results of {count}: ' + ', '.join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fma', choices=['default', 'off', 'both'], default='default')
    args = ap.parse_args()
    if args.fma == 'both':
        rc = 0
        for label, env in (('FMA contraction on (cimo default)', '1'), ('FMA contraction off', '0')):
            print(f'== {label}')
            rc |= subprocess.call([sys.executable, __file__], env={**os.environ, 'CIMO_FMA': env})
        sys.exit(rc)
    if args.fma == 'off':
        os.environ['CIMO_FMA'] = '0'
    cimo = import_cimo()
    rng = np.random.default_rng(1)
    ok = all([check_gray_scott(cimo, rng), check_flood_fill(cimo, rng), check_viewshed(cimo, rng)])
    math_agreement(cimo, rng)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
