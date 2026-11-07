"""scramjit against Numba, Cython and gcc on the same kernels: steady-state time, first-call time, install size.

Every column runs the same algorithm. scramjit and Numba compile the plain Python in compare_kernels.py (and kernels/) as
written; Cython gets typed versions (compare_cy.pyx: memoryviews, no bounds checks) for the array kernels and compiles the
Python-data kernels as written; "gcc" is hand-written C (kernels/c/ and below) through gcc -O3. Floats keep Python's
results unless the column says otherwise: gcc and Cython with -ffp-contract=off, Numba's default and scramjit's parity.
Every result is checked against CPython's: "≈" marks a column allowed to differ in the last bits (parity=False,
fastmath) that did, "!" one that differs and shouldn't.

Run from the repository root, after pip install -e '.[bench]' (numba and cython: never dependencies of scramjit; a
missing one is a column of "-"):  python bench/compare.py
"""

import ctypes
import copy
import importlib.util
import math
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix='scramjit-compare-')
os.environ['CIMO_CACHE_DIR'] = os.path.join(TMP, 'cimo-cache')     # first calls compile from nothing, as on a new machine
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'kernels'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'cimo', 'python'))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import compare_kernels as K  # noqa: E402
import flood_fill as ff  # noqa: E402
import gray_scott as gs  # noqa: E402
import viewshed as vs  # noqa: E402
from scramjit import scram  # noqa: E402

NUMBA_NOTE = ''
try:
    import llvmlite.binding as _llvm
    _llvm.initialize_native_target()
    if _llvm.get_host_cpu_name() == 'oryon-1' and 'NUMBA_CPU_NAME' not in os.environ:
        # LLVM's scheduling model for this CPU is incomplete and crashes it ("incomplete machine model")
        os.environ['NUMBA_CPU_NAME'] = 'generic'
        os.environ['NUMBA_CPU_FEATURES'] = _llvm.get_host_cpu_features().flatten()
        NUMBA_NOTE = ' (for a generic arm64 CPU with this one\'s features: LLVM crashes tuning for oryon-1)'
    import numba
except ImportError:
    numba = None
try:
    import Cython  # noqa: F401
    HAVE_CYTHON = shutil.which('cython') is not None or importlib.util.find_spec('Cython.Compiler.Main') is not None
except ImportError:
    HAVE_CYTHON = False
HAVE_GCC = shutil.which('gcc') is not None

COLUMNS = ['CPython', 'scramjit', 'scramjit parity=False', 'Numba', 'Numba fastmath', 'Cython', 'gcc -O3']
LOOSE = {'scramjit parity=False', 'Numba fastmath'}            # allowed to round differently

HAND_C = r'''
#include <stdint.h>
double dot(const double *x, const double *y, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}
void axpy(double a, const double *x, double *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] += a * x[i];
}
void histogram(const double *x, int64_t n, int64_t *counts, int64_t nb, double lo, double hi) {
    double scale = nb / (hi - lo);
    for (int64_t i = 0; i < n; i++) {
        int64_t k = (int64_t)((x[i] - lo) * scale);
        if (0 <= k && k < nb) counts[k]++;
    }
}
'''


# --- building -------------------------------------------------------------------------------------------------------

def gcc(name, source, *flags):
    src, lib = os.path.join(TMP, name + '.c'), os.path.join(TMP, name + '.so')
    with open(src, 'w') as f:
        f.write(source)
    subprocess.run(['gcc', '-O3', '-ffp-contract=off', '-fPIC', '-shared', src, '-o', lib, '-lm', *flags], check=True)
    return lib


def cython_module(name, path):
    """path (.pyx or .py) through cython and gcc -O3: (the module, seconds it took)."""
    t = time.perf_counter()
    c = os.path.join(TMP, name + '.c')
    subprocess.run([sys.executable, '-m', 'cython', '-3', '--module-name', name, path, '-o', c], check=True,
                   stdout=subprocess.DEVNULL)
    so = os.path.join(TMP, name + sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['gcc', '-O3', '-ffp-contract=off', '-fPIC', '-shared', c, '-o', so,
                    '-I', sysconfig.get_paths()['include']], check=True)
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, time.perf_counter() - t


def ptr(a):
    return ctypes.c_void_p(a.ctypes.data)


def hand(lib, name, argtypes, restype=ctypes.c_int):
    f = getattr(ctypes.CDLL(lib), name)
    f.argtypes, f.restype = argtypes, restype
    return f


# --- the cases ------------------------------------------------------------------------------------------------------

class Case:
    """name; fn, the Python; make() -> fresh arguments; out(args, result) -> what's compared; cy(mod) and c() -> callables
    taking the same arguments, or None."""

    def __init__(self, name, fn, make, out, cy=None, c=None, reset=None):
        self.name, self.fn, self.make, self.out, self.cy, self.c, self.reset = name, fn, make, out, cy, c, reset


def cases(rng, cy, cy_plain, lib, kdir_libs):
    line, ret = np.zeros(1, np.int64), np.zeros(1, np.int64)
    I64, F64, P = ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
    out = []

    n = 256
    u0, v0 = rng.random((n, n)), rng.random((n, n))
    p = (0.16, 0.08, 0.035, 0.065, 1.0)

    def gs_c():
        h = hand(kdir_libs['gray_scott'], 'gray_scott', [P, I64, I64] * 4 + [F64] * 5 + [P])
        return lambda u, v, u2, v2, *q: h(ptr(u), *u.shape, ptr(v), *v.shape, ptr(u2), *u2.shape, ptr(v2), *v2.shape,
                                          *q, ptr(line))
    out.append(Case('gray_scott 256²', gs.gray_scott,
                    lambda: (u0.copy(), v0.copy(), np.zeros_like(u0), np.zeros_like(v0), *p),
                    lambda a, r: (a[2].copy(), a[3].copy()), cy and (lambda: cy.gray_scott), gs_c))

    nf = 512
    grid0 = (rng.random((nf, nf)) < 0.8).astype(np.uint8)
    y0, x0 = divmod(int(np.argmax(grid0)), nf)

    def ff_c():
        h = hand(kdir_libs['flood_fill'], 'flood_fill', [P, I64, I64, P, I64, I64, I64, I64, P, P])

        def call(g, stack, y, x, new):
            h(ptr(g), *g.shape, ptr(stack), len(stack), y, x, new, ptr(ret), ptr(line))
            return int(ret[0])
        return call
    out.append(Case('flood_fill 512²', ff.flood_fill, lambda: (grid0.copy(), np.zeros(nf * nf, np.int64), y0, x0, 7),
                    lambda a, r: (a[0].copy(), r), cy and (lambda: cy.flood_fill), ff_c,
                    reset=lambda a: a[0].__setitem__(Ellipsis, grid0)))

    nv = 96
    yy, xx = np.mgrid[0:nv, 0:nv] / nv
    dem = np.sin(6 * xx) * np.cos(5 * yy) * 40 + rng.random((nv, nv)) * 3

    def vs_c():
        h = hand(kdir_libs['viewshed'], 'viewshed', [P, I64, I64, P, I64, I64, I64, I64, F64, P])
        return lambda d, vis, oy, ox, oh: h(ptr(d), *d.shape, ptr(vis), *vis.shape, oy, ox, oh, ptr(line))
    out.append(Case('viewshed 96²', vs.viewshed, lambda: (dem, np.zeros((nv, nv), np.uint8), nv // 3, nv // 2, 2.0),
                    lambda a, r: a[1].copy(), cy and (lambda: cy.viewshed), vs_c))

    x, y = rng.random(1_000_000), rng.random(1_000_000)

    def dot_c():
        h = hand(lib, 'dot', [P, P, I64], F64)
        return lambda a, b: h(ptr(a), ptr(b), len(a))
    out.append(Case('dot 1M', K.dot, lambda: (x, y), lambda a, r: r, cy and (lambda: cy.dot), dot_c))

    def axpy_c():
        h = hand(lib, 'axpy', [F64, P, P, I64], None)
        return lambda k, a, b: h(k, ptr(a), ptr(b), len(a))
    out.append(Case('axpy 1M', K.axpy, lambda: (1.5, x, y.copy()), lambda a, r: a[2].copy(), cy and (lambda: cy.axpy),
                    axpy_c))

    def hist_c():
        h = hand(lib, 'histogram', [P, I64, P, I64, F64, F64], None)
        return lambda a, counts, lo, hi: h(ptr(a), len(a), ptr(counts), len(counts), lo, hi)
    out.append(Case('histogram 1M', K.histogram, lambda: (x, np.zeros(64, np.int64), 0.0, 1.0),
                    lambda a, r: a[1].copy(), cy and (lambda: cy.histogram), hist_c))

    settings = {'name': 'flu', 'beta': 0.3, 'gamma': 0.1, 'seed': 1e-4, 'days': 400, 'substeps': 2000, 'dt': 0.0005}
    out.append(Case('dict of settings', K.epidemic,
                    lambda: (settings, {'peak': 0.0, 'peak_day': -1, 'recovered': 0.0}, np.zeros(400)),
                    lambda a, r: (dict(a[1]), a[2].copy()), cy_plain and (lambda: cy_plain.epidemic)))

    ps0 = [K.Particle(math.cos(k), math.sin(k * 1.3), mass=1.0 + k % 3) for k in range(100)]
    out.append(Case('list of dataclasses', K.particles, lambda: (copy.deepcopy(ps0), 0.001, 40),
                    lambda a, r: [(q.x, q.y, q.vx, q.vy) for q in a[0]], cy_plain and (lambda: cy_plain.particles)))
    return out


# --- measuring ------------------------------------------------------------------------------------------------------

def best(f, args, reset, repeat):
    ts = []
    for _ in range(repeat):
        if reset:
            reset(args)
        t = time.perf_counter()
        f(*args)
        ts.append(time.perf_counter() - t)
    return min(ts)


def same(a, b):
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return a == b or (a != a and b != b)
    return a == b


def close(a, b):
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-12)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def implementations(case):
    """column -> (a callable, or the reason there's none)."""
    impl = {'CPython': case.fn}

    def sj(**kw):
        return scram(case.fn, fallback=False, cache=False, **kw)
    impl['scramjit'] = sj
    impl['scramjit parity=False'] = lambda: sj(parity=False)
    if numba is None:
        impl['Numba'] = impl['Numba fastmath'] = 'not installed'
    else:
        impl['Numba'] = lambda: numba.njit(cache=False)(case.fn)
        impl['Numba fastmath'] = lambda: numba.njit(cache=False, fastmath=True)(case.fn)
    impl['Cython'] = case.cy or ('not installed' if not HAVE_CYTHON else 'no version')
    impl['gcc -O3'] = case.c if (case.c and HAVE_GCC) else '-'
    return impl


def warm_up():
    """Each tool's one-off start-up, outside every kernel's first call: cimo's runtime library, LLVM."""
    def tiny(a):
        s = 0.0
        for i in range(len(a)):
            s += math.sqrt(a[i])
        return s
    a = np.ones(4)
    scram(tiny, cache=False)(a)
    scram(tiny, cache=False, parity=False)(a)
    if numba is not None:
        numba.njit(tiny)(a)
        numba.njit(tiny, fastmath=True)(a)


def measure(case):
    """column -> (steady seconds, first-call seconds, mark) or a reason string."""
    expected_args = case.make()
    expected = case.out(expected_args, case.fn(*expected_args))
    row = {}
    for col, make in implementations(case).items():
        if isinstance(make, str):
            row[col] = make
            continue
        try:
            t = time.perf_counter()
            f = make() if col != 'CPython' else make
            args = case.make()
            r = f(*args)
            first = time.perf_counter() - t
        except Exception as e:                                     # a tool that can't compile this kernel
            row[col] = "doesn't compile: " + (str(e).strip().splitlines() or [type(e).__name__])[0][:90]
            continue
        got = case.out(args, r)
        mark = '' if same(expected, got) else ('≈' if col in LOOSE and close(expected, got) else '!')
        steady = best(f, case.make(), case.reset, 1 if col == 'CPython' else 7)
        row[col] = (steady, first, mark)
    return row


def call_overhead(cy):
    """name -> column -> nanoseconds a call, best of 7 runs of 200,000 calls (Cython's typed module)."""
    x = np.ones(8)
    rows = {}
    for name, fn, args in (('a + b (two floats)', K.add, (1.5, 2.25)), ('x[0] (a numpy array)', K.first, (x,))):
        impl = {'CPython': fn, 'scramjit': scram(fn, fallback=False, cache=False),
                'scramjit parity=False': scram(fn, fallback=False, cache=False, parity=False)}
        if numba is not None:
            impl['Numba'] = numba.njit(cache=False)(fn)
        if cy is not None:
            impl['Cython'] = getattr(cy, fn.__name__)
        row = {}
        for col, f in impl.items():
            f(*args)
            n = 200_000
            ts = []
            for _ in range(7):
                t = time.perf_counter()
                for _ in range(n):
                    f(*args)
                ts.append(time.perf_counter() - t)
            row[col] = min(ts) / n * 1e9
        rows[name] = row
    return rows


# --- sizes ----------------------------------------------------------------------------------------------------------

def size_mb(*packages):
    total = 0
    for p in packages:
        spec = importlib.util.find_spec(p)
        if spec is None or not spec.submodule_search_locations:
            return None
        for d in spec.submodule_search_locations:
            for root, _, files in os.walk(d):
                total += sum(os.path.getsize(os.path.join(root, f)) for f in files if '__pycache__' not in root)
    return total / 1e6


# --- the report -----------------------------------------------------------------------------------------------------

def fmt(t):
    return f'{t * 1e3:.2f} ms' if t < 1 else f'{t:.2f} s'


def main():
    rng = np.random.default_rng(0)
    kdir = os.path.join(HERE, '..', 'kernels', 'c')
    lib = gcc('hand', HAND_C) if HAVE_GCC else None
    kdir_libs = {k: gcc('hand_' + k, open(os.path.join(kdir, k + '.c')).read(), '-I', kdir)
                 for k in ('gray_scott', 'flood_fill', 'viewshed')} if HAVE_GCC else {}
    cy = cy_plain = None
    cy_build = cy_plain_build = None
    if HAVE_CYTHON:
        cy, cy_build = cython_module('compare_cy', os.path.join(HERE, 'compare_cy.pyx'))
        cy_plain, cy_plain_build = cython_module('compare_kernels_cy', os.path.join(HERE, 'compare_kernels.py'))
    warm_up()

    rows = []
    for case in cases(rng, cy, cy_plain, lib, kdir_libs):
        print(f'{case.name} ...', file=sys.stderr, flush=True)
        rows.append((case.name, measure(case)))

    print(f'\n{platform.machine()}, {platform.system()}, Python {platform.python_version()}, numpy {np.__version__}'
          + (f', Numba {numba.__version__}{NUMBA_NOTE}' if numba else '') + (f', Cython {Cython.__version__}' if HAVE_CYTHON else ''))
    print('\nSteady state (best of 7; CPython one run), and the speedup over CPython:\n')
    width = 22
    print(f'{"kernel":20s}' + ''.join(f'{c:>{width}s}' for c in COLUMNS))
    reasons = []
    for name, row in rows:
        cells = []
        py = row['CPython'][0]
        for c in COLUMNS:
            v = row[c]
            if isinstance(v, str):
                if v.startswith("doesn't"):
                    reasons.append(f'{name}, {c}: {v}')
                    v = "doesn't compile"
                cells.append(f'{v:>{width}s}')
            else:
                steady, _, mark = v
                cells.append(f'{fmt(steady) + mark:>{width - 7}s}' + (f' {py / steady:5.0f}x' if c != 'CPython' else ' ' * 7))
        print(f'{name:20s}' + ''.join(cells))
    print('\nOne call of a tiny function, including the Python loop that calls it (best of 7 runs of 200,000):\n')
    over = call_overhead(cy)
    cols = ['CPython', 'scramjit', 'scramjit parity=False', 'Numba', 'Cython']
    print(f'{"function":22s}' + ''.join(f'{c:>{width}s}' for c in cols))
    for name, row in over.items():
        print(f'{name:22s}' + ''.join(f'{(f"{row[c]:.0f} ns" if c in row else "-"):>{width}s}' for c in cols))
    print('\nFirst call, including compiling (each tool warmed up once on another function first):\n')
    first_cols = ['scramjit', 'scramjit parity=False', 'Numba', 'Numba fastmath']
    print(f'{"kernel":20s}' + ''.join(f'{c:>{width}s}' for c in first_cols))
    for name, row in rows:
        print(f'{name:20s}' + ''.join(f'{(fmt(row[c][1]) if not isinstance(row[c], str) else "-"):>{width}s}'
                                      for c in first_cols))
    if cy_build is not None:
        print(f'\nCython builds ahead of time: the typed module took {cy_build:.2f} s (cython and gcc), the Python-data '
              f'one {cy_plain_build:.2f} s.')
    if reasons:
        print('\nWhy a column has no number:')
        for r in reasons:
            print('  ' + r)
    print('\nOn disk, and where each runs:')
    sizes = [('scramjit + cimo', size_mb('scramjit', 'cimo'), 'nothing else; pure Python, runs in Pyodide'),
             ('Numba + llvmlite', size_mb('numba', 'llvmlite'), 'a binary wheel per platform; not in Pyodide'),
             ('Cython', size_mb('Cython'), 'a C compiler and Python headers at build time; not in Pyodide')]
    for name, mb, note in sizes:
        print(f'  {name:18s} {"-" if mb is None else f"{mb:6.1f} MB"}   {note}')


if __name__ == '__main__':
    main()
