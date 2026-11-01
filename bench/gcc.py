"""The M0 kernels, hand-written C and scramjit's generated C, each compiled by cimo and by
gcc -O3 (-ffp-contract=off, as Python never contracts). Separates scramjit's code generation
from cimo's backend. Run from the repository root:  python3 bench/gcc.py   (needs gcc)
"""

import ctypes
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('CIMO_FMA', '0')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'kernels'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'cimo', 'python'))

import numpy as np  # noqa: E402

import cimo  # noqa: E402
import flood_fill as ff  # noqa: E402
import gray_scott as gs  # noqa: E402
import viewshed as vs  # noqa: E402
from scramjit import scram  # noqa: E402
from scramjit._runtime import PRELUDE  # noqa: E402

TMP = tempfile.mkdtemp(prefix='scramjit-gcc-')
CFLAGS = ['-O3', '-ffp-contract=off', '-fPIC', '-shared']


def gcc(name, source, include=None):
    src = os.path.join(TMP, name + '.c')
    lib = os.path.join(TMP, name + '.so')
    with open(src, 'w') as f:
        f.write(source)
    cmd = ['gcc', *CFLAGS, src, '-o', lib, '-lm'] + (['-I', include] if include else [])
    subprocess.run(cmd, check=True)
    return ctypes.CDLL(lib)


def best(f, repeat=7):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def ptr(a):
    return ctypes.c_void_p(a.ctypes.data)


def scramjit_gcc(fn, args, tag):
    """scramjit's generated C for fn(*args), compiled by gcc; returns a callable taking args."""
    entry = next(v for v in fn._cache.values() if hasattr(v, 'source'))
    lib = gcc(tag, PRELUDE + entry.source)
    k = lib.sj_kernel
    k.restype = ctypes.c_int
    io = bytearray(64)
    iobuf = (ctypes.c_char * len(io)).from_buffer(io)
    libm = entry.libm

    def conv(a, t):
        if t.kind == 'array':
            return [ptr(a)] + [ctypes.c_int64(n) for n in a.shape]
        if t.kind == 'float':
            return [ctypes.c_double(a)]
        return [ctypes.c_int64(int(a))] if t.ctype == 'int64_t' else [ctypes.c_int(int(a))]

    def call():
        cargs = [c for a, t in zip(args, entry.types) for c in conv(a, t)]
        if libm is not None:
            cargs.append(ctypes.c_char_p(libm))
        cargs.append(iobuf)
        return k(*cargs)
    return call


def main():
    rng = np.random.default_rng(0)
    kdir = os.path.join(HERE, '..', 'kernels', 'c')
    line = np.zeros(1, np.int64)
    ret = np.zeros(1, np.int64)
    rows = []

    def hand_gcc(name):
        return getattr(gcc('hand_' + name, open(os.path.join(kdir, name + '.c')).read(), kdir), name)

    def hand_cimo(name):
        return getattr(cimo.load(os.path.join(kdir, name + '.c')), name)

    # gray_scott
    n = 256
    u, v = rng.random((n, n)), rng.random((n, n))
    u2, v2 = np.zeros_like(u), np.zeros_like(v)
    p = (0.16, 0.08, 0.035, 0.065, 1.0)
    hc, hg = hand_cimo('gray_scott'), hand_gcc('gray_scott')
    hg.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64] * 4 + [ctypes.c_double] * 5 + [ctypes.c_void_p]
    hand_args = lambda: (u, *u.shape, v, *v.shape, u2, *u2.shape, v2, *v2.shape, *p, line)  # noqa: E731
    hg_args = lambda: (ptr(u), *u.shape, ptr(v), *v.shape, ptr(u2), *u2.shape, ptr(v2), *v2.shape, *p, ptr(line))  # noqa: E731
    for parity in (True, False):
        f = scram(gs.gray_scott, fallback=False, parity=parity)
        args = (u, v, u2, v2, *p)
        f(*args)
        sg = scramjit_gcc(f, args, f'sj_gray_scott_{parity}')
        rows.append((f'gray_scott 256²{"" if parity else " (parity off)"}', best(lambda: f(*args)), best(sg),
                     best(lambda: hc(*hand_args())), best(lambda: hg(*hg_args()))))

    # flood_fill (the timing includes copying the grid, the same for every column)
    n = 512
    grid = (rng.random((n, n)) < 0.8).astype(np.uint8)
    y0, x0 = divmod(int(np.argmax(grid)), n)
    stack = np.zeros(n * n, np.int64)
    hc, hg = hand_cimo('flood_fill'), hand_gcc('flood_fill')
    hg.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                   ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    for parity in (True, False):
        f = scram(ff.flood_fill, fallback=False, parity=parity)
        g = grid.copy()
        f(g, stack, y0, x0, 7)
        sg = scramjit_gcc(f, (g, stack, y0, x0, 7), f'sj_flood_fill_{parity}')

        def run_sg():
            g[...] = grid
            sg()

        def run_f():
            g[...] = grid
            f(g, stack, y0, x0, 7)

        def run_hc():
            g[...] = grid
            hc(g, n, n, stack, n * n, y0, x0, 7, ret, line)

        def run_hg():
            g[...] = grid
            hg(ptr(g), n, n, ptr(stack), n * n, y0, x0, 7, ptr(ret), ptr(line))
        rows.append((f'flood_fill 512²{"" if parity else " (parity off)"}', best(run_f), best(run_sg), best(run_hc), best(run_hg)))

    # viewshed
    n = 96
    yy, xx = np.mgrid[0:n, 0:n] / n
    dem = np.sin(6 * xx) * np.cos(5 * yy) * 40 + rng.random((n, n)) * 3
    vis = np.zeros((n, n), np.uint8)
    hc, hg = hand_cimo('viewshed'), hand_gcc('viewshed')
    hg.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                   ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_void_p]
    for parity in (True, False):
        f = scram(vs.viewshed, fallback=False, parity=parity)
        args = (dem, vis, n // 3, n // 2, 2.0)
        f(*args)
        sg = scramjit_gcc(f, args, f'sj_viewshed_{parity}')
        rows.append((f'viewshed 96²{"" if parity else " (parity off)"}', best(lambda: f(*args)), best(sg),
                     best(lambda: hc(dem, n, n, vis, n, n, n // 3, n // 2, 2.0, line)),
                     best(lambda: hg(ptr(dem), n, n, ptr(vis), n, n, n // 3, n // 2, 2.0, ptr(line)))))

    print(f'{"kernel":28s} {"scramjit":>10s} {"scramjit":>10s} {"hand C":>9s} {"hand C":>9s}')
    print(f'{"":28s} {"(cimo)":>10s} {"(gcc -O3)":>10s} {"(cimo)":>9s} {"(gcc)":>9s}   cimo/gcc on hand C')
    for name, a, b, c, d in rows:
        print(f'{name:28s} {a * 1e3:8.2f}ms {b * 1e3:8.2f}ms {c * 1e3:7.2f}ms {d * 1e3:7.2f}ms   {c / d:5.2f}x')


if __name__ == '__main__':
    main()
