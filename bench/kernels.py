"""The M0 kernels: CPython, scramjit (with and without parity), and the hand-written C of kernels/c/.
Run from the repository root:  python3 bench/kernels.py
"""

import os
import sys
import time

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


def best(f, repeat=5):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def hand(name):
    return getattr(cimo.load(os.path.join(HERE, '..', 'kernels', 'c', name + '.c'), fma=False), name)


def main():
    rng = np.random.default_rng(0)
    line = np.zeros(1, np.int64)
    rows = []

    n = 256
    u, v = rng.random((n, n)), rng.random((n, n))
    u2, v2 = np.zeros_like(u), np.zeros_like(v)
    p = (0.16, 0.08, 0.035, 0.065, 1.0)
    g, h = scram(gs.gray_scott, fallback=False), hand('gray_scott')
    gr = scram(gs.gray_scott, fallback=False, parity=False)
    g(u, v, u2, v2, *p)
    gr(u, v, u2, v2, *p)
    rows.append(('gray_scott 256x256', best(lambda: gs.gray_scott(u, v, u2, v2, *p), 1),
                 best(lambda: g(u, v, u2, v2, *p)), best(lambda: gr(u, v, u2, v2, *p)),
                 best(lambda: h(u, *u.shape, v, *v.shape, u2, *u2.shape, v2, *v2.shape, *p, line))))

    n = 512
    grid = (rng.random((n, n)) < 0.8).astype(np.uint8)    # one large connected region
    y0, x0 = divmod(int(np.argmax(grid)), n)
    stack = np.zeros(n * n, np.int64)
    f, h = scram(ff.flood_fill, fallback=False), hand('flood_fill')
    fr = scram(ff.flood_fill, fallback=False, parity=False)
    ret = np.zeros(1, np.int64)
    f(grid.copy(), stack, y0, x0, 7)
    fr(grid.copy(), stack, y0, x0, 7)
    rows.append(('flood_fill 512x512', best(lambda: ff.flood_fill(grid.copy(), stack, y0, x0, 7), 1),
                 best(lambda: f(grid.copy(), stack, y0, x0, 7)), best(lambda: fr(grid.copy(), stack, y0, x0, 7)),
                 best(lambda: h(grid.copy(), n, n, stack, n * n, y0, x0, 7, ret, line))))

    n = 96
    yy, xx = np.mgrid[0:n, 0:n] / n
    dem = np.sin(6 * xx) * np.cos(5 * yy) * 40 + rng.random((n, n)) * 3
    vis = np.zeros((n, n), np.uint8)
    w, h = scram(vs.viewshed, fallback=False), hand('viewshed')
    wr = scram(vs.viewshed, fallback=False, parity=False)
    w(dem, vis, n // 3, n // 2, 2.0)
    wr(dem, vis, n // 3, n // 2, 2.0)
    rows.append(('viewshed 96x96', best(lambda: vs.viewshed(dem, vis, n // 3, n // 2, 2.0), 1),
                 best(lambda: w(dem, vis, n // 3, n // 2, 2.0)), best(lambda: wr(dem, vis, n // 3, n // 2, 2.0)),
                 best(lambda: h(dem, n, n, vis, n, n, n // 3, n // 2, 2.0, line))))

    print(f'{"kernel":20s} {"CPython":>10s} {"scramjit":>10s} {"parity off":>11s} {"hand C":>10s}')
    for name, t_py, t_sj, t_r, t_c in rows:
        print(f'{name:20s} {t_py * 1e3:8.1f}ms {t_sj * 1e3:8.2f}ms {t_r * 1e3:9.2f}ms {t_c * 1e3:8.2f}ms'
              f'   ({t_py / t_sj:.0f}x / {t_py / t_r:.0f}x CPython)')


if __name__ == '__main__':
    main()
