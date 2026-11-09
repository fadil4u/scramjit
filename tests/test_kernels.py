"""The M0 kernels (kernels/*.py, unchanged), compiled by scramjit, against CPython."""

import os
import sys

import numpy as np

from scramjit import scram
from util import same

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kernels'))
import flood_fill as ff  # noqa: E402
import gray_scott as gs  # noqa: E402
import viewshed as vs  # noqa: E402

gray_scott = scram(gs.gray_scott, fallback=False)
flood_fill = scram(ff.flood_fill, fallback=False)
viewshed = scram(vs.viewshed, fallback=False)


def test_gray_scott():
    rng = np.random.default_rng(1)
    for n in (1, 2, 7, 32):
        u, v = rng.random((n, n + 1)), rng.random((n, n + 1))
        same(gray_scott, u, v, np.zeros_like(u), np.zeros_like(v), 0.16, 0.08, 0.035, 0.065, 1.0)
    u, v = rng.random((8, 8)), rng.random((8, 8))
    same(gray_scott, u, v, np.zeros_like(u), np.zeros((7, 8)), 0.16, 0.08, 0.035, 0.065, 1.0)


def test_flood_fill():
    rng = np.random.default_rng(2)
    for n in (1, 5, 40):
        grid = (rng.random((n, n)) < 0.6).astype(np.uint8)
        y, x = divmod(int(np.argmax(grid)), n)
        same(flood_fill, grid, np.zeros(n * n, np.int64), y, x, 7)
        same(flood_fill, grid, np.zeros(n * n, np.int64), -1, -1, 9)
        same(flood_fill, grid, np.zeros(n * n, np.int64), y, x, int(grid[y, x]))
    grid = np.ones((20, 20), np.uint8)
    same(flood_fill, grid, np.zeros(30, np.int64), 3, 3, 7)        # IndexError mid-fill
    same(flood_fill, grid, np.zeros(400, np.int64), 3, 3, 300)     # OverflowError (NEP 50)
    same(flood_fill, grid, np.zeros(400, np.int64), 30, 3, 7)      # IndexError, axis 0


def test_viewshed():
    rng = np.random.default_rng(3)
    for n in (1, 3, 24):
        yy, xx = np.mgrid[0:n, 0:n] / n
        dem = np.sin(6 * xx) * np.cos(5 * yy) * 40 + rng.random((n, n)) * 3
        same(viewshed, dem, np.zeros((n, n), np.uint8), n // 3, n // 2, 2.0)
    dem = rng.random((10, 10))
    same(viewshed, dem, np.zeros((10, 9), np.uint8), 4, 4, 1.0)    # IndexError on vis


def test_the_list_flood_fill_compiles():
    import warnings
    from scramjit import FallbackWarning
    f = scram(ff.flood_fill_list)
    grid = np.ones((6, 6), np.uint8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        same(f, grid, 2, 2, 5)
    assert not any(issubclass(x.category, FallbackWarning) for x in w)      # a list that grows, and `while todo:` (0.0.4)
