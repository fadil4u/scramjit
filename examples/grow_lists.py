"""Lists that grow (from 0.0.4): the pairs of points closer than a radius, found cell by cell on a grid and returned as
a list of (i, j) tuples, with the cells themselves a list of lists built by appending. Points come in as numpy arrays (a
list of 200,000 floats costs about 15 ms to check and copy at each call); the pairs go out as Python's own list."""

import math
import time

import numpy as np

from scramjit import scram


@scram
def close_pairs(xs, ys, r):
    cells = int(1.0 / r) + 1
    grid = []
    for _ in range(cells * cells):
        grid.append([])
    for i in range(len(xs)):
        grid[int(xs[i] / r) * cells + int(ys[i] / r)].append(i)
    out = []
    for cx in range(cells):
        for cy in range(cells):
            here = grid[cx * cells + cy]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or ny < 0 or nx >= cells or ny >= cells:
                        continue
                    there = grid[nx * cells + ny]
                    for i in here:
                        for j in there:
                            if i < j and math.hypot(xs[i] - xs[j], ys[i] - ys[j]) < r:
                                out.append((i, j))
    return out


rng = np.random.default_rng(1)
n = 100_000
xs, ys = rng.random(n), rng.random(n)

t0 = time.perf_counter()
close_pairs(xs[:10], ys[:10], 0.01)
first = time.perf_counter() - t0
t0 = time.perf_counter()
want = close_pairs.__wrapped__(xs, ys, 0.004)
py = time.perf_counter() - t0
t0 = time.perf_counter()
got = close_pairs(xs, ys, 0.004)
sj = time.perf_counter() - t0
assert got == want
print(f'{len(got):,} pairs closer than 0.004 among {n:,} points')
print(f'Python {py:.2f} s, scramjit {sj * 1e3:.1f} ms ({py / sj:.0f}x), first call {first:.2f} s')
