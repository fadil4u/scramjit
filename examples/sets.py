"""Sets: islands on a sparse map, the land a set of (x, y) cells. Each island is found by a flood fill
that pops a cell from the land, then keeps popping cells from a frontier set and moving the neighbours that are
land into it. Sets are kept as CPython keeps them, so they pop in the order Python's do, and the islands come out
in the order Python finds them."""

import time

import numpy as np

from scramjit import scram


@scram
def islands(xs, ys):
    land = set()
    for i in range(len(xs)):
        land.add((int(xs[i]), int(ys[i])))
    sizes = []
    while land:
        frontier = {land.pop()}
        size = 1
        while frontier:
            x, y = frontier.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (x + dx, y + dy)
                if c in land:
                    land.discard(c)
                    frontier.add(c)
                    size += 1
        sizes.append(size)
    return sizes


rng = np.random.default_rng(5)
n = 200_000
xs, ys = rng.integers(0, 600, n), rng.integers(0, 600, n)

t0 = time.perf_counter()
islands(xs[:10], ys[:10])
first = time.perf_counter() - t0
t0 = time.perf_counter()
want = islands.__wrapped__(xs, ys)
py = time.perf_counter() - t0
t0 = time.perf_counter()
got = islands(xs, ys)
sj = time.perf_counter() - t0
assert got == want
print(f'{len(got):,} islands among {n:,} cells; the first found has {got[0]} cells, the largest {max(got)}')
print(f'Python {py:.2f} s, scramjit {sj * 1e3:.1f} ms ({py / sj:.0f}x), first call {first:.2f} s')
