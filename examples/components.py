"""Connected components with union-find. The loops chase parent pointers for as long as the data
says, which array expressions can't do."""

import time

import numpy as np

from scramjit import scram


@scram
def count_components(grid, parent):
    """grid: 2-D uint8, nonzero cells are filled. parent: int64, one entry per cell."""
    rows, cols = grid.shape
    for i in range(rows * cols):
        parent[i] = i
    for y in range(rows):
        for x in range(cols):
            if grid[y, x] == 0:
                continue
            for k in range(2):                  # right and down cover every edge once
                ny = y + k
                nx = x + 1 - k
                if ny >= rows or nx >= cols or grid[ny, nx] == 0:
                    continue
                a = y * cols + x
                b = ny * cols + nx
                while parent[a] != a:           # find, halving the path as it goes
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                while parent[b] != b:
                    parent[b] = parent[parent[b]]
                    b = parent[b]
                if a != b:
                    parent[a] = b
    count = 0
    for y in range(rows):
        for x in range(cols):
            i = y * cols + x
            if grid[y, x] != 0 and parent[i] == i:
                count += 1
    return count


rng = np.random.default_rng(1)
grid = (rng.random((700, 700)) < 0.6).astype(np.uint8)
parent = np.zeros(grid.size, np.int64)

t = time.perf_counter()
count_components(grid[:1, :1], parent)              # the first call for these types compiles
compile_time = time.perf_counter() - t

t0 = time.perf_counter()
native = count_components(grid, parent)
t1 = time.perf_counter()
python = count_components.__wrapped__(grid, parent)
t2 = time.perf_counter()

assert native == python
print(f'700x700 grid, 60% filled: {native} components')
print(f'python {t2 - t1:.2f} s, scramjit {(t1 - t0) * 1000:.1f} ms, {(t2 - t1) / (t1 - t0):.0f}x (compiling took {compile_time * 1000:.0f} ms)')
