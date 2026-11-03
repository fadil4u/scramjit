"""A maze walked by a breadth-first search, with an Enum for what each cell is. Members compile as their values:
Cell.WALL, `is`, `==` and members passed between compiled functions, with the member itself coming back."""

import enum
import time

import numpy as np

from scramjit import scram


class Cell(enum.Enum):
    OPEN = 0
    WALL = 1
    GOAL = 2


@scram
def cell_at(grid, r: int, c: int):
    v = grid[r, c]
    if v == 1:
        return Cell.WALL
    if v == 2:
        return Cell.GOAL
    return Cell.OPEN


@scram
def steps_to_goal(grid, dist, queue):
    """Fewest steps from the top left corner to a GOAL cell, or -1."""
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            dist[i, j] = -1
    dist[0, 0] = 0
    queue[0] = 0
    head, tail = 0, 1
    while head < tail:
        at = queue[head]
        head += 1
        r, c = at // w, at % w
        if cell_at(grid, r, c) is Cell.GOAL:
            return dist[r, c]
        for k in range(4):
            nr = r + (1 if k == 0 else -1 if k == 1 else 0)
            nc = c + (1 if k == 2 else -1 if k == 3 else 0)
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < 0 and cell_at(grid, nr, nc) != Cell.WALL:
                dist[nr, nc] = dist[r, c] + 1
                queue[tail] = nr * w + nc
                tail += 1
    return -1


if __name__ == '__main__':
    rng = np.random.default_rng(4)
    grid = (rng.random((400, 400)) < 0.3).astype(np.int64)
    grid[0, 0], grid[-1, -1] = 0, 2
    dist, queue = np.zeros(grid.shape, np.int64), np.zeros(grid.size, np.int64)

    start = time.perf_counter()
    want = steps_to_goal.__wrapped__(grid, dist.copy(), queue.copy())
    python_time = time.perf_counter() - start

    steps_to_goal(grid, dist, queue)                     # compiles
    start = time.perf_counter()
    got = steps_to_goal(grid, dist, queue)
    scramjit_time = time.perf_counter() - start
    assert got == want
    print(f'{grid.shape[0]}x{grid.shape[1]} maze, {got} steps to the goal: Python {python_time:.2f} s, '
          f'scramjit {scramjit_time * 1000:.2f} ms, {python_time / scramjit_time:.0f}x')
    print('the member itself comes back:', cell_at(grid, 0, 0), cell_at(grid, 399, 399))
