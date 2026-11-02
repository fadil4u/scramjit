"""Conway's Game of Life on a torus. The function makes its own arrays: a copy of the board to start from and a new
board each generation, and what it returns is an ordinary numpy array. Memory from earlier generations is freed as it goes."""

import time

import numpy as np

from scramjit import scram


@scram
def life(start, generations):
    height, width = start.shape
    board = start.copy()
    for _ in range(generations):
        next_board = np.zeros_like(board)
        for i in range(height):
            above, below = (i - 1) % height, (i + 1) % height
            for j in range(width):
                left, right = (j - 1) % width, (j + 1) % width
                neighbours = (board[above, left] + board[above, j] + board[above, right]
                              + board[i, left] + board[i, right]
                              + board[below, left] + board[below, j] + board[below, right])
                if neighbours == 3 or (neighbours == 2 and board[i, j] == 1):
                    next_board[i, j] = 1
        board = next_board
    return board


rng = np.random.default_rng(5)
start = (rng.random((128, 128)) < 0.3).astype(np.int64)
generations = 30


def run(f):
    t = time.perf_counter()
    r = f(start, generations)
    return time.perf_counter() - t, r


t0 = time.perf_counter()
run(life)
compiled_in = time.perf_counter() - t0

python_s, plain = run(life.__wrapped__)
native_s = min(run(life)[0] for _ in range(5))
_, native = run(life)
assert native.tobytes() == plain.tobytes() and native.dtype == plain.dtype and native.shape == plain.shape
assert native.flags.owndata and native.flags.writeable

print(f'{generations} generations of {start.shape[0]}x{start.shape[1]}: {start.sum()} cells alive to {native.sum()}')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.2f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')
