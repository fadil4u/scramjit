"""The Mandelbrot set, drawn in the terminal. How long each pixel iterates depends on the pixel."""

import time

import numpy as np

from scramjit import scram


@scram
def render(out, cx: float, cy: float, scale: float, maxit: int):
    """out: 2-D int64, filled with each pixel's escape count."""
    rows, cols = out.shape
    for y in range(rows):
        ci = cy + (y - rows / 2) * scale * 2.0          # terminal cells are twice as tall as wide
        for x in range(cols):
            cr = cx + (x - cols / 2) * scale
            zr = 0.0
            zi = 0.0
            i = 0
            while i < maxit and zr * zr + zi * zi < 4.0:
                t = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = t
                i += 1
            out[y, x] = i


small = np.zeros((32, 78), np.int64)
render(small, -0.6, 0.0, 0.037, 200)
ramp = ' .:-=+*#%@'
for row in small:
    print(''.join(' ' if v == 200 else ramp[int(v) % len(ramp)] for v in row))

big, big2 = np.zeros((400, 400), np.int64), np.zeros((400, 400), np.int64)
t0 = time.perf_counter()
render(big, -0.7436, 0.1318, 0.00001, 300)
t1 = time.perf_counter()
render.__wrapped__(big2, -0.7436, 0.1318, 0.00001, 300)
t2 = time.perf_counter()

assert np.array_equal(big, big2)
print(f'\n400x400, {big.sum():,} iterations')
print(f'python {t2 - t1:.2f} s, scramjit {(t1 - t0) * 1000:.1f} ms, {(t2 - t1) / (t1 - t0):.0f}x')
