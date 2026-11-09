"""Whole-array arithmetic: 1,000 damped oscillators under one drive, stepped with numpy expressions on
whole arrays. In Python each `*` and `+` is a numpy call making a temporary array; compiled, each line is one pass over the
elements, with numpy's results bit for bit."""

import math
import time

import numpy as np

from scramjit import scram


@scram
def simulate(x, v, k, c, steps, dt):
    for s in range(steps):
        drive = math.sin(s * dt * 3.0)
        a = -k * x - c * v + drive
        v = v + a * dt
        x = x + v * dt
    return x, v


n = 1000
rng = np.random.default_rng(0)
x0, v0 = rng.standard_normal(n), np.zeros(n)
k, c = rng.uniform(0.5, 2.0, n), rng.uniform(0.01, 0.1, n)
args = (x0, v0, k, c, 2000, 0.01)

t0 = time.perf_counter()
simulate(x0, v0, k, c, 1, 0.01)
first = time.perf_counter() - t0

t0 = time.perf_counter()
want = simulate.__wrapped__(*args)
py = time.perf_counter() - t0
t0 = time.perf_counter()
got = simulate(*args)
sj = time.perf_counter() - t0

assert all(g.tobytes() == w.tobytes() for g, w in zip(got, want))
print(f'mean |x| after 2,000 steps: {np.abs(got[0]).mean():.6f}')
print(f'numpy from Python: {py:.3f} s')
print(f'scramjit:          {sj * 1e3:.1f} ms  ({py / sj:.1f}x), first call {first:.2f} s')
