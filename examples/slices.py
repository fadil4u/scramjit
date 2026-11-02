"""One-dimensional heat flow with a slice for every step: boundaries set with x[:2] = ..., interior written through views, and each
step's output copied back with u[1:-1] = next. The function returns a view of its result, which shares memory with the array passed in."""

import time

import numpy as np

from scramjit import scram


@scram
def diffuse(u, scratch, steps: int, rate: float):
    n = len(u)
    inner = scratch[1:n - 1]
    for _ in range(steps):
        left, middle, right = u[:n - 2], u[1:n - 1], u[2:]
        for i in range(len(inner)):
            inner[i] = middle[i] + rate * (left[i] - 2.0 * middle[i] + right[i])
        u[1:n - 1] = inner
        u[:1] = 0.0
        u[n - 1:] = u[n - 2]
    return u[n // 4:3 * n // 4]


n = 2000
start = np.sin(np.linspace(0.0, 12.0, n)) + np.random.default_rng(3).random(n)


def run(f):
    u, scratch = start.copy(), np.zeros(n)
    t = time.perf_counter()
    middle = f(u, scratch, 300, 0.2)
    return time.perf_counter() - t, u, middle


t0 = time.perf_counter()
run(diffuse)
compiled_in = time.perf_counter() - t0

python_s, want, want_middle = run(diffuse.__wrapped__)
native_s = min(run(diffuse)[0] for _ in range(5))
_, got, got_middle = run(diffuse)
assert got.tobytes() == want.tobytes() and got_middle.tobytes() == want_middle.tobytes()
assert np.shares_memory(got_middle, got)

print(f'after 300 steps the middle half spans {got_middle.min():.4f} to {got_middle.max():.4f}, and is a view: {got_middle.base is not None}')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.2f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')
