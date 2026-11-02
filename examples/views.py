"""Solve a linear system by Gaussian elimination, with rows and columns as views: m[i], m[:, k], m[k][j]. Each elimination
step changes what the next one sees, and the pivot search depends on the data, so this isn't one numpy expression."""

import time

import numpy as np

from scramjit import scram


@scram
def solve(m, b, x):
    n = len(m)
    for k in range(n):
        column = m[:, k]
        pivot = k
        for i in range(k + 1, n):
            if abs(column[i]) > abs(column[pivot]):
                pivot = i
        if pivot != k:
            top = m[k]
            other = m[pivot]
            for j in range(k, n):
                top[j], other[j] = other[j], top[j]
            b[k], b[pivot] = b[pivot], b[k]
        top = m[k]
        for i in range(k + 1, n):
            row = m[i]
            factor = row[k] / top[k]
            for j in range(k, n):
                row[j] -= factor * top[j]
            b[i] -= factor * b[k]
    for k in range(n - 1, -1, -1):
        s = b[k]
        row = m[k]
        for j in range(k + 1, n):
            s -= row[j] * x[j]
        x[k] = s / row[k]


rng = np.random.default_rng(5)
n = 300
a = rng.random((n, n)) + np.eye(n) * n
want_x = rng.random(n)
rhs = a @ want_x


def run(f):
    m, b, x = a.copy(), rhs.copy(), np.zeros(n)
    t = time.perf_counter()
    f(m, b, x)
    return time.perf_counter() - t, x


t0 = time.perf_counter()
run(solve)
compiled_in = time.perf_counter() - t0

python_s, plain = run(solve.__wrapped__)
native_s = min(run(solve)[0] for _ in range(5))
_, native = run(solve)
assert native.tobytes() == plain.tobytes()
assert np.allclose(native, want_x)

print(f'{n} equations, largest error {abs(native - want_x).max():.1e}')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.2f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')
