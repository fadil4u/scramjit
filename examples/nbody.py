"""Gravity between every pair of bodies, integrated with leapfrog. A good integrator nearly
conserves energy, so the drift printed at the end is a check on the physics."""

import math
import time

import numpy as np

from scramjit import scram


@scram
def advance(pos, vel, mass, dt: float, steps: int):
    n = pos.shape[0]
    for _ in range(steps):
        for i in range(n):
            for k in range(3):
                pos[i, k] += 0.5 * dt * vel[i, k]
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                r2 = dx * dx + dy * dy + dz * dz + 0.0025       # softened, so close passes stay finite
                inv = dt / (r2 * math.sqrt(r2))
                vel[i, 0] += dx * inv * mass[j]
                vel[i, 1] += dy * inv * mass[j]
                vel[i, 2] += dz * inv * mass[j]
                vel[j, 0] -= dx * inv * mass[i]
                vel[j, 1] -= dy * inv * mass[i]
                vel[j, 2] -= dz * inv * mass[i]
        for i in range(n):
            for k in range(3):
                pos[i, k] += 0.5 * dt * vel[i, k]


@scram
def energy(pos, vel, mass):
    n = pos.shape[0]
    e = 0.0
    for i in range(n):
        e += 0.5 * mass[i] * (vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1] + vel[i, 2] * vel[i, 2])
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            e -= mass[i] * mass[j] / math.sqrt(dx * dx + dy * dy + dz * dz + 0.0025)
    return e


rng = np.random.default_rng(2)
n = 100
pos = rng.standard_normal((n, 3))
vel = rng.standard_normal((n, 3)) * 0.1
mass = np.full(n, 1.0 / n)
p2, v2 = pos.copy(), vel.copy()

before = energy(pos, vel, mass)
t = time.perf_counter()
advance(pos, vel, mass, 0.001, 0)                   # the first call for these types compiles
compile_time = time.perf_counter() - t

t0 = time.perf_counter()
advance(pos, vel, mass, 0.001, 100)
t1 = time.perf_counter()
advance.__wrapped__(p2, v2, mass, 0.001, 100)
t2 = time.perf_counter()
assert np.array_equal(pos, p2) and np.array_equal(vel, v2)
print(f'{n} bodies, 100 steps: python {t2 - t1:.2f} s, scramjit {(t1 - t0) * 1000:.1f} ms, {(t2 - t1) / (t1 - t0):.0f}x (compiling took {compile_time * 1000:.0f} ms)')

advance(pos, vel, mass, 0.001, 5000)
print(f'after 5,100 steps, relative energy drift: {abs((energy(pos, vel, mass) - before) / before):.1e}')
