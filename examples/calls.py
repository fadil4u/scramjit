"""A small program as several functions: a cart and pole, with reset, step and a policy, and a function that runs many episodes.
Each is compiled, the calls between them are C calls, usually inlined, and each can still be called from Python on its own.
Default values work both ways: average_steps leaves out policy's force, and the call from Python leaves out the seed."""

import math
import time

import numpy as np

from scramjit import scram

GRAVITY = 9.8
DT = 0.02


@scram
def reset(state, seed: int):
    for i in range(4):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xffffffff
        state[i] = (seed / 4294967296.0 - 0.5) * 0.1
    return seed


@scram
def step(state, push):
    x, speed, angle, spin = state[0], state[1], state[2], state[3]
    cos, sin = math.cos(angle), math.sin(angle)
    t = (push + 0.05 * spin * spin * sin) / 1.1
    spin_up = (GRAVITY * sin - cos * t) / (0.5 * (4.0 / 3.0 - 0.1 * cos * cos / 1.1))
    speed_up = t - 0.05 * spin_up * cos / 1.1
    state[0], state[1] = x + DT * speed, speed + DT * speed_up
    state[2], state[3] = angle + DT * spin, spin + DT * spin_up
    return abs(state[0]) < 2.4 and abs(state[2]) < 0.21


@scram
def policy(state, weights, force=10.0):
    score = 0.0
    for i in range(4):
        score += state[i] * weights[i]
    return force if score > 0.0 else -force


@scram
def average_steps(weights, episodes: int, seed: int = 42):
    state = np.zeros(4)
    total = 0
    for _ in range(episodes):
        seed = reset(state, seed)
        for _ in range(500):
            total += 1
            if not step(state, policy(state, weights)):
                break
    return total / episodes


weights = np.array([0.1, 0.5, 1.0, 0.5])
episodes = 2000


def run(f):
    t = time.perf_counter()
    r = f(weights, episodes)
    return time.perf_counter() - t, r


t0 = time.perf_counter()
run(average_steps)
compiled_in = time.perf_counter() - t0


def plain_average_steps(weights, episodes, seed=42):      # the same program with nothing compiled
    plain_reset, plain_step, plain_policy = reset.__wrapped__, step.__wrapped__, policy.__wrapped__
    state = np.zeros(4)
    total = 0
    for _ in range(episodes):
        seed = plain_reset(state, seed)
        for _ in range(500):
            total += 1
            if not plain_step(state, plain_policy(state, weights)):
                break
    return total / episodes


python_s, plain = run(plain_average_steps)
native_s = min(run(average_steps)[0] for _ in range(5))
_, native = run(average_steps)
assert native == plain

print(f'{episodes} episodes, {plain:.1f} steps on average')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.2f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')
