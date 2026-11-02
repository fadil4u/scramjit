"""numpy functions on single numbers compile: a greedy delivery route over points on a sphere, written with np.sin,
np.arcsin and friends as you'd write it anyway. Each stop depends on the one before, so it isn't one numpy expression."""

import time

import numpy as np

from scramjit import scram


@scram
def route(lat, lon, visited, order):
    here = 0
    visited[0] = 1
    total = 0.0
    for stop in range(1, len(lat)):
        best = np.inf
        chosen = -1
        for j in range(len(lat)):
            if visited[j]:
                continue
            a = np.sin((lat[j] - lat[here]) / 2.0)
            b = np.sin((lon[j] - lon[here]) / 2.0)
            h = a * a + np.cos(lat[here]) * np.cos(lat[j]) * b * b
            d = 2.0 * 6371.0 * np.arcsin(np.minimum(np.sqrt(h), 1.0))
            if d < best:
                best = d
                chosen = j
        visited[chosen] = 1
        order[stop] = chosen
        total += best
        here = chosen
    return total


n = 1500
rng = np.random.default_rng(11)
lat = np.radians(rng.uniform(-60.0, 60.0, n))
lon = np.radians(rng.uniform(-180.0, 180.0, n))


def run(f, count):
    visited, order = np.zeros(count, np.int64), np.zeros(count, np.int64)
    t = time.perf_counter()
    total = f(lat[:count], lon[:count], visited, order)
    return time.perf_counter() - t, total, order


t0 = time.perf_counter()
run(route, 4)
compiled_in = time.perf_counter() - t0

python_s, want, want_order = run(route.__wrapped__, 300)        # small: this is slow in Python
native_s, got, got_order = min((run(route, 300) for _ in range(5)), key=lambda r: r[0])
assert got == want and got_order.tobytes() == want_order.tobytes()
print(f'{want:,.0f} km over 300 stops')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.2f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')

native_s, got, _ = run(route, n)
print(f'{got:,.0f} km over {n} stops in {native_s * 1000:.1f} ms')
