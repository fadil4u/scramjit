"""Monte Carlo price of an Asian option, whose payoff depends on the average price along each path.
Each step depends on the last, and the generator's state runs through the whole simulation."""

import math
import time

from scramjit import scram


@scram
def price(spot: float, strike: float, rate: float, vol: float, years: float, steps: int, paths: int, seed: int):
    dt = years / steps
    drift = (rate - 0.5 * vol * vol) * dt
    shock = vol * math.sqrt(dt)
    state = seed
    total = 0.0
    total2 = 0.0
    for _ in range(paths):
        s = spot
        average = 0.0
        for _ in range(steps):
            state = (state * 1103515245 + 12345) % 2147483648
            u1 = (state + 1.0) / 2147483649.0
            state = (state * 1103515245 + 12345) % 2147483648
            u2 = (state + 1.0) / 2147483649.0
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)       # Box-Muller
            s *= math.exp(drift + shock * z)
            average += s
        payoff = max(average / steps - strike, 0.0)
        total += payoff
        total2 += payoff * payoff
    discount = math.exp(-rate * years)
    mean = total / paths
    error = math.sqrt((total2 / paths - mean * mean) / paths)
    return discount * mean, discount * error


t = time.perf_counter()
price(100.0, 100.0, 0.05, 0.2, 1.0, 1, 1, 42)       # the first call for these types compiles
compile_time = time.perf_counter() - t

args = (100.0, 100.0, 0.05, 0.2, 1.0, 252, 10_000, 42)
t0 = time.perf_counter()
native = price(*args)
t1 = time.perf_counter()
python = price.__wrapped__(*args)
t2 = time.perf_counter()

assert native == python
print(f'10,000 paths of 252 days: price {native[0]:.3f} ± {native[1]:.3f}')
print(f'python {t2 - t1:.2f} s, scramjit {(t1 - t0) * 1000:.1f} ms, {(t2 - t1) / (t1 - t0):.0f}x (compiling took {compile_time * 1000:.0f} ms)')

t0 = time.perf_counter()
p, e = price(100.0, 100.0, 0.05, 0.2, 1.0, 252, 200_000, 42)
print(f'200,000 paths in {time.perf_counter() - t0:.2f} s: price {p:.3f} ± {e:.3f}')
