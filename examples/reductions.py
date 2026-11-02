"""Find spikes in a signal by comparing each sample with the window before it, using x[a:b].mean(), .max() and .min() as you
would in numpy. Called once per sample these are slow from Python, where each call costs about a microsecond. Compiled, a call
inside the loop costs what its elements do. The result is the same to the last bit."""

import time

import numpy as np

from scramjit import scram


@scram
def spikes(x, width: int, out):
    found = 0
    for i in range(width, len(x)):
        window = x[i - width:i]
        spread = window.max() - window.min()
        if x[i] - window.mean() > 2.0 * spread:
            out[found] = i
            found += 1
    return found


rng = np.random.default_rng(21)
signal = rng.normal(0.0, 1.0, 400_000).cumsum() * 0.002 + rng.random(400_000) * 0.01
signal[rng.integers(0, 400_000, 300)] += 1.0
out = np.zeros(400_000, np.int64)
width = 32


def run(f):
    t = time.perf_counter()
    found = f(signal, width, out)
    return time.perf_counter() - t, found, out[:found].copy()


t0 = time.perf_counter()
spikes(signal[:100], width, out)
compiled_in = time.perf_counter() - t0

python_s, want, want_at = run(spikes.__wrapped__)
native_s = min(run(spikes)[0] for _ in range(5))
_, got, got_at = run(spikes)
assert got == want and got_at.tobytes() == want_at.tobytes()

print(f'{got} spikes in {len(signal):,} samples, first at {got_at[:4].tolist()}')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.1f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')
