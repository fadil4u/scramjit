"""A filter whose settings are at module level, the way scripts usually keep them: a number, a tuple, an array and a dict. A compiled
function reads them like arguments, at each call, so changing one between calls just works, and a new value doesn't recompile anything."""

import time

import numpy as np

from scramjit import scram

TAPS = 16
GAIN = 0.75
LIMITS = (-1.0, 1.0)
WINDOW = np.hanning(TAPS)
SETTINGS = {'passes': 3, 'name': 'smoother'}


@scram
def smooth(x, out):
    total = WINDOW.sum()
    low, high = LIMITS
    for _ in range(SETTINGS['passes']):
        for i in range(TAPS, len(x)):
            s = 0.0
            for k in range(TAPS):
                s += WINDOW[k] * x[i - k]
            out[i] = np.clip(GAIN * s / total, low, high)
    return out[TAPS:].max()


signal = np.random.default_rng(9).normal(0.0, 1.0, 40_000)
out = np.zeros(len(signal))


def run(f):
    t = time.perf_counter()
    peak = f(signal, out)
    return time.perf_counter() - t, peak, out.copy()


t0 = time.perf_counter()
run(smooth)
compiled_in = time.perf_counter() - t0

python_s, want, want_out = run(smooth.__wrapped__)
native_s = min(run(smooth)[0] for _ in range(5))
_, got, got_out = run(smooth)
assert got == want and got_out.tobytes() == want_out.tobytes()
print(f'{SETTINGS["name"]}: peak {got:.4f}')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.1f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')

GAIN = 0.5                                 # read at the next call
assert smooth(signal, out) == smooth.__wrapped__(signal, out)
print(f'after GAIN = 0.5: peak {smooth(signal, out):.4f}, still {len(smooth.signatures)} compiled signature')
