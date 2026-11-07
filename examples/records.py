"""Dicts and instances in lists (from 0.0.4): a list of dataclass instances passed in and grown, a list of dicts
made and returned, and the caller's own objects updated in place."""

import time
from dataclasses import dataclass

import numpy as np

from scramjit import scram


@dataclass
class Event:
    t: int
    size: float


@scram
def detect(signal, level, events):
    """Each run above level becomes an Event, appended to the events passed in; a summary per run comes back."""
    runs = []
    start = -1
    peak = 0.0
    for i in range(len(signal)):
        if signal[i] > level:
            if start < 0:
                start = i
                peak = 0.0
            peak = max(peak, float(signal[i]))
        elif start >= 0:
            events.append(Event(start, peak))
            runs.append({'start': start, 'length': i - start, 'peak': peak})
            start = -1
    for e in events:
        e.size = e.size - level
    return runs


rng = np.random.default_rng(3)
signal = np.cumsum(rng.standard_normal(400_000)) * 0.05
signal -= signal.mean()

detect(signal[:100], 1.0, [Event(-1, 0.0)])          # compile, for these argument types
py_events, sj_events = [Event(-1, 0.0)], [Event(-1, 0.0)]
first = sj_events[0]
t0 = time.perf_counter()
want = detect.__wrapped__(signal, 1.0, py_events)
py = time.perf_counter() - t0
t0 = time.perf_counter()
got = detect(signal, 1.0, sj_events)
sj = time.perf_counter() - t0
assert got == want and sj_events == py_events and sj_events[0] is first
print(f'{len(got):,} runs above the level, {len(sj_events) - 1:,} events appended')
print(f'Python {py * 1e3:.0f} ms, scramjit {sj * 1e3:.1f} ms ({py / sj:.0f}x)')
