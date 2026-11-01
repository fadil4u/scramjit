"""Plucked strings with the Karplus-Strong algorithm, written to a WAV file. Each sample depends on
the sample one period before it: a recurrence, so it has to run one sample at a time."""

import sys
import time
import wave

import numpy as np

from scramjit import scram

RATE = 44100


@scram
def pluck(out, start: int, count: int, period: int, decay: float, seed: int):
    """Add a string of this period (in samples) to out, from start, for count samples."""
    string = [0.0] * period
    state = seed
    for i in range(period):                     # the pluck: a burst of noise
        state = (state * 1103515245 + 12345) % 2147483648
        string[i] = state / 1073741824.0 - 1.0
    for n in range(count):
        k = n % period
        following = string[(k + 1) % period]
        out[start + n] += string[k]
        string[k] = decay * 0.5 * (string[k] + following)


song = [(330, 0.0), (392, 0.25), (494, 0.5), (659, 0.75), (494, 1.0), (392, 1.25), (330, 1.5), (247, 1.75), (330, 2.0),
        (392, 2.5), (494, 2.75), (659, 3.0), (784, 3.25), (659, 3.5), (494, 3.75), (392, 4.0), (330, 4.25), (247, 4.5)]
seconds = 7.0
out = np.zeros(int(seconds * RATE))
python = np.zeros_like(out)


def play(fn, buffer):
    for i, (hz, when) in enumerate(song):
        start = int(when * RATE)
        fn(buffer, start, len(buffer) - start, round(RATE / hz), 0.996, 7 + i)


t = time.perf_counter()
pluck(np.zeros(4), 0, 4, 2, 0.9, 1)                 # the first call for these types compiles
compile_time = time.perf_counter() - t

t0 = time.perf_counter()
play(pluck, out)
t1 = time.perf_counter()
play(pluck.__wrapped__, python)
t2 = time.perf_counter()

assert np.array_equal(out, python)
print(f'{seconds:.0f} s of audio, {len(song)} notes: python {t2 - t1:.2f} s, scramjit {(t1 - t0) * 1000:.1f} ms, {(t2 - t1) / (t1 - t0):.0f}x (compiling took {compile_time * 1000:.0f} ms)')

path = sys.argv[1] if len(sys.argv) > 1 else 'pluck.wav'
with wave.open(path, 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(RATE)
    f.writeframes((out / np.abs(out).max() * 32000).astype(np.int16).tobytes())
print('wrote', path)
