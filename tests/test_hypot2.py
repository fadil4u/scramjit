"""math.hypot of two values takes a fast path (sj_hypot2) that must give CPython's bits for every pair, its rare
cases (NaN, inf, zero, subnormal, near overflow) included."""

import math
import random
import struct

import numpy as np

from scramjit import scram


@scram(cache=False, fallback=False)
def hypots(xs, ys, out):
    for i in range(len(xs)):
        out[i] = math.hypot(xs[i], ys[i])


EDGE = [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 2.225073858507201e-308, 1e-310, 2.0 ** -1022,
        2.0 ** -1021, 1e-300, 0.1, 1.0, 3.0, 4.0, 1e300, 2.0 ** 1021, 2.0 ** 1022, 2.0 ** 1023,
        8.98846567431158e307, 1.7976931348623157e308, math.inf, -math.inf, math.nan]


def bits(x):
    return 'nan' if x != x else struct.pack('<d', x)


def test_hypot_of_two_matches_cpython_bit_for_bit():
    r = random.Random(1)
    vals = EDGE + [struct.unpack('<d', struct.pack('<Q', r.getrandbits(64)))[0] for _ in range(300)] + \
        [r.uniform(-100, 100) for _ in range(300)]
    pairs = [(a, b) for a in EDGE for b in EDGE]
    pairs += [(r.choice(vals), r.choice(vals) * 2.0 ** r.randint(-60, 60)) for _ in range(20000)]
    xs, ys = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    out = np.empty(len(pairs))
    hypots(xs, ys, out)
    bad = [(a, b) for (a, b), o in zip(pairs, out.tolist()) if bits(o) != bits(math.hypot(a, b))]
    assert not bad, bad[:5]
