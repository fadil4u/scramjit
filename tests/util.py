"""Differential testing: a compiled function must behave exactly like the same function in CPython."""

import math
import os
import struct
import traceback
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _value(v):
    """A comparable record of a value: its type and, for floats, its exact bits (NaN, -0.0)."""
    if isinstance(v, np.ndarray):
        return ('array', str(v.dtype), v.tobytes())
    if isinstance(v, (float, np.floating)):
        return (type(v).__name__, struct.pack('<d', float(v)))
    return (type(v).__name__, v)


def outcome(f, args):
    try:
        r = f(*args)
    except Exception as e:  # noqa: BLE001
        frames = traceback.extract_tb(e.__traceback__)
        line = frames[-1].lineno if frames else None
        return ('raises', type(e).__name__, str(e), line)
    return ('returns', _value(r))


def same(f, *args):
    """Run f compiled and f.__wrapped__ on copies of args; require identical outcomes."""
    a1 = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    a2 = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # numpy's warnings on scalar inf/nan
        got = outcome(f, a1)
        want = outcome(f.__wrapped__, a2)
    assert got == want, f'{f.__name__}{tuple(args)!r}: compiled {got} != python {want}'
    for x, y in zip(a1, a2):
        if isinstance(x, np.ndarray):
            assert x.tobytes() == y.tobytes(), f'{f.__name__}: arrays differ after the call'
    return got


INTS = [0, 1, -1, 2, -2, 3, 7, -7, 10, 1 << 31, -(1 << 31), (1 << 53) + 1, -(1 << 53) - 1]
FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.75, 1e300, -1e-300, math.inf, -math.inf, math.nan,
          float(1 << 53), 2.0 ** 63, -(2.0 ** 63)]
