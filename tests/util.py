"""Differential testing: a compiled function must behave exactly like the same function in CPython."""

import math
import os
import struct
import traceback
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def bits(a):
    """An array's bytes, every NaN made the same one. A NaN's sign and payload from arithmetic aren't Python's
    rule: they follow the C compiler CPython was built with (Ubuntu's 3.12 gives nan * -1 a sign bit, a
    python.org build on the same machine doesn't) and the CPU (x86-64's default NaN has its sign set)."""
    if a.dtype.kind in 'fc' and np.isnan(a).any():
        a = a.copy()
        a[np.isnan(a)] = np.nan
    return a.tobytes()


def _value(v):
    """A comparable record of a value: its type and, for floats, its exact bits (-0.0; any NaN is 'nan', see bits)."""
    if isinstance(v, np.ndarray):
        return ('array', str(v.dtype), bits(v))
    if isinstance(v, (float, np.floating)):
        return (type(v).__name__, 'nan' if v != v else struct.pack('<d', float(v)))
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
            assert bits(x) == bits(y), f"{f.__name__}: arrays differ after the call"
    return got


INTS = [0, 1, -1, 2, -2, 3, 7, -7, 10, 1 << 31, -(1 << 31), (1 << 53) + 1, -(1 << 53) - 1]
FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.75, 1e300, -1e-300, math.inf, -math.inf, math.nan,
          float(1 << 53), 2.0 ** 63, -(2.0 ** 63)]


def maxrss_kb():
    """Peak resident memory in KB: getrusage gives KB on Linux and bytes on macOS."""
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // 1024 if sys.platform == 'darwin' else peak


def _numpy_orders_zeros():
    """Does this numpy order zeros as IEEE 754-2019 does, min(-0.0, 0.0) -0.0 and max 0.0 whichever comes first, as
    compiled code does? numpy on arm64 Linux does; its x86-64 SIMD returns the second operand on a tie, as minps does,
    and other builds have their own rules, so there a zero's sign from min and max is the numpy build's choice."""
    ties = ((0.0, -0.0), (-0.0, 0.0))
    return all(np.signbit(lo(a, b)) and not np.signbit(hi(a, b)) for a, b in ties
               for lo, hi in ((np.minimum, np.maximum), (np.fmin, np.fmax)))


NUMPY_ORDERS_ZEROS = _numpy_orders_zeros()


def zero_ties(v):
    """v with a zero's sign dropped where this numpy's min and max don't order zeros (see _numpy_orders_zeros)."""
    if NUMPY_ORDERS_ZEROS:
        return v
    if isinstance(v, tuple):
        return tuple(zero_ties(x) for x in v)
    if isinstance(v, bytes) and len(v) == 8 and v in (struct.pack('<d', 0.0), struct.pack('<d', -0.0)):
        return 'zero'
    return v
