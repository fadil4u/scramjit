"""numpy's own loops, for the functions where numpy's results aren't the C library's.

On x86-64 with AVX-512, numpy computes exp, log, sin, power and others with Intel's SVML, whose last bits differ from
the C library compiled code calls. For each function, the numpy present is asked once, over a few thousand values,
whether its results are the C library's. Where they aren't, compiled code calls numpy's own loop for that function and
dtype, one element at a time, so it gets numpy's bits, whatever CPU numpy dispatched for.

A ufunc's loops are read from its C struct (PyUFuncObject in numpy's ufuncobject.h, a public layout), checked against
what the ufunc says of itself in Python before anything is used.
"""

import math
import os
import struct
import sys

# (numpy's name, signature): each takes two slots after LIBM's in the table a kernel gets (_dispatch._libm_table):
# the loop's address and its data pointer
LOOPS = (('exp', 'd->d'), ('exp2', 'd->d'), ('expm1', 'd->d'), ('log', 'd->d'), ('log2', 'd->d'), ('log10', 'd->d'),
         ('log1p', 'd->d'), ('sin', 'd->d'), ('cos', 'd->d'), ('tan', 'd->d'), ('arcsin', 'd->d'), ('arccos', 'd->d'),
         ('arctan', 'd->d'), ('sinh', 'd->d'), ('cosh', 'd->d'), ('arcsinh', 'd->d'), ('arccosh', 'd->d'),
         ('arctanh', 'd->d'), ('cbrt', 'd->d'), ('arctan2', 'dd->d'), ('hypot', 'dd->d'), ('fmod', 'dd->d'),
         ('power', 'dd->d'), ('power', 'ff->f'))
# the C library function compiled code calls otherwise (LIBM's names)
LIBM_OF = {'exp': 'exp', 'exp2': 'exp2', 'expm1': 'expm1', 'log': 'log', 'log2': 'log2', 'log10': 'log10',
           'log1p': 'log1p', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'arcsin': 'asin', 'arccos': 'acos',
           'arctan': 'atan', 'sinh': 'sinh', 'cosh': 'cosh', 'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh',
           'cbrt': 'cbrt', 'arctan2': 'atan2', 'hypot': 'hypot', 'fmod': 'fmod', 'power': 'pow'}
CANONICAL = {'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan', 'asinh': 'arcsinh', 'acosh': 'arccosh',
             'atanh': 'arctanh', 'atan2': 'arctan2'}

_loops = {}
_differs = {}


def loop(name, sig):
    """(address, data pointer) of the numpy present's loop for name and sig, or None."""
    key = (name, sig)
    if key not in _loops:
        _loops[key] = _read(name, sig)
    return _loops[key]


def _read(name, sig):
    np = sys.modules.get('numpy')
    if np is None or sys.platform == 'emscripten' or struct.calcsize('P') != 8:
        return None
    try:
        import ctypes
        uf = getattr(np, name)
        if sig not in uf.types:
            return None
        head, base = object.__basicsize__, id(uf)

        def rd(off, t):
            return t.from_address(base + off).value
        nin, nout, nargs, ntypes = (rd(head + o, ctypes.c_int) for o in (0, 4, 8, 32))
        cname = ctypes.c_char_p(rd(head + 40, ctypes.c_void_p)).value
        if (nin, nout, nargs, ntypes, cname) != (uf.nin, uf.nout, uf.nargs, uf.ntypes, uf.__name__.encode()):
            return None
        i = uf.types.index(sig)
        funcs, data = rd(head + 16, ctypes.c_void_p), rd(head + 24, ctypes.c_void_p)
        f = ctypes.c_void_p.from_address(funcs + 8 * i).value
        d = ctypes.c_void_p.from_address(data + 8 * i).value if data else 0
        return (f, d or 0) if f else None
    except Exception:                                   # noqa: BLE001  (another layout: use the C library)
        return None


def _sample(two):
    import random
    rnd = random.Random(7)
    xs = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, 3.0, 0.1, 1e-300, 1e300, 5e-324, math.pi, 1 - 2 ** -53, 1 + 2 ** -52]
    for _ in range(1500):
        xs.append(rnd.uniform(-1, 1))
        xs.append(rnd.uniform(-20, 20))
        xs.append(math.copysign(10.0 ** rnd.uniform(-300, 300), rnd.random() - 0.5))
    if not two:
        return [(x,) for x in xs]
    return [(x, rnd.choice([0.5, 2.0, -1.5, 3.0, 0.3, 1.7, -0.25]) if i % 2 else xs[-1 - i]) for i, x in enumerate(xs)]


def differs(name, sig, libm_address):
    """Does the numpy present give, for name and sig, anything but what the C library function at libm_address
    gives? Only then is numpy's loop worth calling instead."""
    key = (name, sig)
    if key not in _differs:
        if loop(name, sig) is None:
            _differs[key] = False
        elif os.environ.get('SCRAMJIT_NUMPY_LOOPS') == '1':     # tests: numpy's loops wherever there are any
            _differs[key] = True
        else:
            _differs[key] = _probe(name, sig, libm_address)
    return _differs[key]


_pow = {}


def pow_needs_loop(ka, kb, sig, libm_address):
    """For a ** b with a and b of these kinds ('float32', 'int32', ... for numpy scalars, 'int' or 'float' for Python
    numbers), does numpy give what its power loop for sig gives, and not what the C library's pow does? Asked of numpy
    itself, as the very operation: its scalar math computes some combinations with the C library, others not."""
    key = (ka, kb, sig)
    if key not in _pow:
        if loop('power', sig) is None:
            _pow[key] = False
        else:
            libm_ok, loop_ok = _probe_pow(ka, kb, sig, libm_address)
            forced = os.environ.get('SCRAMJIT_NUMPY_LOOPS') == '1'
            _pow[key] = loop_ok and (forced or not libm_ok)
    return _pow[key]


def _probe_pow(ka, kb, sig, libm_address):
    """(numpy's a ** b is the C library's pow, numpy's power loop's) on these kinds. Their differences are rare (SVML's
    power is within an ulp), so they are found first, with numpy's loop over many inputs at once, and numpy's own a ** b
    is then asked on those. If the two never differ, there is nothing to tell apart: (True, True)."""
    import ctypes
    import random
    np = sys.modules['numpy']
    f32 = sig == 'ff->f'
    ct = ctypes.c_float if f32 else ctypes.c_double
    lib = ctypes.CFUNCTYPE(ct, ct, ct)(libm_address)
    dt = np.float32 if f32 else np.float64
    ints = {'int': (-(1 << 40), 1 << 40), 'bool': (0, 1)}

    def is_int(k):
        return k in ints or np.dtype(k).kind in 'iu' if k not in ('float',) else False

    def bounds(k):
        if k in ints:
            return ints[k]
        info = np.iinfo(k)
        return int(info.min), int(info.max)

    def make(k, v):
        return {'int': int, 'float': float, 'bool': bool}[k](v) if k in ('int', 'float', 'bool') else getattr(np, k)(v)
    rnd = random.Random(11)
    xs, ys = [], []
    for n in range(20000):
        if is_int(ka):
            lo, hi = bounds(ka)
            x = rnd.choice([hi, max(lo, 1), rnd.randint(max(lo, 1), max(1, min(hi, 100))), rnd.randint(max(lo, 1), hi)])
        else:
            x = rnd.uniform(0.01, 60) if n % 2 else 10.0 ** rnd.uniform(-35 if f32 else -300, 35 if f32 else 300)
        if is_int(kb):
            y = rnd.randint(0, 9)
        else:
            y = rnd.uniform(-4, 6) if n % 3 else rnd.choice([-1.0, 0.5, -0.5, 2.0, 1.5, -2.5, 2.5, 0.25])
        xs.append(x)
        ys.append(y)
    f, d = loop('power', sig)
    uloop = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(f)
    one, zeros = (ctypes.c_int64 * 1)(1), (ctypes.c_int64 * 3)(0, 0, 0)

    def via_loop(x, y):                                 # as compiled code calls it: one element, strides 0
        p, q, o = ct(x), ct(y), ct()
        uloop((ctypes.c_void_p * 3)(ctypes.addressof(p), ctypes.addressof(q), ctypes.addressof(o)), one, zeros, d)
        return o.value
    with np.errstate(all='ignore'):
        a = [float(dt(make(ka, x))) for x in xs]
        b = [float(dt(make(kb, y))) for y in ys]
        by_loop = [float(dt(via_loop(p, q))) for p, q in zip(a, b)]
        apart = [k for k in range(len(xs)) if struct.pack('<d', by_loop[k]) != struct.pack('<d', float(dt(lib(a[k], b[k]))))]
        if not apart:
            return True, True
        libm_ok = loop_ok = True
        for k in apart[:200]:
            try:
                want = make(ka, xs[k]) ** make(kb, ys[k])
            except Exception:                           # noqa: BLE001  (ints to negative powers and the like)
                continue
            if not isinstance(want, (float, np.floating)):
                return False, False
            w = struct.pack('<d', float(want))
            libm_ok = libm_ok and struct.pack('<d', float(dt(lib(a[k], b[k])))) == w
            loop_ok = loop_ok and struct.pack('<d', by_loop[k]) == w
    return libm_ok, loop_ok


def _probe(name, sig, libm_address):
    import ctypes
    np = sys.modules['numpy']
    f32 = sig == 'ff->f'
    two = sig.index('-') == 2
    ct = ctypes.c_float if f32 else ctypes.c_double
    lib = ctypes.CFUNCTYPE(ct, *([ct] * (2 if two else 1)))(libm_address)
    dt = np.float32 if f32 else np.float64
    rows = _sample(two)
    args = [np.array([r[k] for r in rows], dt) for k in range(len(rows[0]))]
    with np.errstate(all='ignore'):
        want = getattr(np, name)(*args)
    for k, r in enumerate(rows):
        got = lib(*(float(dt(v)) for v in r))
        w = float(want[k])
        if (w != w) != (got != got) or (w == w and struct.pack('<d', w) != struct.pack('<d', got)):
            return True
    return False
