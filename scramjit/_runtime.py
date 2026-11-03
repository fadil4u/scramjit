"""The C emitted ahead of every compiled function, and the exceptions its status codes stand for.

Calling convention: `int f(args..., int64_t *sj_io)`. The int is 0 or a status below. sj_io has
at least 5 slots: [0] the return value's bits, [1] the Python line of a raised exception, [2],
[3] and [4] details (IndexError: the index, the size and the axis; others: a message number).
Tuples return through slots 4 onward.
"""

import errno
import os
import struct

INDEX_ERROR, ZERO_DIVISION, VALUE_ERROR, OVERFLOW_ERROR, LIST_INDEX_ERROR, MEMORY_ERROR, INT_BOUNDS_ERROR = \
    1, 2, 3, 4, 5, 6, 7
INTERRUPTED = 8      # the kernel saw Ctrl-C (see _interrupt.py)
RERUN = 9            # the kernel did nothing and this call should run as Python
BROADCAST_ERROR = 10  # assigning an array to a slice: io[2] and io[3] hold the shapes, two dimensions packed in each
MAKE_ERROR = 64      # allocating an array failed: 64 + index in MAKE_DTYPES, with the shape in io[0], io[2], io[3], io[4] (NO_AXIS: no such axis)
NO_AXIS = -(1 << 63)
ARANGE_ERROR = 11    # np.arange can't compute a length: its numbers in io[0], io[2], io[3], how to read each in io[4]
LINSPACE_ERROR = 12  # np.linspace of a number of samples, in io[2], that is negative or too many
MAKE_DTYPES = ['float64', 'int64', 'float32', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8']
RETURNS_INT = 2      # a returned float's flag, when the value is really a Python int
NARROW_INT_DTYPES = ['int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8']

LIST_INDEX_MESSAGES = ['list index out of range', 'list assignment index out of range']

ZERO_DIVISION_MESSAGES = [
    'division by zero',                       # int / int
    'float division by zero',                 # / with a float
    'integer division or modulo by zero',     # int // int
    'integer modulo by zero',                 # int % int
    'float floor division by zero',           # // with a float
    'float modulo by zero',                   # % with a float
    '0.0 cannot be raised to a negative power',  # 0.0 ** negative
]
VALUE_ERROR_MESSAGES = [
    'cannot convert float NaN to integer',
    'range() arg 3 must not be zero',
    'min() iterable argument is empty',
    'max() iterable argument is empty',
    'math domain error',
    'factorial() not defined for negative values',
    'isqrt() argument must be nonnegative',
    'n must be a non-negative integer',
    'k must be a non-negative integer',
    'tolerances must be non-negative',
    'both points must have the same number of dimensions',
    'Inputs are not the same length',
    'steps must be a non-negative integer',
    'invalid operation in fma',
    '-inf + inf in fsum',
    'Integers to negative integer powers are not allowed.',     # numpy
    'n ** k with a negative k: Python gives a float here, which compiled code can\'t; write float(n) ** k '
    '(see "Gotchas" in the docs)',
    'x ** y with a negative x and a fractional y: Python gives a complex here, which compiled code can\'t '
    '(see "Gotchas" in the docs)',
    'slice step cannot be zero',                                # numpy
    'zero-size array to reduction operation maximum which has no identity',    # numpy
    'zero-size array to reduction operation minimum which has no identity',
    'attempt to get argmax of an empty sequence',
    'attempt to get argmin of an empty sequence',
    'negative shift count',
]
OVERFLOW_ERROR_MESSAGES = [
    'cannot convert float infinity to integer',
    'math range error',
    'overflow in fma',
    'intermediate overflow in fsum',
    None,                                     # float ** overflowing: OverflowError(errno.ERANGE, ...)
]

# Math functions CPython takes from the platform's C library. Compiled code calls the same
# functions, through a table of their addresses in the host library (see _dispatch._libm_table),
# wrapped in CPython's own handling of special values and errors (mathmodule.c).
LIBM = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh', 'sinh', 'cosh',
        'tanh', 'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p', 'cbrt', 'erf', 'erfc',
        'atan2', 'pow', 'ldexp', 'nextafter', 'hypot', 'fmod']


# Python's wording changes between versions (`1.0 % 0.0` says "float modulo" before 3.13): use this one's
for _i, _f in enumerate((lambda: 1 / 0, lambda: 1.0 / 0.0, lambda: 1 // 0, lambda: 1 % 0, lambda: 1.0 // 0.0, lambda: 1.0 % 0.0, lambda: 0.0 ** -1.0)):
    try:
        _f()
    except ZeroDivisionError as _e:
        ZERO_DIVISION_MESSAGES[_i] = str(_e)


def _numpys_error(make, args):
    """The exception numpy raises when asked for this, or a plain MemoryError if it happens to succeed."""
    try:
        make(*args)
    except Exception as e:  # noqa: BLE001
        return e.with_traceback(None)
    return MemoryError()


def exception(status, io):
    """The Python exception for a nonzero status and the sj_io slots."""
    if status == INTERRUPTED:           # Python's handler has a signal pending: it raises at a loop back edge
        for _ in range(1000):
            pass
        return KeyboardInterrupt()
    if status == INDEX_ERROR:
        return IndexError(f'index {io[2]} is out of bounds for axis {io[4]} with size {io[3]}')
    if status == ZERO_DIVISION:
        return ZeroDivisionError(ZERO_DIVISION_MESSAGES[io[2]])
    if status == VALUE_ERROR:
        return ValueError(VALUE_ERROR_MESSAGES[io[2]])
    if status == OVERFLOW_ERROR:
        if OVERFLOW_ERROR_MESSAGES[io[2]] is None:      # float_pow: the C library's errno, as CPython raises it
            return OverflowError(errno.ERANGE, os.strerror(errno.ERANGE))
        return OverflowError(OVERFLOW_ERROR_MESSAGES[io[2]])
    if status == LIST_INDEX_ERROR:
        return IndexError(LIST_INDEX_MESSAGES[io[2]])
    if status == MEMORY_ERROR:
        return MemoryError()
    if status == BROADCAST_ERROR:
        def shape(packed, nd):
            dims = (packed >> 32, packed & 0xffffffff)[2 - nd:]
            return '(' + ','.join(map(str, dims)) + (',)' if nd == 1 else ')')
        return ValueError(f'could not broadcast input array from shape {shape(io[2], io[4] >> 8)} into shape {shape(io[3], io[4] & 255)}')
    if MAKE_ERROR <= status < MAKE_ERROR + len(MAKE_DTYPES):     # numpy's own error, for what it was asked
        import numpy
        return _numpys_error(numpy.empty, (tuple(n for n in (io[0], io[2], io[3], io[4]) if n != NO_AXIS), MAKE_DTYPES[status - MAKE_ERROR]))
    if status == ARANGE_ERROR:          # numpy's own error, for what it was asked
        import numpy

        def number(bits, how):
            v = struct.unpack('<d', struct.pack('<q', bits))[0] if how & 1 else bits
            return numpy.float64(v) if how & 2 and how & 1 else numpy.int64(v) if how & 2 else v
        return _numpys_error(numpy.arange, [number(b, io[4] >> (2 * k)) for k, b in enumerate((io[0], io[2], io[3]))])
    if status == LINSPACE_ERROR:        # numpy's own error, for what it was asked
        import numpy
        return _numpys_error(numpy.linspace, (0.0, 1.0, io[2]))
    if status == INT_BOUNDS_ERROR:      # numpy (NEP 50): a Python int stored into a narrower dtype
        if not -(1 << 8 * struct.calcsize('l') - 1) <= io[2] < 1 << 8 * struct.calcsize('l') - 1:    # 32 bits on wasm32 and Windows
            return OverflowError('Python int too large to convert to C long')
        return OverflowError(f'Python integer {io[2]} out of bounds for {NARROW_INT_DTYPES[io[3]]}')
    return SystemError(f'scramjit: unknown status {status}')


PRELUDE = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* A slice of an extent n, as CPython's PySlice_Unpack and PySlice_AdjustIndices: its length, start and step. s != 0. */
static inline int64_t sj_slice(int has_a, int64_t a, int has_b, int64_t b, int64_t s, int64_t n, int64_t *start, int64_t *step) {
    if (s < -INT64_MAX) s = -INT64_MAX;
    if (!has_a) a = s < 0 ? INT64_MAX : 0;
    if (!has_b) b = s < 0 ? INT64_MIN : INT64_MAX;
    if (a < 0) { a += n; if (a < 0) a = s < 0 ? -1 : 0; } else if (a >= n) a = s < 0 ? n - 1 : n;
    if (b < 0) { b += n; if (b < 0) b = s < 0 ? -1 : 0; } else if (b >= n) b = s < 0 ? n - 1 : n;
    *start = a;
    *step = s;
    if (s < 0) return b < a ? (int64_t)(((uint64_t)a - (uint64_t)b - 1) / (0 - (uint64_t)s)) + 1 : 0;
    return a < b ? (int64_t)(((uint64_t)b - (uint64_t)a - 1) / (uint64_t)s) + 1 : 0;
}

#define SJ_FAIL(st, ln, a, b) do { sj_st = (st); sj_ln = (ln); sj_x = (a); sj_y = (b); goto sj_fail; } while (0)

/* int64 arithmetic that wraps instead of being undefined on overflow. Macros, not functions: a
 * function stops being inlined once its caller is large, and these are one instruction each. */
#define sj_add(a, b) ((int64_t)((uint64_t)(int64_t)(a) + (uint64_t)(int64_t)(b)))
#define sj_sub(a, b) ((int64_t)((uint64_t)(int64_t)(a) - (uint64_t)(int64_t)(b)))
#define sj_mul(a, b) ((int64_t)((uint64_t)(int64_t)(a) * (uint64_t)(int64_t)(b)))
#define sj_neg(a) ((int64_t)(0 - (uint64_t)(int64_t)(a)))
static inline int64_t sj_abs(int64_t a) { return a < 0 ? sj_neg(a) : a; }

/* Python's // and % on ints: round toward negative infinity. b != 0. */
static inline int64_t sj_floordiv(int64_t a, int64_t b) {
    if (b == -1) return sj_neg(a);
    int64_t q = a / b, r = a - q * b;
    return q - ((r != 0) & ((a < 0) != (b < 0)));
}
static inline int64_t sj_mod(int64_t a, int64_t b) {
    if (b == -1) return 0;
    int64_t r = a % b;
    return r + ((r != 0 && ((r < 0) != (b < 0))) ? b : 0);
}

/* Python's // and % on floats (the algorithm of CPython's float divmod; numpy's matches it for
 * nonzero divisors). b != 0. */
static inline double sj_fmod_py(double a, double b) {
    double m = fmod(a, b);
    if (m != 0.0) { if ((b < 0) != (m < 0)) m += b; }
    else m = copysign(0.0, b);
    return m;
}
static inline float sj_fmod_py32(float a, float b) {
    float m = (float)fmod(a, b);
    if (m != 0.0f) { if ((b < 0) != (m < 0)) m += b; }
    else m = (float)copysign(0.0, b);
    return m;
}
static inline float sj_ffloordiv32(float a, float b) {
    float m = (float)fmod(a, b);
    float div = (a - m) / b;
    if (m != 0.0f) { if ((b < 0) != (m < 0)) div -= 1.0f; }
    if (div != 0.0f) {
        float f = (float)floor(div);
        if (div - f > 0.5f) f += 1.0f;
        return f;
    }
    return (float)copysign(0.0, a / b);
}
static inline double sj_ffloordiv(double a, double b) {
    double m = fmod(a, b);
    double div = (a - m) / b;
    if (m != 0.0) { if ((b < 0) != (m < 0)) div -= 1.0; }
    if (div != 0.0) {
        double f = floor(div);
        if (div - f > 0.5) f += 1.0;
        return f;
    }
    return copysign(0.0, a / b);
}

/* int / int, correctly rounded for every int64, as Python's true division is. b != 0. */
static int sj_bitlen(uint64_t x) { int n = 0; while (x) { n++; x >>= 1; } return n; }
static double sj_truediv_big(int64_t a, int64_t b) {
    int neg = (a < 0) != (b < 0);
    uint64_t ua = a < 0 ? 0 - (uint64_t)a : (uint64_t)a;
    uint64_t ub = b < 0 ? 0 - (uint64_t)b : (uint64_t)b;
    uint64_t m = ua / ub, rem = ua % ub;
    int e = 0;
    int bits = sj_bitlen(m);
    if (bits > 55) {                       /* drop low bits of the quotient into the sticky bit */
        int s = bits - 55;
        if (m & ((((uint64_t)1) << s) - 1)) rem |= 1;
        m >>= s; e += s;
    } else {
        while (sj_bitlen(m) < 55) {        /* long division for the missing fraction bits */
            rem <<= 1; m <<= 1; e -= 1;
            if (rem >= ub) { rem -= ub; m |= 1; }
        }
    }
    uint64_t low = m & 3;                  /* 55 bits -> 53, round half to even */
    m >>= 2; e += 2;
    if (low > 2 || (low == 2 && (rem != 0 || (m & 1)))) m += 1;
    double r = ldexp((double)m, e);
    return neg ? -r : r;
}

/* Compare an int with a float exactly, as Python does (no rounding of the int). Returns -1, 0
 * or 1, or 2 if f is NaN (every ordering false). */
static inline int sj_cmp_if(int64_t i, double f) {
    if (f != f) return 2;
    const int64_t lim = (int64_t)1 << 53;
    if ((i <= lim && i >= -lim) || fabs(f) < 9007199254740992.0) {
        double d = (double)i;
        return d < f ? -1 : d > f ? 1 : 0;
    }
    if (f >= 9223372036854775808.0) return -1;
    if (f < -9223372036854775808.0) return 1;
    int64_t fi = (int64_t)f;               /* exact: |f| >= 2**53 is an integer */
    return i < fi ? -1 : i > fi ? 1 : 0;
}

/* the result of an ordering test op (0 <, 1 <=, 2 >, 3 >=, 4 ==, 5 !=) from sj_cmp_if's r */
static inline int sj_cmp_test(int r, int op) {
    switch (op) {
    case 0: return r == -1;
    case 1: return r == -1 || r == 0;
    case 2: return r == 1;
    case 3: return r == 0 || r == 1;
    case 4: return r == 0;
    default: return r != 0;
    }
}

/* int(f) for a float that isn't NaN or infinite: truncate; beyond int64, wrap like every int. */
static inline int64_t sj_float_to_int(double f) {
    double t = trunc(f);
    if (t >= -9223372036854775808.0 && t < 9223372036854775808.0) return (int64_t)t;
    double w = fmod(t, 18446744073709551616.0);
    if (w < 0) w += 18446744073709551616.0;
    uint64_t u = w >= 9223372036854775808.0 ? (uint64_t)(w - 9223372036854775808.0) + ((uint64_t)1 << 63)
                                            : (uint64_t)w;
    return (int64_t)u;
}
/* int / int: exact in double when both are within 2**53, else the long path */
static inline double sj_truediv(int64_t a, int64_t b) {
    return a <= 9007199254740992LL && a >= -9007199254740992LL && b <= 9007199254740992LL && b >= -9007199254740992LL
        ? (double)a / (double)b : sj_truediv_big(a, b);
}
/* ---- math: CPython's mathmodule.c wrappers around the C library (PSF license). The library
 * functions themselves come from the host, through sj_m (the addresses, in LIBM's order). */
typedef double (*sj_f1)(double);
typedef double (*sj_f2)(double, double);
#define SJ_LIBM1(i) ((sj_f1)(uintptr_t)sj_m[i])
#define SJ_LIBM2(i) ((sj_f2)(uintptr_t)sj_m[i])

/* m_log, m_log2, m_log10: special values directly, positive finite values to the library */
static double sj_m_log(sj_f1 f, double x) {
    if (isfinite(x)) {
        if (x > 0.0) return f(x);
        if (x == 0.0) return -INFINITY;
        return NAN;
    }
    if (x != x) return x;
    if (x > 0.0) return x;
    return NAN;
}

/* m_atan2 */
static double sj_m_atan2(sj_f2 f, double y, double x) {
    const double pi = 3.141592653589793;
    if (x != x || y != y) return NAN;
    if (isinf(y)) {
        if (isinf(x)) return copysign(copysign(1., x) == 1. ? 0.25 * pi : 0.75 * pi, y);
        return copysign(0.5 * pi, y);
    }
    if (isinf(x) || y == 0.) return copysign(copysign(1., x) == 1. ? 0. : pi, y);
    return f(y, x);
}

/* math.pow: returns 0, or 1 for ValueError, 2 for OverflowError */
static int sj_math_pow(sj_f2 f, double x, double y, double *out) {
    double r = 0.;
    int err = 0;                                  /* 1: EDOM, 2: ERANGE */
    if (!isfinite(x) || !isfinite(y)) {
        if (x != x) r = y == 0. ? 1. : x;
        else if (y != y) r = x == 1. ? 1. : y;
        else if (isinf(x)) {
            int odd_y = isfinite(y) && fmod(fabs(y), 2.0) == 1.0;
            if (y > 0.) r = odd_y ? x : fabs(x);
            else if (y == 0.) r = 1.;
            else r = odd_y ? copysign(0., x) : 0.;
        } else {
            if (fabs(x) == 1.0) r = 1.;
            else if (y > 0. && fabs(x) > 1.0) r = y;
            else if (y < 0. && fabs(x) < 1.0) r = -y;
            else r = 0.;
        }
    } else {
        r = f(x, y);
        if (!isfinite(r)) {
            if (r != r) err = 1;
            else if (isinf(r)) err = x == 0. ? 1 : 2;
        }
    }
    *out = r;
    if (err == 2 && fabs(r) < 1.5) err = 0;       /* is_error: underflow isn't an error */
    return err;
}

/* math.fmod: returns 0, or 1 for ValueError */
static int sj_math_fmod(double x, double y, double *out) {
    if (isinf(y) && isfinite(x)) { *out = x; return 0; }
    double r = fmod(x, y);
    *out = r;
    return (r != r && x == x && y == y) ? 1 : 0;
}

/* vector_norm (math.hypot): error-free squares and sums, then a corrected sqrt */
static double sj_vector_norm(int64_t n, double *vec, double max, int found_nan) {
    double x, h, scale, csum = 1.0, frac1 = 0.0, frac2 = 0.0, hi, lo, s;
    int max_e;
    if (isinf(max)) return max;
    if (found_nan) return NAN;
    if (max == 0.0 || n <= 1) return max;
    frexp(max, &max_e);
    if (max_e < -1023) {
        for (int64_t i = 0; i < n; i++) vec[i] /= 2.2250738585072014e-308;
        return 2.2250738585072014e-308 * sj_vector_norm(n, vec, max / 2.2250738585072014e-308, found_nan);
    }
    scale = ldexp(1.0, -max_e);
    for (int64_t i = 0; i < n; i++) {
        x = vec[i] * scale;
        hi = x * x; lo = fma(x, x, -hi);          /* dl_mul */
        s = csum + hi; frac2 += (csum - s) + hi;  /* dl_fast_sum */
        csum = s;
        frac1 += lo;
    }
    h = sqrt(csum - 1.0 + (frac1 + frac2));
    hi = -h * h; lo = fma(-h, h, -hi);
    s = csum + hi; frac2 += (csum - s) + hi;
    csum = s;
    frac1 += lo;
    x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return h / scale;
}
'''
