"""The C emitted ahead of every compiled function, and the exceptions its status codes stand for.

Calling convention: `int f(args..., int64_t *sj_io)`. The int is 0 or a status below. sj_io has
4 slots: [0] the return value's bits, [1] the Python line of a raised exception, [2] and [3]
details (IndexError: the index and the size; others: a message number).
"""

INDEX_ERROR, ZERO_DIVISION, VALUE_ERROR, OVERFLOW_ERROR = 1, 2, 3, 4

ZERO_DIVISION_MESSAGES = [
    'division by zero',                       # int / int
    'float division by zero',                 # / with a float
    'integer division or modulo by zero',     # int // int
    'integer modulo by zero',                 # int % int
    'float floor division by zero',           # // with a float
    'float modulo by zero',                   # % with a float
]
VALUE_ERROR_MESSAGES = [
    'cannot convert float NaN to integer',
    'range() arg 3 must not be zero',
]
OVERFLOW_ERROR_MESSAGES = [
    'cannot convert float infinity to integer',
]


def exception(status, io):
    """The Python exception for a nonzero status and the sj_io slots."""
    if status == INDEX_ERROR:
        return IndexError(f'index {io[2]} is out of bounds for axis 0 with size {io[3]}')
    if status == ZERO_DIVISION:
        return ZeroDivisionError(ZERO_DIVISION_MESSAGES[io[2]])
    if status == VALUE_ERROR:
        return ValueError(VALUE_ERROR_MESSAGES[io[2]])
    if status == OVERFLOW_ERROR:
        return OverflowError(OVERFLOW_ERROR_MESSAGES[io[2]])
    return SystemError(f'scramjit: unknown status {status}')


PRELUDE = r'''
#include <stdint.h>
#include <string.h>
#include <math.h>

#define SJ_FAIL(st, ln, a, b) do { sj_st = (st); sj_ln = (ln); sj_x = (a); sj_y = (b); goto sj_fail; } while (0)

/* int64 arithmetic that wraps instead of being undefined on overflow */
static inline int64_t sj_add(int64_t a, int64_t b) { return (int64_t)((uint64_t)a + (uint64_t)b); }
static inline int64_t sj_sub(int64_t a, int64_t b) { return (int64_t)((uint64_t)a - (uint64_t)b); }
static inline int64_t sj_mul(int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); }
static inline int64_t sj_neg(int64_t a) { return (int64_t)(0 - (uint64_t)a); }
static inline int64_t sj_abs(int64_t a) { return a < 0 ? sj_neg(a) : a; }

/* Python's // and % on ints: round toward negative infinity. b != 0. */
static inline int64_t sj_floordiv(int64_t a, int64_t b) {
    if (b == -1) return sj_neg(a);
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}
static inline int64_t sj_mod(int64_t a, int64_t b) {
    if (b == -1) return 0;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

/* Python's // and % on floats (the algorithm of CPython's float divmod; numpy's matches it for
 * nonzero divisors). b != 0. */
static inline double sj_fmod_py(double a, double b) {
    double m = fmod(a, b);
    if (m != 0.0) { if ((b < 0) != (m < 0)) m += b; }
    else m = copysign(0.0, b);
    return m;
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
static inline int sj_bitlen(uint64_t x) { int n = 0; while (x) { n++; x >>= 1; } return n; }
static inline double sj_truediv(int64_t a, int64_t b) {
    const int64_t lim = (int64_t)1 << 53;
    if (a <= lim && a >= -lim && b <= lim && b >= -lim) return (double)a / (double)b;
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
'''
