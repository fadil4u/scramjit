"""math functions CPython implements itself, ported to C from Modules/mathmodule.c.

Portions copyright Python Software Foundation, under the PSF License Agreement. A kernel gets only
the helpers it uses (SNIPPETS, with their dependencies), so these add nothing to other compiles.
"""

from ._runtime import LIBM


def _f1(name):
    return f'((sj_f1)(uintptr_t)sj_m[{LIBM.index(name)}])'


def _f2(name):
    return f'((sj_f2)(uintptr_t)sj_m[{LIBM.index(name)}])'


# Each returns 0, or 1 for a domain error and 2 for a range error, with the result in *r.
SNIPPETS = {
    'sinpi': ((), f"""
static double sj_m_sinpi(const uint64_t *sj_m, double x) {{
    const double pi = 3.141592653589793238462643383279502884197;
    double y = fmod(fabs(x), 2.0), r = 0.0;
    int n = (int)round(2.0 * y);
    switch (n) {{
    case 0: r = {_f1('sin')}(pi * y); break;
    case 1: r = {_f1('cos')}(pi * (y - 0.5)); break;
    case 2: r = {_f1('sin')}(pi * (1.0 - y)); break;
    case 3: r = -{_f1('cos')}(pi * (y - 1.5)); break;
    case 4: r = {_f1('sin')}(pi * (y - 2.0)); break;
    }}
    return copysign(1.0, x) * r;
}}
"""),
    'lanczos': ((), """
static double sj_lanczos_sum(double x) {
    static const double nc[13] = {23531376880.410759688572007674451636754734846804940, 42919803642.649098768957899047001988850926355848959, 35711959237.355668049440185451547166705960488635843, 17921034426.037209699919755754458931112671403265390, 6039542586.3520280050642916443072979210699388420708, 1439720407.3117216736632230727949123939715485786772, 248874557.86205415651146038641322942321632125127801, 31426415.585400194380614231628318205362874684987640, 2876370.6289353724412254090516208496135991145378768, 186056.26539522349504029498971604569928220784236328, 8071.6720023658162106380029022722506138218516325024, 210.82427775157934587250973392071336271166969580291, 2.5066282746310002701649081771338373386264310793408};
    static const double dc[13] = {0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0, 13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0};
    double num = 0.0, den = 0.0;
    if (x < 5.0) {
        for (int i = 13; --i >= 0; ) { num = num * x + nc[i]; den = den * x + dc[i]; }
    } else {
        for (int i = 0; i < 13; i++) { num = num / x + nc[i]; den = den / x + dc[i]; }
    }
    return num / den;
}
"""),
    'gamma': (('sinpi', 'lanczos'), f"""
static int sj_m_tgamma(const uint64_t *sj_m, double x, double *out) {{
    static const double gi[23] = {{1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0, 1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0, 121645100408832000.0, 2432902008176640000.0, 51090942171709440000.0, 1124000727777607680000.0,}};
    const double g = 6.024680040776729583740234375, gmh = 5.524680040776729583740234375;
    const double pi = 3.141592653589793238462643383279502884197;
    double absx, r, y, z, sqrtpow;
    if (!isfinite(x)) {{
        if (x != x || x > 0.0) {{ *out = x; return 0; }}
        *out = NAN; return 1;
    }}
    if (x == 0.0) {{ *out = copysign(INFINITY, x); return 1; }}
    if (x == floor(x)) {{
        if (x < 0.0) {{ *out = NAN; return 1; }}
        if (x <= 23.0) {{ *out = gi[(int)x - 1]; return 0; }}
    }}
    absx = fabs(x);
    if (absx < 1e-20) {{ r = 1.0 / x; *out = r; return isinf(r) ? 2 : 0; }}
    if (absx > 200.0) {{
        if (x < 0.0) {{ *out = 0.0 / sj_m_sinpi(sj_m, x); return 0; }}
        *out = INFINITY; return 2;
    }}
    y = absx + gmh;
    if (absx > gmh) {{ double q = y - absx; z = q - gmh; }}
    else {{ double q = y - gmh; z = q - absx; }}
    z = z * g / y;
    if (x < 0.0) {{
        r = -pi / sj_m_sinpi(sj_m, absx) / absx * {_f1('exp')}(y) / sj_lanczos_sum(absx);
        r -= z * r;
        if (absx < 140.0) r /= {_f2('pow')}(y, absx - 0.5);
        else {{ sqrtpow = {_f2('pow')}(y, absx / 2.0 - 0.25); r /= sqrtpow; r /= sqrtpow; }}
    }} else {{
        r = sj_lanczos_sum(absx) / {_f1('exp')}(y);
        r += z * r;
        if (absx < 140.0) r *= {_f2('pow')}(y, absx - 0.5);
        else {{ sqrtpow = {_f2('pow')}(y, absx / 2.0 - 0.25); r *= sqrtpow; r *= sqrtpow; }}
    }}
    *out = r;
    return isinf(r) ? 2 : 0;
}}
"""),
    'lgamma': (('sinpi', 'lanczos'), f"""
static int sj_m_lgamma(const uint64_t *sj_m, double x, double *out) {{
    const double g = 6.024680040776729583740234375, logpi = 1.144729885849400174143427351353058711647;
    double r, absx;
    if (!isfinite(x)) {{ *out = x != x ? x : INFINITY; return 0; }}
    if (x == floor(x) && x <= 2.0) {{
        if (x <= 0.0) {{ *out = INFINITY; return 1; }}
        *out = 0.0; return 0;
    }}
    absx = fabs(x);
    if (absx < 1e-20) {{ *out = -{_f1('log')}(absx); return 0; }}
    r = {_f1('log')}(sj_lanczos_sum(absx)) - g;
    r += (absx - 0.5) * ({_f1('log')}(absx + g - 0.5) - 1);
    if (x < 0.0) r = logpi - {_f1('log')}(fabs(sj_m_sinpi(sj_m, absx))) - {_f1('log')}(absx) - r;
    *out = r;
    return isinf(r) ? 2 : 0;
}}
"""),
    'remainder': ((), """
static double sj_m_remainder(double x, double y) {
    if (isfinite(x) && isfinite(y)) {
        double absx, absy, c, m, r;
        if (y == 0.0) return NAN;
        absx = fabs(x); absy = fabs(y);
        m = fmod(absx, absy);
        c = absy - m;
        if (m < c) r = m;
        else if (m > c) r = -c;
        else r = m - 2.0 * fmod(0.5 * (absx - m), absy);
        return copysign(1.0, x) * r;
    }
    if (x != x) return x;
    if (y != y) return y;
    if (isinf(x)) return NAN;
    return x;
}
"""),
    'nextafter': ((), """
static double sj_nextafter(double x, double y, uint64_t steps) {
    union { double f; uint64_t i; } ux, uy, res;
    const uint64_t sign = (uint64_t)1 << 63;
    if (steps == 0 || x != x) return x;
    if (y != y) return y;
    ux.f = x; uy.f = y;
    if (ux.i == uy.i) return x;
    uint64_t ax = ux.i & ~sign, ay = uy.i & ~sign;
    if ((ux.i ^ uy.i) & sign) {
        if (ax + ay <= steps) return y;
        if (ax < steps) { res.i = (uy.i & sign) | (steps - ax); return res.f; }
        ux.i -= steps;
        return ux.f;
    }
    if (ax > ay) {
        if (ax - ay >= steps) { ux.i -= steps; return ux.f; }
        return y;
    }
    if (ay - ax >= steps) { ux.i += steps; return ux.f; }
    return y;
}
"""),
    'ulp': (('nextafter',), """
static double sj_ulp(double x) {
    if (x != x) return x;
    x = fabs(x);
    if (isinf(x)) return x;
    double x2 = sj_nextafter(x, INFINITY, 1);
    if (isinf(x2)) return x - sj_nextafter(x, -INFINITY, 1);
    return x2 - x;
}
"""),
    'frexp': ((), """
/* finite and nonzero x: exact, so any correct implementation gives the C library's result */
static double sj_frexp(double x, int64_t *ex) {
    union { double f; uint64_t i; } u;
    int64_t adjust = 0;
    u.f = x;
    if (!((u.i >> 52) & 0x7ff)) { u.f = x * 18014398509481984.0; adjust = -54; }   /* subnormal: scale by 2**54 */
    *ex = (int64_t)((u.i >> 52) & 0x7ff) - 1022 + adjust;
    u.i = (u.i & 0x800fffffffffffffULL) | 0x3fe0000000000000ULL;
    return u.f;
}
"""),
    'floatpow': ((), """
/* float ** float (float_pow in floatobject.c). Returns 0, 1 for 0.0 ** negative, 2 on overflow, 3
 * where Python's result is a complex. */
static int sj_float_pow(sj_f2 f, double v, double w, double *out) {
    int negate = 0, odd;
    *out = 1.0;
    if (w == 0.0) return 0;
    if (v != v) { *out = v; return 0; }
    if (w != w) { *out = v == 1.0 ? 1.0 : w; return 0; }
    if (isinf(w)) {
        v = fabs(v);
        if (v == 1.0) *out = 1.0;
        else if ((w > 0.0) == (v > 1.0)) *out = fabs(w);
        else *out = 0.0;
        return 0;
    }
    odd = fmod(fabs(w), 2.0) == 1.0;
    if (isinf(v)) {
        if (w > 0.0) *out = odd ? v : fabs(v);
        else *out = odd ? copysign(0.0, v) : 0.0;
        return 0;
    }
    if (v == 0.0) {
        if (w < 0.0) return 1;
        *out = odd ? v : 0.0;
        return 0;
    }
    if (v < 0.0) {
        if (w != floor(w)) return 3;
        v = -v;
        negate = odd;
    }
    if (v == 1.0) { *out = negate ? -1.0 : 1.0; return 0; }
    double r = f(v, w);
    if (isinf(r)) return 2;
    *out = negate ? -r : r;
    return 0;
}
"""),
    # numpy's reductions, written from what numpy returns: tests/test_numpy_reductions.py compares the bits
    'np_sum': ((), """
/* numpy's float sum: pairwise, with blocks of 128 summed in 8 parallel accumulators. Rounding depends on this order. */
static double sj_np_pairwise(const double *a, int64_t n, int64_t st) {
    if (n < 8) {
        double r = 0.0;
        for (int64_t i = 0; i < n; i++) r += a[i * st];
        return r;
    }
    if (n <= 128) {
        double r0 = a[0], r1 = a[st], r2 = a[2 * st], r3 = a[3 * st], r4 = a[4 * st], r5 = a[5 * st], r6 = a[6 * st], r7 = a[7 * st];
        int64_t i = 8, m = n - n % 8;
        for (; i < m; i += 8) {
            r0 += a[i * st]; r1 += a[(i + 1) * st]; r2 += a[(i + 2) * st]; r3 += a[(i + 3) * st];
            r4 += a[(i + 4) * st]; r5 += a[(i + 5) * st]; r6 += a[(i + 6) * st]; r7 += a[(i + 7) * st];
        }
        double r = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; i++) r += a[i * st];
        return r;
    }
    int64_t half = n / 2;
    half -= half % 8;
    return sj_np_pairwise(a, half, st) + sj_np_pairwise(a + half * st, n - half, st);
}
/* chunk: 0 for one pass over everything, else numpy sums this many at a time (it depends on the build) */
static double sj_np_sum(const double *a, int64_t n, int64_t st, int64_t chunk) {
    if (!chunk || n <= chunk) return sj_np_pairwise(a, n, st);
    double r = 0.0;
    for (int64_t i = 0; i < n; i += chunk) r += sj_np_pairwise(a + i * st, n - i < chunk ? n - i : chunk, st);
    return r;
}
"""),
    'np_stats': (('np_sum',), """
/* The same pairwise order over a computed element: an int as a double, or a squared difference from the mean */
#define SJ_NP_PW(NAME, TYPE, ELEM) \\
static double NAME(const TYPE *a, int64_t n, int64_t st, double m) { \\
    if (n < 8) { \\
        double r = 0.0; \\
        for (int64_t i = 0; i < n; i++) r += ELEM(a[i * st], m); \\
        return r; \\
    } \\
    if (n <= 128) { \\
        double r0 = ELEM(a[0], m), r1 = ELEM(a[st], m), r2 = ELEM(a[2 * st], m), r3 = ELEM(a[3 * st], m); \\
        double r4 = ELEM(a[4 * st], m), r5 = ELEM(a[5 * st], m), r6 = ELEM(a[6 * st], m), r7 = ELEM(a[7 * st], m); \\
        int64_t i = 8, e = n - n % 8; \\
        for (; i < e; i += 8) { \\
            r0 += ELEM(a[i * st], m); r1 += ELEM(a[(i + 1) * st], m); r2 += ELEM(a[(i + 2) * st], m); r3 += ELEM(a[(i + 3) * st], m); \\
            r4 += ELEM(a[(i + 4) * st], m); r5 += ELEM(a[(i + 5) * st], m); r6 += ELEM(a[(i + 6) * st], m); r7 += ELEM(a[(i + 7) * st], m); \\
        } \\
        double r = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)); \\
        for (; i < n; i++) r += ELEM(a[i * st], m); \\
        return r; \\
    } \\
    int64_t half = n / 2; \\
    half -= half % 8; \\
    return NAME(a, half, st, m) + NAME(a + half * st, n - half, st, m); \\
}
#define SJ_NP_AS_DOUBLE(v, m) ((double)(v))
#define SJ_NP_SQUARED(v, m) (((double)(v) - (m)) * ((double)(v) - (m)))
SJ_NP_PW(sj_np_pw_i, int64_t, SJ_NP_AS_DOUBLE)
SJ_NP_PW(sj_np_pw_sq_f, double, SJ_NP_SQUARED)
SJ_NP_PW(sj_np_pw_sq_i, int64_t, SJ_NP_SQUARED)
/* numpy converts ints a buffer at a time, whatever the build */
static double sj_np_sum_as_float(const int64_t *a, int64_t n, int64_t st) {
    if (n <= 8192) return sj_np_pw_i(a, n, st, 0.0);
    double r = 0.0;
    for (int64_t i = 0; i < n; i += 8192) r += sj_np_pw_i(a + i * st, n - i < 8192 ? n - i : 8192, st, 0.0);
    return r;
}
static double sj_np_var_f(const double *a, int64_t n, int64_t st, int64_t chunk, int64_t ddof) {
    double mean = sj_np_sum(a, n, st, chunk) / (double)n, r = 0.0;
    int64_t div = n > ddof ? n - ddof : 0;
    if (!chunk || n <= chunk) r = sj_np_pw_sq_f(a, n, st, mean);
    else for (int64_t i = 0; i < n; i += chunk) r += sj_np_pw_sq_f(a + i * st, n - i < chunk ? n - i : chunk, st, mean);
    return r / (double)div;
}
static double sj_np_var_i(const int64_t *a, int64_t n, int64_t st, int64_t chunk, int64_t ddof) {
    double mean = sj_np_sum_as_float(a, n, st) / (double)n, r = 0.0;
    int64_t div = n > ddof ? n - ddof : 0;
    if (!chunk || n <= chunk) r = sj_np_pw_sq_i(a, n, st, mean);
    else for (int64_t i = 0; i < n; i += chunk) r += sj_np_pw_sq_i(a + i * st, n - i < chunk ? n - i : chunk, st, mean);
    return r / (double)div;
}
"""),
    'np_make': ((), """
typedef struct sj_block { struct sj_block *next; int64_t bytes; int64_t mark; int64_t ndim; int64_t shape[4]; struct sj_heap *owner; } sj_block;
typedef struct sj_heap { sj_block *all; int64_t since; int64_t limit; uint64_t most; int64_t sweeps; } sj_heap;
#define SJ_ORPHAN ((sj_heap *)1)   /* kept at a function's exit for its caller (or Python), in no heap's list */
static sj_block *sj_make(sj_heap *h, int ndim, const int64_t *shape, int64_t item, int zero) {
    uint64_t bytes = (uint64_t)item, limit = h->most - sizeof(sj_block);   /* numpy's own limit, too */
    int empty = 0;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return 0;
        if (shape[i] == 0) { empty = 1; continue; }
        if (bytes > limit / (uint64_t)shape[i]) return 0;
        bytes *= (uint64_t)shape[i];
    }
    if (empty) bytes = 0;
    size_t total = (size_t)bytes + sizeof(sj_block);
    sj_block *b = zero ? calloc(1, total) : malloc(total);
    if (!b) return 0;
    b->next = h->all; b->bytes = (int64_t)total; b->mark = 0; b->ndim = ndim; b->owner = h;
    for (int i = 0; i < ndim; i++) b->shape[i] = shape[i];
    h->all = b;
    h->since += (int64_t)total + 64;
    return b;
}
static void sj_sweep(sj_heap *h) {
    int64_t live = 0;
    sj_block **p = &h->all;
    while (*p) {
        sj_block *b = *p;
        if (b->mark) { b->mark = 0; live += b->bytes; p = &b->next; }
        else { *p = b->next; free(b); }
    }
    h->since = 0;
    h->limit = live > 1048576 ? live : 1048576;
}
static void sj_free_blocks(sj_heap *h, int keep) {
    sj_block *b = h->all;
    while (b) {
        sj_block *next = b->next;
        if (!keep || b->mark != 2) free(b);
        b = next;
    }
}
static void sj_adopt(sj_heap *h, sj_block *b) {      /* a block a called function returned: it's the caller's now */
    if (b->mark == 2 && b->owner != h) { b->mark = 0; b->owner = h; b->next = h->all; h->all = b; h->since += b->bytes + 64; }
}
static int64_t sj_phase(sj_heap *h) {               /* a mark no other sweep, of this heap or another, uses */
    return ((int64_t)(uintptr_t)h << 8) ^ ((++h->sweeps & 0x7f) << 1) | 1;
}
int sj_release(int64_t block) { free((void *)(uintptr_t)block); return 0; }
"""),
    'np_folds': ((), """
static int64_t sj_np_sum_i(const int64_t *a, int64_t n, int64_t st) {
    uint64_t r = 0;
    for (int64_t i = 0; i < n; i++) r += (uint64_t)a[i * st];
    return (int64_t)r;
}
static int64_t sj_np_prod_i(const int64_t *a, int64_t n, int64_t st) {
    uint64_t r = 1;
    for (int64_t i = 0; i < n; i++) r *= (uint64_t)a[i * st];
    return (int64_t)r;
}
static double sj_np_prod_f(const double *a, int64_t n, int64_t st) {
    double r = 1.0;
    if (n > 0) r = a[0];
    for (int64_t i = 1; i < n; i++) r *= a[i * st];
    return r;
}
/* A NaN wins. Of 0.0 and -0.0: rule 0 keeps the first seen, 1 prefers the sign max or min would, 2 takes the last seen.
   The fast loop is the strict maximum (the first of equal values, NaNs never taken: rule 0) beside z, +0 unless an
   element is a NaN or an infinity, summed in any order: cimo vectorizes both exactly; the loop that stops at the first
   NaN runs only when z says one may be there, or when a zero result needs another rule. */
static double sj_np_max_slow(const double *a, int64_t n, int64_t st, int rule) {
    double r = a[0];
    for (int64_t i = 0; i < n && r == r; i++) {
        double e = a[i * st];
        if (e != e || e > r) r = e;
        else if (e == 0.0 && r == 0.0 && (rule == 2 || (rule == 1 && signbit(r)))) r = e;
    }
    return r;
}
static double sj_np_max_f(const double *a, int64_t n, int64_t st, int rule) {
    double r = a[0], z = 0.0;
#pragma omp simd reduction(+:z)
    for (int64_t i = 0; i < n; i++) {
        double e = a[i * st];
        if (e > r) r = e;
        z += e - e;
    }
    if (z != z || (r == 0.0 && rule != 0)) return sj_np_max_slow(a, n, st, rule);
    return r;
}
static double sj_np_min_slow(const double *a, int64_t n, int64_t st, int rule) {
    double r = a[0];
    for (int64_t i = 0; i < n && r == r; i++) {
        double e = a[i * st];
        if (e != e || e < r) r = e;
        else if (e == 0.0 && r == 0.0 && (rule == 2 || (rule == 1 && signbit(e)))) r = e;
    }
    return r;
}
static double sj_np_min_f(const double *a, int64_t n, int64_t st, int rule) {
    double r = a[0], z = 0.0;
#pragma omp simd reduction(+:z)
    for (int64_t i = 0; i < n; i++) {
        double e = a[i * st];
        if (e < r) r = e;
        z += e - e;
    }
    if (z != z || (r == 0.0 && rule != 0)) return sj_np_min_slow(a, n, st, rule);
    return r;
}
static int64_t sj_np_max_i(const int64_t *a, int64_t n, int64_t st) {
    int64_t r = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i * st] > r) r = a[i * st];
    return r;
}
static int64_t sj_np_min_i(const int64_t *a, int64_t n, int64_t st) {
    int64_t r = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i * st] < r) r = a[i * st];
    return r;
}
/* The first of the largest, or the first NaN: the fast loop as for sj_np_max_f, the one that stops at a NaN when z
   says there may be one */
static int64_t sj_np_argmax_slow(const double *a, int64_t n, int64_t st) {
    int64_t at = 0;
    double r = a[0];
    for (int64_t i = 0; i < n; i++) {
        double e = a[i * st];
        if (e != e) return i;
        if (e > r) { r = e; at = i; }
    }
    return at;
}
static int64_t sj_np_argmax_f(const double *a, int64_t n, int64_t st) {
    int64_t at = 0;
    double r = a[0], z = 0.0;
#pragma omp simd reduction(+:z)
    for (int64_t i = 0; i < n; i++) {
        double e = a[i * st];
        if (e > r) { r = e; at = i; }
        z += e - e;
    }
    if (z != z) return sj_np_argmax_slow(a, n, st);
    return at;
}
static int64_t sj_np_argmin_slow(const double *a, int64_t n, int64_t st) {
    int64_t at = 0;
    double r = a[0];
    for (int64_t i = 0; i < n; i++) {
        double e = a[i * st];
        if (e != e) return i;
        if (e < r) { r = e; at = i; }
    }
    return at;
}
static int64_t sj_np_argmin_f(const double *a, int64_t n, int64_t st) {
    int64_t at = 0;
    double r = a[0], z = 0.0;
#pragma omp simd reduction(+:z)
    for (int64_t i = 0; i < n; i++) {
        double e = a[i * st];
        if (e < r) { r = e; at = i; }
        z += e - e;
    }
    if (z != z) return sj_np_argmin_slow(a, n, st);
    return at;
}
static int64_t sj_np_argmax_i(const int64_t *a, int64_t n, int64_t st) {
    int64_t at = 0, r = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i * st] > r) { r = a[i * st]; at = i; }
    return at;
}
static int64_t sj_np_argmin_i(const int64_t *a, int64_t n, int64_t st) {
    int64_t at = 0, r = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i * st] < r) { r = a[i * st]; at = i; }
    return at;
}
static int sj_np_any_f(const double *a, int64_t n, int64_t st) {
    for (int64_t i = 0; i < n; i++) if (a[i * st] != 0.0) return 1;
    return 0;
}
static int sj_np_all_f(const double *a, int64_t n, int64_t st) {
    for (int64_t i = 0; i < n; i++) if (a[i * st] == 0.0) return 0;
    return 1;
}
static int sj_np_any_i(const int64_t *a, int64_t n, int64_t st) {
    for (int64_t i = 0; i < n; i++) if (a[i * st] != 0) return 1;
    return 0;
}
static int sj_np_all_i(const int64_t *a, int64_t n, int64_t st) {
    for (int64_t i = 0; i < n; i++) if (a[i * st] == 0) return 0;
    return 1;
}
"""),
    'cmp_uf': ((), """
/* an int in [0, 2**64) against a double, exactly, as sj_cmp_if: -1, 0, 1, or 2 for a NaN */
static inline int sj_cmp_uf(uint64_t u, double f) {
    if (f != f) return 2;
    if (f < 0.0) return 1;
    if (f >= 18446744073709551616.0) return -1;
    uint64_t t = (uint64_t)f;                 /* f's whole part, exact */
    if (u != t) return u < t ? -1 : 1;
    return (double)t < f ? -1 : 0;
}
"""),
    'ipow': ((), """
/* int ** a nonnegative int, wrapping at 64 bits */
static int64_t sj_ipow(int64_t a, int64_t b) {
    uint64_t base = (uint64_t)a, acc = 1;
    for (uint64_t e = (uint64_t)b; e; e >>= 1) {
        if (e & 1) acc *= base;
        base *= base;
    }
    return (int64_t)acc;
}
"""),
    'gcd': ((), """
static int64_t sj_gcd2(int64_t a, int64_t b) {
    uint64_t x = a < 0 ? 0 - (uint64_t)a : (uint64_t)a, y = b < 0 ? 0 - (uint64_t)b : (uint64_t)b;
    while (y) { uint64_t t = x % y; x = y; y = t; }
    return (int64_t)x;
}
"""),
    'lcm': (('gcd',), """
static int64_t sj_lcm2(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    uint64_t x = a < 0 ? 0 - (uint64_t)a : (uint64_t)a, y = b < 0 ? 0 - (uint64_t)b : (uint64_t)b;
    return (int64_t)(x / (uint64_t)sj_gcd2(a, b) * y);
}
"""),
    'isqrt': ((), """
static int64_t sj_isqrt(int64_t n) {
    uint64_t r = (uint64_t)sqrt((double)n);
    if (r > 3037000499u) r = 3037000499u;
    while (r * r > (uint64_t)n) r--;
    while (r < 3037000499u && (r + 1) * (r + 1) <= (uint64_t)n) r++;
    return (int64_t)r;
}
"""),
    'perm': ((), """
static int64_t sj_perm(int64_t n, int64_t k) {
    if (k > n) return 0;
    uint64_t acc = 1;
    for (int64_t i = 0; i < k && acc != 0; i++) acc *= (uint64_t)(n - i);
    return (int64_t)acc;
}
"""),
    'comb': ((), """
/* comb(n, k) modulo 2**64, as every int here: odd parts multiplied and divided through their
 * inverses, powers of two counted. Exact whenever the result fits. */
static uint64_t sj_inv64(uint64_t a) {
    uint64_t x = a;
    for (int i = 0; i < 6; i++) x *= 2 - a * x;
    return x;
}
static int64_t sj_comb(int64_t n, int64_t k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t acc = 1;
    int64_t twos = 0;
    for (int64_t i = 1; i <= k; i++) {
        uint64_t a = (uint64_t)(n - k + i), b = (uint64_t)i;
        while (!(a & 1)) { a >>= 1; twos++; }
        while (!(b & 1)) { b >>= 1; twos--; }
        acc *= a * sj_inv64(b);
    }
    return (int64_t)(twos >= 64 ? 0 : acc << twos);
}
"""),
    'fsum': ((), """
typedef struct { int64_t n; double special, inf; double p[2200]; } sj_fsum_t;
/* returns 2 on intermediate overflow */
static int sj_fsum_add(sj_fsum_t *s, double x) {
    double xsave = x, y, t, hi, yr, lo;
    int64_t i = 0;
    for (int64_t j = 0; j < s->n; j++) {
        y = s->p[j];
        if (fabs(x) < fabs(y)) { t = x; x = y; y = t; }
        hi = x + y;
        yr = hi - x;
        lo = y - yr;
        if (lo != 0.0) s->p[i++] = lo;
        x = hi;
    }
    s->n = i;
    if (x != 0.0) {
        if (!isfinite(x)) {
            if (isfinite(xsave)) return 2;
            if (isinf(xsave)) s->inf += xsave;
            s->special += xsave;
            s->n = 0;
        } else s->p[s->n++] = x;
    }
    return 0;
}
/* returns 1 for -inf + inf */
static int sj_fsum_end(sj_fsum_t *s, double *out) {
    double x, y, hi = 0.0, yr, lo = 0.0;
    int64_t n = s->n;
    if (s->special != 0.0) {
        if (s->inf != s->inf) return 1;
        *out = s->special;
        return 0;
    }
    if (n > 0) {
        hi = s->p[--n];
        while (n > 0) {
            x = hi;
            y = s->p[--n];
            hi = x + y;
            yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && s->p[n - 1] < 0.0) || (lo > 0.0 && s->p[n - 1] > 0.0))) {
            y = lo * 2.0;
            x = hi + y;
            yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    *out = hi;
    return 0;
}
"""),
}


TAGS = {'uint64_t': 'u64', 'float': 'f32', 'int32_t': 'i32', 'int16_t': 'i16', 'int8_t': 'i8', 'uint32_t': 'u32', 'uint16_t': 'u16', 'uint8_t': 'u8'}


def typed(name, ctype):
    """A copy of numpy's reduction helper `name` for arrays of a narrow type, its functions renamed sj_np<tag>_. For
    float32 everything is float: numpy adds float32 in float32. For a narrow int only the element type changes: the
    total is int64 or double, as numpy's is."""
    key = f'{name}@{ctype}'
    if key not in SNIPPETS:
        deps, code = SNIPPETS[name]
        code = code.replace('sj_np_', f'sj_np{TAGS[ctype]}_')
        if ctype == 'float':
            code = code.replace('double', 'float')
        else:
            code = code.replace('const int64_t *', f'const {ctype} *').replace(', int64_t, SJ_NP_', f', {ctype}, SJ_NP_')
            if ctype == 'uint64_t':                    # compared unsigned
                code = code.replace('    int64_t r = a[0];', '    uint64_t r = a[0];').replace('int64_t at = 0, r = a[0];', 'int64_t at = 0; uint64_t r = a[0];')
        SNIPPETS[key] = (tuple(typed(d, ctype) for d in deps), code)
    return key


def snippets(names):
    """The C for these helpers and what they need, each once, dependencies first."""
    out, seen = [], set()

    def add(n):
        if n not in seen:
            seen.add(n)
            for d in SNIPPETS[n][0]:
                add(d)
            out.append(SNIPPETS[n][1])
    for n in sorted(names):
        add(n)
    return ''.join(out)
