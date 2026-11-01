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
/* float ** float (float_pow in floatobject.c), for an exponent or base that can't give a complex
 * result. Returns 0, 1 for 0.0 ** negative, 2 on overflow. */
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
    if (v < 0.0) { v = -v; negate = odd; }
    if (v == 1.0) { *out = negate ? -1.0 : 1.0; return 0; }
    double r = f(v, w);
    if (isinf(r)) return 2;
    *out = negate ? -r : r;
    return 0;
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
