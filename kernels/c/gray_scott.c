/* What scramjit should generate for kernels/gray_scott.py, called with four C-contiguous
 * float64 2-D arrays and five floats.
 *
 * - (i - 1) % h etc.: i is in [0, h) from range(h), so the result is i - 1 or h - 1: a compare,
 *   not a division (value-range analysis on Python's %).
 * - u[...] indices are proven in bounds (range over u's own shape). v, u2 and v2 may be smaller
 *   than u, so their checks are hoisted into one test before the loops. If it fails, the checked
 *   copy runs instead, so an IndexError is raised at the same access, after the same writes, as
 *   in Python (loop versioning).
 * - Float expressions keep Python's evaluation order; no reassociation, no FMA contraction.
 * - No restrict: the caller may pass the same array as u and u2.
 */
#include "scramjit.h"

static int gray_scott_checked(const double *u, int64_t u_0, int64_t u_1,
                              const double *v, int64_t v_0, int64_t v_1,
                              double *u2, int64_t u2_0, int64_t u2_1,
                              double *v2, int64_t v2_0, int64_t v2_1,
                              double du, double dv, double f, double k, double dt,
                              int64_t *sj_line)
{
    int64_t h = u_0, w = u_1;
    for (int64_t i = 0; i < h; i++) {
        int64_t im = i == 0 ? h - 1 : i - 1, ip = i + 1 == h ? 0 : i + 1;
        for (int64_t j = 0; j < w; j++) {
            int64_t jm = j == 0 ? w - 1 : j - 1, jp = j + 1 == w ? 0 : j + 1;
#line 12 "gray_scott.py"
            double uc = u[i * w + j];
#line 13 "gray_scott.py"
            SJ_CHECK(i, v_0, 13); SJ_CHECK(j, v_1, 13);
            double vc = v[i * v_1 + j];
#line 14 "gray_scott.py"
            double lap_u = u[im * w + j] + u[ip * w + j] + u[i * w + jm] + u[i * w + jp] - 4.0 * uc;
#line 16 "gray_scott.py"
            SJ_CHECK(im, v_0, 16); SJ_CHECK(ip, v_0, 16); SJ_CHECK(jm, v_1, 16); SJ_CHECK(jp, v_1, 16);
            double lap_v = v[im * v_1 + j] + v[ip * v_1 + j] + v[i * v_1 + jm] + v[i * v_1 + jp] - 4.0 * vc;
#line 18 "gray_scott.py"
            double uvv = uc * vc * vc;
#line 19 "gray_scott.py"
            SJ_CHECK(i, u2_0, 19); SJ_CHECK(j, u2_1, 19);
            u2[i * u2_1 + j] = uc + dt * (du * lap_u - uvv + f * (1.0 - uc));
#line 20 "gray_scott.py"
            SJ_CHECK(i, v2_0, 20); SJ_CHECK(j, v2_1, 20);
            v2[i * v2_1 + j] = vc + dt * (dv * lap_v + uvv - (f + k) * vc);
        }
    }
    return SJ_OK;
}

#line 8 "gray_scott.py"
int gray_scott(const double *u, int64_t u_0, int64_t u_1,
               const double *v, int64_t v_0, int64_t v_1,
               double *u2, int64_t u2_0, int64_t u2_1,
               double *v2, int64_t v2_0, int64_t v2_1,
               double du, double dv, double f, double k, double dt,
               int64_t *sj_line)
{
    int64_t h = u_0, w = u_1;
    /* every index used on v, u2 and v2 is in [0, h) x [0, w) */
    if (h > 0 && w > 0 && !(h <= v_0 && w <= v_1 && h <= u2_0 && w <= u2_1 && h <= v2_0 && w <= v2_1))
        return gray_scott_checked(u, u_0, u_1, v, v_0, v_1, u2, u2_0, u2_1, v2, v2_0, v2_1,
                                  du, dv, f, k, dt, sj_line);
    for (int64_t i = 0; i < h; i++) {
        int64_t im = i == 0 ? h - 1 : i - 1, ip = i + 1 == h ? 0 : i + 1;
        for (int64_t j = 0; j < w; j++) {
            int64_t jm = j == 0 ? w - 1 : j - 1, jp = j + 1 == w ? 0 : j + 1;
#line 12 "gray_scott.py"
            double uc = u[i * w + j];
            double vc = v[i * v_1 + j];
#line 14 "gray_scott.py"
            double lap_u = u[im * w + j] + u[ip * w + j] + u[i * w + jm] + u[i * w + jp] - 4.0 * uc;
            double lap_v = v[im * v_1 + j] + v[ip * v_1 + j] + v[i * v_1 + jm] + v[i * v_1 + jp] - 4.0 * vc;
#line 18 "gray_scott.py"
            double uvv = uc * vc * vc;
            u2[i * u2_1 + j] = uc + dt * (du * lap_u - uvv + f * (1.0 - uc));
            v2[i * v2_1 + j] = vc + dt * (dv * lap_v + uvv - (f + k) * vc);
        }
    }
    return SJ_OK;
}
