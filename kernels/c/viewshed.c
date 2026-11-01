/* What scramjit should generate for kernels/viewshed.py: viewshed(dem, vis, oy, ox, oh) with dem
 * a C-contiguous float64 2-D array, vis a uint8 2-D array, two ints and a float.
 *
 * - round(y) is Python's round: halves go to the even neighbour. That is C's rint() in the
 *   default rounding mode, not C's round(), which rounds halves away from zero.
 * - The rounded indices aren't provably in [0, h): they may be negative, which Python counts from
 *   the end, so SJ_INDEX. This check stays in the innermost loop.
 * - dy * s / n is Python's true division of two ints: both are exact as doubles here (|values|
 *   below 2**53), so converting first and dividing in double gives the same correctly rounded
 *   result.
 * - Float division by zero raises ZeroDivisionError in Python, so divisors are checked. (hypot is
 *   never 0 here, but proving that needs s >= 1 to flow into a float function's range.)
 * - math.hypot is CPython's own implementation; this calls the C library's hypot. Whether the two
 *   agree to the last bit is what kernels/check.py tests.
 * - vis gets per-store checks (once per target cell, not in the inner loop). The if/else keeps
 *   its two stores apart, so an IndexError names the line that actually ran.
 */
#include "scramjit.h"

#line 11 "viewshed.py"
int viewshed(const double *dem, int64_t dem_0, int64_t dem_1,
             uint8_t *vis, int64_t vis_0, int64_t vis_1,
             int64_t oy, int64_t ox, double oh,
             int64_t *sj_line)
{
    int64_t h = dem_0, w = dem_1;
#line 13 "viewshed.py"
    int64_t iy, ix;
    SJ_INDEX(iy, oy, h, 13); SJ_INDEX(ix, ox, w, 13);
    double z0 = dem[iy * w + ix] + oh;
#line 14 "viewshed.py"
    for (int64_t ty = 0; ty < h; ty++) {
        for (int64_t tx = 0; tx < w; tx++) {
            int64_t dy = ty - oy;
            int64_t dx = tx - ox;
            int64_t ady = dy < 0 ? -dy : dy, adx = dx < 0 ? -dx : dx;
            int64_t n = ady > adx ? ady : adx;
#line 19 "viewshed.py"
            if (n == 0) {
                SJ_CHECK(ty, vis_0, 20); SJ_CHECK(tx, vis_1, 20);
                vis[ty * vis_1 + tx] = 1;
                continue;
            }
#line 22 "viewshed.py"
            double best = -INFINITY;
            for (int64_t s = 1; s < n; s++) {
#line 24 "viewshed.py"
                double y = (double)oy + (double)(dy * s) / (double)n;
                double x = (double)ox + (double)(dx * s) / (double)n;
#line 26 "viewshed.py"
                int64_t ry, rx;
                SJ_INDEX(ry, (int64_t)rint(y), h, 26);
                SJ_INDEX(rx, (int64_t)rint(x), w, 26);
                double d = hypot(y - (double)oy, x - (double)ox);
                if (d == 0.0) SJ_RAISE(SJ_ZERO_DIVISION_ERROR, 26);
                double slope = (dem[ry * w + rx] - z0) / d;
#line 27 "viewshed.py"
                if (slope > best)
                    best = slope;
            }
#line 29 "viewshed.py"
            double d = hypot((double)dy, (double)dx);
            if (d == 0.0) SJ_RAISE(SJ_ZERO_DIVISION_ERROR, 29);
            if ((dem[ty * w + tx] - z0) / d >= best) {
#line 30 "viewshed.py"
                SJ_CHECK(ty, vis_0, 30); SJ_CHECK(tx, vis_1, 30);
                vis[ty * vis_1 + tx] = 1;
            } else {
#line 32 "viewshed.py"
                SJ_CHECK(ty, vis_0, 32); SJ_CHECK(tx, vis_1, 32);
                vis[ty * vis_1 + tx] = 0;
            }
        }
    }
    return SJ_OK;
}
