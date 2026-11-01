/* What scramjit should generate for kernels/flood_fill.py: flood_fill(grid, stack, y0, x0, new)
 * with grid a C-contiguous uint8 2-D array, stack an int64 1-D array and three ints.
 *
 * - grid[y0, x0] may use negative indices: SJ_INDEX counts them from the end, as Python does.
 * - `new` is stored into a uint8 array. numpy raises OverflowError for an int that doesn't fit,
 *   so the range is checked; `new` never changes, so it is checked once, before the first store.
 * - p // w and p % w: w > 0 is proven (grid[y0, x0] succeeded), so there is no zero check, but
 *   p comes out of an array, so its sign is unknown and Python's rounding needs sj_floordiv.
 * - Pushes check stack's length (top grows without a proven bound). Pops don't: top only
 *   decreases from a value that was checked when it was pushed.
 * - The neighbour tests keep their bounds checks: y and x come from array values, so y - 1 < h
 *   isn't provable without reasoning about what the stack holds.
 */
#include "scramjit.h"

enum { SJ_OVERFLOW_ERROR = 3 };

#line 10 "flood_fill.py"
int flood_fill(uint8_t *grid, int64_t grid_0, int64_t grid_1,
               int64_t *stack, int64_t stack_0,
               int64_t y0, int64_t x0, int64_t new_,
               int64_t *ret, int64_t *sj_line)
{
    int64_t h = grid_0, w = grid_1;
#line 12 "flood_fill.py"
    int64_t gy, gx;
    SJ_INDEX(gy, y0, h, 12); SJ_INDEX(gx, x0, w, 12);
    int64_t old = grid[gy * w + gx];
#line 13 "flood_fill.py"
    if (old == new_) { *ret = 0; return SJ_OK; }
#line 15 "flood_fill.py"
    if (y0 < 0) y0 += h;
    if (x0 < 0) x0 += w;
#line 19 "flood_fill.py"
    if (new_ < 0 || new_ > 255) SJ_RAISE(SJ_OVERFLOW_ERROR, 19);
    grid[y0 * w + x0] = (uint8_t)new_;
#line 20 "flood_fill.py"
    SJ_CHECK(0, stack_0, 20);
    stack[0] = y0 * w + x0;
    int64_t top = 1, n = 0;
#line 23 "flood_fill.py"
    while (top > 0) {
        top -= 1;
        int64_t p = stack[top];
#line 26 "flood_fill.py"
        int64_t y = sj_floordiv(p, w);
        int64_t x = sj_mod(p, w);
        n += 1;
#line 29 "flood_fill.py"
        if (y > 0) {
            SJ_CHECK(y - 1, h, 29); SJ_CHECK(x, w, 29);
            if (grid[(y - 1) * w + x] == old) {
                grid[(y - 1) * w + x] = (uint8_t)new_;
                SJ_CHECK(top, stack_0, 31);
                stack[top] = p - w;
                top += 1;
            }
        }
#line 33 "flood_fill.py"
        if (y < h - 1) {
            SJ_CHECK(y + 1, h, 33); SJ_CHECK(x, w, 33);
            if (grid[(y + 1) * w + x] == old) {
                grid[(y + 1) * w + x] = (uint8_t)new_;
                SJ_CHECK(top, stack_0, 35);
                stack[top] = p + w;
                top += 1;
            }
        }
#line 37 "flood_fill.py"
        if (x > 0) {
            SJ_CHECK(y, h, 37); SJ_CHECK(x - 1, w, 37);
            if (grid[y * w + x - 1] == old) {
                grid[y * w + x - 1] = (uint8_t)new_;
                SJ_CHECK(top, stack_0, 39);
                stack[top] = p - 1;
                top += 1;
            }
        }
#line 41 "flood_fill.py"
        if (x < w - 1) {
            SJ_CHECK(y, h, 41); SJ_CHECK(x + 1, w, 41);
            if (grid[y * w + x + 1] == old) {
                grid[y * w + x + 1] = (uint8_t)new_;
                SJ_CHECK(top, stack_0, 43);
                stack[top] = p + 1;
                top += 1;
            }
        }
    }
#line 45 "flood_fill.py"
    *ret = n;
    return SJ_OK;
}
