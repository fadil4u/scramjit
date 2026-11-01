/* The prelude scramjit emits ahead of compiled code (inlined into each output in practice; a
 * header here so the hand-written kernels stay readable).
 *
 * Calling convention, per compiled function:
 *   - returns an int status: 0, or the Python exception to raise (SJ_INDEX_ERROR, ...);
 *   - arrays arrive as a pointer plus one int64 extent per dimension (this is the C-contiguous
 *     specialization; other layouts add strides);
 *   - Python ints arrive as int64_t, floats as double;
 *   - a returned value is written through `ret`, the Python line of a raised exception through
 *     `sj_line`, both as the last parameters.
 */
#include <stdint.h>
#include <math.h>

enum { SJ_OK = 0, SJ_INDEX_ERROR = 1, SJ_ZERO_DIVISION_ERROR = 2 };

#define SJ_RAISE(kind, line) do { *sj_line = (line); return (kind); } while (0)

/* An index already known to be >= 0: one unsigned compare. */
#define SJ_CHECK(i, n, line) do { if ((uint64_t)(i) >= (uint64_t)(n)) SJ_RAISE(SJ_INDEX_ERROR, line); } while (0)

/* An index that may be negative: Python counts it from the end. Sets `out`. */
#define SJ_INDEX(out, i, n, line) do { int64_t sj_i_ = (i); if (sj_i_ < 0) sj_i_ += (n); \
    SJ_CHECK(sj_i_, n, line); (out) = sj_i_; } while (0)

/* Python's // and % on int64: round toward negative infinity. The divisor is checked for zero
 * by the caller (SJ_RAISE) unless proven non-zero. INT64_MIN // -1 wraps, like every int64. */
static inline int64_t sj_floordiv(int64_t a, int64_t b) {
    if (b == -1) return (int64_t)(0 - (uint64_t)a);
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
