# cython: language_level=3, boundscheck=False, wraparound=False
"""The array kernels of bench/compare.py, typed for Cython as its users write them: memoryviews, C types, no bounds or
negative-index checks. Python's % and // are kept (cdivision off), as gray_scott's wraparound needs them."""

from libc.math cimport hypot, round as cround, INFINITY


def gray_scott(double[:, ::1] u, double[:, ::1] v, double[:, ::1] u2, double[:, ::1] v2,
               double du, double dv, double f, double k, double dt):
    cdef Py_ssize_t h = u.shape[0], w = u.shape[1], i, j
    cdef double uc, vc, lap_u, lap_v, uvv
    for i in range(h):
        for j in range(w):
            uc = u[i, j]
            vc = v[i, j]
            lap_u = (u[(i - 1) % h, j] + u[(i + 1) % h, j]
                     + u[i, (j - 1) % w] + u[i, (j + 1) % w] - 4.0 * uc)
            lap_v = (v[(i - 1) % h, j] + v[(i + 1) % h, j]
                     + v[i, (j - 1) % w] + v[i, (j + 1) % w] - 4.0 * vc)
            uvv = uc * vc * vc
            u2[i, j] = uc + dt * (du * lap_u - uvv + f * (1.0 - uc))
            v2[i, j] = vc + dt * (dv * lap_v + uvv - (f + k) * vc)


def flood_fill(unsigned char[:, ::1] grid, long long[::1] stack, Py_ssize_t y0, Py_ssize_t x0, unsigned char new):
    cdef Py_ssize_t h = grid.shape[0], w = grid.shape[1], top, n = 0, y, x
    cdef long long p
    cdef unsigned char old = grid[y0, x0]
    if old == new:
        return 0
    grid[y0, x0] = new
    stack[0] = y0 * w + x0
    top = 1
    while top > 0:
        top -= 1
        p = stack[top]
        y = p // w
        x = p % w
        n += 1
        if y > 0 and grid[y - 1, x] == old:
            grid[y - 1, x] = new
            stack[top] = p - w
            top += 1
        if y < h - 1 and grid[y + 1, x] == old:
            grid[y + 1, x] = new
            stack[top] = p + w
            top += 1
        if x > 0 and grid[y, x - 1] == old:
            grid[y, x - 1] = new
            stack[top] = p - 1
            top += 1
        if x < w - 1 and grid[y, x + 1] == old:
            grid[y, x + 1] = new
            stack[top] = p + 1
            top += 1
    return n


cdef inline Py_ssize_t pyround(double v) nogil:
    """Python's round(): halves to even."""
    cdef double r = cround(v)
    if r - v == 0.5 or v - r == 0.5:
        r = 2.0 * cround(v / 2.0)
    return <Py_ssize_t>r


def viewshed(double[:, ::1] dem, unsigned char[:, ::1] vis, Py_ssize_t oy, Py_ssize_t ox, double oh):
    cdef Py_ssize_t h = dem.shape[0], w = dem.shape[1], ty, tx, dy, dx, n, s
    cdef double z0 = dem[oy, ox] + oh, best, y, x, slope
    for ty in range(h):
        for tx in range(w):
            dy = ty - oy
            dx = tx - ox
            n = max(abs(dy), abs(dx))
            if n == 0:
                vis[ty, tx] = 1
                continue
            best = -INFINITY
            for s in range(1, n):
                y = oy + <double>(dy * s) / n
                x = ox + <double>(dx * s) / n
                slope = (dem[pyround(y), pyround(x)] - z0) / hypot(y - oy, x - ox)
                if slope > best:
                    best = slope
            if (dem[ty, tx] - z0) / hypot(<double>dy, <double>dx) >= best:
                vis[ty, tx] = 1
            else:
                vis[ty, tx] = 0


def dot(double[::1] x, double[::1] y):
    cdef Py_ssize_t i
    cdef double s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * y[i]
    return s


def axpy(double a, double[::1] x, double[::1] y):
    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        y[i] += a * x[i]


def histogram(double[::1] x, long long[::1] counts, double lo, double hi):
    cdef Py_ssize_t i, nb = counts.shape[0], k
    cdef double scale = nb / (hi - lo)
    for i in range(x.shape[0]):
        k = <Py_ssize_t>((x[i] - lo) * scale)
        if 0 <= k < nb:
            counts[k] += 1
