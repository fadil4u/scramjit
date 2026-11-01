"""Gray-Scott reaction-diffusion: one explicit time step on a periodic grid.

A 5-point stencil over two float64 fields. Every cell is independent of the others in the same
step, so the work is regular, but each output reads neighbours through wraparound indices.
"""


def gray_scott(u, v, u2, v2, du: float, dv: float, f: float, k: float, dt: float):
    h, w = u.shape
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
