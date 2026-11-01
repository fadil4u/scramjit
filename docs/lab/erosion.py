import math


def erode(h, seeds, inertia: float, capacity: float, deposit: float, erosion: float,
          evaporation: float, steps: int):
    """Hydraulic erosion: each droplet flows downhill over the height field h, picking up
    sediment where it speeds up and dropping it where it slows. Every droplet's path depends on
    what the ones before it carved, so it doesn't vectorize."""
    rows, cols = h.shape
    for d in range(len(seeds) // 2):
        x = seeds[2 * d] * (cols - 2)
        y = seeds[2 * d + 1] * (rows - 2)
        dx = 0.0
        dy = 0.0
        speed = 1.0
        water = 1.0
        sediment = 0.0
        for _ in range(steps):
            ix = int(x)
            iy = int(y)
            fx = x - ix
            fy = y - iy
            h00 = h[iy, ix]
            h10 = h[iy, ix + 1]
            h01 = h[iy + 1, ix]
            h11 = h[iy + 1, ix + 1]
            gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
            here = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)
            norm = math.sqrt(dx * dx + dy * dy)
            if norm == 0:
                break
            dx = dx / norm
            dy = dy / norm
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= cols - 1 or ny >= rows - 1:
                break
            jx = int(nx)
            jy = int(ny)
            gx2 = nx - jx
            gy2 = ny - jy
            there = (h[jy, jx] * (1 - gx2) * (1 - gy2) + h[jy, jx + 1] * gx2 * (1 - gy2)
                     + h[jy + 1, jx] * (1 - gx2) * gy2 + h[jy + 1, jx + 1] * gx2 * gy2)
            dh = there - here
            cap = max(-dh * speed * water * capacity, 0.01)
            if sediment > cap or dh > 0:
                amount = min(dh, sediment) if dh > 0 else (sediment - cap) * deposit
                sediment = sediment - amount
                h[iy, ix] += amount * (1 - fx) * (1 - fy)
                h[iy, ix + 1] += amount * fx * (1 - fy)
                h[iy + 1, ix] += amount * (1 - fx) * fy
                h[iy + 1, ix + 1] += amount * fx * fy
            else:
                amount = min((cap - sediment) * erosion, -dh)
                sediment = sediment + amount
                h[iy, ix] -= amount * (1 - fx) * (1 - fy)
                h[iy, ix + 1] -= amount * fx * (1 - fy)
                h[iy + 1, ix] -= amount * (1 - fx) * fy
                h[iy + 1, ix + 1] -= amount * fx * fy
            speed = math.sqrt(max(speed * speed + dh * -4.0, 0.0))
            water = water * (1 - evaporation)
            x = nx
            y = ny
