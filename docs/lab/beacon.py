import math


def viewshed(dem, vis, oy: int, ox: int, oh: float, r0: int, r1: int):
    """Which cells a beacon at (oy, ox), oh above the ground, can see: rows r0 to r1. Each ray
    keeps the steepest slope seen so far, a running maximum: sequential within the ray."""
    rows, cols = dem.shape
    z0 = dem[oy, ox] + oh
    for ty in range(r0, r1):
        for tx in range(cols):
            dy = ty - oy
            dx = tx - ox
            n = max(abs(dy), abs(dx))
            if n == 0:
                vis[ty, tx] = 1
                continue
            best = -math.inf
            for s in range(1, n):
                y = oy + dy * s / n
                x = ox + dx * s / n
                slope = (dem[round(y), round(x)] - z0) / math.sqrt((y - oy) * (y - oy) + (x - ox) * (x - ox))
                if slope > best:
                    best = slope
            vis[ty, tx] = 1 if (dem[ty, tx] - z0) / math.sqrt(dy * dy + dx * dx) >= best else 0
