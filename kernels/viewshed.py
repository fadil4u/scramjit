"""Viewshed: which cells of an elevation grid are visible from an observer.

For every target cell, march along the straight line from the observer and keep the steepest
slope seen so far; the target is visible if its own slope is at least that steep. Sequential
within each ray (a running maximum), with float math, rounding and math calls in the inner loop.
"""

import math


def viewshed(dem, vis, oy: int, ox: int, oh: float):
    h, w = dem.shape
    z0 = dem[oy, ox] + oh
    for ty in range(h):
        for tx in range(w):
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
                slope = (dem[round(y), round(x)] - z0) / math.hypot(y - oy, x - ox)
                if slope > best:
                    best = slope
            if (dem[ty, tx] - z0) / math.hypot(dy, dx) >= best:
                vis[ty, tx] = 1
            else:
                vis[ty, tx] = 0
