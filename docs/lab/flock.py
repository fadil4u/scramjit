import math


def flock(pos, vel, i0: int, i1: int, width: float, height: float, radius: float,
          separation: float, align: float, cohere: float, speed: float):
    """Boids i0 to i1 steer by every other boid within radius: away from the very close, along
    with the heading of the rest, toward their centre. All pairs, on a wrapping world."""
    n = pos.shape[0]
    r2 = radius * radius
    s2 = separation * separation
    for i in range(i0, i1):
        px = pos[i, 0]
        py = pos[i, 1]
        ax = 0.0
        ay = 0.0
        cx = 0.0
        cy = 0.0
        sx = 0.0
        sy = 0.0
        count = 0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - px
            dy = pos[j, 1] - py
            if dx > width / 2:
                dx -= width
            elif dx < -width / 2:
                dx += width
            if dy > height / 2:
                dy -= height
            elif dy < -height / 2:
                dy += height
            d2 = dx * dx + dy * dy
            if d2 < r2:
                count += 1
                ax += vel[j, 0]
                ay += vel[j, 1]
                cx += dx
                cy += dy
                if d2 < s2 and d2 > 0:
                    sx -= dx / d2
                    sy -= dy / d2
        vx = vel[i, 0]
        vy = vel[i, 1]
        if count > 0:
            vx += (ax / count - vx) * align + cx / count * cohere + sx * 1.5
            vy += (ay / count - vy) * align + cy / count * cohere + sy * 1.5
        v = math.sqrt(vx * vx + vy * vy)
        if v > 0:
            vel[i, 0] = vx / v * speed
            vel[i, 1] = vy / v * speed
