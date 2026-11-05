"""Particles as a plain list of dataclass instances, the way the code would be written without a compiler in mind: methods,
a helper function taking two particles, a constant at module level. Each step depends on the one before, and every
particle pulls on every other, so numpy can't vectorize it. `name` can be anything: only what the code uses matters."""

import dataclasses
import math
import time

from scramjit import scram

SOFTENING = 0.05


@dataclasses.dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0
    name: str = ''
    bounces: int = 0

    def dist2(self, other):
        dx, dy = other.x - self.x, other.y - self.y
        return dx * dx + dy * dy + SOFTENING * SOFTENING

    def drift(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x * self.x + self.y * self.y > 100.0:     # a wall at radius 10: turn back
            self.vx, self.vy = -self.vx, -self.vy
            self.bounces += 1


@scram
def pull(a, b, dt):
    """b's gravity on a, over dt."""
    r2 = a.dist2(b)
    f = b.mass / (r2 * math.sqrt(r2)) * dt
    a.vx += (b.x - a.x) * f
    a.vy += (b.y - a.y) * f


@scram
def simulate(particles, dt, steps):
    for _ in range(steps):
        for a in particles:
            for b in particles:
                pull(a, b, dt)
        for p in particles:
            p.drift(dt)
    return sum((p.vx * p.vx + p.vy * p.vy for p in particles), 0.0)


def cloud(n):
    return [Particle(math.cos(i * 2.4) * (1 + i % 5), math.sin(i * 2.4) * (1 + i % 5), mass=0.5 + i % 3, name=f'p{i}') for i in range(n)]


def plain_simulate(particles, dt, steps):          # the same program with nothing compiled
    for _ in range(steps):
        for a in particles:
            for b in particles:
                pull.__wrapped__(a, b, dt)
        for p in particles:
            p.drift(dt)
    return sum((p.vx * p.vx + p.vy * p.vy for p in particles), 0.0)


def main():
    first = cloud(200)
    t = time.perf_counter()
    expected = plain_simulate(first, 0.001, 20)
    python_s = time.perf_counter() - t

    t = time.perf_counter()
    simulate(cloud(2), 0.001, 1)                 # the first call compiles
    compile_s = time.perf_counter() - t

    second = cloud(200)
    t = time.perf_counter()
    result = simulate(second, 0.001, 20)
    scramjit_s = time.perf_counter() - t

    assert result == expected and second == first
    assert simulate.signatures, 'it should have compiled'
    print(f'{len(second)} particles, 20 steps: kinetic energy {result:.6f}, {sum(p.bounces for p in second)} bounces')
    print(f'Python {python_s:.3f} s, scramjit {scramjit_s * 1e3:.1f} ms ({python_s / scramjit_s:.0f}x), compile {compile_s:.2f} s')


if __name__ == '__main__':
    main()
