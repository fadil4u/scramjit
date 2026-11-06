"""A charged particle in a magnetic field, written with a vector dataclass and its operators: v = v + (v @ b) * dt makes a
new Vec at every step, through __add__, __matmul__ and __mul__, the way such code is written by hand. Compiled, each new
Vec is three variables and each operator call is inlined, so nothing is allocated in the loop."""

import dataclasses
import math
import time

from scramjit import scram


@dataclasses.dataclass
class Vec:
    x: float
    y: float
    z: float

    def __add__(self, o):
        return Vec(self.x + o.x, self.y + o.y, self.z + o.z)

    def __mul__(self, k):
        return Vec(self.x * k, self.y * k, self.z * k)

    def __matmul__(self, o):                     # the cross product
        return Vec(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@scram
def gyrate(pos, vel, field, steps, dt):
    """Steps a unit charge through a uniform field; returns the farthest it gets from where it started."""
    start = pos + Vec(0.0, 0.0, 0.0)
    far = 0.0
    for _ in range(steps):
        vel = vel + (vel @ field) * dt
        pos = pos + vel * dt
        far = max(far, (pos + start * -1.0).norm())
    return far


def main():
    args = (Vec(0.0, 0.0, 0.0), Vec(1.0, 0.0, 0.1), Vec(0.0, 0.0, 2.0), 300_000, 0.001)

    t = time.perf_counter()
    expected = gyrate.__wrapped__(*args)
    python_s = time.perf_counter() - t

    t = time.perf_counter()
    gyrate(*args[:3], 1, 0.001)                  # the first call compiles
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    result = gyrate(*args)
    scramjit_s = time.perf_counter() - t

    assert result == expected
    assert gyrate.signatures, 'it should have compiled'
    print(f'farthest from the start: {result:.6f}')
    print(f'Python {python_s:.3f} s, scramjit {scramjit_s * 1e3:.1f} ms ({python_s / scramjit_s:.0f}x), compile {compile_s:.2f} s')


if __name__ == '__main__':
    main()
