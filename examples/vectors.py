"""Rays against a sphere, written with a small vector dataclass: each pixel makes a Vec, normalizes it with a method and
hands it to the sphere's own method. Each pixel's hit depends on its ray, and the shading on the hit, the sort of code that
is written with classes, not arrays. Making a Vec in the loop costs nothing compiled: its attributes are variables."""

import dataclasses
import math
import time

import numpy as np

from scramjit import scram


@dataclasses.dataclass
class Vec:
    x: float
    y: float
    z: float

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def normalize(self):
        n = math.sqrt(self.dot(self))
        self.x /= n
        self.y /= n
        self.z /= n


@dataclasses.dataclass
class Sphere:
    center: Vec
    radius: float

    def hit(self, d):
        """Distance along the unit ray d from the origin to the sphere, or -1.0."""
        b = d.dot(self.center)
        c = self.center.dot(self.center) - self.radius * self.radius
        disc = b * b - c
        return b - math.sqrt(disc) if disc >= 0.0 else -1.0


@scram
def render(sphere, light, image):
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            d = Vec((j - w / 2) / w, (i - h / 2) / h, 1.0)
            d.normalize()
            t = sphere.hit(d)
            if t > 0.0:
                n = Vec(d.x * t - sphere.center.x, d.y * t - sphere.center.y, d.z * t - sphere.center.z)
                n.normalize()
                image[i, j] = max(n.dot(light), 0.0)
            else:
                image[i, j] = 0.0
    return image.sum()


def main():
    sphere, light = Sphere(Vec(0.0, 0.0, 3.0), 1.0), Vec(-0.5, -0.6, -0.62)
    light.normalize()
    plain, fast = np.zeros((300, 400)), np.zeros((300, 400))

    t = time.perf_counter()
    expected = render.__wrapped__(sphere, light, plain)
    python_s = time.perf_counter() - t

    t = time.perf_counter()
    render(sphere, light, np.zeros((2, 2)))      # the first call compiles
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    result = render(sphere, light, fast)
    scramjit_s = time.perf_counter() - t

    assert result == expected and np.array_equal(plain, fast)
    assert render.signatures, 'it should have compiled'
    lit = ' .:-=+*#%@'
    for row in fast[::30]:
        print(''.join(lit[min(int(v * 9.99), 9)] for v in row[::8]))
    print(f'Python {python_s:.3f} s, scramjit {scramjit_s * 1e3:.1f} ms ({python_s / scramjit_s:.0f}x), compile {compile_s:.2f} s')


if __name__ == '__main__':
    main()
