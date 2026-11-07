"""The kernels bench/compare.py runs, as plain Python: each tool compiles these functions as written. The three M0 kernels
come from kernels/; these add three loops a vectorizing compiler is known to do well on (dot, axpy, histogram) and two
written around Python data (a dict of settings, a list of dataclass instances)."""

import dataclasses
import math


def dot(x, y):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s


def axpy(a, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]


def histogram(x, counts, lo, hi):
    nb = len(counts)
    scale = nb / (hi - lo)
    for i in range(len(x)):
        k = int((x[i] - lo) * scale)
        if 0 <= k < nb:
            counts[k] += 1


def epidemic(settings, summary, infected):
    """A dict of settings in, results into another dict (as examples/config_dict.py)."""
    s = 1.0 - settings['seed']
    i = settings['seed']
    r = 0.0
    for day in range(settings['days']):
        for _ in range(settings['substeps']):
            new_cases = settings['beta'] * s * i * settings['dt']
            recoveries = settings['gamma'] * i * settings['dt']
            s -= new_cases
            i += new_cases - recoveries
            r += recoveries
        infected[day] = i
        if i > summary['peak']:
            summary['peak'] = i
            summary['peak_day'] = day
    summary['recovered'] = r


SOFTENING = 0.05


@dataclasses.dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0

    def dist2(self, other):
        dx, dy = other.x - self.x, other.y - self.y
        return dx * dx + dy * dy + SOFTENING * SOFTENING


def particles(ps, dt, steps):
    """A list of dataclass instances with a method, every one pulling on every other (as examples/particles.py)."""
    for _ in range(steps):
        for a in ps:
            for b in ps:
                r2 = a.dist2(b)
                f = b.mass / (r2 * math.sqrt(r2)) * dt
                a.vx += (b.x - a.x) * f
                a.vy += (b.y - a.y) * f
        for a in ps:
            a.x += a.vx * dt
            a.y += a.vy * dt
