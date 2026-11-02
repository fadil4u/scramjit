"""An epidemic model whose settings and results live in dicts, the way real code keeps them. Each day depends
on the one before, so numpy can't vectorize it. Only the keys the function uses matter to scramjit: `name` and
`notes` can be anything, and `summary['peak']` ends up in your dict, as it would in Python."""

import time

import numpy as np

from scramjit import scram


@scram
def simulate(settings, summary, infected):
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


settings = {'name': 'flu', 'notes': ['a dict can hold anything it does not use'], 'beta': 0.3, 'gamma': 0.1,
            'seed': 1e-4, 'days': 400, 'substeps': 2000, 'dt': 0.0005}


def run(f):
    summary = {'peak': 0.0, 'peak_day': -1, 'recovered': 0.0, 'label': settings['name']}
    infected = np.zeros(settings['days'])
    t = time.perf_counter()
    f(settings, summary, infected)
    return time.perf_counter() - t, summary, infected


t0 = time.perf_counter()
run(simulate)
compiled_in = time.perf_counter() - t0

python_s, want, want_curve = run(simulate.__wrapped__)
native_s = min(run(simulate)[0] for _ in range(5))
_, got, got_curve = run(simulate)
assert got == want and got_curve.tobytes() == want_curve.tobytes()

print(f'{settings["name"]}: peak {got["peak"] * 100:.1f}% infected on day {got["peak_day"]}, {got["recovered"] * 100:.1f}% recovered')
print(f'Python {python_s:.3f} s, scramjit {native_s * 1000:.2f} ms, {python_s / native_s:.0f}x (first call {compiled_in:.2f} s)')
