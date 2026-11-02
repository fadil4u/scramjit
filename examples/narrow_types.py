"""Arithmetic on float32, int32, uint8 and the other narrow dtypes compiles, and follows numpy's rules: a uint8 wraps at
256, a float32 is rounded to float32 at each step, and x[i] + 300 on a uint8 raises OverflowError, as in numpy.

Half the memory per element also means twice the numbers in each vector register, where the loop vectorizes.
"""

import time

import numpy as np

from scramjit import scram


@scram
def axpy(a, x, y):
    for i in range(len(x)):
        y[i] = a * x[i] + y[i]


@scram
def brighten(image, amount):
    for i in range(len(image)):
        image[i] = image[i] + amount       # wraps past 255, as numpy does


@scram
def dot(x, y):
    total = 0.0                            # np.float32 from the first +=, and still the float 0.0 if x is empty
    for i in range(len(x)):
        total += x[i] * y[i]
    return total


def per_element(f, n, *args):
    best = float('inf')
    for _ in range(20):
        start = time.perf_counter()
        f(*args)
        best = min(best, time.perf_counter() - start)
    return best / n * 1e9


if __name__ == '__main__':
    n = 1_000_000
    x, y = np.linspace(0, 1, n, dtype=np.float32), np.linspace(1, 0, n, dtype=np.float32)
    a = np.float32(0.5)

    plain = y.copy()
    start = time.perf_counter()
    axpy.__wrapped__(a, x, plain)
    python_time = time.perf_counter() - start

    compiled = y.copy()
    axpy(a, x, compiled)                   # compiles
    compiled[:] = y
    start = time.perf_counter()
    axpy(a, x, compiled)
    scramjit_time = time.perf_counter() - start
    assert compiled.tobytes() == plain.tobytes()
    print(f'axpy on float32: Python {python_time:.3f} s, scramjit {scramjit_time * 1000:.2f} ms, {python_time / scramjit_time:.0f}x, same bits')

    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    print(f'per element: float64 {per_element(axpy, n, 0.5, x64, y64):.2f} ns, float32 {per_element(axpy, n, a, x, y):.2f} ns')

    image = np.array([0, 100, 250], np.uint8)
    brighten(image, 10)
    print('brighten by 10:', image)
    try:
        brighten(image, 300)
    except OverflowError as e:
        print('brighten by 300:', e)

    x = np.array([0.1, 0.2, 0.3], np.float32)
    print('dot:', repr(dot(x, x)), 'and on nothing:', repr(dot(x[:0], x[:0])))
