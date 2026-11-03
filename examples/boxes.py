"""Boxes as nested tuples, ((x0, y0), (x1, y1)): passed in, unpacked, built in a loop, passed between compiled
functions and returned, as in Python. Finds the box in an array of 200,000 that overlaps a query box most."""

import time

import numpy as np

from scramjit import scram


@scram
def iou(a, b):
    """Intersection over union of two boxes."""
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    w = min(ax1, bx1) - max(ax0, bx0)
    h = min(ay1, by1) - max(ay0, by0)
    if w <= 0.0 or h <= 0.0:
        return 0.0
    inter = w * h
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


@scram
def best_match(boxes, query):
    best = (0.0, -1)
    for i in range(len(boxes)):
        box = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]))
        score = iou(box, query)
        if score > best[0]:
            best = (score, i)
    return best


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    corner = rng.random((200_000, 2)) * 100
    boxes = np.hstack([corner, corner + rng.random((200_000, 2)) * 10])
    query = ((40.0, 40.0), (46.0, 47.0))

    start = time.perf_counter()
    want = best_match.__wrapped__(boxes, query)
    python_time = time.perf_counter() - start

    best_match(boxes[:1], query)                         # compiles
    start = time.perf_counter()
    got = best_match(boxes, query)
    scramjit_time = time.perf_counter() - start
    assert got == want
    print(f'best overlap {got[0]:.3f} with box {got[1]}: Python {python_time:.2f} s, scramjit {scramjit_time * 1000:.2f} ms, '
          f'{python_time / scramjit_time:.0f}x')
