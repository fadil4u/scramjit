"""One loop split across threads by calling a compiled function on parts of the same data: arrays are shared in place,
and lists, lists of instances and lists of lists are copied in and back, so a call writes back only what it changed and
keeps what another thread's call wrote to other elements meanwhile."""

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scramjit import scram


@dataclasses.dataclass
class P:
    x: float
    n: int = 0


@scram(cache=False)
def fill(xs, lo, hi, reps):
    for i in range(lo, hi):
        s = 0.0
        for _ in range(reps):
            s += 1.0
        xs[i] = s


@scram(cache=False)
def fill_objects(ps, lo, hi, reps):
    for i in range(lo, hi):
        s = 0.0
        for _ in range(reps):
            s += 1.0
        ps[i].x = s
        ps[i].n += 1


@scram(cache=False)
def fill_rows(rows, lo, hi, reps):
    for i in range(lo, hi):
        s = 0.0
        for _ in range(reps):
            s += 1.0
        for j in range(len(rows[i])):
            rows[i][j] = s


SLOW = 300_000_000          # about 0.3 s compiled


def overlapped(f, data):
    """A slow call on element 0 and, while it runs, a fast one on element 1: the slow one writes back last."""
    f(data, 0, 0, 1)                           # compiles
    slow = threading.Thread(target=f, args=(data, 0, 1, SLOW))
    slow.start()
    time.sleep(0.05)
    f(data, 1, 2, 7)
    slow.join()
    return data


def test_a_list_keeps_what_another_thread_wrote():
    assert overlapped(fill, [0.0, 0.0]) == [float(SLOW), 7.0]


def test_a_list_of_instances_keeps_what_another_thread_wrote():
    assert overlapped(fill_objects, [P(0.0), P(0.0)]) == [P(float(SLOW), 1), P(7.0, 1)]


def test_a_list_of_lists_keeps_rows_another_thread_wrote():
    assert overlapped(fill_rows, [[0.0, 0.0], [0.0, 0.0]]) == [[float(SLOW)] * 2, [7.0, 7.0]]


@scram(cache=False)
def work(xs, ys, lo, hi):
    for i in range(lo, hi):
        ys[i] = xs[i] * xs[i] + 1.0


@pytest.mark.parametrize('make', [lambda n: np.arange(n, dtype=float), lambda n: [float(i) for i in range(n)]],
                         ids=['array', 'list'])
@pytest.mark.parametrize('threads', [1, 2, 3, 8])
def test_split_over_a_thread_pool(make, threads):
    n = 1001
    xs, ys, expected = make(n), make(n), make(n)
    work.__wrapped__(xs, expected, 0, n)
    bounds = [(k * n // threads, (k + 1) * n // threads) for k in range(threads)]
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda b: work(xs, ys, *b), bounds))
    assert list(ys) == list(expected) and work.signatures


def test_unchanged_keeps_signed_zero_and_nan():
    @scram(cache=False)
    def touch(xs, i, v):
        xs[i] = v
    nan = float('nan')
    xs = [0.0, nan, 1.0]
    touch(xs, 0, -0.0)
    assert str(xs[0]) == '-0.0' and xs[1] is nan
    touch(xs, 2, nan)
    assert xs[2] != xs[2] and xs[1] is nan
