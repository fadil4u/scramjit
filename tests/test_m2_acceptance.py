"""M2's acceptance check, where it can be a test: the C for each M0 kernel has no iterator or boxing code."""

import os
import re
import sys

import numpy as np

from scramjit import scram

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kernels'))
import flood_fill as ff  # noqa: E402
import gray_scott as gs  # noqa: E402
import viewshed as vs  # noqa: E402

FORBIDDEN = re.compile(r'\bPy[A-Z]\w*|__next__|__iter__|\biter\b|\bnext\b|\bbox\b|\bunbox\b')


def generated(f, *args):
    g = scram(f, fallback=False, cache=False)
    g(*args)
    return g.c()


def check(c):
    assert not FORBIDDEN.findall(c)
    assert 'for (' in c


def test_gray_scott():
    u, v = np.ones((4, 4)), np.ones((4, 4))
    check(generated(gs.gray_scott, u, v, u.copy(), v.copy(), 0.16, 0.08, 0.035, 0.065, 1.0))


def test_flood_fill():
    check(generated(ff.flood_fill, np.ones((5, 5), np.uint8), np.zeros(25, np.int64), 1, 1, 7))


def test_viewshed():
    check(generated(vs.viewshed, np.ones((5, 5)), np.zeros((5, 5), np.uint8), 1, 1, 2.0))
