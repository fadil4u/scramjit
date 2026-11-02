"""For test_calls: a compiled function in another module, reading that module's globals."""

import numpy as np

from scramjit import scram

SCALE = 3.0
K = 7          # test_calls.py has a different K
TABLE = np.arange(4.0)


@scram
def scaled(x):
    return x * SCALE + K


@scram
def from_table(i):
    return TABLE[i] * SCALE


def set_scale(v):
    global SCALE
    SCALE = v
