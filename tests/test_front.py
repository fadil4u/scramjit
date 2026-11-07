"""A Function's __call__ of its own arity (the front): calls with the last plan's types go straight to it. Every other
call takes the general path, and must behave exactly as before."""

import numpy as np
import pytest

from scramjit import scram
from scramjit._dispatch import Function

K = 3


@scram(fallback=False)
def add(a, b):
    return a + b


@scram
def add_any(a, b):
    return a + b


@scram(fallback=False)
def scaled(x, k=2.0):
    return x * k


@scram(fallback=False)
def reads_global(x):
    return x + K


@scram(fallback=False)
def names(_sj_a0, a0, self):
    return _sj_a0 * 100 + a0 * 10 + self


@scram(fallback=False)
def none():
    return 7


class Point:
    def __init__(self, x):
        self.x = x

    @scram(fallback=False)
    def moved(self, d):
        return self.x + d


def test_still_a_function():
    assert isinstance(add, Function) and type(add).__name__ == 'Function' and add.__name__ == 'add'
    assert type(add) is not Function            # its front: a subclass


def test_alternating_types_and_values():
    for _ in range(3):
        assert add(1, 2) == 3 and type(add(1, 2)) is int
        assert add(1.5, 2.25) == 3.75 and type(add(1.5, 2.25)) is float
        assert add(np.float64(1.5), 1.0) == 2.5
        assert add_any(1, 2) == 3 and add_any(2 ** 70, 1) == 2 ** 70 + 1    # past 64 bits: the general way, as Python


def test_keywords_defaults_and_wrong_counts():
    assert scaled(3.0) == 6.0 and scaled(3.0, 4.0) == 12.0 and scaled(x=3.0) == 6.0 and scaled(3.0, k=0.5) == 1.5
    assert add(b=2, a=5) == 7 and add(5, b=2) == 7
    for call in (lambda: add(1), lambda: add(1, 2, 3), lambda: add(1, c=2), lambda: add(1, 2, a=3)):
        with pytest.raises(TypeError):
            call()


def test_parameter_names_like_the_fronts():
    assert names(1, 2, 3) == 123 and names(1, 2, self=3) == 123 and names(1, a0=2, self=3) == 123
    assert names(_sj_a0=1, a0=2, self=3) == 123


def test_no_arguments():
    assert none() == 7 and none() == 7
    with pytest.raises(TypeError):
        none(1)


def test_methods():
    p = Point(1.5)
    assert p.moved(2.0) == 3.5 and p.moved(2.0) == 3.5 and Point.moved(p, 1.0) == 2.5


def test_a_global_of_another_type_after_the_fast_path():
    global K
    try:
        assert reads_global(1) == 4 and reads_global(1) == 4
        K = 2.5
        assert reads_global(1) == 3.5 and reads_global(1) == 3.5
    finally:
        K = 3
    assert reads_global(1) == 4
