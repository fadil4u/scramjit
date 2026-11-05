"""isinstance() compiles to a constant: of an instance, whose class is in the signature, against classes at module level;
of a number against int, float and bool. The answer is Python's, subclasses and bool being an int included."""

import dataclasses
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from test_classes import same


@dataclasses.dataclass
class Shape:
    size: float


class Square(Shape):
    pass


class Circle(Shape):
    pass


class Other:
    pass


def area(s):
    if isinstance(s, Square):
        return s.size * s.size
    if isinstance(s, (Circle, Other)):
        return 3.14159 * s.size * s.size
    return 0.0


def total(shapes):
    t = 0.0
    for s in shapes:
        if isinstance(s, Shape) and not isinstance(s, Circle):
            t += s.size
    return t


def only_asks(s):
    return 1 if isinstance(s, Other) else 2


def numbers(x, n, b):
    return (isinstance(x, float), isinstance(n, int), isinstance(b, int), isinstance(b, bool), isinstance(n, (float, bool)))


@pytest.mark.parametrize('f,make,compiled', [
    (area, lambda: [Square(2.0)], True),
    (area, lambda: [Circle(2.0)], True),
    (area, lambda: [Shape(2.0)], True),
    (total, lambda: [[Square(1.0), Square(2.0)]], True),
    (total, lambda: [[Circle(1.0)]], True),
    (only_asks, lambda: [Other()], True),
    (only_asks, lambda: ['text'], True),
    (numbers, lambda: [1.5, 3, True], True),
    (numbers, lambda: [np.float64(1.5), np.int64(3), False], True),
])
def test_isinstance(f, make, compiled):
    same(f, make, compiled)


def test_a_new_class_under_the_name_is_a_new_signature():
    global Other
    g = scram(cache=False)(only_asks)
    o = Other()
    assert g(o) == 1
    old, Other = Other, Square
    try:
        assert g(o) == 2 and g(o) == 2
    finally:
        Other = old
    assert g(o) == 1
