"""Enum and IntEnum members: held as their values, compared as Python compares them, and returned as members."""

import enum
import itertools
import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import outcome


class Level(enum.Enum):
    LOW = 1
    MID = 2
    HIGH = 3
    TOP = 3          # an alias: the same member as HIGH


class Pri(enum.IntEnum):
    A = 0
    B = 5
    C = -2


class Other(enum.Enum):
    LOW = 1


def classify(x: float):
    if x < 0.3:
        return Level.LOW
    if x < 0.6:
        return Level.MID
    return Level.HIGH


def count_high(a):
    n = 0
    for v in a:
        if classify_inline(v) is Level.HIGH:
            n += 1
    return n


@scram
def classify_inline(v):
    return Level.HIGH if v > 0.5 else Level.LOW


def weight(level, p):
    w = 0.0
    if level == Level.LOW:
        w = 1.0
    elif level is Level.HIGH:
        w = 3.0
    else:
        w = 2.0
    return w * (p + 1) + p.value + level.value


def compiled(f):
    g = scram(cache=False)(f)
    return g


def agree(f, *args):
    g = compiled(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        got = outcome(g, list(args))
    assert got == outcome(f, list(args)), (f.__name__, args)
    assert g.signatures, (f.__name__, [str(m.message) for m in w])
    return g


def test_returning_members():
    for x in (0.1, 0.4, 0.9):
        g = agree(classify, x)
        assert g(x) is classify(x)


def test_members_as_arguments():
    for level, p in itertools.product(Level, Pri):
        agree(weight, level, p)


OTHERS = [Level.LOW, Level.MID, Level.HIGH, Level.TOP, Other.LOW, Pri.A, Pri.B, Pri.C, 0, 1, 5, -2, 3, 2.0, 5.0, True]


@pytest.mark.parametrize('op', ['==', '!=', 'is', 'is not', '<', '>=', '+', '*', '-'])
def test_against_everything(op):
    space = {}
    exec(f'def f(a, b):\n    return a {op} b\n', space)
    f = space['f']
    for a, b in itertools.product(OTHERS, OTHERS):
        if isinstance(a, (int, float)) and not isinstance(a, enum.Enum) and isinstance(b, (int, float)) and not isinstance(b, enum.Enum):
            continue                                   # numbers alone: not what this is about
        g = scram(cache=False)(f)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert outcome(g, [a, b])[:3] == outcome(f, [a, b])[:3], (a, op, b)


def test_truth_and_not():
    def f(a):
        return 1 if a else 0, not a
    for a in list(Level) + list(Pri):
        assert scram(cache=False)(f)(a) == f(a)


def test_storing_a_member_stays_python():
    def f(out, level):
        out[0] = level
        return out[0]
    g = scram(cache=False)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert outcome(g, [np.zeros(1), Pri.B]) == outcome(f, [np.zeros(1), Pri.B])


class Custom(enum.Enum):
    X = 1

    def __eq__(self, other):
        return True

    __hash__ = enum.Enum.__hash__


class Words(enum.Enum):
    X = 'x'


@pytest.mark.parametrize('cls', [Custom, Words])
def test_unusual_enums_stay_python(cls):
    def f(a):
        return a == a
    g = scram(cache=False)(f)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(cls.X) == f(cls.X)
    assert not g.signatures


def test_count_in_a_loop():
    a = np.random.default_rng(1).random(1000)
    agree(count_high, a)


@scram
def bonus(level, p):
    return 10 if level is Level.HIGH else p * 2


@scram
def total(a, p):
    s = 0
    for v in a:
        s += bonus(classify_inline(v), p) + bonus(Level.LOW, Pri.B)
    return s


def test_members_passed_between_compiled_functions():
    a = np.random.default_rng(2).random(100)
    for p in Pri:
        with warnings.catch_warnings():
            warnings.simplefilter('error', FallbackWarning)
            assert total(a, p) == total.__wrapped__(a, p)
