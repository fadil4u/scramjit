"""Tuples inside tuples: as locals, arguments and results, indexed and unpacked, as in Python."""

import warnings

import numpy as np
import pytest

from scramjit import FallbackWarning, scram

from util import outcome


def literal(x: float):
    t = ((x, 2), 3.5)
    return t[0][0] + t[0][1] * t[1], t


def unpack(p):
    (a, b), c = p
    return a * b + c, ((c, a), b)


def through_variables(p):
    q = p[0]
    r = (q, p[1], (q[1], q[0]))
    return r[2][0] - r[0][1] + r[1]


def mixed_kinds(p, k: int):
    s = 0.0
    for i in range(k):
        s += p[0][0] * i + p[1][1][0]
    return s, (p[1][1], s)


def loop_state(a):
    best = ((0.0, -1), (0.0, -1))
    for i in range(len(a)):
        v = a[i]
        if v > best[0][0]:
            best = ((v, i), best[0])
        elif v > best[1][0]:
            best = (best[0], (v, i))
    return best


@scram
def area(box):
    (x0, y0), (x1, y1) = box
    return (x1 - x0) * (y1 - y0)


@scram
def total_area(n: int):
    s = 0.0
    for i in range(n):
        s += area(((0.0, 0.0), (i * 1.0, 2.0)))
    return s


def strict(f, *args):
    g = scram(cache=False)(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        got = outcome(g, list(args))
    assert got == outcome(f, list(args)), (f.__name__, args)
    assert g.signatures, (f.__name__, [str(m.message) for m in w])


@pytest.mark.parametrize('x', [0.5, -2.0])
def test_literals(x):
    strict(literal, x)


@pytest.mark.parametrize('p', [((1, 2), 3), ((1.5, -2.0), 0.25), ((True, 2), 3)])
def test_arguments_unpacked(p):
    strict(unpack, p)
    strict(through_variables, p)


def test_three_deep():
    strict(mixed_kinds, ((1.5, 2), (3, (4.0, 5))), 4)


def test_as_loop_state():
    strict(loop_state, np.random.default_rng(0).random(50))
    strict(loop_state, np.zeros(0))


def test_passed_to_a_compiled_function():
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        assert total_area(5) == total_area.__wrapped__(5)
        assert area(((1.0, 1.0), (3.0, 4.0))) == 6.0


BOXES = (((0.0, 0.0), (1.0, 2.0)), ((1.0, 1.0), (4.0, 3.0)))


@scram
def boxes_area():
    s = 0.0
    for box in (BOXES[0], BOXES[1]):
        (x0, y0), (x1, y1) = box
        s += (x1 - x0) * (y1 - y0)
    return s


def test_a_nested_global():
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        assert boxes_area() == boxes_area.__wrapped__()


def test_preflight_reads_a_nested_global(tmp_path):
    import subprocess, sys
    (tmp_path / 'p.py').write_text('from scramjit import scram\nT = ((1.0, 2), 3)\n@scram\ndef f(x: float):\n    (a, b), c = T\n    return a * x + b + c\n')
    r = subprocess.run([sys.executable, '-m', 'scramjit', 'preflight', 'p.py'], cwd=tmp_path, capture_output=True, text=True)
    assert '✓ f' in r.stdout, r.stdout + r.stderr


def over_numbers(x: float):
    s = 0.0
    for w in (0.5, 1.5, x):
        if w > 10:
            break
        s += w * x
    return s


def over_pairs(p, q):
    s = 0.0
    for (a, b) in (p, q, (1.0, 2.0)):
        s += a * b
    for t in (p, q):
        s -= t[0]
        if t[1] < 0:
            continue
        s += 1
    return s


def mixed_elements(k: int):
    s = 0
    for v in (1, 2, k):
        s += v
    return s


def differing_elements(x: float):
    for v in (1, x):
        x += v
    return x


@pytest.mark.parametrize('x', [0.5, 20.0])
def test_loops_over_tuples(x):
    strict(over_numbers, x)
    strict(over_pairs, (x, 2.0), (3.0, -1.0))
    strict(mixed_elements, 3)


def test_a_tuple_of_differing_types_stays_python():
    g = scram(cache=False)(differing_elements)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FallbackWarning)
        assert g(1.5) == differing_elements(1.5)
    assert not g.signatures and 'differ in type' in str(g._why)


def comprehensions(p):
    xs = [a * b for a, b in (p, (2.0, 3.0))]
    return xs[0] + xs[1] + sum((v for v in (1.0, p[0])), 0.0) + max(b for _, b in (p, (0.0, 9.0)))


def test_comprehensions_over_tuples():
    strict(comprehensions, (1.5, 4.0))
