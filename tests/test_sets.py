"""Sets (_gset), kept as CPython keeps them: the same elements in the same order, the same pop, and a set passed in
or returned left with the very table Python would have left, so what Python does with it next goes as it would.

Each result is compared as list(s): the iteration order, not only the elements."""

import importlib.util
import math
import random
from dataclasses import dataclass

import pytest

from scramjit import scram, _pysets
from scramjit._errors import CompileError


def ordered(v):
    """A value with every set in it as the list of its elements, in order. Without CPython's set layout (free-threaded
    builds: _pysets.exact is False) a set handed back is rebuilt from its elements, in an order of its own: its elements."""
    if isinstance(v, set):
        return ('set', [ordered(x) for x in v] if _pysets.exact else sorted((ordered(x) for x in v), key=repr))
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, [ordered(x) for x in v])
    if isinstance(v, dict):
        return ('dict', [(k, ordered(x)) for k, x in v.items()])
    if isinstance(v, float):
        return ('float', repr(v))
    return (type(v).__name__, v)


def same(f, *args, copy_args=None):
    import copy
    py_args = copy.deepcopy(args) if copy_args is None else copy_args(args)
    try:
        want, werr = f(*py_args), None
    except Exception as e:                                        # noqa: BLE001
        want, werr = None, e
    try:
        got, gerr = scram(f, fallback=False, cache=False)(*args), None
    except CompileError:
        raise
    except Exception as e:                                        # noqa: BLE001
        got, gerr = None, e
    assert type(werr) is type(gerr) and str(werr) == str(gerr), (werr, gerr)
    assert ordered(want) == ordered(got), (want, got)
    assert ordered(list(py_args)) == ordered(list(args)), (py_args, args)
    return got


def build(n):
    s = set()
    for i in range(n):
        s.add(i * 7919 % 101)
    s.discard(3)
    s.discard(5000)
    return s, len(s), 3 in s, 17 in s, 2.0 in s, 17.5 not in s


def operators(n, m):
    a = {i * 3 for i in range(n)}
    b = {i * 5 for i in range(m)}
    out = [a | b, a & b, a - b, b - a, a ^ b, a.union(b), a.intersection(b), a.difference(b), a.symmetric_difference(b)]
    c = set(a)
    c |= b
    d = set(a)
    d &= b
    e = set(a)
    e -= b
    f = set(a)
    f ^= b
    out += [c, d, e, f]
    return out, a == b, a <= c, c >= a, a < c, d.issubset(a), a.isdisjoint(b), a.issuperset(d)


def popping(n):
    s = {i * 13 % 47 for i in range(n)}
    got = []
    for _ in range(n // 2):
        got.append(s.pop())
    s.add(1000)
    got.append(s.pop())
    return got, s


def literals(x):
    a = {1, 2, 3, -4, 17, 33, 2 * 8, (5)}
    b = {x, 2, 3}
    c = {x, x + 8}
    return a, b, c, a | b | c


def floats(n):
    s = set()
    for i in range(n):
        s.add(i * 0.37 - 5.0)
    s.add(-0.0)
    s.add(0.0)
    s.add(math.inf)
    s.add(2.0 ** 70)
    s.discard(0.37 - 5.0)
    return s


def tuples(n):
    s = set()
    for i in range(n):
        s.add((i % 5, (i % 3, i * 0.5)))
    return s, (2, (1, 3.5)) in s, len(s)


def from_iterables(xs, n):
    a = set(xs)
    b = set(range(n, 0, -3))
    c = set(x * 2 for x in xs)
    d = {x for x in xs if x % 2}
    a.update(b)
    a.update([1, 2, 3])
    total = 0
    for x in a:
        total = (total * 31 + x) % 1000003
    return a, b, c, d, total, sorted(a)[:5]


def grows(s, n):
    for i in range(n):
        s.add(i * 11)
        if i % 3 == 0:
            s.discard(i)
    return len(s)


def in_lists(n):
    rows = []
    for i in range(n):
        rows.append({j * i for j in range(5)})
    return rows, rows[1] == rows[2], {0} in rows


def errors(x):
    s = {1, 2, 3}
    s.remove(x)
    return s


def empty_pop(n):
    s = {n}
    s.pop()
    return s.pop()


def test_build_and_order():
    for n in (0, 1, 5, 9, 50, 300):
        same(build, n)


def test_operators():
    for n, m in ((0, 3), (5, 3), (40, 7), (3, 40), (100, 100), (200, 9)):
        same(operators, n, m)


def test_pop_literals_floats_tuples():
    for n in (2, 9, 40):
        same(popping, n)
    for x in (0, 9, -4, 100):
        same(literals, x)
    for n in (0, 10, 100):
        same(floats, n)
        same(tuples, n)


def test_iterables_lists_and_errors():
    same(from_iterables, [5, 3, 8, 3, 1, 99, 1024], 20)
    same(in_lists, 6)
    for x in (2, 7):
        same(errors, x)
    same(empty_pop, 3)


@pytest.mark.skipif(not _pysets.exact, reason="no CPython set layout here (free-threaded): sets are rebuilt, in their own order")
def test_passed_in_and_written_back():
    for start in (set(), {1, 2, 3}, set(range(0, 200, 7)), {5, 13, 21, 29}):
        s = set(start)
        for x in list(s)[::3]:
            s.discard(x)                                  # dummies in its table, as Python left them
        table = _pysets.read(s)
        u = _pysets.make(*table[:4], table[4])          # the same set, table and all
        r1 = scram(grows, fallback=False, cache=False)(s, 40)
        r2 = grows(u, 40)
        assert r1 == r2 and list(s) == list(u)
        for x in (4, 8000, 77):                           # Python goes on with it as with its own
            s.add(x)
            u.add(x)
        assert list(s) == list(u) and s.pop() == u.pop()
    assert _pysets.exact


def random_program(rnd, k):
    lines = [f'def f{k}(n):', '    a = set()', '    b = set()']
    for _ in range(rnd.randrange(5, 40)):
        w = rnd.choice('ab')
        o = 'b' if w == 'a' else 'a'
        r = rnd.random()
        v = rnd.choice(['n * {}', '{}', 'n + {}', '{} * 64', '-{}'])
        v = v.format(rnd.randrange(0, 300))
        if r < 0.4:
            lines.append(f'    {w}.add({v})')
        elif r < 0.55:
            lines.append(f'    {w}.discard({v})')
        elif r < 0.6:
            lines.append(f'    if {w}:\n        {w}.pop()')
        elif r < 0.7:
            lines.append(f'    {w} = {rnd.choice(["a | b", "a & b", "a - b", "b - a", "a ^ b"])}')
        elif r < 0.8:
            lines.append(f'    {w} {rnd.choice(["|=", "&=", "-=", "^="])} {o}')
        elif r < 0.85:
            lines.append(f'    {w} = set({o})')
        elif r < 0.9:
            lines.append(f'    {w}.update(range({v} % 50))')
        elif r < 0.93:
            lines.append(f'    {w}.clear()')
        else:
            lines.append(f'    for x in range({v} % 30):\n        {w}.add(x * {rnd.randrange(1, 70)})')
    lines.append('    return a, b')
    return '\n'.join(lines)


def test_random_programs(tmp_path):
    rnd = random.Random(7)
    src = '\n\n\n'.join(random_program(rnd, k) for k in range(60))
    path = tmp_path / 'set_programs.py'
    path.write_text(src + '\n')
    spec = importlib.util.spec_from_file_location('set_programs', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in range(60):
        f = getattr(mod, f'f{k}')
        for n in (0, 3, 50):
            same(f, n)


@dataclass
class Box:
    tags: set


def nan_set(x):
    s = {1.0}
    s.add(x)
    return s


def test_what_stays_python():
    with pytest.raises(CompileError):
        scram(lambda_mixed, fallback=False, cache=False)(1)


def lambda_mixed(n):
    s = {1, 2.5}                                          # one element type per set
    return s
