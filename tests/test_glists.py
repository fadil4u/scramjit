"""Lists held by reference (_glist): lists that grow, nest and are shared, as Python's are. Each function's result
must be Python's: the same values, types and sharing, or the same exception with the same message."""

try:
    import resource
except ImportError:                     # Pyodide
    resource = None

import numpy as np
import pytest

from scramjit import scram
from scramjit._errors import CompileError


def same(f, *args):
    c = scram(f, fallback=False, cache=False)
    try:
        want = f(*args)
    except Exception as e:                                  # noqa: BLE001
        with pytest.raises(type(e)) as got:
            c(*args)
        assert str(got.value) == str(e)
        return None
    got = c(*args)
    assert got == want and shape(got) == shape(want), (got, want)
    return got


def shape(x):
    """Types, nesting and which inner lists are one object, as a comparable value."""
    ids = {}

    def walk(v):
        if isinstance(v, list):
            k = ids.setdefault(id(v), len(ids))
            return ('list', k, tuple(walk(y) for y in v))
        if isinstance(v, tuple):
            return ('tuple', tuple(walk(y) for y in v))
        return type(v).__name__
    return walk(x)


def evens(n):
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(i * 1.5)
    return out


def methods(n):
    out = [1, 2, 3]
    out.append(n)
    out.insert(0, -1)
    out.insert(-1, 99)
    out.insert(100, 7)
    out.insert(-100, 8)
    out.extend([10, 11])
    out += [12]
    out.extend(out)
    x = out.pop()
    y = out.pop(0)
    z = out.pop(-2)
    del out[1]
    del out[-1]
    out.append(x + y + z)
    return out


def cleared(n):
    out = []
    for i in range(n):
        out.append(i)
        if len(out) > 3:
            out.clear()
    return out


def nested(n):
    rows = []
    for i in range(n):
        row = []
        for j in range(i):
            row.append(j * i)
        rows.append(row)
    rows[0].append(5)
    return rows


def shared(n):
    rows = [[0.0] * 3] * n                    # one row, n times, as in Python
    rows[0][1] = 7.5
    other = []                                # n rows
    for _ in range(n):
        other.append([0.0] * 3)
    other[0][1] = 7.5
    return rows, other


def shared_return(n):
    rows = [[0.0] * 3] * n
    rows[0][1] = 7.5
    return rows


def alias(n):
    a = []
    b = a
    for i in range(n):
        b.append(i)
    c = [a, a, []]
    c[2].append(len(a))
    return c


def pairs(a):
    out = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] + a[j] > 1.0:
                out.append((i, j))
    return out


def tuples_in(n):
    out = [(i, i * 0.5) for i in range(n)]
    s = 0.0
    for i, h in out:
        s += i * h
    a, b = out[-1]
    out.append((a + 1, s))
    return out


def comprehensions(n):
    return [[i * j for j in range(i)] for i in range(n)]


def loops(n):
    rows = [[i + j for j in range(3)] for i in range(n)]
    s = 0
    for row in rows:
        for v in row:
            s += v
    for k, row in enumerate(rows):
        s += k * len(row)
    return s


def numpy_elements(a):
    out = []
    for i in range(len(a)):
        if a[i] > 0:
            out.append(a[i])
    return out


def flags(a):
    out = []
    for i in range(len(a)):
        out.append(a[i] > 0.5)
    return out


def never_appended(n):
    out = []
    return out


def index_errors(n, k):
    out = []
    for i in range(n):
        out.append(i)
    return out[k]


def store_errors(n, k):
    out = []
    for i in range(n):
        out.append(i)
    out[k] = 5
    return out


def pop_empty(n):
    out = []
    for i in range(n):
        out.append(i)
    out.pop()
    return out.pop()


def pop_index(n, k):
    out = []
    for i in range(n):
        out.append(i)
    return out.pop(k)


def del_error(n, k):
    out = []
    for i in range(n):
        out.append(i)
    del out[k]
    return out


def test_grow_and_return():
    same(evens, 10)
    same(evens, 0)
    same(evens, 10_000)
    same(methods, 4)
    same(cleared, 10)
    same(never_appended, 3)


def test_nesting_and_sharing():
    same(nested, 5)
    got = same(shared, 3)
    assert got[0][0] is got[0][1] and got[1][0] is not got[1][1]
    got = same(shared_return, 3)
    assert got[0] is got[2]
    got = same(alias, 4)
    assert got[0] is got[1]


def test_tuples_and_comprehensions():
    a = np.array([0.2, 0.9, 0.5, 0.7])
    same(pairs, a)
    same(tuples_in, 5)
    same(comprehensions, 5)
    same(loops, 4)


def test_element_types():
    a = np.array([0.5, -1.0, 2.0, 0.0])
    got = same(numpy_elements, a)
    assert all(type(v) is np.float64 for v in got)
    ints = np.arange(-2, 3)                                 # int32 on wasm32, int64 elsewhere
    got = same(numpy_elements, ints)
    assert all(type(v) is type(ints[0]) for v in got)
    got = same(flags, a)
    assert got == [False, False, True, False]


def test_errors_as_python():
    for k in (0, 2, -3, 3, -4):
        same(index_errors, 3, k)
        same(store_errors, 3, k)
        same(pop_index, 3, k)
        same(del_error, 3, k)
    same(pop_empty, 1)
    same(pop_empty, 0)


def mixed(n):
    out = []
    out.append(1)
    out.append(n * 0.5)
    return out


def changes_while_looping(n):
    out = [1.0, 2.0]
    for v in out:
        if len(out) < n:
            out.append(v)
    return out


def read_before_append(n):
    out = []
    x = out[0]
    return x


def table_and_list(n):
    rows = [[0.0] * 3] * n
    other = [[0.0] * 3 for _ in range(n)]       # a rectangular list of lists: the other representation
    return rows, other


def test_what_stays_python():
    for f, args in ((mixed, (3,)), (changes_while_looping, (5,)), (read_before_append, (1,)), (table_and_list, (2,))):
        with pytest.raises(CompileError):
            scram(f, fallback=False, cache=False)(*args)
        assert scram(f, cache=False)(*args) == f(*args) if f is not read_before_append else True


def churn(n):
    total = 0
    for i in range(n):
        row = []
        for j in range(100):
            row.append(j)
        total += len(row)
    return total


@pytest.mark.skipif(resource is None, reason='no resource module (Pyodide)')
def test_memory_is_freed():
    f = scram(churn, fallback=False, cache=False)
    f(10)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert f(200_000) == 20_000_000                     # 20M ints made and dropped: 160 MB if nothing were freed
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before < 60_000


def arrays_equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(arrays_equal(x, y) for x, y in zip(a, b))
    return a == b and type(a) is type(b)


def same_arrays(f, *args):
    want, got = f(*args), scram(f, fallback=False, cache=False)(*args)
    assert arrays_equal(want, got), (want, got)
    return got


def chunks(x, k):
    out = []
    for i in range(0, len(x), k):
        out.append(x[i:i + k])
    return out


def made(n):
    out = []
    for i in range(n):
        a = np.zeros(i + 1)
        a[i] = i
        out.append(a)
    return out


def grid(n):
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(np.full(2, i * 10.0 + j))
        rows.append(row)
    return rows


def literal_arrays(x):
    out = [x, x * 2.0, np.sqrt(x)]
    tot = 0.0
    for a in out:
        tot += a[1]
    last = out[-1]
    return out, tot + last[0] + out[0][1]


def twice(x):
    a = x * 2.0
    return [a, a, x, x]


def churn_arrays(n):
    total = 0.0
    for i in range(n):
        parts = []
        for j in range(10):
            parts.append(np.ones(100) * j)
        total += parts[9][0]
    return total


def test_lists_of_arrays():
    x = np.arange(10.0)
    got = same_arrays(chunks, x, 3)
    got[0][0] = 99.0
    assert x[0] == 99.0                                    # views of the argument, as numpy's are
    same_arrays(made, 4)
    same_arrays(grid, 3)
    same_arrays(literal_arrays, np.arange(4.0))
    got = same_arrays(twice, x)
    assert got[0] is got[1] and got[2] is x and got[3] is x


@pytest.mark.skipif(resource is None, reason='no resource module (Pyodide)')
def test_arrays_in_lists_are_freed():
    f = scram(churn_arrays, fallback=False, cache=False)
    f(10)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert f(20_000) == churn_arrays(20_000)               # 200,000 arrays of 800 bytes: 160 MB if nothing were freed
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before < 60_000


# ---- passed in, written back (on errors too), shared and returned as themselves
import copy                                                       # noqa: E402
from dataclasses import dataclass                                 # noqa: E402


def deep_equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return type(b) is dict and list(a) == list(b) and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, '__dict__') and not isinstance(a, type):
        return type(a) is type(b) and deep_equal(vars(a), vars(b))
    return a == b and type(a) is type(b)


def same_passed(f, *args):
    """f compiled changes its arguments as Python's f does, and returns or raises the same."""
    py_args = copy.deepcopy(args)
    try:
        want, werr = f(*py_args), None
    except Exception as e:                                        # noqa: BLE001
        want, werr = None, e
    try:
        got, gerr = scram(f, fallback=False, cache=False)(*args), None
    except Exception as e:                                        # noqa: BLE001
        got, gerr = None, e
    assert type(werr) is type(gerr) and str(werr) == str(gerr), (werr, gerr)
    assert deep_equal(list(py_args), list(args)), (py_args, args)
    assert deep_equal(want, got), (want, got)
    return got


def fill(out, n):
    for i in range(n):
        out.append(i * 0.5)


def grow_ragged(rows):
    s = 0
    for r in rows:
        for v in r:
            s += v
    rows[0].append(s)
    rows.append([len(rows)])
    return s


def write_shared(rows):
    rows[0][0] = 9
    return rows[1][0]


def grow_arrays(parts):
    parts.append(parts[0] * 2.0)
    parts[1][0] = -1.0
    return len(parts)


def fail_midway(out, n):
    for i in range(n):
        out.append(i)
    return out[n + 5]


def give_back(out):
    out.append(3)
    return out


def grow_tuples(ps):
    s = 0.0
    for a, b in ps:
        s += a * b
    ps.append((1, s))
    return s


def test_lists_passed_in():
    same_passed(fill, [], 5)
    same_passed(fill, [1.5], 3)
    same_passed(grow_ragged, [[1, 2], [3], []])
    r = [1, 2]
    rows = [r, r]
    same_passed(write_shared, rows)
    assert rows[0] is rows[1]
    same_passed(grow_arrays, [np.arange(3.0), np.zeros(2)])
    same_passed(fail_midway, [], 3)                         # IndexError, after the appends Python made
    x = []
    assert scram(give_back, fallback=False, cache=False)(x) is x and x == [3]
    same_passed(grow_tuples, [(1, 2.0), (3, 4.0)])


@scram
def evens_of(n):
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(i)
    return out


@scram
def fill_doubles(out, n):
    for i in range(n):
        out.append(i * 2)


@scram
def calls_them(n):
    a = evens_of(n)
    b = []
    fill_doubles(b, 3)
    s = 0
    for v in a:
        s += v
    big = 0
    for _ in range(200):                                    # sweeps while it holds what they made
        big += len(evens_of(1000))
    return a, b, s, big


def test_calls_between_compiled_functions():
    assert calls_them(10) == calls_them.__wrapped__(10)
    assert calls_them.signatures


def array_sums(parts):
    return [p.sum() for p in parts]


def test_comprehension_over_arrays():
    got = same_passed(array_sums, [np.arange(3.0), np.ones(4)])
    assert all(type(v) is np.float64 for v in got)


def make_records(n):
    out = []
    for i in range(n):
        out.append({'i': i, 'sq': i * i * 0.5})
    out[0]['sq'] = -1.0
    r = out[1]
    r['i'] = 100
    s = 0.0
    for rec in out:
        s += rec['sq']
    return out, s


def dict_of_list(n):
    d = {'xs': [], 'n': 0}
    for i in range(n):
        d['xs'].append(i)
        d['n'] += 1
    return d


def groups_of(n):
    groups = []
    for g in range(n):
        groups.append({'id': g, 'members': [], 'pos': np.zeros(2)})
        for k in range(g):
            groups[g]['members'].append((k, k * 1.5))
        groups[g]['pos'][0] = g
    return groups


def bump_records(rows):
    for r in rows:
        r['n'] += 1
        r['hist'].append(r['n'] * 0.5)
    rows.append({'n': 0, 'hist': []})


def into_dict(d, k):
    for i in range(k):
        d['items'].append(i)
    d['count'] = len(d['items'])
    return d['count']


def test_dicts():
    same_passed(make_records, 4)
    same_passed(dict_of_list, 5)
    same_passed(groups_of, 3)
    rows = [{'n': 1, 'hist': [0.5]}, {'n': 5, 'hist': []}]
    inner, first = rows[0]['hist'], rows[0]
    same_passed(bump_records, rows)
    assert rows[0] is first and rows[0]['hist'] is inner
    same_passed(into_dict, {'items': [], 'count': 0}, 4)


@dataclass
class Pt:
    x: float
    y: float
    hits: int = 0


class Bin:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.items = []


def spawn(n):
    out = []
    for i in range(n):
        out.append(Pt(i * 0.5, i * 2.0))
    out[0].hits += 3
    q = out[1]
    q.x = -1.0
    s = 0.0
    for p in out:
        s += p.x + p.y
    return out, s


def bins(n):
    bs = []
    for i in range(n):
        b = Bin(i, i + 1)
        for k in range(i):
            b.items.append(k)
        bs.append(b)
    return bs


def move(ps, dx):
    for p in ps:
        p.x += dx
        p.hits += 1
    ps.append(Pt(0.0, 0.0))


def test_instances():
    got = same_passed(spawn, 4)
    assert type(got[0][0]) is Pt
    same_passed(bins, 3)
    ps = [Pt(1.0, 2.0), Pt(3.0, 4.0)]
    first = ps[0]
    same_passed(move, ps, 0.5)
    assert ps[0] is first


@dataclass
class Body:
    x: float
    v: float

    def step(self, dt):
        self.x += self.v * dt

    def energy(self):
        return 0.5 * self.v * self.v

    def bounce(self, other):
        self.v, other.v = other.v, self.v


def simulate_bodies(n, steps):
    bs = []
    for i in range(n):
        bs.append(Body(i * 1.0, 1.0 - i * 0.5))
    for _ in range(steps):
        for b in bs:
            b.step(0.1)
    bs[0].bounce(bs[1])
    e = 0.0
    for b in bs:
        e += b.energy()
    return bs, e


def drive_bodies(bs, steps):
    for _ in range(steps):
        for b in bs:
            b.step(0.01)
    bs.append(Body(0.0, 0.0))
    return bs[0].energy()


def test_methods_of_instances_in_lists():
    same_passed(simulate_bodies, 3, 5)
    bs = [Body(1.0, 2.0), Body(0.0, -1.0)]
    first = bs[0]
    same_passed(drive_bodies, bs, 10)
    assert bs[0] is first


def slices(n):
    xs = [i * 1.5 for i in range(n)]
    a = xs[2:5]
    b = xs[::-2]
    c = xs[-3:]
    d = xs[100:]
    e = xs[1:-1:3]
    xs[1:3] = [9.0, 9.5, 10.0]
    xs[::2] = [0.0] * len(xs[::2])
    del xs[:2]
    del xs[::3]
    xs[0:0] = xs
    return a, b, c, d, e, xs


def step_of(xs, s):
    return xs[::s]


def short_extended(n):
    xs = [1, 2, 3, 4]
    xs[::2] = [5]
    return xs


def rows_slice(m):
    top = m[:1]
    top[0].append(99)                                       # a shallow copy: the row is the same list
    return m[1:], top


def test_slices():
    for n in (0, 1, 5, 12):
        same_passed(slices, n)
    for s in (1, 2, -1, -3, 0):                             # 0: ValueError, as Python
        same_passed(step_of, [1, 2, 3, 4, 5], s)
    same_passed(short_extended, 1)
    m = [[1], [2, 3]]
    got = same_passed(rows_slice, m)
    assert got[1][0] is m[0]


def members(n, q):
    xs = []
    for i in range(n):
        xs.append(i * i)
    hits = 0
    for k in range(n * n):
        if k in xs:
            hits += 1
    return hits, q in xs, q not in xs, (2, 4) in [(1, 2), (2, 4)], q in (1, 4, 9), 2.0 in xs


def has_keys(d):
    return 'a' in d, 'z' in d, 'z' not in d


def test_in():
    for q in (0, 9, 10, -1):
        same_passed(members, 6, q)
    same_passed(has_keys, {'a': [1], 'b': 2.0})
