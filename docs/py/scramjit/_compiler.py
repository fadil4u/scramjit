"""From a function's syntax tree and its argument types to C.

Three passes over the tree: infer a type for every variable (Python and numpy semantics, see
_types), check that every variable is assigned before it is read, then emit C. Anything outside
the compiled subset raises Unsupported with the Python line, and the function runs as Python.

Expressions are emitted so that Python's evaluation order holds: anything that can raise (an
index, a division, a conversion) is computed into a temporary, checked, in the order Python
evaluates it; what is left inline is pure arithmetic on values already computed.

Every emitted expression is a pair (value, flag): the C value, and a C expression that is 1
when the value is a numpy scalar rather than a Python number. The flag is the constant '0' or '1'
whenever the type says so statically, and a runtime value for variables that hold both (see
_types: INT_DYN and friends). numpy-ness decides division by zero (numpy returns inf, nan or 0
where Python raises) and the type a value is returned as.
"""

import ast
import copy
import types
import math
import re

from . import _types as T
from . import _webs
from ._errors import Unsupported
import sys

from ._mathlib import snippets
from . import _mathlib
from ._runtime import (ATTRIBUTE_NONE, POOL_FULL, attribute_number, ARANGE_ERROR, BROADCAST_ERROR, INDEX_ERROR, INTERRUPTED, LINSPACE_ERROR, MAKE_DTYPES, MAKE_ERROR, NO_AXIS, RERUN, RETURNS_INT, VALUE_ERROR_MESSAGES, INT_BOUNDS_ERROR, LIBM, LIST_INDEX_ERROR, MEMORY_ERROR, NARROW_INT_DTYPES,
                       OVERFLOW_ERROR, VALUE_ERROR, ZERO_DIVISION)

BUILTINS = ('range', 'len', 'abs', 'min', 'max', 'int', 'float', 'round', 'sum', 'any', 'all')
MATH_CONSTANTS = {'pi': math.pi, 'e': math.e, 'tau': math.tau, 'inf': math.inf, 'nan': math.nan}
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
ARITH = {ast.Add: 'Add', ast.Sub: 'Sub', ast.Mult: 'Mult', ast.Div: 'Div', ast.FloorDiv: 'FloorDiv',
         ast.Mod: 'Mod', ast.BitAnd: 'BitAnd', ast.BitOr: 'BitOr', ast.BitXor: 'BitXor', ast.Pow: 'Pow',
         ast.LShift: 'LShift', ast.RShift: 'RShift'}
CMP = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
IS = {ast.Is: 'is', ast.IsNot: 'is not'}      # only between enum members
SWAP = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==', '!=': '!='}

# math functions: name -> (kind, can_overflow). 'f1' wraps a one-argument C library function the
# way CPython's math_1 does; '1a' like math_1a (no special-value checks); the rest are CPython's
# own logic (see _runtime).
MATH_FUNCS = {
    **{n: ('f1', 0) for n in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
                               'tanh', 'cbrt', 'log1p', 'sqrt', 'fabs')},
    **{n: ('f1', 1) for n in ('sinh', 'cosh', 'exp', 'exp2', 'expm1')},
    'erf': ('1a', 0), 'erfc': ('1a', 0),
    'log': ('log', 0), 'log2': ('log', 0), 'log10': ('log', 0),
    'atan2': ('atan2', 0), 'pow': ('pow', 0), 'fmod': ('fmod', 0), 'copysign': ('copysign', 0),
    'hypot': ('hypot', 0), 'floor': ('round', 0), 'ceil': ('round', 0), 'trunc': ('round', 0),
    'degrees': ('scale', 0), 'radians': ('scale', 0),
    'isnan': ('test', 0), 'isinf': ('test', 0), 'isfinite': ('test', 0),
    # ported from mathmodule.c (_mathlib.py)
    'gamma': ('gamma', 0), 'lgamma': ('gamma', 0), 'remainder': ('remainder', 0), 'nextafter': ('nextafter', 0),
    'ulp': ('ulp', 0), 'fma': ('fma', 0), 'ldexp': ('ldexp', 0), 'isclose': ('isclose', 0), 'frexp': ('frexp', 0),
    'modf': ('modf', 0), 'isqrt': ('intf', 0), 'factorial': ('intf', 0), 'gcd': ('intf', 0), 'lcm': ('intf', 0),
    'comb': ('intf', 0), 'perm': ('intf', 0), 'fsum': ('iter', 0), 'prod': ('iter', 0), 'dist': ('iter', 0),
    'sumprod': ('iter', 0),
}


def c_float(x):
    if math.isnan(x):
        return 'NAN'
    if math.isinf(x):
        return 'INFINITY' if x > 0 else '(-INFINITY)'
    return x.hex()        # a C99 hex float: exact


def c_int(x):
    if x == INT64_MIN:
        return '(-9223372036854775807LL - 1)'
    return f'{x}LL'


def numpy_dtype(elem):
    """The numpy dtype name of an element type; a list of lists' Python numbers are as numpy would hold them."""
    return {'int': 'int64', 'float': 'float64', 'bool': 'int32'}.get(elem.name, elem.name[3:])     # a list's bools are C ints


FORWARD_DEPTH = 4     # y[i - 1] to y[i - 4] can be forwarded from temporaries
# cimo's compile time grows with the square of a function's size (a 60-loop function: 3,140 lines of C in 4.0 s,
# 6,020 with every loop versioned in 14.7 s), so versioning's copies may bring a function to VERSION_TOTAL syntax nodes
# (about 3 s), with VERSION_MIN at least for its innermost loops; past that a loop keeps its checks.
VERSION_TOTAL = 3200
VERSION_MIN = 400


def const_int(node):
    """The Python value of an int expression made only of literals ((1 << 64) - 1, 2 ** 63), or None. Kept small."""
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is int else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
        v = const_int(node.operand)
        return None if v is None else {ast.USub: -v, ast.UAdd: v, ast.Invert: ~v}[type(node.op)]
    if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.LShift, ast.RShift, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor):
        a, b = const_int(node.left), const_int(node.right)
        if a is None or b is None or (type(node.op) in (ast.LShift, ast.Pow) and not 0 <= b <= 128) or (type(node.op) is ast.RShift and b < 0):
            return None
        v = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.BitAnd: a & b, ast.BitOr: a | b, ast.BitXor: a ^ b}.get(type(node.op)) \
            if type(node.op) not in (ast.LShift, ast.RShift, ast.Pow) else (a << b if type(node.op) is ast.LShift else a >> b if type(node.op) is ast.RShift else a ** b)
        return v if abs(v) < 1 << 128 else None
    return None


def folds(node):
    """An expression of literals whose value only Python's unbounded ints get right: it compiles as that value."""
    if isinstance(node, ast.Constant) or not isinstance(node, (ast.BinOp, ast.UnaryOp)):
        return None
    v = const_int(node)
    return v if v is not None and -(1 << 63) <= v < 1 << 64 else None


def const_number(node):
    """The value of a number literal, signed or not, else None."""
    sign = 1
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -sign if isinstance(node.op, ast.USub) else sign
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sign * node.value
    return None


def flag_of(t):
    """The static flag of a type, or None if it is only known at run time."""
    return '1' if t.np is True else '0' if t.np is False else None


def flag_or(a, b):
    if a == '1' or b == '1':
        return '1'
    if a == '0':
        return b
    if b == '0':
        return a
    return f'({a} | {b})'


UNSUPPORTED_EXPRESSIONS = {'Dict': "dicts aren't compiled", 'DictComp': "dict comprehensions aren't compiled",
                           'Set': "sets aren't compiled", 'SetComp': "set comprehensions aren't compiled",
                           'JoinedStr': "f-strings aren't compiled", 'Lambda': "a lambda outside map() or filter() isn't compiled",
                           'Slice': "slices aren't compiled", 'Starred': "argument unpacking isn't compiled",
                           'Await': "await isn't compiled", 'Yield': "yield isn't compiled", 'YieldFrom': "yield from isn't compiled",
                           'NamedExpr': "the walrus operator isn't compiled"}
OPERATOR_NAMES = {'MatMult': "the @ operator isn't compiled"}


class Lazy(ast.NodeTransformer):
    """map() and filter() as what they are: a comprehension, or, in a for statement, a plain loop.
    Only the builtins themselves (not names rebound in the function or its module), with a
    lambda, a builtin or math function, or (filter) None."""

    def __init__(self, names, global_names):
        self.bound = names | global_names
        self.n = 0

    def fresh(self):
        self.n += 1
        return f'_sj_{self.n}'

    def builtin(self, e, which):
        return isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == which \
            and which not in self.bound and not e.keywords

    def apply(self, f, args, at):
        """f(*args) as an expression; args are Name nodes. A lambda is inlined."""
        if isinstance(f, ast.Lambda):
            a = f.args
            names = [p.arg for p in a.posonlyargs + a.args]
            if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or len(names) != len(args):
                return None
            inner = {n: x.id for n, x in zip(names, args)}

            class Rename(ast.NodeTransformer):
                def visit_Name(self, n):
                    if n.id in inner:
                        return ast.copy_location(ast.Name(inner[n.id], n.ctx), n)
                    return n
            return Rename().visit(copy.deepcopy(f.body))
        if isinstance(f, (ast.Name, ast.Attribute)):
            return ast.copy_location(ast.Call(copy.deepcopy(f), list(args), []), at)
        return None

    def parts(self, e):
        """(comprehension target, iterable, value expression, condition) for a map or filter call."""
        if self.builtin(e, 'map') and len(e.args) >= 2:
            vs = [self.fresh() for _ in e.args[1:]]
            load = [ast.copy_location(ast.Name(v, ast.Load()), e) for v in vs]
            value = self.apply(e.args[0], load, e)
            if value is None:
                return None
            if len(vs) == 1:
                target, it = ast.copy_location(ast.Name(vs[0], ast.Store()), e), e.args[1]
            else:                                       # map(f, a, b) stops at the shortest, as zip does
                target = ast.copy_location(ast.Tuple([ast.copy_location(ast.Name(v, ast.Store()), e) for v in vs], ast.Store()), e)
                it = ast.copy_location(ast.Call(ast.copy_location(ast.Name('zip', ast.Load()), e), list(e.args[1:]), []), e)
            return target, it, value, None
        if self.builtin(e, 'filter') and len(e.args) == 2:
            v = self.fresh()
            load = ast.copy_location(ast.Name(v, ast.Load()), e)
            none = isinstance(e.args[0], ast.Constant) and e.args[0].value is None
            cond = load if none else self.apply(e.args[0], [load], e)
            if cond is None:
                return None
            return ast.copy_location(ast.Name(v, ast.Store()), e), e.args[1], load, cond
        return None

    def visit_Call(self, e):
        self.generic_visit(e)
        if self.builtin(e, 'pow') and len(e.args) == 2:     # pow(a, b) is a ** b
            return ast.copy_location(ast.BinOp(e.args[0], ast.Pow(), e.args[1]), e)
        p = self.parts(e)
        if p is None:
            return e
        target, it, value, cond = p
        return self.comp(ast.copy_location(ast.GeneratorExp(value, [ast.comprehension(target, it, [cond] if cond is not None else [], 0)]), e))

    def comp(self, e):
        """A comprehension over map() or filter(): one flat loop. The inner value replaces the loop
        variable, where that evaluates it just as often (once, or it is only a name)."""
        self.generic_visit(e)
        out = []
        elts = [e.elt]
        for g in e.generators:
            inner = g.iter
            if not (isinstance(inner, ast.GeneratorExp) and len(inner.generators) == 1 and isinstance(g.target, ast.Name)):
                out.append(g)
                continue
            value, name = inner.elt, g.target.id
            uses = sum(isinstance(n, ast.Name) and n.id == name for x in elts + g.ifs for n in ast.walk(x))
            if uses != 1 and not isinstance(value, ast.Name):
                out.append(g)                           # (not flattened: it won't compile, and falls back)
                continue

            class Put(ast.NodeTransformer):
                def visit_Name(self, n):
                    return copy.deepcopy(value) if n.id == name else n
            i = inner.generators[0]
            elts = [Put().visit(x) for x in elts]
            out.append(ast.comprehension(i.target, i.iter, i.ifs + [Put().visit(x) for x in g.ifs], 0))
        e.elt, e.generators = elts[0], out
        return e

    def visit_GeneratorExp(self, e):
        return self.comp(e)

    def visit_ListComp(self, e):
        return self.comp(e)

    def visit_For(self, s):
        self.generic_visit(s)
        g = s.iter
        if not isinstance(g, ast.GeneratorExp) or len(g.generators) != 1:
            return s
        c = g.generators[0]                             # the call above turned it into a comprehension
        head = []
        for x in c.ifs:
            stop = ast.copy_location(ast.If(ast.copy_location(ast.UnaryOp(ast.Not(), x), s), [ast.copy_location(ast.Continue(), s)], []), s)
            head.append(stop)
        head.append(ast.copy_location(ast.Assign([s.target], g.elt), s))
        return ast.copy_location(ast.For(c.target, c.iter, head + s.body, s.orelse), s)


# numpy on single numbers. Each of these matched the C library bit for bit on 160,024 inputs, natively on arm64 and in
# Pyodide. np.tanh didn't, natively: numpy has its own, so it stays Python.
NP_LIBM1 = {'exp': 'exp', 'exp2': 'exp2', 'expm1': 'expm1', 'log': 'log', 'log2': 'log2', 'log10': 'log10', 'log1p': 'log1p',
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan', 'asin': 'asin',
            'acos': 'acos', 'atan': 'atan', 'sinh': 'sinh', 'cosh': 'cosh', 'arcsinh': 'asinh', 'arccosh': 'acosh',
            'arctanh': 'atanh', 'asinh': 'asinh', 'acosh': 'acosh', 'atanh': 'atanh', 'cbrt': 'cbrt'}
NP_DIRECT1 = {'sqrt': 'sqrt', 'fabs': 'fabs', 'rint': 'rint'}
NP_ROUND = {'floor', 'ceil', 'trunc'}                # numpy 2.1 changed what these do to ints: floats only
NP_KEEPS_INT = {'abs', 'absolute', 'sign', 'square'}  # of a Python int, the type depends on the platform
# np.power stays Python: it has shortcuts for some exponents (np.power(-0.0, 0.5) is -0.0, pow's is 0.0)
NP_LIBM2 = {'arctan2': 'atan2', 'atan2': 'atan2', 'hypot': 'hypot', 'fmod': 'fmod'}
NP_FLOATS_ONLY2 = {'fmod'}                            # of ints it stays int
NP_MINMAX = {'minimum', 'maximum', 'fmin', 'fmax'}
NP_TESTS = {'isnan', 'isinf', 'isfinite', 'signbit'}
NP_CONVERT = {'float64', 'int64'}
NP_NARROW_OK = {'sqrt', 'fabs', 'abs', 'absolute', 'square', 'sign', 'floor', 'ceil', 'trunc', 'rint', 'copysign', 'float64', 'int64'} \
    | NP_MINMAX | NP_TESTS | {'float32', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8'}   # exact in double, then rounded once
NP_NARROW_INT_OK = {'abs', 'absolute', 'square', 'sign', 'float64', 'int64'} | NP_MINMAX | NP_TESTS \
    | {'float32', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8'}
NP_CONSTANTS = {'pi': math.pi, 'e': math.e, 'inf': math.inf, 'nan': math.nan}
NP_REDUCE = {'sum': 'sum', 'prod': 'prod', 'mean': 'mean', 'max': 'max', 'min': 'min', 'any': 'any', 'all': 'all', 'argmax': 'argmax',
             'argmin': 'argmin', 'amax': 'max', 'amin': 'min', 'std': 'std', 'var': 'var'}
CONTRACTING = {'lanczos', 'gamma', 'lgamma', 'remainder', 'np_stats'}     # helpers with a * b + c in one C expression
NP_MAKE_SHAPED = {'zeros', 'ones', 'empty', 'full'}
NP_MAKE_LIKE = {'zeros_like', 'ones_like', 'empty_like', 'full_like'}
NP_MAKE = NP_MAKE_SHAPED | NP_MAKE_LIKE | {'copy', 'array', 'asarray', 'arange', 'linspace'}
MAX_MADE_DIMS = 4      # sj_block has room for this many
ARRAY_METHODS = {'sum', 'prod', 'mean', 'max', 'min', 'any', 'all', 'argmax', 'argmin', 'std', 'var'}
NP_FLOAT_ONLY = {'round': 1, 'around': 1, 'clip': 3, 'degrees': 1, 'rad2deg': 1, 'radians': 1, 'deg2rad': 1}
NP_FUNCS = {'float32', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8', 'uint64'} | NP_MAKE | set(NP_LIBM1) | set(NP_DIRECT1) | NP_ROUND | NP_KEEPS_INT | set(NP_LIBM2) | {'copysign'} | NP_MINMAX | NP_TESTS | NP_CONVERT \
    | set(NP_REDUCE) | set(NP_FLOAT_ONLY)


class Tables(ast.NodeTransformer):
    """[[c] * w for _ in range(h)] and [[c for _ in range(w)] for _ in range(h)] as sj_table(h, w, c): a list of lists
    whose rows are separate lists, made as one block. Python's [[c] * w] * h repeats one row, and isn't one of these."""
    @staticmethod
    def range_of(g):
        return g.iter.args[0] if (isinstance(g.target, ast.Name) and not g.ifs and not g.is_async and isinstance(g.iter, ast.Call)
                                  and isinstance(g.iter.func, ast.Name) and g.iter.func.id == 'range' and len(g.iter.args) == 1
                                  and not g.iter.keywords) else None

    def visit_ListComp(self, node):
        self.generic_visit(node)
        if len(node.generators) != 1 or self.range_of(node.generators[0]) is None:
            return node
        h, t = self.range_of(node.generators[0]), node.generators[0].target.id
        x = node.elt
        w = c = None
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Mult):
            one, w = (x.left, x.right) if isinstance(x.left, ast.List) else (x.right, x.left)
            if isinstance(one, ast.List) and len(one.elts) == 1:
                c = one.elts[0]
        elif isinstance(x, ast.ListComp) and len(x.generators) == 1 and self.range_of(x.generators[0]) is not None:
            w, c = self.range_of(x.generators[0]), x.elt
            if any(isinstance(n, ast.Name) and n.id == x.generators[0].target.id for n in ast.walk(c)):
                return node
        def plain(n):                                 # nothing evaluating it again per row could tell
            return isinstance(n, (ast.Name, ast.Constant, ast.UnaryOp, ast.BinOp, ast.operator, ast.unaryop, ast.Load)) or (
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'len' and len(n.args) == 1
                and isinstance(n.args[0], ast.Name) and not n.keywords)
        if c is None or not all(plain(n) for p in (w, c) for n in ast.walk(p)) \
                or any(isinstance(n, ast.Name) and n.id == t for p in (w, c) for n in ast.walk(p)):
            return node                                   # only plain values: each row would evaluate them again
        return ast.copy_location(ast.Call(ast.Name('sj_table', ast.Load()), [h, w, c], []), node)


class ListReturns(ast.NodeTransformer):
    """return [...] as `sj_list = [...]; return sj_list`: a list is returned from a variable holding it."""
    def visit_Return(self, node):
        v = node.value
        if isinstance(v, (ast.List, ast.ListComp)) or (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult)
                                                        and (isinstance(v.left, ast.List) or isinstance(v.right, ast.List))):
            name = ast.copy_location(ast.Name('sj_list', ast.Store()), node)
            return [ast.copy_location(ast.Assign(targets=[name], value=v), node),
                    ast.copy_location(ast.Return(ast.copy_location(ast.Name('sj_list', ast.Load()), node)), node)]
        return node


class TupleOf:
    """A loop element that is a tuple: nested lists of (C value, flag), and its type."""
    __slots__ = ('vals', 't')

    def __init__(self, vals, t):
        self.vals, self.t = vals, t


class RowOf:
    """What a loop over a multi-dimensional array yields: its k-th first-axis element, a view."""

    def __init__(self, name, k, first):
        self.name, self.k, self.first = name, k, first


class StaticNumpy:
    """numpy, or one of its functions, for preflight reading source without importing anything."""

    def __init__(self, name=None):
        self.name = name


def is_numpy(obj):
    return isinstance(obj, StaticNumpy) and obj.name is None or (obj is not None and obj is sys.modules.get('numpy'))


def zero_rule(name):
    """Which zero np.minimum gives for 0.0 and -0.0 depends on the platform: always 'negative', always 'positive', or by
    position, 'first' or 'second'. The compiled code must agree with the numpy it stands in for."""
    np = sys.modules.get('numpy')
    if np is None:
        return 'negative' if name in ('minimum', 'fmin') else 'positive'
    one = math.copysign(1.0, float(getattr(np, name)(0.0, -0.0))) < 0
    two = math.copysign(1.0, float(getattr(np, name)(-0.0, 0.0))) < 0
    return {(True, True): 'negative', (False, False): 'positive', (False, True): 'first', (True, False): 'second'}[one, two]


_SUM_CHUNK = []


def sum_chunk():
    """How many elements numpy sums in one pairwise pass: some builds do 8192 at a time, which shows in the last bits. Asked
    once, of the numpy present, with a fixed array on which the two differ. None if it does neither."""
    np = sys.modules.get('numpy')
    if np is None:
        return 0
    if not _SUM_CHUNK:
        x = [((i * 2654435761) % 1000003) / 1000003.0 for i in range(20000)]

        def pairwise(i, n):
            if n < 8:
                return sum(x[i:i + n], 0.0)
            if n <= 128:
                r = x[i:i + 8]
                k, m = 8, n - n % 8
                while k < m:
                    for j in range(8):
                        r[j] += x[i + k + j]
                    k += 8
                t = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
                while k < n:
                    t += x[i + k]
                    k += 1
                return t
            half = n // 2
            half -= half % 8
            return pairwise(i, half) + pairwise(i + half, n - half)

        def chunks(size):
            total = 0.0
            for i in range(0, len(x), size):
                total += pairwise(i, min(size, len(x) - i))
            return total
        got = float(np.sum(np.array(x)))
        _SUM_CHUNK.append(0 if got == pairwise(0, len(x)) else 8192 if got == chunks(8192) else None)
    return _SUM_CHUNK[0]


def clip_ties():
    """Whether np.clip takes the bound when the value equals it, which shows only with zeros: it depends on the platform."""
    np = sys.modules.get('numpy')
    if np is None:
        return False, False
    takes_lo = math.copysign(1.0, float(np.clip(0.0, -0.0, 1.0))) < 0
    takes_hi = math.copysign(1.0, float(np.clip(-0.0, -1.0, 0.0))) > 0
    return takes_lo, takes_hi


def uint64_shift_masks():
    """Some builds of numpy cut a uint64 shift count to 32 bits before checking it's below 64, so 2**32 counts as 0
    (np.uint64(1) >> np.uint64(2**32) is 1 in Pyodide's, where a C long is 32 bits, and 0 natively)."""
    np = sys.modules.get('numpy')
    if np is None:
        return False
    with np.errstate(all='ignore'):
        return int(np.uint64(1) >> np.uint64(1 << 32)) == 1


def abs_flips_nan():
    """Some builds of numpy compute abs(x) as x > 0 ? x : -x, which flips a NaN's sign."""
    np = sys.modules.get('numpy')
    return np is not None and math.copysign(1.0, float(np.abs(math.nan))) < 0


WEB_SUFFIX = re.compile(r'__v\d+\b')


def literal_value(node):
    """The number in the statement x = <literal>, or None."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return const_number(node.value)
    return None


def exact_int_literal(node):
    """Is this statement x = <an int literal a double holds exactly>?"""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return False
    v = node.value
    if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub):
        v = v.operand
    return isinstance(v, ast.Constant) and type(v.value) is int and abs(v.value) <= 1 << 53



def ret_pairs(ret):
    """How many pairs of sj_io slots a return value uses beyond the first five: one per number, nested tuples too, and one
    for a list (its length, and which argument it is if it's one)."""
    if ret.kind == 'list':
        return 1
    return len([et for et in ret.elem if et.kind != 'tuple']) + sum(ret_pairs(et) for et in ret.elem if et.kind == 'tuple') if ret.kind == 'tuple' else 0


def flat(vals):
    """The (value, flag) leaves of a nested list of them, in order."""
    out = []
    for v in vals:
        out.extend(flat(v) if isinstance(v, list) else [v])
    return out


def ret_arrays(ret):
    """The arrays a return value holds, in order."""
    return [ret] if ret.kind == 'array' else [t for t in ret.elem if t.kind == 'array'] if ret.kind == 'tuple' else []


def array_record_slots(t):
    """Slots describing a returned array: its address, which argument's memory (index, is it that very array), the
    block it was made in (address, data, ndim, shape), then extent and stride for each axis."""
    return 10 + 2 * t.ndim


def line_tag(frame):
    """The file name #line gives a function's lines: the function called from Python, or the frame-th it calls."""
    return 'sj' if frame is None else f'sj{frame}'


def cimo_diagnostic(text, filename, offset, frames):
    """cimo's message with sj:12 or sj3:4 put back as the Python file and line."""
    def put(m):
        k = m.group(1)
        f, o = (filename, offset) if k == '' else (frames[int(k)][0], frames[int(k)][2])
        return f'{f}:{int(m.group(2)) + o}'
    return re.sub(r'\bsj(\d*):(\d+)', put, text)


class Safepoint(ast.Constant):
    """Stands for a place where no array is held anywhere but in variables, so unused ones can be freed."""


def flatten_dicts(fdef, arg_types):
    """A dict parameter becomes one parameter per key the function uses, and p['a'] a plain variable, so
    everything the compiler knows about variables applies. An instance does too, by attribute: p.a, and a list of
    instances becomes a list of numbers per attribute: ps[i].a, and p.a for p bound by for p in ps, enumerate(ps) or
    p = ps[i], where p becomes the index. Any other use of p or ps can't work this way. Returns the function, the new
    parameter types, and {variable: (argument index, key, how the user wrote it)}."""
    a = fdef.args
    params = a.posonlyargs + a.args
    if not any(t is not None and t.kind in ('dict', 'object', 'objlist', 'madeclass', 'elemref', 'pool', 'ref', 'nopool', 'same', 'inlist', 'poolref', 'reflist') for t in arg_types) \
            or len(params) != len(arg_types):
        return fdef, arg_types, {}
    tree = copy.deepcopy(fdef)
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {p.arg for p in params}

    def fresh(name):
        while name in taken:
            name += '_'
        taken.add(name)
        return name
    slots, flat, methods, nested, same = {}, {}, {}, {}, {}
    new_params, new_types = [], []
    names = [p.arg for p in params]

    def target(i, path, root):
        """Where an object was first met: parameter i's path, or root's, as an owner's name ('p.pos')."""
        return '.'.join([root if i is None else names[i]] + list(path))
    plain = {}                                   # a parameter kept as it is -> the argument it is
    renamed = {}                                 # a parameter that is another's object: the other's path, wherever it's named
    for p, t in zip(params, arg_types):
        if t.kind == 'same':
            if any(isinstance(n, ast.Name) and n.id == p.arg and not isinstance(n.ctx, ast.Load) for n in ast.walk(tree)):
                raise Unsupported(tree, f'{p.arg} is assigned, and is another argument\'s object')
            renamed[p.arg] = target(*t.elem, None)
    if renamed:
        def path_expr(owner, at):
            base, *path = owner.split('.')
            e = ast.Name(base, ast.Load())
            for a in path:
                e = ast.Attribute(e, a, ast.Load())
            return ast.copy_location(e, at)

        class Rename(ast.NodeTransformer):
            def visit_Name(self, n):
                return path_expr(renamed[n.id], n) if n.id in renamed else n
        tree.body = [Rename().visit(s) for s in tree.body]
        ast.fix_missing_locations(tree)
    pools = {}                                   # a list with repeats -> its index list
    inlists = {}                                 # an argument that's an element of another -> (that list, its place's name)
    inner_lists = {}                             # 'ps.pos' -> the type of the objects in that attribute of ps's elements
    refs = {}                                    # an element given by reference -> (its own list's name, its index's, its type)
    pool, pools_by, refnames, ref_fields = None, {}, {}, {}      # class -> (its pool's name, numbers, {reference: class})
    pool_type = None
    reflists = {}                                # a list of references -> the class they reach
    for i, (p, t) in enumerate(zip(tree.args.posonlyargs + tree.args.args, arg_types)):
        if t.kind in ('same', 'nopool'):
            continue
        if t.kind == 'pool':                     # per class: a list per number, an index list per reference
            pool, pool_type = p.arg, t
            made_k = dict(t.elem[2])
            for k, (cls, numbers, prefs) in enumerate(t.elem[0]):
                name = f'{pool}{k}'
                for key, ft in list(numbers) + [(a, T.INT) for a, _ in prefs]:
                    flat[name, key] = fresh(f'sjpool{k}__{key}')
                    slots[flat[name, key]] = (i, f'{k}.{key}', f'[a {cls.__name__ if cls else "referenced"} object].{key}')
                    new_params.append(ast.copy_location(ast.arg(flat[name, key]), p))
                    new_types.append(T.list_of(ft))
                grows = None
                if k in made_k:                  # it makes instances of this class: how many are in use, then room
                    count = fresh(f'sjpool{k}__count')
                    slots[count] = (i, f'{k}.#count', f'[how many {cls.__name__}]')
                    new_params.append(ast.copy_location(ast.arg(count), p))
                    new_types.append(T.INT)
                    grows = (count, made_k[k], flat[name, (list(numbers) + [(a, T.INT) for a, _ in prefs])[0][0]])
                pools_by[cls] = (name, {a for a, _ in numbers}, dict(prefs), grows)
            continue
        if t.kind == 'inlist':                   # an element of another argument: its place there
            inlists[p.arg] = (names[t.elem], fresh(f'sji__{p.arg}'))
            new_params.append(ast.copy_location(ast.arg(inlists[p.arg][1]), p))
            new_types.append(T.INT)
            continue
        if t.kind == 'reflist':                  # a list of references: of indexes
            reflists[p.arg] = t.cls
            plain[p.arg] = i
            new_params.append(p)
            new_types.append(T.list_of(T.INT))
            continue
        if t.kind == 'poolref':                  # a reference given by the caller: its pool's lists (once), then its index
            ptype, cls, lists = t.elem
            if lists:
                pool, pool_type = 'sjpool', ptype
                for k, (pc, numbers, prefs) in enumerate(ptype.elem[0]):
                    name = f'{pool}{k}'
                    for key, ft in list(numbers) + [(a, T.INT) for a, _ in prefs]:
                        flat[name, key] = fresh(f'sjpool{k}__{key}')
                        new_params.append(ast.copy_location(ast.arg(flat[name, key]), p))
                        new_types.append(T.list_of(ft))
                    pools_by[pc] = (name, {a for a, _ in numbers}, dict(prefs), None)
            elif pool_type is not ptype:
                raise Unsupported(tree, 'references into two pools')
            refnames[p.arg] = cls
            new_params.append(p)
            new_types.append(T.INT)
            continue
        if t.kind == 'ref':                      # a reference chosen by data: its index in the pool, -1 for None
            refnames[p.arg] = t.cls
            plain[p.arg] = i
            new_params.append(p)
            new_types.append(T.INT)
            continue
        if t.kind == 'elemref':                  # the attributes' lists, then the index: an element of a list of its own
            pl = fresh(f'sjref__{p.arg}')
            refs[p.arg] = (pl, fresh(f'sji__{p.arg}'), t.elem)
            for key, ft in t.elem.elem:
                flat[pl, key] = fresh(f'{p.arg}__{key}')
                if ft.kind == 'function':
                    methods[pl, key] = ft
                new_params.append(ast.copy_location(ast.arg(flat[pl, key]), p))
                new_types.append(ft if ft.kind == 'function' else T.list_of(ft))
            new_params.append(ast.copy_location(ast.arg(refs[p.arg][1]), p))
            new_types.append(T.INT)
            continue
        if t.kind not in ('dict', 'object', 'objlist'):
            plain[p.arg] = i
            new_params.append(p)
            new_types.append(t)
            continue
        fields = t.elem.elem if t.kind == 'objlist' else t.elem
        todo = [(p.arg, '', fields)]
        popped = []
        while todo:                              # an object in an attribute is an owner of its own: 'p.pos'
            owner, prefix, fields = todo.pop(0)
            popped.append((owner, prefix[:-1]))
            for k, (key, ft) in enumerate(fields):
                if ft.kind == 'same' and t.kind == 'object':     # another name for an object met before: its variables
                    same[f'{owner}.{key}'] = target(*ft.elem, p.arg)
                    continue
                if ft.kind == 'object' and t.kind == 'object':
                    nested[f'{owner}.{key}'] = ft
                    todo.append((f'{owner}.{key}', f'{prefix}{key}.', ft.elem))
                    continue
                if ft.kind == 'object' and t.kind == 'objlist':     # each element's object there: a list per number, by path
                    inner_lists[f'{owner}.{key}'] = ft
                    todo.append((f'{owner}.{key}', f'{prefix}{key}.', ft.elem))
                    continue
                base = owner.replace('.', '__')
                name = fresh(f'{base}__{key}' if isinstance(key, str) and key.isidentifier() and key.isascii() else f'{base}__k{k}')
                flat[owner, key] = name
                if ft.kind == 'ref':                # a reference in a fixed place: its index, a variable
                    ref_fields[owner, key] = ft.cls
                    ft = T.INT
                if ft.kind == 'function':           # a method: called as a function, with p's attributes as self
                    methods[owner, key] = ft
                shown = f'{owner}.{key}' if t.kind == 'object' else f'{p.arg}[...].{key}' if t.kind == 'objlist' else f'{p.arg}[{key!r}]'
                slots[name] = (i, prefix + key if prefix else key, shown)     # a dict's key can be an int
                new_params.append(ast.copy_location(ast.arg(name), p))
                new_types.append(T.list_of(ft) if t.kind == 'objlist' and ft.kind != 'function' else ft)
        for owner, path in popped if t.kind == 'objlist' else ():     # after the lists: which object is at each place
            if path in t.strided:
                pools[owner] = fresh(f'{owner.replace(".", "__")}__ix')
                new_params.append(ast.copy_location(ast.arg(pools[owner]), p))
                new_types.append(T.list_of(T.INT))

    dicts = {p.arg for p, t in zip(params, arg_types) if t.kind == 'dict'}
    objects = {p.arg for p, t in zip(params, arg_types) if t.kind == 'object'}
    lists = {p.arg: t for p, t in zip(params, arg_types) if t.kind == 'objlist'}
    types_of = {p.arg: (t.elem if t.kind == 'objlist' else t) for p, t in zip(params, arg_types)}
    for pl, _, et in refs.values():
        lists[pl], types_of[pl] = T.objlist_of(et), et
    types_of.update(inner_lists)
    objects |= set(nested)
    types_of.update(nested)
    made = {p.arg: t for p, t in zip(params, arg_types) if t.kind == 'madeclass'}
    makers = {}                                  # id(assignment) -> 'make' for v = C(...), 'call' for v = a call returning a new instance
    copied = {}                                  # a parameter rebound to new instances -> the local that holds it from the start
    for p in list(objects):
        if '.' in p or not any(isinstance(n, ast.Name) and n.id == p and not isinstance(n.ctx, ast.Load) for n in ast.walk(tree)):
            continue
        t = types_of[p]
        if t.writes or any(isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id == p for n in ast.walk(tree)) \
                or any(isinstance(n, ast.Compare) and any(isinstance(o, (ast.Is, ast.IsNot)) for o in n.ops)
                       and any(isinstance(x, ast.Name) and x.id == p for x in [n.left] + n.comparators) for n in ast.walk(tree)):
            raise Unsupported(tree, f'{p} is given a new instance, and is written through, returned or compared with is')
        q = fresh(f'sjp__{p}')
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id == p:
                n.id = q
        start = ast.copy_location(ast.Assign([ast.Name(q, ast.Store())], ast.Name(p, ast.Load())), tree.body[0])
        doc = 1 if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str) else 0
        tree.body.insert(doc, ast.fix_missing_locations(start))      # after a docstring, which must stay first
        copied[q] = p
        makers[id(start)] = 'copy'

    def pool_made(e):
        """For C(...) where C's instances are kept in the pool: its pool's entry, else None."""
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in made:
            spec = pools_by.get(made[e.func.id].cls)
            if spec is not None and spec[3] is not None:
                return spec
        return None
    # v = C(...), C a class it makes instances of: v's attributes become variables, assigned as C.__init__ assigns them
    made_local = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Name) and n.value.func.id in made and not pool_made(n.value):
            v = n.targets[0].id
            if made_local.get(v, n.value.func.id) != n.value.func.id or v in types_of:
                raise Unsupported(n, f'{v} is an instance of two classes, or a parameter too')
            made_local[v] = n.value.func.id
    def register(v, c):
        t = made[c]
        made_local[v] = c
        for a, _ in t.recipe['assigns']:
            if (v, a) not in flat:
                flat[v, a] = fresh(f'{v}__{a}')
        for m, ft in t.recipe['methods']:
            methods[v, m] = ft
            if (v, m) not in flat:
                flat[v, m] = fresh(f'sjm__{v}__{m}')
        types_of[v] = T.object_of(t.py, list(t.recipe['methods']), t.methods, t.cls, t.writes)
        objects.add(v)
    for v, c in list(made_local.items()):
        register(v, c)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) \
                and n.value.func.id in made and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and not pool_made(n.value):
            makers[id(n)] = 'make'
    first = {}                                   # a list's first attribute, for len() and an index's check
    for p, t in lists.items():
        data = [k for k, _ in leaf_types(t.elem)]
        if not data:
            raise Unsupported(tree, f'{p} is a list of instances whose attributes it never reads')
        first[p] = None
        for path in [''] + data:                 # a list as long as ps: its index list, or a list of a number, or the index
            *inner, last = path.split('.')       # list of the first pooled object on the way to one
            chain = ['.'.join([p] + inner[:k]) for k in range(len(inner) + 1)]
            pooled_on_way = [o for o in chain if o in pools]
            if not path:
                first[p] = pools.get(p)
            elif pooled_on_way:
                first[p] = pools[pooled_on_way[0]]
            else:
                first[p] = flat['.'.join([p] + inner), last]
            if first[p] is not None:
                break

    # q = p for an instance parameter p, at the top of the body, q assigned nowhere else and not named before: q is p
    for k, s in enumerate(tree.body):
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Name) \
                and s.value.id in objects and s.targets[0].id not in types_of:
            q, p = s.targets[0].id, s.value.id
            stores = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == q and not isinstance(n.ctx, ast.Load)]
            before = any(isinstance(n, ast.Name) and n.id == q for b in tree.body[:k] for n in ast.walk(b))
            if len(stores) == 1 and not before:
                for b in tree.body[k + 1:]:
                    for n in ast.walk(b):
                        if isinstance(n, ast.Name) and n.id == q:
                            n.id = p
                tree.body[k] = ast.copy_location(ast.Pass(), s)

    # names bound to an element: for p in ps, for i, p in enumerate(ps), p = ps[i], q = p. Each becomes an index.
    alias, binders, copies = {}, set(), set()

    def bind(name, ps, node):
        if alias.get(name, ps) != ps:
            raise Unsupported(node, f'{name} is an element of two lists')
        alias[name] = ps
        binders.add(id(node))

    def element_loop(target, it):
        """(element name, list, counter name or None, start) if this for or comprehension walks a list of instances."""
        if isinstance(target, ast.Name) and isinstance(it, ast.Name) and it.id in lists:
            return target.id, it.id, None, None
        if isinstance(target, ast.Tuple) and len(target.elts) == 2 and all(isinstance(x, ast.Name) for x in target.elts) \
                and isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'enumerate' and it.args \
                and isinstance(it.args[0], ast.Name) and it.args[0].id in lists and len(it.args) + len(it.keywords) <= 2:
            start = it.args[1] if len(it.args) == 2 else (it.keywords[0].value if it.keywords and it.keywords[0].arg == 'start' else None)
            if it.keywords and it.keywords[0].arg != 'start':
                return None
            if start is not None and not isinstance(start, (ast.Constant, ast.Name)):
                raise Unsupported(it, 'enumerate() of a list of instances with a start that is an expression')
            return target.elts[1].id, it.args[0].id, target.elts[0].id, start
        return None
    for n in ast.walk(tree):
        if isinstance(n, (ast.For, ast.comprehension)):
            found = element_loop(n.target, n.iter)
            if found:
                bind(found[0], found[1], n.target if isinstance(n, ast.comprehension) else n)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Subscript) \
                and isinstance(n.value.value, ast.Name) and n.value.value.id in lists and not isinstance(n.value.slice, ast.Slice):
            bind(n.targets[0].id, n.value.value.id, n)
    changed = True
    while changed:                                # best = p, p an element: best is one too, its index copied
        changed = False
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Name) \
                    and n.value.id in alias and id(n) not in copies:
                bind(n.targets[0].id, alias[n.value.id], n)
                copies.add(id(n))
                changed = True
    if set(alias) & (set(lists) | dicts | objects):
        raise Unsupported(tree, 'a parameter is also an element of a list')
    for p, (pl, _, _) in refs.items():
        alias[p] = pl
    for p, (pl, _) in inlists.items():
        if any(isinstance(n, ast.Name) and n.id == p and not isinstance(n.ctx, ast.Load) for n in ast.walk(tree)):
            raise Unsupported(tree, f'{p}, an element of {pl}, is assigned')
        alias[p] = pl
    index = {name: fresh(f'sji__{name}') for name in alias}
    index.update({p: ix for p, (_, ix, _) in refs.items()})
    index.update({p: ix for p, (_, ix) in inlists.items()})

    def pooled(ps, idx, node=None):
        """Where the attributes of an element (of ps, or of an object in its elements: 'ps.vel') are in their lists: at its
        place, or where repeats are pooled, through each level's index list."""
        idx = copy.deepcopy(idx)
        if idx is None:
            return None
        parts = ps.split('.')
        for k in range(1, len(parts) + 1):
            o = '.'.join(parts[:k])
            if o in pools:
                idx = ast.Subscript(ast.Name(pools[o], ast.Load()), idx, ast.Load())
                if node is not None:
                    ast.copy_location(idx, node)
        return idx

    NOT = object()

    def refclass(e):
        """The class a reference chosen by data reaches (None: only None seen), or NOT if e isn't one: a name holding one,
        a reference attribute in a fixed place or reached from one, or a if c else b of them and None."""
        if pool_made(e):                         # a new instance, kept in the pool
            return made[e.func.id].cls
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in funcs and getattr(funcs[e.func.id], '_tree', None) is not None \
                and pool_type is not None and any(refexpr(a) for a in e.args):
            w = funcs[e.func.id]                 # a function given references, returning one of them or one reached from them
            ps = [q.arg for q in w._tree.args.posonlyargs + w._tree.args.args]
            given = {q: refclass(a) for q, a in zip(ps, e.args) if refexpr(a)}
            return ref_return_class(w._tree, given, pools_by, NOT)
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id in reflists and not isinstance(e.slice, ast.Slice):
            return reflists[e.value.id]
        if isinstance(e, ast.Name):
            return refnames[e.id] if e.id in refnames else NOT
        if isinstance(e, ast.IfExp):
            cs = [None if none(x) else refclass(x) for x in (e.body, e.orelse)]
            if NOT in cs or all(none(x) for x in (e.body, e.orelse)):
                return NOT
            known = {c for c in cs if c is not None}
            if len(known) > 1:
                raise Unsupported(e, 'a reference to objects of two classes')
            return known.pop() if known else None
        if isinstance(e, ast.Attribute):
            o = owner_of(e.value)
            if o is not None and (o, e.attr) in ref_fields:
                return ref_fields[o, e.attr]
            c = refclass(e.value)
            if c is NOT:
                return NOT
            if c is None:                        # only ever None: reading through it raises, whatever it would reach
                targets = {sp[2][e.attr] for sp in pools_by.values() if e.attr in sp[2]}
                return (targets.pop() if len(targets) == 1 else None) if targets else NOT
            spec = pools_by.get(c)
            return spec[2][e.attr] if spec is not None and e.attr in spec[2] else NOT
        return NOT

    def refexpr(e):
        return refclass(e) is not NOT

    def none(e):
        return isinstance(e, ast.Constant) and e.value is None

    def locate(e):
        """Where the attributes of the object e names live, the one question every attribute and method resolves through:
        ('ref', class, e) in a pool, at e's index checked against None; ('element', owner, index, e) in a list per attribute
        (owner a list, or an object in its elements' attributes: 'ps.pos'); ('static', owner) in variables (an argument, an
        object in its attributes, through other names for an object, or an instance made here); None if e names none."""
        if refexpr(e):
            return ('ref', refclass(e), e)
        el = element_of(e)
        if el is not None:
            return ('element', el[0], el[1], e)
        o = owner_of(e)
        return ('static', o) if o is not None else None

    def field(pl, attr, ctx, node, visit):
        """The storage of attribute attr of the object pl locates, as an expression; None if it holds no such number."""
        if pl[0] == 'ref':
            spec = pools_by.get(pl[1])
            if spec is None and pl[1] is None:   # only ever None: any pool with the attribute, never read past the check
                spec = next((sp for sp in pools_by.values() if attr in sp[1] or attr in sp[2]), None)
            if spec is None or (attr not in spec[1] and attr not in spec[2]):
                raise Unsupported(node, f'.{attr} of an object a reference reaches: only its numbers and references compile')
            return ast.copy_location(ast.Subscript(ast.Name(flat[spec[0], attr], ast.Load()), deref(visit(pl[2]), attr, node), ctx), node)
        if (pl[1], attr) not in flat:
            return None
        if pl[0] == 'element':
            return column(pl[1], attr, visit(pl[2]) if not isinstance(pl[3], ast.Name) else pl[2], ctx, node)
        return ast.copy_location(ast.Name(flat[pl[1], attr], ctx), node)

    def column(ps, key, idx, ctx, node):
        return ast.copy_location(ast.Subscript(ast.Name(flat[ps, key], ast.Load()), pooled(ps, idx, node), ctx), node)

    def path_owner(owner, path):
        """('p.pos', 'x') for owner p and path 'pos.x': the key flat has it under, through other names for an object."""
        *inner, last = path.split('.')
        for a in inner:
            owner = same.get(f'{owner}.{a}', f'{owner}.{a}')
        return (owner, last)

    def handed(owner, t, root=None, done=None, stack=()):
        """The type an object goes to a callee as, the callee seeing it as its own root: another name for an object inside it
        becomes same_as within it; one for an object outside is that object's own fields, which are the caller's variables
        for it, met once and same_as after."""
        root = owner if root is None else root
        done = {} if done is None else done
        fields = []
        for k, ft in t.elem:
            at = f'{owner}.{k}'
            if ft.kind == 'same':
                to = same[at]
                if to == root or to.startswith(root + '.'):
                    ft = T.same_as(None, tuple(to.split('.')[len(root.split('.')):]))
                elif to in done:
                    ft = T.same_as(None, done[to])
                elif to in stack:
                    raise Unsupported(tree, f'{to} holds itself through what is passed')
                else:
                    done[to] = tuple(at.split('.')[len(root.split('.')):])
                    ft = handed(to, types_of[to], root, done, stack + (to,))
            elif ft.kind == 'object':
                ft = handed(at, ft, root, done, stack)
            fields.append((k, ft))
        return T.object_of(t.py, fields, t.methods, t.cls, t.writes)

    def owner_of(e):
        """The owner e names: an instance parameter, one made here, or an object in an attribute of one ('p.pos')."""
        if isinstance(e, ast.Name) and e.id in objects:
            return e.id
        if isinstance(e, ast.Attribute):
            o = owner_of(e.value)
            name = same.get(f'{o}.{e.attr}', f'{o}.{e.attr}')
            if o is not None and (name in nested or name in objects):
                return name
        return None

    def element_of(e):
        """(list, index expression) if e is an element: p bound above, or ps[i]."""
        if isinstance(e, ast.Name) and e.id in alias:
            return alias[e.id], ast.Name(index[e.id], ast.Load())
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id in lists and not isinstance(e.slice, ast.Slice):
            return e.value.id, e.slice
        if isinstance(e, ast.Attribute):         # b.pos for an element b: an element of ps.pos, at b's index
            inner = element_of(e.value)
            if inner is not None and f'{inner[0]}.{e.attr}' in inner_lists:
                return f'{inner[0]}.{e.attr}', inner[1]
        return None

    def range_of(ps, node):
        size = ast.Call(ast.Name('len', ast.Load()), [ast.Name(first[ps], ast.Load())], [])
        return ast.copy_location(ast.Call(ast.Name('range', ast.Load()), [size], []), node)

    class Flatten(ast.NodeTransformer):
        def visit_Subscript(self, n):
            if isinstance(n.value, ast.Name) and n.value.id in dicts and isinstance(n.slice, ast.Constant) \
                    and type(n.slice.value) in (str, int) and (n.value.id, n.slice.value) in flat:
                return ast.copy_location(ast.Name(flat[n.value.id, n.slice.value], n.ctx), n)
            return self.generic_visit(n)

        def visit_For(self, n):
            found = element_loop(n.target, n.iter)
            if not found:
                return self.generic_visit(n)
            name, ps, counter, start = found
            body = [self.visit(s) for s in n.body]
            body = [x for s in body for x in (s if isinstance(s, list) else [s])]
            if counter is not None:
                value = ast.Name(index[name], ast.Load())
                if start is not None:
                    value = ast.BinOp(value, ast.Add(), copy.deepcopy(start))
                body = [ast.copy_location(ast.Assign([ast.Name(counter, ast.Store())], value), n)] + body
            orelse = [x for s in n.orelse for x in (lambda r: r if isinstance(r, list) else [r])(self.visit(s))]
            return ast.copy_location(ast.For(ast.Name(index[name], ast.Store()), range_of(ps, n.iter), body, orelse), n)

        def comprehension(self, n):
            """In a comprehension, for i, p in enumerate(ps): i is p's index (plus the start) wherever it's read."""
            for g in n.generators:
                found = element_loop(g.target, g.iter)
                if found and found[2] is not None:
                    name, ps, counter, start = found
                    value = ast.Name(index[name], ast.Load())
                    if start is not None:
                        value = ast.BinOp(value, ast.Add(), copy.deepcopy(start))
                    for m in ast.walk(n):
                        for field, v in ast.iter_fields(m):
                            if isinstance(v, ast.Name) and v.id == counter and isinstance(v.ctx, ast.Load):
                                setattr(m, field, ast.copy_location(copy.deepcopy(value), v))
                            elif isinstance(v, list):
                                v[:] = [ast.copy_location(copy.deepcopy(value), x) if isinstance(x, ast.Name) and x.id == counter
                                        and isinstance(x.ctx, ast.Load) else x for x in v]
                    g.target, g.iter = ast.Name(name, ast.Store()), ast.Name(ps, ast.Load())
            return self.generic_visit(n)

        visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = comprehension

        def visit_comprehension(self, n):
            found = element_loop(n.target, n.iter)
            if not found:
                return self.generic_visit(n)
            name, ps, counter, start = found
            return ast.comprehension(ast.Name(index[name], ast.Store()), range_of(ps, n.iter), [self.visit(x) for x in n.ifs], n.is_async)

        def visit_If(self, n):
            n.test = truth(n.test, self.visit)
            n.body = [x for s in n.body for x in (lambda r: r if isinstance(r, list) else [r])(self.visit(s))]
            n.orelse = [x for s in n.orelse for x in (lambda r: r if isinstance(r, list) else [r])(self.visit(s))]
            return n

        visit_While = visit_If

        def visit_IfExp(self, n):
            ref = refexpr(n)
            n.test = truth(n.test, self.visit)
            n.body, n.orelse = [ast.copy_location(ast.Constant(-1), x) if ref and none(x) else self.visit(x) for x in (n.body, n.orelse)]
            return n

        def visit_BoolOp(self, n):
            n.values = [truth(v, self.visit) for v in n.values] if any(refexpr(v) for v in n.values) else [self.visit(v) for v in n.values]
            return n

        def visit_UnaryOp(self, n):
            if isinstance(n.op, ast.Not) and refexpr(n.operand):
                return ast.copy_location(ast.Compare(self.visit(n.operand), [ast.Eq()], [ast.Constant(-1)]), n)
            return self.generic_visit(n)

        def visit_Compare(self, n):
            sides = [n.left] + n.comparators
            if any(refexpr(x) for x in sides):
                if len(n.ops) != 1 or not isinstance(n.ops[0], (ast.Is, ast.IsNot)):
                    raise Unsupported(n, 'a reference compared other than with is or is not (== calls __eq__)')
                a, b = sides
                if not all(refexpr(x) or none(x) for x in sides):
                    raise Unsupported(n, 'a reference compared with something other than a reference or None')
                side = [ast.Constant(-1) if none(x) else self.visit(x) for x in (a, b)]
                return ast.copy_location(ast.Compare(side[0], [ast.Eq() if isinstance(n.ops[0], ast.Is) else ast.NotEq()], [side[1]]), n)
            return self.generic_visit(n)

        def build(self, call, node):
            """C(args) kept in the pool: its arguments in order, then a new place, each attribute's list given what __init__
            assigns it, and the count one more. (statements, the new place's name)"""
            name, numbers, prefs, (count, order, first_list) = pool_made(call)
            t = made[call.func.id]
            params = t.recipe['params']
            if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
                raise Unsupported(call, f'{call.func.id}() with * or **')
            positional = [q for q in params if not q[2]]
            out, given = [], {}
            for q, a in list(zip(positional, call.args)) + [((k.arg,), k.value) for k in call.keywords]:
                temp = fresh(f'sjc__{q[0]}')
                out.append(ast.copy_location(ast.Assign([ast.Name(temp, ast.Store())], ast.Constant(-1) if none(a) else self.visit(a)), node))
                given[q[0]] = ast.Name(temp, ast.Load())
            for pname, default, _ in params:
                if pname not in given:
                    if default is None or (type(default[1]) not in (int, float, bool) and default[1] is not None):
                        raise Unsupported(call, f'{call.func.id}() leaving out {pname}, whose default isn\'t a number or None')
                    given[pname] = ast.Constant(-1 if default[1] is None else default[1])
            idx = fresh('sjnew')
            out.append(ast.copy_location(ast.Assign([ast.Name(idx, ast.Store())], ast.Call(ast.Name('__sj_room__', ast.Load()),
                                         [ast.Name(count, ast.Load()), ast.Name(first_list, ast.Load())], [])), node))
            for a, src in t.recipe['assigns']:
                if src[0] == 'param':
                    v = copy.deepcopy(given[src[1]])
                elif src[0] == 'const' and (type(src[1]) in (int, float, bool) or src[1] is None):
                    v = ast.Constant(-1 if src[1] is None else src[1])
                else:
                    raise Unsupported(call, f'{call.func.id}.__init__ computes {a}: only a parameter or a constant, for an instance kept here')
                out.append(ast.copy_location(ast.Assign([ast.Subscript(ast.Name(flat[name, a], ast.Load()), ast.Name(idx, ast.Load()), ast.Store())], v), node))
            out.append(ast.copy_location(ast.Assign([ast.Name(count, ast.Store())], ast.BinOp(ast.Name(count, ast.Load()), ast.Add(), ast.Constant(1))), node))
            for s_ in out:
                ast.fix_missing_locations(ast.copy_location(s_, node))
            return out, ast.copy_location(ast.Name(idx, ast.Load()), node)

        def visit_Assign(self, n):
            if len(n.targets) == 1 and pool_made(n.value):     # a new instance kept in the pool: its place
                out, idx = self.build(n.value, n)
                n.value = idx
                return out + [self.visit_Assign(n)]
            if len(n.targets) == 1 and none(n.value) and (isinstance(n.targets[0], ast.Name) and n.targets[0].id in refnames or
                                                         isinstance(n.targets[0], (ast.Attribute, ast.Subscript)) and refexpr(n.targets[0])):
                n.value = ast.copy_location(ast.Constant(-1), n.value)     # None, as a reference
            if makers.get(id(n)) == 'make':
                return self.construct(n)
            if makers.get(id(n)) == 'copy':
                q, p = n.targets[0].id, copied[n.targets[0].id]
                out = []
                for a in dict.fromkeys(a for a, _ in made[made_local[q]].recipe['assigns']):
                    if (p, a) not in flat:
                        raise Unsupported(n, f'{p} is given a new instance, and has no {a} of its own')
                    out.append(ast.copy_location(ast.Assign([ast.Name(flat[q, a], ast.Store())], ast.Name(flat[p, a], ast.Load())), n))
                return out
            if makers.get(id(n)) == 'call':        # its attributes, in __init__'s order: what the callee returns for it
                v = n.targets[0].id
                attrs = list(dict.fromkeys(a for a, _ in made[made_local[v]].recipe['assigns']))
                if not attrs:
                    raise Unsupported(n, f'{v} is an instance with no attributes')
                target = ast.Tuple([ast.Name(flat[v, a], ast.Store()) for a in attrs], ast.Store())
                return ast.copy_location(ast.Assign([ast.copy_location(target, n)], self.visit(n.value)), n)
            if id(n) not in binders:
                return self.generic_visit(n)
            if id(n) in copies:
                return ast.copy_location(ast.Assign([ast.Name(index[n.targets[0].id], ast.Store())], ast.Name(index[n.value.id], ast.Load())), n)
            name, ps = n.targets[0].id, n.value.value.id
            idx = ast.copy_location(ast.Assign([ast.Name(index[name], ast.Store())], self.visit(n.value.slice)), n)
            check = ast.copy_location(ast.Assign([ast.Name(fresh('sjchk__'), ast.Store())],    # IndexError here, as in Python
                                                 ast.Subscript(ast.Name(first[ps], ast.Load()), ast.Name(index[name], ast.Load()), ast.Load())), n)
            return [idx, check]

        def visit_Call(self, n):
            f = n.func
            if (refnames or pool) and isinstance(f, ast.Attribute) and refexpr(f.value):      # a method of a referenced object
                from ._dispatch import _in_class, _method
                cls = refclass(f.value)
                m = _in_class(cls, f.attr) if cls is not None else None
                if not (type(m).__name__ == 'Function' or isinstance(m, types.FunctionType)):
                    raise Unsupported(n, f'{f.attr} of an object a reference reaches is not a plain method')
                fn = _method(m)
                CALLEES[id(fn)] = fn
                me = ast.copy_location(RefArg(deref(self.visit(f.value), f.attr, n)), f.value)
                me.cls, me.pool, me.cols = cls, pool_type, pool_lists()
                call = ast.copy_location(ast.Call(ast.copy_location(ast.Name(f'sjm__ref__{f.attr}', ast.Load()), f),
                                                  [me] + [self.ref_arg(a) for a in n.args], [self.visit(k) for k in n.keywords]), n)
                call.sj_callee = id(fn)
                return call
            if (refnames or pool) and any(refexpr(a) or (pool_type is not None and none(a)) for a in n.args):
                return ast.copy_location(ast.Call(self.visit(f), [self.ref_arg(a) for a in n.args], [self.visit(k) for k in n.keywords]), n)
            if isinstance(f, ast.Name) and f.id == 'isinstance' and len(n.args) == 2 and self.instance(n.args[0]):
                owner = self.instance(n.args[0])[0]
                mark = ast.copy_location(ClassOf(0), n.args[0])
                mark.cls = types_of[owner].cls
                return ast.copy_location(ast.Call(f, [mark, self.visit(n.args[1])], []), n)
            if isinstance(f, ast.Name) and f.id == 'len' and len(n.args) == 1 and isinstance(n.args[0], ast.Name) and n.args[0].id in lists:
                return ast.copy_location(ast.Call(f, [ast.Name(first[n.args[0].id], ast.Load())], []), n)
            if isinstance(f, ast.Attribute):
                pl = locate(f.value)              # whose method: an element's, or an object's in variables
                el = (pl[1], pl[2]) if pl is not None and pl[0] == 'element' else None
                owner = pl[1] if pl is not None and pl[0] != 'ref' else None
                if owner is not None and (owner, f.attr) in methods:
                    if el and not isinstance(el[1], (ast.Name, ast.Constant)):
                        raise Unsupported(n, f'a method of {owner}[...] with an index that is an expression: name the element first')
                    t = types_of[owner] if el or owner in made_local else handed(owner, types_of[owner])
                    uses, writes = t.methods[f.attr]
                    fields = [(k, ft) for k, ft in t.elem if ft.kind == 'function' or k in uses]   # only what it uses: the rest may have changed type
                    if el:
                        place = {k: column(owner, k, copy.deepcopy(el[1]), ast.Load(), n) for k, ft in fields if ft.kind != 'function'}
                    elif owner in made_local:            # made here: its attributes are variables, typed as the compiler finds them
                        place, held = made_place(owner, sorted(uses), n)
                    else:
                        sub = T.object_of(t.py, fields, t.methods, t.cls, t.writes)
                        place = {k: ast.copy_location(ast.Name(flat[path_owner(owner, k)], ast.Load()), n) for k, _ in leaf_types(sub)}
                    me = SelfArg(list(place.values()), ast.Load())
                    me.object_type, me.slots = T.object_of(t.py, fields, t.methods, t.cls, t.writes), place
                    me.names_used, me.writes_used = set(uses), set(writes)
                    me.owner = owner
                    if owner in made_local:
                        me.local, me.held = t, held
                    passed = [self.argument(x) for x in n.args]
                    call = ast.Call(ast.copy_location(ast.Name(flat[owner, f.attr], ast.Load()), f), [ast.copy_location(me, n)] + [x for x, _ in passed],
                                    [self.visit(k) for k in n.keywords])
                    call.sj_writes = [(flat[owner, k], pooled(owner, el[1]) if el else None) for k in sorted(writes) if (owner, k) in flat] \
                        + [w for _, ws in passed for w in ws]
                    if not el and owner not in made_local:       # objects in its attributes: any of their numbers may be assigned
                        call.sj_writes += [(flat[path_owner(owner, k)], None) for k, _ in leaf_types(T.object_of(t.py, fields, t.methods)) if '.' in k]
                    if owner in made_local:             # objects it holds: any of their numbers may be assigned
                        call.sj_writes += [(v.id, None) for k, v in place.items() if '.' in k]
                    if owner in made_local:             # no parameter holds the method: the call names it
                        fn = methods[owner, f.attr].elem
                        CALLEES[id(fn)] = fn
                        call.sj_callee = id(fn)
                    return ast.copy_location(call, n)
            if isinstance(f, ast.Name) and f.id not in BUILTIN_NAMES and any(self.instance(a) for a in n.args):
                passed = [self.argument(x) for x in n.args]
                call = ast.copy_location(ast.Call(f, [x for x, _ in passed], [self.visit(k) for k in n.keywords]), n)
                call.sj_writes = [w for _, ws in passed for w in ws]
                return call
            return self.generic_visit(n)

        def ref_arg(self, a):
            """An argument that is a reference: its index, marked with its class and the pool it indexes."""
            if not refexpr(a) and not (none(a) and pool_type is not None):
                return self.visit(a)
            r = ast.copy_location(RefArg(ast.copy_location(ast.Constant(-1), a) if none(a) else self.visit(a)), a)
            r.cls, r.pool, r.cols = None if none(a) else refclass(a), pool_type, pool_lists()
            return r

        def construct(self, n):
            """v = C(args): the arguments in order, then each attribute as C.__init__ assigns it."""
            v, call = n.targets[0].id, n.value
            t = made[call.func.id]
            params = t.recipe['params']
            if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
                raise Unsupported(call, f'{call.func.id}() with * or **')
            positional = [p for p in params if not p[2]]
            if len(call.args) > len(positional):
                raise Unsupported(call, f'{call.func.id}() with more arguments than it takes')
            out, given, inner = [], {}, {}
            for p, a in list(zip(positional, call.args)) + [((k.arg,), k.value) for k in call.keywords]:
                if p[0] in given or p[0] not in {q[0] for q in params}:
                    raise Unsupported(call, f'{call.func.id}() with these arguments')
                o = owner_of(a)
                if o is not None:                # an object: v's attribute is another name for it, one set of variables
                    base = o.split('.')[0]
                    if stores.get(v, 0) != 1 or base in made_local and not base.startswith('sjnew') and (
                            stores.get(base, 0) != 1 or id(n) in looped):
                        raise Unsupported(call, f'{v}.{p[0]} would hold an object that one of the names is later given another of')
                    inner[p[0]] = o
                    given[p[0]] = None
                    continue
                temp = fresh(f'sjc__{v}__{p[0]}')
                out.append(ast.copy_location(ast.Assign([ast.Name(temp, ast.Store())], self.visit(a)), n))
                given[p[0]] = ast.Name(temp, ast.Load())
            for name, default, _ in params:
                if name not in given:
                    if default is None or type(default[1]) not in (int, float, bool):
                        raise Unsupported(call, f'{call.func.id}() leaving out {name}, whose default isn\'t a number')
                    given[name] = ast.Constant(default[1])

            def value(src):
                if src[0] == 'param':
                    return copy.deepcopy(given[src[1]])
                if src[0] == 'expr' and any(isinstance(m, ast.Name) and m.id in inner for m in ast.walk(src[1])):
                    raise Unsupported(call, f'{call.func.id}.__init__ computes with an object it is given')
                if src[0] == 'const':
                    if type(src[1]) not in (int, float, bool):
                        raise Unsupported(call, f'{call.func.id}() assigns a {type(src[1]).__name__}')
                    return ast.Constant(src[1])
                e = copy.deepcopy(src[1])
                for m in ast.walk(e):
                    for field, x in ast.iter_fields(m):
                        if isinstance(x, ast.Name) and x.id in given:
                            setattr(m, field, copy.deepcopy(given[x.id]))
                        elif isinstance(x, list):
                            x[:] = [copy.deepcopy(given[y.id]) if isinstance(y, ast.Name) and y.id in given else y for y in x]
                return ast.Name(given[e.id].id, ast.Load()) if isinstance(e, ast.Name) and e.id in given else e
            for a, src in t.recipe['assigns']:
                if src[0] == 'param' and src[1] in inner:
                    same[f'{v}.{a}'] = inner[src[1]]
                    continue
                out.append(ast.copy_location(ast.Assign([ast.Name(flat[v, a], ast.Store())], value(src)), n))
            for s in out:
                ast.fix_missing_locations(ast.copy_location(s, n))
                for m in ast.walk(s):
                    if hasattr(m, 'lineno'):
                        m.lineno, m.col_offset, m.end_lineno, m.end_col_offset = n.lineno, n.col_offset, n.end_lineno, n.end_col_offset
            if t.recipe['post_init']:
                post = ast.copy_location(ast.Expr(ast.copy_location(ast.Call(ast.copy_location(ast.Attribute(ast.Name(v, ast.Load()), '__post_init__', ast.Load()), n), [], []), n)), n)
                out.append(self.visit(ast.fix_missing_locations(post)))
            return out

        def instance(self, a):
            """(owner, index expression or None) if a names an instance in variables or a list (locate), else None."""
            pl = locate(a)
            if pl is None or pl[0] == 'ref':
                return None
            if pl[0] == 'static':
                return pl[1], None
            if isinstance(a, ast.Subscript) and not isinstance(pl[2], (ast.Name, ast.Constant)):
                raise Unsupported(a, f'passes {a.value.id}[...] with an index that is an expression: name the element first')
            return pl[1], pl[2]

        def argument(self, a):
            """An argument, and what a call may write through it: an instance goes as all its attributes, which the
            compiler narrows to what the callee uses once it knows the callee."""
            found = self.instance(a)
            if found is None:
                return self.visit(a), []
            owner, idx = found
            t = types_of[owner] if idx is not None or owner in made_local else handed(owner, types_of[owner])
            data = [k for k, ft in t.elem if ft.kind != 'function']
            held = {}
            if owner in made_local:
                data = list(dict.fromkeys(a for a, _ in made[made_local[owner]].recipe['assigns']))
            if idx is None and owner not in made_local:
                place = {k: ast.copy_location(ast.Name(flat[path_owner(owner, k)], ast.Load()), a) for k, _ in leaf_types(t)}
            elif idx is None:
                place, held = made_place(owner, data, a)
            else:
                place = {k: column(owner, k, copy.deepcopy(idx), ast.Load(), a) for k in data}
            me = ast.copy_location(SelfArg(list(place.values()), ast.Load()), a)
            me.object_type, me.slots, me.passed, me.full, me.owner = t, place, True, (t, place), owner
            if owner in made_local:
                me.local, me.full, me.held = t, (None, place), held
            if owner in made_local:
                return me, [(flat[owner, k], None) for k in sorted(t.writes) if (owner, k) in flat] + [(v.id, None) for k, v in place.items() if '.' in k]
            return me, [(flat[path_owner(owner, k)], copy.deepcopy(idx)) for k, _ in leaf_types(t) if k.split('.')[0] in t.writes or '.' in k] \
                if idx is None else [(flat[owner, k], pooled(owner, idx)) for k in sorted(t.writes) if (owner, k) in flat]

        def visit_Attribute(self, n):
            pl = locate(n.value)
            if pl is not None and pl[0] == 'ref':
                return field(pl, n.attr, n.ctx, n, self.visit)
            owner = pl[1] if pl is not None else n.value.id if isinstance(n.value, ast.Name) else None
            if owner is not None and (owner, n.attr) in methods:
                raise Unsupported(n, f'{owner}.{n.attr} is a method, and only calling it compiles')
            got = field(pl, n.attr, n.ctx, n, self.visit) if pl is not None else None
            if got is not None:
                if isinstance(n.ctx, ast.Del):
                    raise Unsupported(n, f'del of {self.shown_owner(owner)}.{n.attr}')
                return got
            if owner_of(n) is not None and not isinstance(n.ctx, ast.Load):
                raise Unsupported(n, f'assigns {owner_of(n)}, an object in an attribute')
            return self.generic_visit(n)

        def shown_owner(self, owner):
            return f'an element of {owner}' if owner in lists or owner in inner_lists else owner

        def visit_Return(self, n):
            v = n.value
            if ref_returns:                      # a reference, or None: the object is found by its index after the call
                if v is not None and not refexpr(v) and not none(v):
                    raise Unsupported(n, 'returns a reference at one return and something else at another')
                returned.append(('pool', ref_returns_cls))
                if pool_made(v):
                    out, idx = self.build(v, n)
                    return out + [ast.copy_location(ast.Return(idx), n)]
                return ast.copy_location(ast.Return(ast.Constant(-1) if v is None or none(v) else self.visit(v)), n)
            if isinstance(v, ast.Name) and v.id in made_local:       # a new instance: its attributes, made into one by the call
                leaves = []

                def spec(o, stack):
                    """How the instance is built after the call: [attribute or (attribute, held object's spec)], its numbers
                    being the leaves, in order. A held object is a new instance too, or an argument's object, as it is."""
                    out = []
                    for a in dict.fromkeys(a for a, _ in made[made_local[o]].recipe['assigns']):
                        to = same.get(f'{o}.{a}')
                        if to is None:
                            leaves.append(flat[o, a])
                            out.append(a)
                        elif to in made_local and to not in stack:
                            out.append((a, ('made', types_of[to].cls, spec(to, stack + (to,)))))
                        elif to.split('.')[0] in pindex and to not in made_local:
                            base, *path = to.split('.')
                            out.append((a, ('path', pindex[base], '.'.join(path))))
                        else:
                            raise Unsupported(n, f'returns {o}, holding {to}')
                    return tuple(out)
                attrs = spec(v.id, (v.id,))
                if not leaves:
                    raise Unsupported(n, f'returns an instance of {made_local[v.id]} with no numbers in it')
                returned.append(('made', types_of[v.id].cls, attrs))
                return ast.copy_location(ast.Return(ast.copy_location(ast.Tuple([ast.Name(x, ast.Load()) for x in leaves], ast.Load()), v)), n)
            if isinstance(v, ast.Name) and v.id in objects and v.id in pindex:
                returned.append(('arg', None))
                return ast.copy_location(ast.Return(ast.copy_location(ast.Constant(pindex[v.id]), v)), n)
            if isinstance(v, ast.Attribute) and owner_of(v) is not None and owner_of(v).split('.')[0] in pindex:
                base, *path = owner_of(v).split('.')
                returned.append(('path', pindex[base], '.'.join(path)))
                return ast.copy_location(ast.Return(ast.copy_location(ast.Constant(0), v)), n)
            el = element_of(v) if v is not None else None
            if el:
                returned.append(('elem', pindex[el[0]]))
                return ast.copy_location(ast.Return(self.visit(el[1]) if isinstance(v, ast.Subscript) else el[1]), n)
            returned.append(None)
            return self.generic_visit(n)

        def visit_Lambda(self, n):
            return self.generic_visit(n)

        def visit_Name(self, n):
            if n.id in dicts:
                raise Unsupported(n, f"{n.id} is used as a dict object, and only {n.id}['key'] with a key it has compiles")
            if n.id in objects:
                raise Unsupported(n, f'{n.id} is used as an object, and only {n.id}.attribute compiles')
            if n.id in lists:
                raise Unsupported(n, f'{n.id} is used as a list, and only len({n.id}), {n.id}[i].attribute and for x in {n.id} compile')
            if n.id in alias:
                raise Unsupported(n, f'{n.id}, an element of {alias[n.id]}, is used as an object, and only {n.id}.attribute compiles')
            return n

    # v = a.add(b), v = make(x), v = a + b: a call returning a new instance makes v one too, its attributes the call's results
    by_cls = {t.cls: name for name, t in made.items()}
    funcs = {p.arg: t.elem for p, t in zip(params, arg_types) if t.kind == 'function'}

    def instance_owner(e):
        """The owner of the instance e names, in variables or a list (locate), else None."""
        pl = locate(e)
        return pl[1] if pl is not None and pl[0] != 'ref' else None
    for q, p in copied.items():                  # starts as the argument's numbers, then is each new instance
        if types_of[p].cls not in by_cls:
            raise Unsupported(tree, f'{p} is given a new instance of a class it can\'t make here')
        register(q, by_cls[types_of[p].cls])

    def call_class(call):
        """The made class's parameter, if the call returns a new instance: a function's, or a method's."""
        f, w = call.func, None
        if isinstance(f, ast.Name) and f.id in funcs:
            w = funcs[f.id]
        elif isinstance(f, ast.Attribute):
            o = instance_owner(f.value)
            if o is not None and (o, f.attr) in methods:
                w = methods[o, f.attr].elem
        if getattr(w, '_tree', None) is None or not (hasattr(w, '_fn') or hasattr(w, 'class_globals')):
            return None
        cls = returns_made(w._tree, w.class_globals if hasattr(w, 'class_globals') else w._fn.__globals__)
        if cls is None:
            return None
        if cls not in by_cls:
            raise Unsupported(call, f'calls something returning a new {cls.__name__}, which can\'t be made here')
        if pools_by.get(cls) is not None and pools_by[cls][3] is not None:
            raise Unsupported(call, f'calls something returning a new {cls.__name__}, kept in a linked structure here: not yet')
        return by_cls[cls]

    class Operators(ast.NodeTransformer):
        """a + b, -a for an instance a: its class's __add__, __neg__, which return a new instance as a method would."""
        def visit_BinOp(self, n):
            self.generic_visit(n)
            return self.method(n, n.left, [n.right])

        def visit_UnaryOp(self, n):
            self.generic_visit(n)
            return self.method(n, n.operand, [])

        def method(self, n, x, rest):
            name = OPERATORS.get(type(n.op))
            o = instance_owner(x) if name else None
            if o is None or (o, name) not in methods:
                return n
            other = instance_owner(rest[0]) if rest else None
            if other is not None and None not in (types_of[other].cls, types_of[o].cls) and types_of[other].cls is not types_of[o].cls:
                raise Unsupported(n, f'{name} between instances of two classes (Python may ask the other\'s reflected method first)')
            return ast.copy_location(ast.Call(ast.copy_location(ast.Attribute(x, name, ast.Load()), n), rest, []), n)

    def split_returns(stmts):
        """return <a new instance> becomes v = it; return v, so v is an instance made here."""
        out = []
        for s in stmts:
            for field in ('body', 'orelse', 'finalbody'):
                inner = getattr(s, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    setattr(s, field, split_returns(inner))
            for h in getattr(s, 'handlers', None) or ():
                h.body = split_returns(h.body)
            if isinstance(s, ast.Return) and isinstance(s.value, ast.Call) and not pool_made(s.value):
                c = s.value.func.id if isinstance(s.value.func, ast.Name) and s.value.func.id in made else call_class(s.value)
                if c is not None:
                    v = fresh('sjret')
                    a = ast.copy_location(ast.Assign([ast.Name(v, ast.Store())], s.value), s)
                    register(v, c)
                    makers[id(a)] = 'make' if s.value.func.__class__ is ast.Name and s.value.func.id in made else 'call'
                    out += [a, ast.copy_location(ast.Return(ast.copy_location(ast.Name(v, ast.Load()), s.value)), s)]
                    continue
            out.append(s)
        return out
    def makes_new(e):
        """The made class's parameter, if e is C(...) or a call returning a new instance."""
        if not isinstance(e, ast.Call) or pool_made(e):
            return None
        if isinstance(e.func, ast.Name) and e.func.id in made:
            return e.func.id
        return call_class(e)

    def pure(call):
        """A call that assigns nothing anything else here reads: a builtin, math, numpy, a method assigning nothing."""
        f = call.func
        if isinstance(f, ast.Name):
            return f.id in BUILTIN_NAMES and f.id not in funcs and f.id not in types_of
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ('math', 'np', 'numpy'):
            return True
        if isinstance(f, ast.Attribute):
            o = instance_owner(f.value)
            return o is not None and f.attr in (types_of[o].methods or {}) and not types_of[o].methods[f.attr][1]
        return False

    def hoist(stmts):
        """A new instance used inside an expression ((a - b).norm(), f(a + b)) is made into a variable first, as Python
        would make it then: so nothing that runs before it in the statement may assign anything."""
        out = []
        for s in stmts:
            for field in ('body', 'orelse', 'finalbody'):
                inner = getattr(s, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    setattr(s, field, hoist(inner))
            for h in getattr(s, 'handlers', None) or ():
                h.body = hoist(h.body)
            value = getattr(s, 'value', None)
            whole = isinstance(value, ast.Call) and (isinstance(s, (ast.Expr, ast.Return)) or isinstance(s, ast.Assign) and (
                id(s) in makers or len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)))
            if whole:                    # the call is the statement's: only what's inside it is made first
                slots = [(value, 'func')] + [(value.args, k) for k in range(len(value.args))] + [(kw, 'value') for kw in value.keywords]
            elif isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return, ast.Expr)) and value is not None:
                slots = [(s, 'value')]
            elif isinstance(s, ast.If):
                slots = [(s, 'test')]
            elif isinstance(s, ast.For):
                slots = [(s, 'iter')]
            else:
                slots = []
                own = [v for f, v in ast.iter_fields(s) if f not in STATEMENTS]
                if any(isinstance(n, ast.Call) and makes_new(n) for v in own for x in (v if isinstance(v, list) else [v])
                       if isinstance(x, ast.AST) for n in ast.walk(x)):
                    raise Unsupported(s, 'a new instance used in this kind of statement: assign it to a name first')
            made_here, unsafe = [], [False]

            def walk(e, cond):
                """Post-order, as Python evaluates. cond: inside something that may not run."""
                if isinstance(e, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    cond = True
                    for x in ast.iter_child_nodes(e):
                        walk(x, cond)
                    return e
                for field, v in ast.iter_fields(e):
                    later = isinstance(e, ast.IfExp) and field in ('body', 'orelse') or isinstance(e, ast.BoolOp) or \
                        isinstance(e, ast.Compare) and field == 'comparators'
                    if isinstance(v, ast.AST):
                        setattr(e, field, walk(v, cond or later))
                    elif isinstance(v, list):
                        setattr(e, field, [walk(x, cond or (later and (k > 0 or not isinstance(e, ast.BoolOp)))) if isinstance(x, ast.AST) else x
                                           for k, x in enumerate(v)])
                if isinstance(e, ast.Call):
                    c = makes_new(e)
                    if c is not None:
                        if cond:
                            raise Unsupported(e, 'a new instance made only on one side of a condition: assign it to a name first')
                        if unsafe[0]:
                            raise Unsupported(e, 'a new instance made after a call that may assign: assign it to a name first')
                        v = fresh('sjnew')
                        a = ast.copy_location(ast.Assign([ast.Name(v, ast.Store())], e), s)
                        register(v, c)
                        makers[id(a)] = 'make' if isinstance(e.func, ast.Name) and e.func.id in made else 'call'
                        made_here.append(a)
                        return ast.copy_location(ast.Name(v, ast.Load()), e)
                    if not pure(e):
                        unsafe[0] = True
                return e
            for node, field in slots:
                if isinstance(node, list):
                    node[field] = walk(node[field], False)
                else:
                    setattr(node, field, walk(getattr(node, field), False))
            out += made_here + [s]
        return out
    for _ in range(16):
        before = (len(made_local), ast.dump(tree))
        tree = Operators().visit(tree)
        tree.body = hoist(tree.body)
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and id(n) not in makers and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and isinstance(n.value, ast.Call):
                c = call_class(n.value)
                if c is not None:
                    v = n.targets[0].id
                    if made_local.get(v, c) != c or (v in types_of and v not in made_local):
                        raise Unsupported(n, f'{v} is an instance of two classes, or a parameter too')
                    register(v, c)
                    makers[id(n)] = 'call'
        tree.body = split_returns(tree.body)
        if (len(made_local), ast.dump(tree)) == before:
            break
    ast.fix_missing_locations(tree)
    for n in ast.walk(tree):                      # assigned only a new instance
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.With, ast.NamedExpr)) and id(n) not in makers:
            for t in (n.targets if isinstance(n, ast.Assign) else [getattr(n, 'target', None)]):
                if t is not None and any(isinstance(x, ast.Name) and x.id in made_local and not isinstance(x.ctx, ast.Load) for x in ast.walk(t)):
                    raise Unsupported(n, 'an instance it makes is assigned something other than a new instance')

    stores = {}                                  # how often each name is assigned, and the statements inside loops
    looped = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            stores[n.id] = stores.get(n.id, 0) + 1
        elif isinstance(n, (ast.For, ast.While)):
            looped.update(id(m) for b in n.body + n.orelse for m in ast.walk(b))

    def made_place(owner, keys, node):
        """For an instance made here, given to a callee: {path: variable} of its numbers, through the objects it holds, and
        {path: the held object's type (a made class's, or a parameter's object type)}."""
        place, held = {}, {}

        def expand(prefix, o, keys, stack):
            for k in keys:
                path, to = f'{prefix}{k}', same.get(f'{o}.{k}')
                if to is None:
                    if (o, k) in flat:
                        place[path] = ast.copy_location(ast.Name(flat[o, k], ast.Load()), node)
                elif to in made_local and to not in stack:
                    t2 = made[made_local[to]]
                    held[path] = t2
                    expand(path + '.', to, list(dict.fromkeys(a for a, _ in t2.recipe['assigns'])), stack + (to,))
                elif to in types_of and types_of[to].kind == 'object' and not same_targets(types_of[to]):
                    held[path] = types_of[to]
                    for lk, _ in leaf_types(types_of[to]):
                        place[f'{path}.{lk}'] = ast.copy_location(ast.Name(flat[path_owner(to, lk)], ast.Load()), node)
                else:
                    raise Unsupported(node, f'{owner} holds {to}, which can\'t be passed this way')
        expand('', owner, keys, (owner,))
        return place, held

    def holds_object(v, keys):
        """Of an instance made here, an attribute among keys that is another name for an object."""
        return next((k for k in keys if f'{v}.{k}' in same), None)

    if pool is not None or refnames:
        for n in ast.walk(tree):                 # for n in heads: n is each reference
            if isinstance(n, (ast.For, ast.comprehension)) and isinstance(n.target, ast.Name) and isinstance(n.iter, ast.Name) \
                    and n.iter.id in reflists:
                refnames[n.target.id] = reflists[n.iter.id]
        for _ in range(16):
            grew = False
            for n in ast.walk(tree):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and refexpr(n.value):
                    t, c = n.targets[0].id, refclass(n.value)
                    if t in refnames and refnames[t] is not None and c is not None and refnames[t] is not c:
                        raise Unsupported(n, f'{t} holds references to objects of two classes')
                    if t not in refnames or (refnames[t] is None and c is not None):
                        refnames[t] = c
                        grew = True
            if not grew:
                break
        for n in ast.walk(tree):
            if isinstance(n, (ast.AugAssign, ast.For, ast.comprehension, ast.NamedExpr, ast.AnnAssign)) and not (
                    isinstance(n, (ast.For, ast.comprehension)) and isinstance(n.iter, ast.Name) and n.iter.id in reflists) and any(
                    isinstance(x, ast.Name) and x.id in refnames for x in [n.target] + list(getattr(n.target, 'elts', []))):
                raise Unsupported(n, 'a reference is assigned in a way other than name = reference or None')
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in refnames for t in n.targets) and \
                    not (len(n.targets) == 1 and (refexpr(n.value) or none(n.value))):
                raise Unsupported(n, f'{n.targets[0].id if isinstance(n.targets[0], ast.Name) else "a reference"} holds a reference, and is assigned something else')
    if any(spec[3] is not None for spec in pools_by.values()):     # it may run again with more room: only if it wrote nothing
        arrays = {q.arg for q, t in zip(new_params, new_types) if t.kind == 'array'}
        if any(isinstance(n, ast.Subscript) and not isinstance(n.ctx, ast.Load) and isinstance(n.value, ast.Name) and n.value.id in arrays
               for n in ast.walk(tree)):
            raise Unsupported(tree, 'makes instances kept in a linked structure and writes an array: not yet')
    ref_returns = any(isinstance(n, ast.Return) and n.value is not None and refexpr(n.value) for n in ast.walk(tree))
    returned_cls = {refclass(n.value) for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value is not None and refexpr(n.value)} - {None}
    if len(returned_cls) > 1:
        raise Unsupported(tree, 'returns references to objects of two classes')
    ref_returns_cls = returned_cls.pop() if returned_cls else None

    def pool_lists():
        """The pool's lists, in the order a callee given a reference takes them."""
        return [flat[f'{pool}{k}', key] for k, (_, numbers, prefs) in enumerate(pool_type.elem[0])
                for key in [a for a, _ in numbers] + [a for a, _ in prefs]]

    def deref(idx, attr, node):
        """A reference read through: its index, after checking it isn't None, as Python's AttributeError would."""
        return ast.copy_location(ast.Call(ast.Name('__sj_deref__', ast.Load()), [idx, ast.Constant(attribute_number(attr))], []), node)

    def truth(x, visit):
        if refexpr(x):                           # an instance is true, None false
            return ast.copy_location(ast.Compare(visit(x), [ast.NotEq()], [ast.Constant(-1)]), x)
        return visit(x)
    pindex = {p.arg: i for i, p in enumerate(params)}
    returned = []
    tree = Flatten().visit(tree)
    if any(returned):                            # returns an instance: the argument, or the element at an index
        if len(set(returned)) != 1:
            raise Unsupported(tree, 'returns an instance at one return and something else, or another kind of instance, at another')
        tree.sj_instance_ret = returned[0]
    tree.sj_names = {i: name for name, i in index.items()}      # for messages: the element's name, not its index's
    tree.sj_plain = plain
    if methods or objects or lists:
        tree.body = shown_writes(tree.body)
    tree.args.posonlyargs, tree.args.args = [], new_params
    return ast.fix_missing_locations(tree), new_types, slots


OPERATORS = {ast.Add: '__add__', ast.Sub: '__sub__', ast.Mult: '__mul__', ast.Div: '__truediv__', ast.FloorDiv: '__floordiv__',
             ast.Mod: '__mod__', ast.Pow: '__pow__', ast.MatMult: '__matmul__', ast.USub: '__neg__', ast.UAdd: '__pos__'}


def own_nodes(fdef):
    """The nodes of a function, not those of functions or lambdas defined in it."""
    todo = list(fdef.body)
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(n))


def returns_made(fdef, globals_):
    """The class every return of a function makes a new instance of: return C(...), or return v where v is assigned
    nothing but C(...); None if not, or if it can return NotImplemented."""
    def made_class(e):
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
            c = globals_.get(e.func.id)
            if isinstance(c, type) and not issubclass(c, (__import__('enum').Enum, BaseException)):
                return c
        return None
    made, stores, rets = {}, {}, []
    for n in own_nodes(fdef):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            stores[n.id] = stores.get(n.id, 0) + 1
        elif isinstance(n, ast.Name) and n.id == 'NotImplemented':
            return None
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and made_class(n.value):
            made.setdefault(n.targets[0].id, []).append(made_class(n.value))
        elif isinstance(n, ast.Return):
            rets.append(n)
    found = set()
    for r in rets:
        c = made_class(r.value)
        if c is None and isinstance(r.value, ast.Name) and r.value.id in made and len(made[r.value.id]) == stores.get(r.value.id, 0):
            c = made[r.value.id][0] if len(set(made[r.value.id])) == 1 else None
        if c is None:
            return None
        found.add(c)
    return found.pop() if len(found) == 1 else None


def ref_return_class(fdef, given, pools_by, missing):
    """The class of the reference a function returns, when given references of these classes ({parameter: class}) into
    pools laid out as pools_by: followed through its names, attributes, a if c else b and None. missing if it can't tell."""
    names = dict(given)

    def cls_of(e):
        if isinstance(e, ast.Constant) and e.value is None:
            return None
        if isinstance(e, ast.Name):
            return names.get(e.id, missing)
        if isinstance(e, ast.IfExp):
            cs = [cls_of(e.body), cls_of(e.orelse)]
            known = {c for c in cs if c is not None}
            return missing if missing in cs or len(known) > 1 else (known.pop() if known else None)
        if isinstance(e, ast.Attribute):
            c = cls_of(e.value)
            spec = pools_by.get(c) if c is not missing else None
            return spec[2][e.attr] if spec is not None and e.attr in spec[2] else missing
        return missing
    for _ in range(8):
        for n in own_nodes(fdef):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                c = cls_of(n.value)
                if c is not missing and (names.get(n.targets[0].id) is None):
                    names[n.targets[0].id] = c
    found = {cls_of(n.value) for n in own_nodes(fdef) if isinstance(n, ast.Return) and n.value is not None}
    if missing in found or not found:
        return missing
    known = found - {None}
    return missing if len(known) > 1 else (known.pop() if known else None)


CALLEES = {}     # id -> a method a call to an instance made in the function names (Call.sj_callee): no parameter holds it

BUILTIN_NAMES = frozenset(dir(__import__('builtins')))     # a call to one of these never gets an instance's attributes


def leaf_types(t):
    """An instance type's numbers and arrays, through the objects in its attributes: [(path, type)], in the order its
    parameters come when it's flattened (breadth first, as flatten_dicts registers them)."""
    out, todo = [], [('', t.elem)]
    while todo:
        prefix, fields = todo.pop(0)
        for key, ft in fields:
            if ft.kind == 'object':
                todo.append((f'{prefix}{key}.', ft.elem))
            elif ft.kind not in ('function', 'same'):      # another name for an object: its numbers are where it was first met
                out.append((prefix + key, ft))
    return out


def same_targets(t):
    """The (argument, path) of each same_as in an instance type, through the objects in its attributes."""
    out = []
    for _, ft in t.elem:
        if ft.kind == 'same':
            out.append(ft.elem)
        elif ft.kind == 'object':
            out += same_targets(ft)
    return out


class RefArg(ast.UnaryOp):
    """A reference given to a compiled callee: its index (operand), its class (cls) and the pool it indexes (pool, a type).
    A unary + of the index, so whatever looks for reads sees them."""

    def __init__(self, index):
        super().__init__(ast.UAdd(), index)

    def __deepcopy__(self, memo):
        new = RefArg(copy.deepcopy(self.operand, memo))
        new.cls, new.pool, new.lists, new.cols = self.cls, self.pool, getattr(self, 'lists', True), self.cols
        return ast.copy_location(new, self)


class ClassOf(ast.Constant):
    """Stands for an instance in isinstance(): its class (cls), which is in the signature."""


class SelfArg(ast.Tuple):
    """An instance passed as a method's self: its attributes, as the caller's variables (object_type; slots: attribute ->
    variable). A tuple of names, so whatever looks for reads sees them."""

    def __deepcopy__(self, memo):            # the names are copied; the type, which holds functions, is shared
        new = SelfArg(copy.deepcopy(self.elts, memo), ast.Load())
        new.object_type, new.slots = self.object_type, self.slots
        new.passed, new.full, new.owner = getattr(self, 'passed', False), getattr(self, 'full', None), getattr(self, 'owner', None)
        new.local, new.same, new.ref = getattr(self, 'local', None), getattr(self, 'same', None), getattr(self, 'ref', False)
        new.held = getattr(self, 'held', None)
        if hasattr(self, 'writes_used'):
            new.names_used, new.writes_used = self.names_used, self.writes_used
        return ast.copy_location(new, self)


STATEMENTS = ('body', 'orelse', 'finalbody', 'handlers')


def shown_writes(stmts):
    """After a statement calling a method that assigns attributes, p__a = p__a for each: the call writes the variables,
    and this says so to everything that looks for assignments, so nothing takes p.a as unchanged across it."""
    out = []
    for s in stmts:
        for field in STATEMENTS:
            inner = getattr(s, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], (ast.stmt, ast.excepthandler)):
                setattr(s, field, shown_writes(inner) if isinstance(inner[0], ast.stmt) else inner)
        if isinstance(s, ast.Try):
            for h in s.handlers:
                h.body = shown_writes(h.body)
        own = [v for f, v in ast.iter_fields(s) if f not in STATEMENTS]
        writes, deferred, places, inside = set(), False, {}, set()
        for v in own:
            for n in ast.walk(v) if isinstance(v, ast.AST) else [x for y in (v or []) if isinstance(y, ast.AST) for x in ast.walk(y)]:
                if isinstance(n, ast.Lambda):
                    deferred = deferred or any(getattr(m, 'sj_writes', None) for m in ast.walk(n))
                if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    inside |= {id(m) for m in ast.walk(n)}      # runs within the statement: shown after it, as for a call
                writes |= {(name, ast.dump(i) if i is not None else None) for name, i in getattr(n, 'sj_writes', None) or ()}
                for name, i in getattr(n, 'sj_writes', None) or ():
                    key = (name, ast.dump(i) if i is not None else None)
                    places[key] = i
                    if id(n) in inside and i is not None:       # an element the comprehension chose: its index is gone after
                        places[key] = None
        if writes and (deferred or not isinstance(s, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return, ast.If, ast.While, ast.For))):
            raise Unsupported(s, 'a method that assigns attributes, called in a lambda or a with or try header')

        def target(name, i, ctx):
            return ast.Name(name, ctx) if i is None else ast.Subscript(ast.Name(name, ast.Load()), copy.deepcopy(i), ctx)
        shown = []
        for name, d in sorted(writes, key=lambda w: (w[0], w[1] or '')):
            if d is not None and places[name, d] is None:     # a list's element: shown written where nothing runs
                at = ast.Constant(0)
                mark = ast.Assign([target(name, at, ast.Store())], target(name, at, ast.Load()))
                mark.sj_shown = True
                shown.append(ast.fix_missing_locations(ast.copy_location(ast.If(ast.Constant(False), [ast.copy_location(mark, s)], []), s)))
                continue
            shown.append(ast.copy_location(ast.Assign([target(name, places[name, d], ast.Store())], target(name, places[name, d], ast.Load())), s))
            shown[-1].sj_shown = True        # says a call may have written it; not itself a write
        if isinstance(s, ast.Return):     # before it, then: nothing runs after, but the function's exits must write these back
            out += shown + [s]
        elif isinstance(s, (ast.If, ast.While, ast.For)) and shown:
            # the test or the header ran: what follows it in each way on may see the writes
            s.body = copy.deepcopy(shown) + s.body
            s.orelse = copy.deepcopy(shown) + s.orelse
            out += [s] + (shown if isinstance(s, (ast.While, ast.For)) else [])
        else:
            out += [s] + shown
    return out



class Callee:
    """What compiling a call to another function needs from it. hidden: for the globals its compiled tree takes as parameters
    after its real ones, their identity. hidden_map: identity of a global -> the parameter holding it, within this function."""

    def __init__(self, identity, tree, globals_, filename, offset, boundscheck, parity, hidden=(), hidden_map=None, name=None, npos=None):
        self.identity, self.tree, self.globals, self.filename, self.offset = identity, tree, globals_, filename, offset
        self.boundscheck, self.parity, self.hidden, self.hidden_map, self.name = boundscheck, parity, hidden, hidden_map, name
        self.npos = npos                 # how many of the parameters can be given by position: those after are keyword-only


class Unit:
    """One C file: the function called from Python, and a static C function for each function and signature it calls."""

    def __init__(self, identity=None):
        self.callees = {}        # (function, argument types) -> its Compiler
        self.stack = [identity]  # the calls being worked out, to find recursion
        self.code = []           # the callees' C, each after those it calls
        self.helpers = set()
        self.frames = []         # (file, function name, line offset) a failed call's trace refers to
        self.modules = []        # cimo modules whose functions are called: their C goes in this file
        self.fma = self.wrapv = None

    def add_module(self, m):
        if any(m is x for x in self.modules):
            return
        o = m.options or {}
        for name in ('fma', 'wrapv'):
            if self.modules and getattr(self, name) != o.get(name):
                raise Unsupported(None, f'C modules compiled with different {name}=')
        self.fma, self.wrapv = o.get('fma'), o.get('wrapv')
        self.modules.append(m)

    def c_source(self):
        """The C of the modules called, with their defines."""
        out = []
        for m in self.modules:
            defines = (m.options or {}).get('defines', ())
            out += [f'#define {k} {"1" if v is None else v}' for k, v in defines] + [m.source] + [f'#undef {k}' for k, _ in defines]
        return '\n'.join(out) + '\n' if out else ''

    def callee(self, spec, types):
        key = (spec.identity, tuple(types))
        c = self.callees.get(key)
        if c is None:
            if spec.identity in self.stack:
                raise Unsupported(None, 'recursion')
            self.stack.append(spec.identity)
            try:
                c = Compiler(spec.tree, list(types), spec.globals, spec.filename, spec.offset, spec.boundscheck, spec.parity, False, self, spec.hidden_map)
                c.infer_with_webs()
                if c.tagged:
                    raise Unsupported(None, 'a variable in it is an int at some times and a float at others')
                if any(t.kind == 'dict' for t in c.env.values()):
                    raise Unsupported(None, 'it reads a dict')
                if c.instance_ret and c.instance_ret[0] == 'pool':
                    pass                         # a reference into the caller's pool: its index
                elif c.instance_ret and (c.instance_ret[0] != 'made' or any(type(a) is tuple for a in c.instance_ret[2])):
                    raise Unsupported(None, 'it returns an instance' + (' holding objects' if c.instance_ret[0] == 'made' else ''))
            finally:
                self.stack.pop()
            c.cname = f'sj_call_{len(self.callees)}'
            c.frame = len(self.frames)
            c.spec = spec
            self.frames.append((spec.filename, spec.name or spec.tree.name, spec.offset))
            self.callees[key] = c
        return c


class Compiler:
    def __init__(self, fdef, arg_types, globals_, filename, line_offset, boundscheck=True, parity=True, counts=False, unit=None, hidden_map=None):
        fdef, arg_types, self.dict_slots = flatten_dicts(fdef, arg_types)
        self.instance_ret = getattr(fdef, 'sj_instance_ret', None)
        self.plain = getattr(fdef, 'sj_plain', None)      # a parameter flattening kept -> the argument it is
        self.element_names = getattr(fdef, 'sj_names', {})   # ('arg', None): the int returned is an argument's index; ('elem', i): an index into argument i
        self.again = (arg_types, globals_, filename, line_offset, boundscheck, parity, counts, unit, hidden_map)
        self.unit = unit if unit is not None else Unit()
        self.hidden_map = hidden_map
        self.cname = 'sj_kernel'
        self.inferred = False
        self.calls = False         # the C calls a function of the unit
        self.call_stmt = False     # this statement calls one: anything read in it is read now, in Python's order
        self.counts = counts       # preflight --counts: the kernel counts how often each line runs
        self.count_lines = []      # counter -> the Python lines it stands for
        self.count_slot = {}       # id(statement) -> (its counter, whether it's the one that adds to it)
        self.count_heads = {}      # Python line of an if or a loop -> ('if' or 'loop', the counter of its body)
        self.count_tests = {}      # id(if statement) -> the counter its test adds to
        self.count_rest = {}       # counter of an else -> (counter of its if, counter of the if's body)
        self.split_done = False    # names are split into webs, so a conflict left is within one variable
        self.tagged = set()        # variables holding an exact int or a float: a double and v___int (see may_tag)
        self.not_literal = set()   # variables assigned an int that isn't a literal
        self.python_values = {}    # each variable's Python numbers: the literal, or None
        self.u64_params = set()    # parameters that must not be negative (else the call runs as Python)
        self.uses_rerun = False    # the kernel can ask for the call to run as Python, before doing anything
        self.tracks = False        # the function returns an array, so views remember which argument they're of
        self.makes = False         # the function makes arrays, so views remember the block they're in
        self.array_args = []       # for those: (argument index, dict key or None) of each array parameter
        self.ref_written = set()   # list parameters a callee given an element by reference writes
        local = {p.arg for p in fdef.args.posonlyargs + fdef.args.args}
        local |= {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.f = ListReturns().visit(Tables().visit(Lazy(local, set(globals_)).visit(copy.deepcopy(fdef))))
        self.exact = parity    # False: C's float semantics where CPython parity costs speed
        self.globals = globals_
        self.filename = filename
        self.offset = line_offset
        self.boundscheck = boundscheck
        a = fdef.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.kw_defaults:      # defaults are filled in by the call
            raise Unsupported(fdef, 'only plain positional parameters, with or without defaults, are compiled')
        self.params = [p.arg for p in a.posonlyargs + a.args]
        if len(self.params) != len(arg_types):
            raise Unsupported(fdef, 'wrong number of arguments')
        self.env = dict(zip(self.params, arg_types))
        self.returns = []
        self.lines = []
        self.depth = 1
        self.ntemp = 0
        self.uses_libm = False
        self.helpers = set()      # helpers from _mathlib this kernel uses
        self.checks = {}       # Python line -> bounds checks that run each time (for preflight)
        self.gotchas = {}      # Python line -> a way this line can differ from Python
        self.hoisted_lines = set()   # Python lines whose checks live only in a versioned loop's slow copy
        self.slow_copy = 0     # inside that slow copy
        self.forwards = {}     # array -> (loop variable, temps holding its elements at var - 1, var - 2, ...)
        self.no_forwarding = False  # emitting the copy of a loop for when the arrays it stores to share memory
        self.sunk = []             # per loop being emitted: variables whose numpy flag is set after it, not in it
        self.trips = []            # per loop being emitted: the C test that it ran at least once
        self.versionable = None    # ids of the loops versioning may copy, planned at the first one (plan_versioning)
        self.view_names = {}   # id of a view expression -> the variable that holds it while it's used
        self.rebound = set()   # arrays assigned somewhere: their extents aren't fixed, so facts mustn't name them
        self.broke = []        # for each open Python loop, the flag its else tests, or None
        self.da_breaks = []    # for each loop definite assignment is in: what's assigned at each break
        self.polls = sys.platform != 'emscripten'       # no signals to wait for in Pyodide
        self.uses_stop = False  # the kernel takes sj_stop, the Ctrl-C flag it checks (see open_counted)
        self.counted = {}      # depth of an open counted loop -> what closing it must emit
        self.loops = []        # for each open loop, the label its break jumps to, or None for a plain C break
        self.labels = 0
        self.short_range = False    # the next counted loop is a range with constant bounds, and short
        self.aliases = {}
        self.hoisted = {}      # ast.dump of a wraparound index -> its temporary, per loop iteration
        self.owned = []        # temporaries holding a list being built (freed on every exit)
        self.cnames = {}       # comprehension variable -> (C name, flag), shadowing the function's
        self.facts = {}        # loop variable -> (lowest value, array whose len() bounds it, slack)

    # ---- helpers
    def line(self, node):
        """A line for the C: counted from the function's def, so moving the function in its file, or renaming the file,
        leaves the C as it was and cimo's cache still has it. The dispatcher adds the offset back for tracebacks."""
        return node.lineno

    def py_line(self, node):
        """The line in the file, for what Python reads: preflight's notes, checks and counts."""
        return node.lineno + self.offset

    def fail(self, node, reason):
        raise Unsupported(node, self.shown(WEB_SUFFIX.sub('', reason)))

    def shown(self, text):
        """Text with the variables a dict's keys became written as the user wrote them."""
        for flat, (_, _, written) in self.dict_slots.items():
            text = text.replace(flat, written)
        for index, name in getattr(self, 'element_names', {}).items():
            text = text.replace(index, name)
        return text

    def shown_types(self):
        return {self.shown(WEB_SUFFIX.sub("'", k)): t.name for k, t in self.env.items() if not k.startswith(('sjv__', 'sjh__'))}

    def find_outs(self):
        """The dict keys this function writes: [(flat name, argument index, key)]. Their values go back to the
        dict at every exit, since Python would have stored them by then, whatever happens next."""
        stored = {n.id for n in ast.walk(self.f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.outs = [(flat, i, key) for flat, (i, key, _) in self.dict_slots.items() if flat in stored]
        shown = {id(n.targets[0]) for n in ast.walk(self.f) if getattr(n, 'sj_shown', False)}
        self.assigned = {n.id for n in ast.walk(self.f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and id(n) not in shown}
        self.tracks = bool(ret_arrays(self.ret))
        self.array_args = [(self.dict_slots[p][:2] if p in self.dict_slots else (self.plain_arg(p), None))
                           for p in self.params if self.came_as(p).kind == 'array'] if self.tracks else []
        for flat, _, _ in self.outs:
            if not self.env[flat].is_number:
                self.fail(self.f, f'{flat} is assigned something other than a number')

    def out_slot(self, k):
        """The sj_io slot of the k-th written key, past anything a return uses."""
        return 5 + 2 * ret_pairs(self.ret) + 2 * k

    def record_slot(self, k):
        """The sj_io slot where the k-th returned array's record starts, past everything else."""
        return self.out_slot(len(self.outs)) + sum(array_record_slots(t) for t in ret_arrays(self.ret)[:k])

    def save_outs(self):
        for k, (flat, _, _) in enumerate(self.outs):
            t = self.env[flat]
            at = self.out_slot(k)
            if t.kind == 'float':
                self.emit(f'memcpy(&sj_io[{at}], &v_{flat}, 8);')
            else:
                self.emit(f'sj_io[{at}] = (int64_t)v_{flat};')
            flag = f'v_{flat}__np' if t.np == 'dyn' else flag_of(t)
            self.emit(f'sj_io[{at + 1}] = {self.ret_flag(ast.Name(flat), flag)};')

    def is_builtin(self, name, which):
        return name == which and name not in self.env and name not in self.globals

    # ---- type inference
    def infer(self):
        for _ in range(8):
            before = dict(self.env)
            self.returns = []
            self.infer_block(self.f.body)
            if before == self.env:
                break
        ret = None
        for node, t in self.returns:
            j = t if ret is None else T.join_returns(ret, t)
            if j is None:
                self.fail(node, f'returns {ret.name} in one place and {t.name} in another')
            ret = j
        self.ret = ret or T.NONE
        if self.ret is not T.NONE and not ends_with_return(self.f.body):
            self.fail(self.f, 'can fall off the end (returning None) and also return a value')

    def return_type(self, v):
        if v is None:
            return T.NONE
        if isinstance(v, ast.Subscript) and self.typeof(v).kind == 'array' and T.is_pylist(self.typeof(v)):
            self.fail(v, 'returns a row of a list of lists (write the name of the whole list)')
        if isinstance(v, ast.Name) and self.env.get(v.id) is not None and self.env[v.id].kind == 'list':
            if self.env[v.id].elem.np is not False:
                self.fail(v, f'returns {self.env[v.id].name}, whose elements may be numpy scalars')
            return self.env[v.id]          # handed back whole: copied into a new list, or the one passed in
        if isinstance(v, ast.Tuple) and self.has_views(v):        # only a return can hold arrays in a tuple
            return T.tuple_of([self.typeof(x) for x in v.elts])
        return self.typeof(v)

    def is_list_ctor(self, v):
        return isinstance(v, (ast.List, ast.ListComp)) or (
            isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult)
            and (isinstance(v.left, ast.List) or isinstance(v.right, ast.List)))

    def list_type(self, v):
        if isinstance(v, ast.ListComp):
            return T.list_of(self.comp_types(v))
        lit, k = (v, None) if isinstance(v, ast.List) else \
            ((v.left, v.right) if isinstance(v.left, ast.List) else (v.right, v.left))
        if k is not None and self.typeof(k).kind not in ('int', 'bool'):
            self.fail(v, 'a list repeated by a non-integer')
        if not lit.elts:
            self.fail(v, 'an empty list (its element type is unknown)')
        ts = [self.typeof(x) for x in lit.elts]
        if not all(t.is_number for t in ts):
            self.fail(v, 'a list of non-numbers')
        return T.list_of(self.join_all(v, ts, 'a list'))

    def assign_type(self, node, name, t):
        if t is T.NONE:
            self.fail(node, f'{name} would be None')
        old = self.env.get(name)
        if old is not None and old is not t and (old.kind == 'array' or t.kind == 'array') and self.split_done:
            self.fail(node, f'{name} is {old.name} and later {t.name}; a variable must keep one type')
        if t is T.INT and not exact_int_literal(node):
            self.not_literal.add(name)
        if t in (T.INT, T.FLOAT):
            self.python_values.setdefault(name, []).append(literal_value(node))
        if old is not None and old is not t and {old, t} <= {T.U64, T.INT, T.BOOL} and T.U64 in (old, t):
            if not self.name_nonneg(name, frozenset([name])):
                self.fail(node, f'{name} holds an int past int64 in one place, and in another a value that may be negative')
            t = T.U64
            old = t
        if old is not None and old is not t:
            j = T.join(old, t) or self.narrow_variable(name, old, t)
            if j is None and self.may_tag(name, old, t):
                self.tagged.add(name)
                j = T.scalar('float', 'dyn')
            if j is None:
                self.fail(node, f'{name} is {old.name} and later {t.name}; a variable must keep one type')
            t = j
        self.env[name] = t

    def narrow_variable(self, name, old, t):
        """total = 0.0, then total += x[i] on float32: np.float32 from then on, and after no iterations Python has the
        float. Only for literals the narrow type holds exactly, so one C variable and a flag are enough."""
        j = T.narrow_or_literal(old, t) if self.split_done and name not in self.params else None
        if j is None:
            return None
        return j if all(T.holds_exactly(j.elem, v) for v in self.python_values.get(name, [])) else None

    def may_tag(self, name, old, t):
        """s = 0, then s += a[i] on floats: one variable with two types, and after no iterations Python has the
        int. It's held as a double with a flag saying it's still an int. A double holds such an int exactly when
        it's a small enough literal, or a parameter checked on entry, and an int behaves just like that float only
        in some places: check_tagged makes sure that's all it meets."""
        if not self.split_done or name in self.not_literal:
            return False
        kinds = {old.kind, t.kind}
        return kinds == {'int', 'float'} and (old if old.kind == 'int' else t) is T.INT \
            and not T.has_narrow(old, t)

    def check_tagged(self):
        if not self.tagged:
            return
        parents = {c: p for p in ast.walk(self.f) for c in ast.iter_child_nodes(p)}
        for n in ast.walk(self.f):
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id in self.tagged:
                if self.typeof(n.value).kind != 'float':
                    self.mixed(n, n.target.id)
            if isinstance(n, ast.Name) and n.id in self.tagged and isinstance(n.ctx, ast.Load) \
                    and not self.tag_is_safe(n, parents):
                self.mixed(n, n.id)

    def tag_is_safe(self, n, parents):
        p = parents.get(n)
        if isinstance(p, ast.BinOp):
            other = p.right if p.left is n else p.left
            return not (isinstance(other, ast.Name) and other.id in self.tagged) and self.typeof(other).kind == 'float'
        if isinstance(p, ast.Compare) and len(p.ops) == 1:
            other = p.comparators[0] if p.left is n else p.left
            return not (isinstance(other, ast.Name) and other.id in self.tagged) and self.typeof(other).kind == 'float'
        if isinstance(p, ast.AugAssign):        # two that may still be ints would give an int
            return p.value is n and self.typeof(p.target).kind == 'float' \
                and not (isinstance(p.target, ast.Name) and p.target.id in self.tagged)
        if isinstance(p, ast.Return):
            return True
        return isinstance(p, ast.Tuple) and isinstance(parents.get(p), ast.Return)

    def mixed(self, node, name):
        self.fail(node, f'{name} is int and later float, and this use would differ between them; a variable must keep one type')

    def infer_block(self, stmts):
        for i, s in enumerate(stmts):
            self.infer_stmt(s, docstring_ok=(i == 0 and stmts is self.f.body))

    def infer_stmt(self, s, docstring_ok=False):
        if isinstance(s, ast.Assign):
            if self.is_list_ctor(s.value):
                if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                    self.fail(s, 'a list assigned to something other than one name')
                self.assign_type(s, s.targets[0].id, self.list_type(s.value))
                return
            if self.has_views(s.value):
                for target in s.targets:
                    if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != len(s.value.elts):
                        self.fail(s, 'a tuple holding arrays assigned to something other than as many names')
                    for x, v in zip(target.elts, s.value.elts):
                        self.infer_target(x, self.typeof(v), s)
                return
            vt = self.typeof(s.value)
            if ret_arrays(vt) and vt.kind == 'tuple' and not all(isinstance(t, (ast.Tuple, ast.List)) and len(t.elts) == len(vt.elem) for t in s.targets):
                self.fail(s, 'a tuple holding arrays assigned to something other than as many names')
            for target in s.targets:          # a = b = value: value once, targets left to right
                self.infer_target(target, vt, s)
        elif isinstance(s, ast.AnnAssign):
            if s.value is None:
                self.fail(s, 'annotation without a value')
            self.infer_target(s.target, self.typeof(s.value), s)
        elif isinstance(s, ast.AugAssign):
            op = ARITH.get(type(s.op))
            if op is None:
                self.fail(s, OPERATOR_NAMES.get(type(s.op).__name__, f'the {type(s.op).__name__} operator'))
            res = self.binop_type(self.typeof(s.target), self.typeof(s.value), op, s)
            self.infer_target(s.target, res, s)
        elif isinstance(s, (ast.If, ast.While)):
            self.check_test(s.test)
            self.infer_block(s.body)
            self.infer_block(s.orelse)
        elif isinstance(s, ast.For):
            self.infer_block(s.orelse)
            self.bind_types(s.target, self.elem_struct(self.iterable(s.iter)), s)
            lists = {n.id for n in ast.walk(s.iter) if isinstance(n, ast.Name)
                     and self.env.get(n.id) is not None and self.env[n.id].kind in ('list', 'array')}
            for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
                if isinstance(n, ast.Name) and n.id in lists and isinstance(n.ctx, ast.Store):
                    self.fail(n, f'rebinds {n.id} inside a loop over it (Python keeps iterating the old list)')
            self.infer_block(s.body)
        elif isinstance(s, ast.Return):
            self.returns.append((s, self.return_type(s.value)))
        elif isinstance(s, (ast.Break, ast.Continue, ast.Pass)):
            pass
        elif isinstance(s, ast.Expr) and docstring_ok and isinstance(s.value, ast.Constant) \
                and isinstance(s.value.value, str):
            pass
        elif isinstance(s, ast.Expr) and self.calls_out(s.value):
            self.typeof(s.value)
        elif isinstance(s, ast.Expr):
            self.fail(s, 'an expression statement (calls with side effects are not compiled yet)')
        else:
            self.fail(s, f'the {type(s).__name__} statement')

    def infer_target(self, target, t, s):
        if isinstance(target, ast.Name):
            self.assign_type(s, target.id, t)
        elif isinstance(target, ast.Subscript) and T.is_pylist(self.array_of(target)) and (
                self.subscript_type(target).kind == 'array' or t is not self.array_of(target).elem):
            self.fail(s, 'replacing a row of a list of lists' if self.subscript_type(target).kind == 'array'
                      else f'stores {t.name} into a {self.array_of(target).name}, which would then hold {t.name}s too')
        elif isinstance(target, ast.Subscript) and self.array_of(target).kind == 'array' and self.subscript_type(target).kind == 'array':
            self.check_slice_store(s, self.subscript_type(target), t)
        elif isinstance(target, ast.Subscript):
            at = self.array_of(target)
            self.index_type(target)
            if at.kind == 'list':             # a list element takes the stored value's type
                e = T.join(at.elem, t) if t.is_number else None
                if e is None:
                    self.fail(s, f'stores {t.name} into a {at.name}')
                if target.value.id in self.params and e is not at.elem:
                    self.fail(s, f'stores {t.name} into a {at.name} passed in, which would then hold {t.name}s too')
                self.env[target.value.id] = T.list_of(e)
                return
            if not at.writable:
                self.fail(s, 'writes a read-only array')
            if not t.is_number or (at.elem.kind == 'int' and t.kind == 'float'):
                self.fail(s, f'stores {t.name} into a {at.name} array')
        elif isinstance(target, (ast.Tuple, ast.List)):
            if t.kind != 'tuple' or len(t.elem) != len(target.elts):
                self.fail(s, f'unpacks {t.name} into {len(target.elts)} targets')
            for x, et in zip(target.elts, t.elem):
                self.infer_target(x, et, s)
        else:
            self.fail(s, 'assignment to something other than a name or an array element')

    def check_test(self, test):
        t = self.typeof(test)
        if not t.is_number and t.kind != 'enum':
            self.fail(test, f'a condition of type {t.name}')

    def range_args(self, it):
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and self.is_builtin(it.func.id, 'range')):
            self.fail(it, 'loops over something other than range()')
        if it.keywords or not 1 <= len(it.args) <= 3:
            self.fail(it, 'range() with these arguments')
        for a in it.args:
            if self.typeof(a).kind not in ('int', 'bool'):
                self.fail(a, 'range() of a non-integer')

    # ---- iterables: range(), arrays, enumerate(), zip(), reversed(), nested. Each lowers to one
    # counted loop that reads elements by index; no iterator exists at run time.
    def iterable(self, it):
        if isinstance(it, ast.Name) and it.id in self.env and self.env[it.id].kind in ('array', 'list'):
            if self.env[it.id].kind == 'array' and self.env[it.id].ndim != 1:
                return ('rows', it.id, None)
            return ('array', it.id)
        if self.is_view(it) and self.typeof(it).kind == 'array':
            return ('rows' if self.typeof(it).ndim != 1 else 'array', self.view_name(it), it)
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name):
            name = it.func.id
            if self.is_builtin(name, 'range'):
                self.range_args(it)
                return ('range', it)
            if self.is_builtin(name, 'enumerate'):
                args = list(it.args)
                for k in it.keywords:
                    if k.arg != 'start':
                        self.fail(it, 'enumerate() with these arguments')
                    args.append(k.value)
                if not 1 <= len(args) <= 2:
                    self.fail(it, 'enumerate() with these arguments')
                if len(args) == 2 and self.typeof(args[1]).kind not in ('int', 'bool'):
                    self.fail(it, 'enumerate() with a non-integer start')
                return ('enumerate', self.iterable(args[0]), args[1] if len(args) == 2 else None)
            if self.is_builtin(name, 'zip'):
                if it.keywords or not it.args:
                    self.fail(it, 'zip() with these arguments')
                return ('zip', [self.iterable(a) for a in it.args])
            if self.is_builtin(name, 'reversed'):
                if it.keywords or len(it.args) != 1:
                    self.fail(it, 'reversed() with these arguments')
                inner = self.iterable(it.args[0])
                if inner[0] not in ('range', 'array') or (len(inner) > 2 and inner[2] is not None):
                    self.fail(it, 'reversed() of something other than range() or an array')
                return ('reversed', inner)
        if isinstance(it, ast.Tuple) or self.tuple_typed(it):
            t = self.typeof(it)
            r = t.elem[0]
            for et in t.elem[1:]:
                j = T.join(r, et) if not (r.kind == 'tuple') else (r if r is et else None)
                if j is None:
                    self.fail(it, f'loops over a tuple whose elements differ in type ({r.name}, {et.name})')
                r = j
            return ('tuple', it, r)
        self.fail(it, 'loops over something other than range(), an array, enumerate(), zip(), reversed() or a tuple')

    def no_rows(self, d, node):
        """Only for statements bind a view to a variable: a reduction or a comprehension over rows stays Python."""
        inner = d[1] if d[0] == 'zip' else [d[1]] if d[0] in ('enumerate', 'reversed') else []
        if d[0] == 'rows':
            self.fail(node, 'rows of an array in a reduction or a comprehension')
        for x in inner:
            self.no_rows(x, node)
        return d

    def elem_struct(self, d):
        kind = d[0]
        if kind == 'range':
            return T.INT
        if kind == 'rows':
            t = self.env[d[1]]
            return T.array(t.elem, t.writable, t.ndim - 1, t.strided)
        if kind == 'array':
            return self.env[d[1]].elem
        if kind == 'enumerate':
            return [T.INT, self.elem_struct(d[1])]
        if kind == 'zip':
            return [self.elem_struct(x) for x in d[1]]
        if kind == 'tuple':
            return d[2]
        return self.elem_struct(d[1])

    def has_views(self, e):
        return isinstance(e, ast.Tuple) and any(self.typeof(x).kind == 'array' for x in e.elts)

    def check_slice_store(self, s, dst, t):
        if not dst.writable:
            self.fail(s, 'writes a read-only array')
        if t.is_number:
            if dst.elem.kind == 'int' and t.kind == 'float':
                self.fail(s, f'stores {t.name} into a {dst.name} array')
            return
        if t.kind != 'array':
            self.fail(s, f'stores {t.name} into a slice')
        if t.elem is not dst.elem and not (t.elem is T.NP_INT64 and dst.elem is T.NP_FLOAT64):
            self.fail(s, f'stores {t.elem.name[3:]} elements into a {dst.name} array')
        if t.ndim > dst.ndim or dst.ndim > 2:
            self.fail(s, f'stores {t.name} into a {dst.name} array')

    def bind_types(self, target, struct, s):
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(struct, T.Type) and struct.kind == 'tuple':
            struct = list(struct.elem)           # a tuple element, unpacked
        if isinstance(target, ast.Name):
            if isinstance(struct, list):
                self.fail(s, 'a tuple assigned to one name (tuples are not compiled yet)')
            self.assign_type(s, target.id, struct)
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(struct, list) \
                and len(target.elts) == len(struct):
            for t, x in zip(target.elts, struct):
                self.bind_types(t, x, s)
        else:
            self.fail(s, 'a loop target that does not match what the loop yields')

    def is_view(self, e):
        """Could this expression be a view: something indexed, or .T of an array?"""
        return isinstance(e, ast.Subscript) or (isinstance(e, ast.Attribute) and e.attr == 'T' and (
            self.is_view(e.value) or (isinstance(e.value, ast.Name) and self.env.get(e.value.id) is not None
                                      and self.env[e.value.id].kind == 'array'))) or self.make_of(e) is not None or self.callee_of(e) is not None

    def transpose_type(self, e):
        t = self.typeof(e.value)
        if t.kind != 'array':
            self.fail(e, f'.T of {t.name}')
        v = T.array(t.elem, t.writable, t.ndim, t.strided or t.ndim > 1)
        self.env[self.view_name(e)] = v
        return v

    def array_of(self, sub):
        t = self.array_of_any(sub)
        if T.is_pylist(t):                    # a list of lists indexes one level at a time, without slices
            if isinstance(sub.slice, ast.Tuple):
                self.fail(sub, 'a list indexed with a tuple (TypeError in Python): write m[i][j]')
            if isinstance(sub.slice, ast.Slice):
                self.fail(sub, 'a slice of a list of lists (a copy in Python)')
        return t

    def array_of_any(self, sub):
        v = sub.value
        if self.is_view(v) and not isinstance(v, ast.Name):
            t = self.typeof(v)
            if t.kind != 'array':
                self.fail(sub, 'indexing something other than an array or a list')
            return t
        if not isinstance(v, ast.Name) or v.id not in self.env or self.env[v.id].kind not in ('array', 'list'):
            self.fail(sub, 'indexing something other than an array or a list')
        return self.env[v.id]

    def subscript_type(self, e):
        """What indexing an array gives: its element, with an int for every axis, else a view."""
        at = self.array_of(e)
        if at.kind == 'list':
            self.index_type(e)
            return at.elem
        items = e.slice.elts if isinstance(e.slice, ast.Tuple) else [e.slice]
        if len(items) > at.ndim:
            self.fail(e, f'{len(items)} indices into {at.name}')
        for k in items:
            if isinstance(k, ast.Slice):
                for part in (k.lower, k.upper, k.step):
                    if part is not None and not (self.typeof(part).kind == 'int') or part is not None and self.typeof(part) in (T.U64, T.NP_UINT64):
                        self.fail(e, 'a slice with bounds that are not integers' if self.typeof(part).kind != 'int' else 'a slice bound past int64')
            elif isinstance(k, ast.Constant) and (k.value is None or k.value is Ellipsis):
                self.fail(e, 'None or ... in an index')
            elif self.typeof(k).kind != 'int':
                self.fail(e, f'an index of type {self.typeof(k).name}')
        kept = at.ndim - sum(not isinstance(k, ast.Slice) for k in items)
        if kept and any(not isinstance(k, ast.Slice) and self.typeof(k) in (T.U64, T.NP_UINT64) for k in items):
            self.fail(e, 'a row of an array chosen by an index past int64')
        if not kept:
            return at.elem
        t = T.array(at.elem, at.writable, kept, at.strided or not self.keeps_contiguous(items))
        self.env[self.view_name(e)] = t
        return t

    @staticmethod
    def keeps_contiguous(items):
        """ints, then at most one slice of step 1, then only whole axes: the elements stay next to each other."""
        i = 0
        while i < len(items) and not isinstance(items[i], ast.Slice):
            i += 1
        if i == len(items):
            return True
        step = items[i].step
        if not (step is None or (isinstance(step, ast.Constant) and step.value == 1)):
            return False
        return all(isinstance(k, ast.Slice) and k.lower is None and k.upper is None and k.step is None for k in items[i + 1:])

    def view_name(self, e):
        name = self.view_names.get(id(e))
        if name is None:
            name = f'sjv__{len(self.view_names)}'
            self.view_names[id(e)] = name
        return name

    def tuple_typed(self, e):
        """Is e a tuple variable, or an element of one that is itself a tuple?"""
        if isinstance(e, ast.Name):
            return e.id in self.env and self.env[e.id].kind == 'tuple'
        return isinstance(e, ast.Subscript) and self.tuple_typed(e.value) and self.typeof(e).kind == 'tuple'

    def tuple_index(self, sub):
        t = self.typeof(sub.value)
        k = sub.slice
        neg = isinstance(k, ast.UnaryOp) and isinstance(k.op, ast.USub)
        if neg:
            k = k.operand
        if not (isinstance(k, ast.Constant) and type(k.value) is int):
            self.fail(sub, 'a tuple indexed with something other than a constant')
        i = -k.value if neg else k.value
        if not -len(t.elem) <= i < len(t.elem):
            self.fail(sub, 'a tuple index out of range')
        return i % len(t.elem)

    def indices(self, sub):
        t = self.array_of(sub)
        nd = t.ndim if t.kind == 'array' else 1
        idx = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        if len(idx) != nd:
            self.fail(sub, f'{len(idx)} indices into {t.name}, assigning to a view')
        return idx

    def index_type(self, sub):
        for k in self.indices(sub):
            if isinstance(k, ast.Slice):
                self.fail(sub, 'assigning to a slice')
            t = self.typeof(k)
            if t.kind != 'int':
                self.fail(sub, f'an index of type {t.name}')

    def binop_type(self, a, b, op, node):
        if a.kind == 'enum' or b.kind == 'enum':
            if not all(t.kind != 'enum' or T.is_int_enum(t) for t in (a, b)):
                self.fail(node, f'arithmetic on {(a if a.kind == "enum" else b).name}, an Enum member (write .value)')
            a, b = (T.INT if a.kind == 'enum' else a), (T.INT if b.kind == 'enum' else b)
        if not (a.is_number and b.is_number):
            self.fail(node, f'{op} on {a.name} and {b.name}')
        if T.NP_UINT64 in (a, b):
            r = T.arith_result(a, b, op)
            if r is None:
                self.fail(node, T.arith_problem(a, b, op))
            return r
        if a is T.U64 or b is T.U64:
            return self.u64_binop_type(a, b, op, node)
        if op == 'Pow':
            return self.pow_type(a, b, node)
        if op in ('BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift'):
            if a.kind == 'float' or b.kind == 'float':
                self.fail(node, 'a bitwise operator on a float')
            if a is T.BOOL and b is T.BOOL and op not in ('LShift', 'RShift'):
                return T.BOOL
        r = T.arith_result(a, b, op)
        if r is None:
            self.fail(node, T.arith_problem(a, b, op))
        return r

    def operands(self, node):
        if isinstance(node, ast.BinOp):
            return node.left, node.right
        return node.target, node.value          # AugAssign

    def u64_binop_type(self, a, b, op, node):
        """An int in [0, 2**64) with another number. What stays in that range is held unsigned; what may not wraps at 64
        bits, as every int does (+, -, *, <<, **), and a mask brings it back."""
        left, right = self.operands(node)
        if (a.np or b.np) and not (a.kind == 'float' or b.kind == 'float') or a.np is True or b.np is True or 'dyn' in (a.np, b.np):
            self.fail(node, 'an int that may be past int64 with numpy scalars (numpy raises OverflowError)')
        if a.kind == 'float' or b.kind == 'float':
            if op in ('BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift'):
                self.fail(node, 'a bitwise operator on a float')
            if op == 'Pow':
                self.fail(node, '** of an int past int64 and a float')
            return T.FLOAT
        if op == 'BitAnd':                           # masked by a constant that fits int64, or a length: an ordinary int
            def small(x):
                k = const_int(x)
                return (k is not None and 0 <= k <= INT64_MAX) or (isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                                                                  and x.func.id == 'len' and self.is_builtin('len', 'len'))
            return T.INT if small(left) or small(right) else T.U64
        if op in ('BitOr', 'BitXor'):
            return T.U64 if self.nonneg(left) and self.nonneg(right) else T.INT
        if op == 'RShift':                           # by 1 or more, it fits int64
            k = const_int(right)
            return T.INT if a is not T.U64 or (k is not None and k >= 1) else T.U64
        if op in ('FloorDiv', 'Mod'):
            if not (self.nonneg(left) and self.nonneg(right)):
                self.fail(node, f'{"//" if op == "FloorDiv" else "%"} of an int past int64 and an int that may be negative')
            return T.U64 if (op == 'FloorDiv' and a is T.U64) or (op == 'Mod' and b is T.U64 and a is T.U64) else T.INT
        if op == 'Div':
            self.fail(node, '/ of an int past int64')
        if op == 'Pow' and b is T.U64:
            self.fail(node, '** with an exponent past int64')
        if op in ('Add', 'Mult', 'LShift', 'Pow') and self.nonneg(left) and self.nonneg(right):
            return T.U64                             # exact up to 2**64, where it wraps
        return T.INT

    def nonneg(self, e, assume=frozenset()):
        """Is e's Python value never negative? Read from the source: names are followed to every assignment, assuming
        what's being shown (a loop that only xors and masks keeps a nonnegative value). A parameter counts if it's
        checked when the function is called (u64_params): a negative one then runs as Python."""
        if isinstance(e, ast.Constant):
            return isinstance(e.value, (int, bool)) and e.value >= 0
        if folds(e) is not None:
            return folds(e) >= 0
        if isinstance(e, ast.Name):
            if e.id in assume or self.env.get(e.id) is T.U64:
                return True
            return self.name_nonneg(e.id, assume | {e.id})
        if isinstance(e, ast.BinOp):
            op = type(e.op)
            if op is ast.BitAnd:
                return self.nonneg(e.left, assume) or self.nonneg(e.right, assume)
            if op is ast.Mod:
                return self.nonneg(e.right, assume)
            if op in (ast.RShift,):
                return self.nonneg(e.left, assume)
            if op in (ast.BitOr, ast.BitXor, ast.FloorDiv, ast.Add, ast.Mult, ast.LShift, ast.Pow):
                return self.nonneg(e.left, assume) and self.nonneg(e.right, assume)
            return False
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.UAdd):
            return self.nonneg(e.operand, assume)
        if isinstance(e, ast.IfExp):
            return self.nonneg(e.body, assume) and self.nonneg(e.orelse, assume)
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and len(e.args) == 1 and not e.keywords:
            if e.func.id == 'len' and self.is_builtin('len', 'len'):
                return True
            if e.func.id == 'int' and self.is_builtin('int', 'int'):
                a = e.args[0]
                t = self.typeof(a)
                return t is T.U64 or (T.is_narrow(t) and t.ctype.startswith('uint')) or (t.kind != 'float' and self.nonneg(a, assume))
        return False

    def name_nonneg(self, name, assume):
        stores = []
        for n in ast.walk(self.f):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.With, ast.NamedExpr)):
                targets = n.targets if isinstance(n, ast.Assign) else [getattr(n, 'target', None)] if not isinstance(n, ast.With) \
                    else [i.optional_vars for i in n.items]
                for tg in targets:
                    if tg is None:
                        continue
                    if any(isinstance(x, ast.Name) and x.id == name and isinstance(x.ctx, ast.Store) for x in ast.walk(tg)):
                        if not isinstance(tg, ast.Name):
                            return False
                        stores.append(n)
        for n in stores:
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                if n.value is None or not self.nonneg(n.value, assume):
                    return False
            elif isinstance(n, ast.AugAssign):
                if not self.nonneg(ast.BinOp(ast.Name(name, ast.Load()), n.op, n.value), assume):
                    return False
            elif isinstance(n, (ast.For, ast.comprehension)):
                it = n.iter
                if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'range' and 1 <= len(it.args) <= 3
                        and (len(it.args) == 1 or self.nonneg(it.args[0], assume))
                        and (len(it.args) < 3 or (isinstance(it.args[2], ast.Constant) and type(it.args[2].value) is int and it.args[2].value > 0))):
                    return False
            else:
                return False
        if name in self.params:
            if self.came_as(name) not in (T.INT, T.BOOL):
                return False
            self.u64_params.add(name)
            return True
        return bool(stores)

    def pow_type(self, a, b, node):
        """a ** b. numpy's rules give one type whatever the values. Python's give a float for an int to
        a negative power, and a complex for a negative float to a fractional power. Compiled code has one
        type per expression, so it assumes neither happens and raises if one does: a documented
        difference from Python. A constant exponent or base that rules it out has no such case."""
        r = T.arith_result(a, b, 'Pow')
        if r is None:
            self.fail(node, T.arith_problem(a, b, 'Pow'))
        if r.np is True:
            return r
        exp = const_number(node.right if isinstance(node, ast.BinOp) else node.value)
        if r.kind == 'int':
            if exp is None:
                self.gotcha(node, 'runs as Python if the exponent is negative' if self.writes_nothing()
                            else 'raises ValueError if the exponent is negative, where Python gives a float')
                return r
            if exp >= 0:
                return r
            if r.np:
                self.fail(node, 'int ** a negative int on a value that may be a numpy int')
            return T.FLOAT
        base = const_number(node.left) if isinstance(node, ast.BinOp) else None
        integral = b.kind != 'float' or (exp is not None and (exp != exp or abs(exp) == math.inf or exp == math.floor(exp)))
        if self.exact and not integral and not (base is not None and base >= 0):
            self.gotcha(node, 'runs as Python if the base is negative' if self.writes_nothing()
                        else 'raises ValueError if the base is negative, where Python gives a complex')
        return r

    def writes_nothing(self):
        """Does this function leave no trace a rerun from the start could double: no stores into arrays, lists or dicts,
        no calls to compiled functions or C? Then where it would differ from Python it can ask for the call to run as
        Python instead (RERUN, through sj_fail, which frees what it made). Not a callee: its caller has done things."""
        if getattr(self, 'cname', 'sj_kernel') != 'sj_kernel' or self.hidden_map or self.counts:
            return False
        for n in ast.walk(self.f):
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
                return False
            if isinstance(n, ast.Call) and (self.callee_of(n) is not None or self.c_function_of(n) is not None):
                return False
        return True

    def differs(self, line, message):
        """The C for where this function's result would differ from Python's (VALUE_ERROR_MESSAGES[message]): run the
        call as Python if nothing can tell, else raise."""
        if self.writes_nothing():
            self.uses_rerun = True
            return f'SJ_FAIL({RERUN}, {line}, 0, 0);'
        return f'SJ_FAIL({VALUE_ERROR}, {line}, {message}, 0);'

    def enum_compare_type(self, e, ts):
        """Comparisons with an enum member, as Python makes them: == and != are identity for an Enum (another type is
        never equal), is and is not between members, and an IntEnum orders as an int."""
        for op, a, b in zip(e.ops, ts, ts[1:]):
            ints = all(T.is_int_enum(t) or (t.kind in ('int', 'bool') and t.np is False and t is not T.U64) for t in (a, b))
            if isinstance(op, (ast.Is, ast.IsNot)):
                if not (a.kind == 'enum' and b.kind == 'enum'):
                    self.fail(e, f'is between {a.name} and {b.name}')
            elif isinstance(op, (ast.Eq, ast.NotEq)):
                if not (a.kind == 'enum' and b.kind == 'enum' or ints or (a.kind == 'enum') != (b.kind == 'enum') and (a.is_number or b.is_number)):
                    self.fail(e, f'comparing {a.name} with {b.name}')
                if (a.np or b.np) and not ints:
                    self.fail(e, f'comparing {a.name} with {b.name}')
            elif type(op) in CMP:
                if not ints:
                    self.fail(e, f'ordering {a.name} and {b.name}')
            else:
                self.fail(e, 'in and not in')
        return T.BOOL

    def enum_attribute(self, e):
        """The type of Level.HIGH (a member) or member.value (its int), or None for another attribute."""
        if isinstance(e.value, ast.Name) and self.env.get(e.value.id) is not None and self.env[e.value.id].kind == 'enumclass':
            cls = self.env[e.value.id].elem
            if e.attr not in cls.__members__:
                self.fail(e, f'{cls.__name__}.{e.attr}, which isn\'t a member')
            return T.enum_member_of(cls)
        if e.attr in ('value', '_value_') and isinstance(e.value, (ast.Name, ast.Attribute)):
            try:
                t = self.typeof(e.value)
            except Unsupported:
                return None
            if t.kind == 'enum':
                return T.INT
        return None

    def gotcha(self, node, text):
        """Note that this line can differ from Python, for preflight."""
        line, have = self.py_line(node), self.gotchas.get(self.py_line(node))
        self.gotchas[line] = text if have is None or text in have else f'{have}; {text}'

    def join_all(self, node, ts, what):
        r = ts[0]
        for t in ts[1:]:
            j = T.join(r, t)
            if j is None:
                self.fail(node, f'{what} of values of different types (the result type would vary)')
            r = j
        return r

    def local_object(self, e):
        """The type an instance made in the function goes as: its attributes' types as they are here, and its methods;
        an object it holds, as a made one's attributes are here, or as a parameter's object is."""
        held = getattr(e, 'held', None) or {}

        def build(prefix, t):
            fields = []
            for a in dict.fromkeys(k[len(prefix):].split('.')[0] for k in e.slots if k.startswith(prefix)):
                full = prefix + a
                if full in e.slots:
                    fields.append((a, self.typeof(e.slots[full])))
                elif held[full].kind == 'object':
                    fields.append((a, held[full]))
                else:
                    fields.append((a, build(full + '.', held[full])))
            fields += [kt for kt in (t.recipe['methods'] if t.kind == 'madeclass' else t.elem) if kt[1].kind == 'function']
            return T.object_of(t.py, sorted(fields, key=lambda kv: kv[0]), t.methods, t.cls, t.writes)
        return build('', e.local)

    def typeof(self, e):
        if isinstance(e, RefArg):
            return T.poolref_of(e.pool, e.cls, getattr(e, 'lists', True))
        if isinstance(e, SelfArg) and getattr(e, 'same', None) is not None:
            return T.same_as(e.same)
        if isinstance(e, SelfArg) and getattr(e, 'ref', False):
            return T.elemref_of(e.object_type)
        if isinstance(e, SelfArg) and getattr(e, 'local', None) is not None:
            for x in e.elts:
                if self.typeof(x).np == 'dyn':
                    self.fail(e, f'{self.shown(x.id)} is a Python number at some times and a numpy one at others')
            e.object_type = self.local_object(e)
            e.elts = [e.slots[k] for k, _ in leaf_types(e.object_type)]
            return e.object_type
        if isinstance(e, SelfArg):
            for x, (k, ft) in zip(e.elts, leaf_types(e.object_type)):
                if self.typeof(x) is not ft:
                    self.fail(e, f'{self.shown(x.id)} is {self.typeof(x).name} here, not {ft.name} as at the call')
            return e.object_type
        if isinstance(e, ast.Constant):
            v = e.value
            if isinstance(v, bool):
                return T.BOOL
            if isinstance(v, int):
                if INT64_MAX < v < 1 << 64:
                    return T.U64
                if not INT64_MIN <= v <= INT64_MAX:
                    self.fail(e, 'an int literal outside int64 and uint64')
                return T.INT
            if isinstance(v, float):
                return T.FLOAT
            self.fail(e, f'a {type(v).__name__} constant')
        if isinstance(e, ast.Name):
            if e.id in self.env:
                if self.env[e.id].kind == 'list':
                    self.fail(e, f'{e.id} is a list: lists can only be indexed, iterated over and passed to '
                              'len() (no aliases, returns or other uses yet)')
                return self.env[e.id]
            self.fail(e, f'uses the global or undefined name {e.id!r}')
        if isinstance(e, ast.Attribute):
            if e.attr == 'T' and self.is_view(e):
                return self.transpose_type(e)
            enum_type = self.enum_attribute(e)
            if enum_type is not None:
                return enum_type
            if isinstance(e.value, ast.Name) and e.value.id not in self.env \
                    and ((self.globals.get(e.value.id) is math and e.attr in MATH_CONSTANTS)
                         or (is_numpy(self.globals.get(e.value.id)) and e.attr in NP_CONSTANTS)):
                return T.FLOAT
            if isinstance(e.value, ast.Name) and e.value.id in self.env and self.env[e.value.id].kind == 'array':
                t = self.env[e.value.id]
                if T.is_pylist(t):
                    self.fail(e, f'.{e.attr} of a list (AttributeError in Python)')
                if e.attr == 'shape':
                    return T.tuple_of([T.INT] * t.ndim)
                if e.attr in ('ndim', 'size'):
                    return T.INT
            self.fail(e, f'the attribute .{e.attr}')
        if isinstance(e, ast.BinOp) and folds(e) is not None:
            return T.U64 if folds(e) > INT64_MAX else T.INT
        if isinstance(e, ast.BinOp):
            op = ARITH.get(type(e.op))
            if op is None:
                self.fail(e, OPERATOR_NAMES.get(type(e.op).__name__, f'the {type(e.op).__name__} operator'))
            return self.binop_type(self.typeof(e.left), self.typeof(e.right), op, e)
        if isinstance(e, ast.UnaryOp):
            t = self.typeof(e.operand)
            if t.kind == 'enum' and isinstance(e.op, ast.Not):
                return T.BOOL
            if T.is_int_enum(t):                         # an IntEnum member is an int
                return T.INT
            if not t.is_number:
                self.fail(e, f'a unary operator on {t.name}')
            if isinstance(e.op, ast.Not):
                return T.BOOL
            if t.np and t.kind == 'bool':
                self.fail(e, 'arithmetic on numpy bools')
            if isinstance(e.op, ast.Invert) and t.kind != 'int':
                self.fail(e, '~ on a non-int')
            if T.is_narrow_dyn(t):
                self.fail(e, f'a unary operator on {t.name}: its result may not fit')
            if t is T.U64 and not isinstance(e.op, ast.UAdd):
                return T.INT                         # wraps
            if t is T.NP_UINT64:                     # wraps, as numpy's does
                return t
            return T.INT if t is T.BOOL else t
        if isinstance(e, ast.Compare):
            ts = [self.typeof(x) for x in [e.left] + e.comparators]
            if any(t.kind == 'enum' for t in ts):
                return self.enum_compare_type(e, ts)
            if not all(type(o) in CMP for o in e.ops):
                self.fail(e, 'in, not in, is and is not')
            if not all(t.is_number for t in ts):
                self.fail(e, 'comparisons of non-numbers')
            for a, b in zip(ts, ts[1:]):
                if T.U64 in (a, b) and ((a.np and a.kind == 'float' and a is not T.NP_FLOAT64) or (b.np and b.kind == 'float' and b is not T.NP_FLOAT64)
                                        or 'dyn' in (a.np, b.np)):
                    self.fail(e, f'comparing an int past int64 with {(b if a is T.U64 else a).name}')
                if (T.is_narrow_dyn(a) or T.is_narrow_dyn(b)) and a.np is not True and b.np is not True \
                        and not (a.kind != 'float' and b.kind != 'float'):
                    self.fail(e, f'comparing {a.name} with {b.name}: numpy rounds one side and Python doesn\'t')
            return T.scalar('bool', T.join_np(*[T.compare_result(a, b).np for a, b in zip(ts, ts[1:])])
                            if len(ts) == 2 or len({T.compare_result(a, b) for a, b in zip(ts, ts[1:])}) == 1
                            else 'dyn')
        if isinstance(e, ast.BoolOp):
            t = self.join_all(e, [self.typeof(v) for v in e.values], 'and/or')
            if not t.is_number:
                self.fail(e, f'and/or on {t.name}')
            return t
        if isinstance(e, ast.IfExp):
            self.check_test(e.test)
            t = self.join_all(e, [self.typeof(e.body), self.typeof(e.orelse)], 'a conditional expression')
            if not t.is_number and t.kind != 'enum':
                self.fail(e, f'a conditional expression of type {t.name}')
            return t
        if isinstance(e, ast.Subscript):
            if self.is_shape(e.value):
                if self.shape_index(e) is None:
                    self.fail(e, 'a.shape indexed with something other than a constant in range')
                return T.INT
            if self.tuple_typed(e.value):
                return self.typeof(e.value).elem[self.tuple_index(e)]
            return self.subscript_type(e)
        if isinstance(e, ast.Tuple):
            ts = [self.typeof(x) for x in e.elts]
            if not all(T.is_number_tree(t) for t in ts):
                self.fail(e, 'tuples of non-numbers (arrays, lists)')
            return T.tuple_of(ts)
        if isinstance(e, ast.Call):
            made = self.make_of(e)
            return self.make_type(e, made) if made else self.call_type(e)
        if isinstance(e, (ast.GeneratorExp, ast.ListComp)):
            self.fail(e, 'a generator or list comprehension outside sum(), min(), max(), any() or all()')
        self.fail(e, UNSUPPORTED_EXPRESSIONS.get(type(e).__name__, f'the {type(e).__name__} expression'))

    def is_reduction(self, e):
        name = e.func.id
        return name in ('sum', 'any', 'all') or (name in ('min', 'max') and len(e.args) == 1)

    def math_func(self, func):
        """The math function a call refers to (math.sin, or sin from `from math import sin`)."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id not in self.env and self.globals.get(func.value.id) is math:
            # (a function this Python's math lacks isn't compiled: running it raises AttributeError)
            return func.attr if func.attr in MATH_FUNCS and hasattr(math, func.attr) else self.fail(func, f'math.{func.attr}()')
        if isinstance(func, ast.Name) and func.id not in self.env and func.id in self.globals:
            obj = self.globals[func.id]
            for n in MATH_FUNCS:
                if obj is getattr(math, n, None):
                    return n
        return None

    def math_type(self, e, name):
        r = self.math_type_of(e, name)
        for a in e.args:
            try:
                t = self.typeof(a)
            except Unsupported:                       # a sequence (fsum of a generator): math_type_of took it
                continue
            if t in (T.U64, T.NP_UINT64):
                self.fail(e, f'math.{name}() of an int past int64')
        return r

    def math_type_of(self, e, name):
        kind, _ = MATH_FUNCS[name]
        if kind == 'iter':
            return self.math_iter_type(e, name)
        allowed = {'isclose': {'rel_tol', 'abs_tol'}, 'nextafter': {'steps'}}.get(kind, set())
        if {k.arg for k in e.keywords} - allowed or any(k.arg is None for k in e.keywords):
            self.fail(e, f'math.{name}() with these keyword arguments')
        for k in e.keywords:
            t = self.typeof(k.value)
            if not t.is_number or (k.arg == 'steps' and t.kind == 'float'):
                self.fail(e, f'math.{name}() with this {k.arg}')
        ts = [self.typeof(a) for a in e.args]
        if kind == 'intf':
            n = len(ts)
            if not all(t.kind in ('int', 'bool') and not (t.np and t.kind == 'bool') for t in ts) \
                    or (name in ('isqrt', 'factorial') and n != 1) or (name == 'comb' and n != 2) \
                    or (name == 'perm' and n not in (1, 2)):
                self.fail(e, f'math.{name}() of these arguments')
            return T.INT
        if not all(t.is_number for t in ts) or any(t.np and t.kind == 'bool' for t in ts):
            self.fail(e, f'math.{name}() of these arguments')
        n = len(ts)
        if (kind in ('f1', '1a', 'round', 'scale', 'test') and n != 1) or (kind == 'log' and not 1 <= n <= (2 if name == 'log' else 1)) \
                or (kind in ('atan2', 'pow', 'fmod', 'copysign', 'remainder', 'nextafter', 'isclose', 'ldexp') and n != 2) \
                or (kind in ('gamma', 'ulp', 'frexp', 'modf') and n != 1) or (kind == 'fma' and n != 3):
            self.fail(e, f'math.{name}() with {n} arguments')
        if kind == 'ldexp' and ts[1].kind == 'float':
            self.fail(e, 'math.ldexp() with a float exponent (it raises TypeError)')
        if kind == 'frexp':
            return T.tuple_of([T.FLOAT, T.INT])
        if kind == 'modf':
            return T.tuple_of([T.FLOAT, T.FLOAT])
        if kind == 'isclose':
            return T.BOOL
        if name == 'trunc' and ts[0].np and ts[0].kind != 'float':
            self.fail(e, 'math.trunc() of a numpy int (it raises TypeError)')
        return T.INT if kind == 'round' else T.BOOL if kind == 'test' else T.FLOAT

    def seq_elem_type(self, src):
        """The element type of a sequence math takes: a tuple, or what reductions loop over."""
        if isinstance(src, ast.List) or self.is_tuple_expr(src):
            ts = [self.typeof(x) for x in src.elts] if isinstance(src, ast.List) else self.typeof(src).elem
            if not ts or not all(t.is_number for t in ts):
                self.fail(src, 'this sequence')
            if len({t.name for t in ts}) == 1:
                return ts[0]
            if any(t.np for t in ts):
                return self.join_all(src, ts, 'this sequence')
            return T.FLOAT if any(t.kind == 'float' for t in ts) else T.INT
        return self.source_elem_type(src)

    def is_tuple_expr(self, src):
        return isinstance(src, ast.Tuple) or (isinstance(src, ast.Name) and src.id in self.env
                                              and self.env[src.id].kind == 'tuple')

    def seq_loops(self, src, after_first, item):
        if isinstance(src, ast.List) or self.is_tuple_expr(src):
            if isinstance(src, ast.List):              # computed in full, as Python builds it
                ts = [self.typeof(x) for x in src.elts]
                vals = [(self.hold(c, t), self.hold_flag(f)) for (c, f), t in zip((self.expr(x) for x in src.elts), ts)]
            else:
                ts, vals = self.typeof(src).elem, self.texpr(src)
            after_first()
            for v, t in zip(vals, ts):
                item(v, t)
            return
        self.source_loops(src, after_first, item)

    def math_iter_type(self, e, name):
        kws = {k.arg: k.value for k in e.keywords}
        if name == 'fsum':
            if kws or len(e.args) != 1:
                self.fail(e, 'math.fsum() with these arguments')
            el = self.seq_elem_type(e.args[0])
            if not el.is_number:
                self.fail(e, 'math.fsum() of non-numbers')
            return T.FLOAT
        if name == 'prod':
            if set(kws) - {'start'} or len(e.args) != 1:
                self.fail(e, 'math.prod() with these arguments')
            el = self.seq_elem_type(e.args[0])
            st = self.typeof(kws['start']) if 'start' in kws else T.INT
            if not el.is_number or el.kind == 'bool' or not st.is_number or st.kind == 'bool' or st.np \
                    or el.ctype not in ('double', 'int64_t'):
                self.fail(e, 'math.prod() of these values')
            if st.kind == 'int':
                if el.kind == 'float':
                    self.fail(e, 'math.prod() of floats returns the int 1 when empty, so its type depends '
                              'on the data; write math.prod(..., start=1.0)')
                return T.INT if el.np is False else T.INT_DYN
            return T.FLOAT if el.np is False else T.FLOAT_DYN
        # dist and sumprod: two named sequences
        if kws or len(e.args) != 2 or not all(isinstance(a, (ast.Name, ast.List)) or self.is_tuple_expr(a) for a in e.args):
            self.fail(e, f'math.{name}() of anything but two named sequences or tuples')
        els = [self.seq_elem_type(a) for a in e.args]
        if not all(t.is_number for t in els):
            self.fail(e, f'math.{name}() of non-numbers')
        if name == 'dist':
            return T.FLOAT
        if not all(t.kind == 'int' and t.ctype == 'int64_t' for t in els):
            self.fail(e, 'math.sumprod() of floats returns the int 0 when empty, so its type depends on the data')
        return T.INT if not any(t.np for t in els) else T.INT_DYN

    # ---- calls to other @scram functions: a static C function per function and signature
    def callee_of(self, e):
        """The function this calls, if it's a @scram function at module level."""
        if isinstance(e, ast.Call) and getattr(e, 'sj_callee', None) is not None:
            return CALLEES[e.sj_callee]
        if not isinstance(e, ast.Call) or not isinstance(e.func, ast.Name) or e.func.id in self.cnames:
            return None
        t = self.env.get(e.func.id)
        f = (t.elem if t.kind == 'function' else None) if t is not None else self.globals.get(e.func.id)
        return f if hasattr(type(f), '_callee') or (f is not None and type(f).__name__ == 'StaticCallee') else None

    def calls_out(self, e):
        """Does this call run code that can write arrays it was given?"""
        return self.callee_of(e) is not None or self.c_function_of(e) is not None

    # ---- calls to C: k = cimo.compile('...'), then k.f(x). Its source goes in this file, so it can be inlined.
    def global_object(self, name):
        t = self.env.get(name)
        if t is not None:
            return t.elem if t.kind == 'function' else None
        return self.globals.get(name)

    def c_function_of(self, e):
        """For a call to a C function: (its module, its name)."""
        if not isinstance(e, ast.Call):
            return None
        f = e.func
        if isinstance(f, ast.Name) and f.id not in self.cnames:
            x = self.global_object(f.id)
            if T.is_c_function(x):
                return (x.module, x.__name__) if hasattr(x, 'module') else (None, x.__name__)
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id not in self.cnames:
            x = self.global_object(f.value.id)
            if T.is_c_module(x):
                return x, f.attr
        return None

    def c_signature(self, e):
        """(return type, parameter types), checked: only what comes out as it does when Python makes this call."""
        m, name = self.c_function_of(e)
        label = f'{name}() in C'
        if m is None or not hasattr(m, 'signature'):
            self.fail(e, f'{label}: this cimo is too old to say what its functions take')
        bad = [k for k in (m.options or {}) if k in ('include_dirs', 'link', 'host_headers')]
        if bad:
            self.fail(e, f'{label}: a module compiled with {bad[0]}=')
        if m.mutable_data:
            self.fail(e, f'{label}: its module has data that isn\'t const ({", ".join(m.mutable_data[:3])}), and a compiled copy wouldn\'t share it')
        try:
            self.unit.add_module(m)
            ret, params = m.signature(name)
        except KeyError:
            self.fail(e, f'{label}: no such function')
        except Unsupported as u:
            self.fail(e, f'{label}: {u.reason}')
        if e.keywords or len(e.args) != len(params) or '...' in params or any(isinstance(a, ast.Starred) for a in e.args):
            self.fail(e, f'{label} with these arguments')
        for a, p in zip(e.args, params):
            self.check_c_arg(e, a, p, label)
        if ret == 'void':
            return T.NONE, params
        if ret in ('double', 'float'):
            return T.FLOAT, params
        if ret in ('_Bool', 'int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int32_t', 'uint32_t', 'int64_t'):
            return T.INT, params
        self.fail(e, f'{label}, which returns {ret}')

    def check_c_arg(self, e, a, p, label):
        t = self.typeof(a)
        if p.endswith('*'):
            base = p[:-1].strip()
            const = base.startswith('const ')
            base = base[6:] if const else base
            if t.kind != 'array':
                self.fail(e, f'{label}: passes {t.name} as {p}')
            if t.strided:
                self.fail(e, f'{label}: passes a view with gaps, and C sees only where it starts')
            if base not in ('void', t.elem.ctype):
                self.fail(e, f'{label}: passes {t.name} as {p}')
            if not const and not t.writable:
                self.fail(e, f'{label}: passes a read-only array as {p}')
            return
        if not t.is_number:
            self.fail(e, f'{label}: passes {t.name} as {p}')
        if p in ('double', 'float'):
            return
        if p == '_Bool':                   # Python's call doesn't make it 0 or 1
            if t is not T.BOOL:
                self.fail(e, f'{label}: passes {t.name} as _Bool')
        elif p not in ('int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t'):
            self.fail(e, f'{label}: a parameter of type {p}')
        elif t.kind == 'float' or t.np == 'dyn' and t.kind != 'int' or t is T.NP_BOOL:
            self.fail(e, f'{label}: passes {t.name} as {p}, which Python raises for')
        elif t.np and p in ('int64_t', 'uint64_t'):
            self.fail(e, f'{label}: passes {t.name} as {p}, and cimo takes numpy\'s integers there as an address: use int(...)')

    def c_call(self, e):
        """Evaluate the arguments in order, and return the call as C."""
        _, name = self.c_function_of(e)
        _, params = self.c_signature(e)
        cargs = []
        for a, p in zip(e.args, params):
            t = self.typeof(a)
            if t.kind == 'array':
                cargs.append(f'(void *)v_{self.materialize(a)}')
            elif p == 'float':
                cargs.append(self.hold(f'(float)(double)({self.expr(a)[0]})', T.NARROW['float32']))
            elif p == 'double':
                cargs.append(self.hold(f'(double)({self.expr(a)[0]})', T.FLOAT))
            else:
                cargs.append(self.hold(f'(int64_t)({self.expr(a)[0]})', T.INT))
        return f'{name}({", ".join(cargs)})'

    def c_number(self, e):
        ret, _ = self.c_signature(e)
        return self.hold(self.c_call(e), ret), '0'

    def call_args(self, e, spec, fn=None):
        """The arguments in the order Python evaluates them, and for each parameter which of those it gets. A
        parameter left out gets its default, which must be a number: it was evaluated when the function was defined."""
        a = spec.tree.args
        names = [p.arg for p in a.posonlyargs + a.args][:len(a.posonlyargs + a.args) - len(spec.hidden)]
        npos = spec.npos if spec.npos is not None else len(names)
        if any(isinstance(x, ast.Starred) for x in e.args) or any(k.arg is None for k in e.keywords):
            self.fail(e, f'calls {spec.tree.name}() with * or **')
        if len(e.args) > npos:
            self.fail(e, f'calls {spec.tree.name}() with more arguments by position than it takes')
        nodes = list(e.args) + [k.value for k in e.keywords]
        order = dict(zip(names, range(len(e.args))))
        for i, k in enumerate(e.keywords):
            if k.arg not in names or k.arg in order:
                self.fail(e, f'calls {spec.tree.name}() with these arguments')
            order[k.arg] = len(e.args) + i
        defaults = self.callee_defaults(fn, spec)[:npos] if len(e.args) <= len(names) else ()
        kwdefaults = (getattr(getattr(fn, '_fn', None), '__kwdefaults__', None) or {})
        for i, n in enumerate(names):
            if n not in order and i >= npos and n in kwdefaults:
                v = kwdefaults[n]
                if type(v) not in (int, float, bool):
                    self.fail(e, f'calls {spec.tree.name}() leaving out {n}, whose default is a {type(v).__name__}')
                nodes.append(ast.copy_location(ast.Constant(v), e))
                order[n] = len(nodes) - 1
            elif n not in order and npos - len(defaults) <= i < npos:
                v = defaults[i - (npos - len(defaults))]
                if type(v) not in (int, float, bool):
                    self.fail(e, f'calls {spec.tree.name}() leaving out {n}, whose default is a {type(v).__name__}')
                nodes.append(ast.copy_location(ast.Constant(v), e))
                order[n] = len(nodes) - 1
        if len(order) != len(names) or len(e.args) > len(names):
            self.fail(e, f'calls {spec.tree.name}() with these arguments')
        return nodes, [order[n] for n in names]

    @staticmethod
    def callee_defaults(fn, spec):
        """A callee's default values: the function's own, or, read from source (preflight), its literals."""
        live = getattr(fn, '_fn', None)
        if live is not None:
            return live.__defaults__ or ()
        out = []
        for d in spec.tree.args.defaults:
            try:
                out.append(ast.literal_eval(d))
            except ValueError:
                out.append(d)                    # not a literal: reported as what it is
        return tuple(out)

    def compiled_callee(self, e):
        """The Compiler of what this call calls, for these argument types."""
        fn = self.callee_of(e)
        try:
            spec = fn._callee()
        except Unsupported as u:
            self.fail(e, f'calls {e.func.id}(): {u.reason}')
        nodes, picked = self.call_args(e, spec, fn)
        self.narrow_passed(nodes, picked, fn)
        ts = [self.typeof(x) for x in nodes]
        if hasattr(fn, '_add_method_globals') and fn._add_method_globals([ts[k] for k in picked]):
            spec = fn._callee()                   # its instances' methods read globals: they're its globals too
            nodes, picked = self.call_args(e, spec, fn)
            self.narrow_passed(nodes, picked, fn)
            ts = [self.typeof(x) for x in nodes]
        for x, t in zip(nodes, ts):
            if T.is_pylist(t):
                self.fail(e, f'passes a list of lists to {e.func.id}()')
            if t.kind == 'object' and any(i is not None for i, _ in same_targets(t)):
                self.fail(e, f'passes {e.func.id}() an object holding another argument\'s object')
            if not (t.is_number or t.kind in ('array', 'enum') or (t.kind in ('object', 'same', 'elemref') and isinstance(x, SelfArg))
                    or (t.kind == 'poolref' and isinstance(x, RefArg))
                    or (t.kind == 'tuple' and T.is_number_tree(t) and all(et.np != 'dyn' for et in T.leaves(t)))):
                self.fail(e, f'passes {t.name} to {e.func.id}()')
        types = [ts[k] for k in picked]
        for h in spec.hidden:
            if h[1] == 'sjpool':                 # its references' pool: not made for a call from compiled code, yet
                types.append(T.NOPOOL)
                continue
            p = self.hidden_param(h)
            if p not in self.env:
                self.fail(e, f'calls {e.func.id}(), which reads the global {h[1]}')
            types.append(self.env[p])
        try:
            return self.unit.callee(spec, types)
        except Unsupported as u:
            self.fail(e, f'calls {e.func.id}(): {u.reason}')

    def hidden_param(self, h):
        return self.hidden_map[h[0], h[1]] if self.hidden_map else h[1]

    def callee_type(self, e):
        if any(self.typeof(a) is T.U64 for a in e.args) or any(self.typeof(k.value) is T.U64 for k in e.keywords):
            self.fail(e, 'passes an int past int64 to a compiled function')
        c = self.compiled_callee(e)
        if c.makes and ret_arrays(c.ret):
            self.makes = True
        if c.ret.kind == 'array':
            self.env[self.view_name(e)] = c.ret
        elif c.ret.kind == 'tuple':
            for k, et in enumerate(c.ret.elem):
                if et.kind == 'array':
                    self.env[self.result_name(e, k)] = et
        return c.ret

    def result_name(self, e, k):
        name = self.view_names.get((id(e), k))
        if name is None:
            name = f'sjv__{len(self.view_names)}'
            self.view_names[(id(e), k)] = name
        return name

    def emit_call(self, e):
        """Evaluate the arguments in order and make the call. Returns its Compiler, the temporary holding its sj_io, and the
        names of the arrays it was passed, in the order its array parameters come."""
        c = self.compiled_callee(e)
        if not c.lines:
            source = c.generate()
            self.unit.code.append(source)
        self.calls = True
        self.uses_libm = self.uses_libm or c.uses_libm
        self.uses_stop = self.uses_stop or c.uses_stop
        spec = c.spec
        nodes, picked = self.call_args(e, spec, self.callee_of(e))
        self.narrow_passed(nodes, picked, self.callee_of(e))
        done = []
        mes = {j: nodes[k] for j, k in enumerate(picked) if isinstance(nodes[k], SelfArg) and nodes[k].same is None and not nodes[k].ref}
        self.check_shared(e, mes, c)

        def value(x):
            t = self.typeof(x)
            if t.kind == 'array':
                return t, self.materialize(x)
            if t.kind == 'tuple':
                return t, self.texpr(x)
            v = self.expr(x)
            return t, (self.hold(v[0], t), self.hold_flag(v[1]))
        for x in nodes:
            if isinstance(x, RefArg):              # a reference: the pool's lists if it's the first, then its index
                cols = x.cols if x.lists else []
                self.ref_written |= set(cols)      # the callee writes the caller's objects
                done.append(('self', [(self.env[c], ('list', c)) for c in cols] + [value(x.operand)]))
            elif isinstance(x, SelfArg) and x.same is not None:      # the object of another argument: nothing more to pass
                done.append(('self', []))
            elif isinstance(x, SelfArg) and x.ref:   # an element by reference: its attributes' lists, then its index
                cols = [x.slots[k] for k, _ in leaf_types(x.object_type)]
                done.append(('self', [(self.env[c.value.id], ('list', c.value.id)) for c in cols] + [value(cols[0].slice)]))
                self.ref_written |= {c.value.id for (k, _), c in zip(leaf_types(x.object_type), cols) if k in x.object_type.writes}
            elif isinstance(x, SelfArg):            # an instance: each of its attributes, as the callee's parameters
                self.typeof(x)
                done.append(('self', [value(y) for y in x.elts]))
            else:
                done.append(value(x))
        chosen = []
        for k in picked:
            chosen += done[k][1] if done[k][0] == 'self' else [done[k]]
        chosen += [(self.env[self.hidden_param(h)], self.hidden_param(h)) if h[1] != 'sjpool' else (T.NOPOOL, None) for h in spec.hidden]
        cargs, arrays = [], []
        for t, v in chosen:
            if t.kind in ('function', 'enumclass', 'classref', 'madeclass', 'nopool'):
                continue
            if t.kind == 'array':
                arrays.append(v)
                cargs += [f'v_{v}'] + [self.extent(v, k) for k in range(t.ndim)]
                if t.strided:
                    cargs += [f'v_{v}__s{k}' for k in range(t.ndim)]
            elif t.kind == 'tuple':
                cargs += [val for val, _ in flat(v if isinstance(v, list) else self.tuple_var(v))]
            elif t.kind == 'list':             # a list parameter, by reference: the callee writes its memory
                cargs += [f'v_{v[1]}', f'v_{v[1]}__n']
            elif isinstance(v, str):           # a global that is a number
                cargs.append(f'v_{v}')
                if t.np == 'dyn':
                    cargs.append(f'v_{v}__np')
            else:
                cargs.append(v[0])
                if t.np == 'dyn':
                    cargs.append(v[1])
        if c.uses_libm:
            cargs.append('sj_m')
        if c.uses_stop:
            cargs.append('sj_stop')
        io, st = self.temp(), self.temp()
        self.emit(f'int64_t {io}[{c.record_slot(len(ret_arrays(c.ret)))}];')
        self.emit(f'int {st} = {c.cname}({", ".join(cargs + ["sj_tb", io])});')
        for j, me in mes.items():
            self.take_writes(e, me, c, io, j)

        self.emit(f'if ({st}) {{ sj_tb[++sj_tb[0]] = ((int64_t){c.frame} << 32) | {io}[1]; sj_io[0] = {io}[0]; sj_io[4] = {io}[4]; '
                  f'SJ_FAIL({st}, {self.line(e)}, {io}[2], {io}[3]); }}')
        return c, io, arrays

    def check_shared(self, e, mes, c):
        """Two instances from one list, or one instance twice, may be the same object: then each is a copy of the same
        attributes, which is only right if what the call assigns through one it doesn't read or assign through another."""
        writes = {j: {key for flat, i, key in c.outs if i == j and flat in c.assigned} for j in mes}
        for j1, a in mes.items():
            for j2, b in mes.items():
                if j1 != j2 and a.owner == b.owner and writes[j1] & (set(b.slots) | writes[j2]):
                    self.fail(e, f'passes two elements of {self.shown(a.owner)}, which may be one instance, and assigns '
                                 f'{sorted(writes[j1] & (set(b.slots) | writes[j2]))[0]} through one while using it through the other')

    def narrow_passed(self, nodes, picked, fn):
        """An instance passed as an argument goes as only the attributes the callee uses: the rest may have changed type."""
        given = False
        for k in picked:                         # the first reference given carries the pool's lists, the others only their index
            if isinstance(nodes[k], RefArg):
                nodes[k].lists, given = not given, True
        for j, k in enumerate(picked):
            x = nodes[k]
            if not (isinstance(x, SelfArg) and x.passed):
                continue
            full, place = x.full
            if full is None:                      # made in the function: typed as its variables are here
                x.elts, x.slots = list(place.values()), dict(place)
                full = self.local_object(x)
            uses = fn._uses_of(j, full.cls) if hasattr(fn, '_uses_of') else None
            if uses is None:
                continue
            names = set(uses[0]).union(*(full.methods[m][0] for m in uses[2] if m in (full.methods or {})))
            x.names_used = names
            x.writes_used = set(uses[1]).union(*(full.methods[m][1] for m in uses[2] if m in (full.methods or {})))
            fields = [(key, ft) for key, ft in full.elem if ft.kind == 'function' or key in names]
            x.object_type = T.object_of(full.py, fields, full.methods, full.cls, full.writes)
            x.elts = [place[key] for key, _ in leaf_types(x.object_type)]
            x.slots = {key: place[key] for key, _ in leaf_types(x.object_type)}
            if getattr(x, 'local', None) is not None:
                x.local = T.object_of(full.py, [kt for kt in fields if kt[1].kind == 'function'], full.methods, full.cls, full.writes)
        self.share_passed(nodes, picked)

    def share_passed(self, nodes, picked):
        """Arguments naming one object (one owner, at one index if it's an element): the first goes as everything any of them
        uses, the others as same_as it, so the callee has one set of variables for all its names and nothing can alias."""
        groups = {}
        for j, k in enumerate(picked):
            x = nodes[k]
            if not isinstance(x, SelfArg):
                continue
            x.same, x.ref = None, False
            sub = [v for v in x.slots.values() if isinstance(v, ast.Subscript)]
            groups.setdefault((x.owner, ast.dump(sub[0].slice) if sub else None), []).append((j, x))
        for members in groups.values():
            if len(members) < 2 or members[0][1].owner is None:
                continue
            j0, lead = members[0]
            for _, x in members[1:]:
                x.same = j0
                lead.slots = {**x.slots, **lead.slots}
            if getattr(lead, 'local', None) is not None:      # made here: typed from its variables, which now are all of them
                continue
            types = [x.object_type for _, x in members]
            fields = {}
            for t in types:
                for k, ft in t.elem:
                    if fields.setdefault(k, ft) is not ft:
                        self.fail(lead, f'passes {self.shown(lead.owner)} twice, and {k} is {fields[k].name} for one and {ft.name} for the other')
            full = next((x.full[0] for _, x in members if getattr(x, 'full', None) and x.full[0] is not None), None)
            order = [k for k, _ in full.elem if k in fields] if full is not None else []
            order += [k for k in fields if k not in order]
            t0 = types[0]
            methods = {}
            for t in types:
                methods.update(t.methods or {})
            lead.object_type = T.object_of(t0.py, [(k, fields[k]) for k in order], methods, t0.cls, frozenset().union(*(t.writes for t in types)))
            lead.elts = [lead.slots[k] for k, _ in leaf_types(lead.object_type)]
        elements = {}                            # elements of one list at different indexes: maybe one instance
        for (owner, at), members in groups.items():
            x = members[0][1]
            if at is not None and owner is not None and all(self.env.get(v.value.id, T.NONE).kind == 'list' and self.env[v.value.id].elem.np is False
                                                            for v in x.slots.values() if isinstance(v, ast.Subscript)):
                elements.setdefault(owner, []).append(x)
        def clash(a, b):                         # what's assigned through one is used through the other
            wa = getattr(a, 'writes_used', None)
            wb = getattr(b, 'writes_used', None)
            if wa is None or wb is None:
                return bool(a.object_type.writes)
            return bool(wa & (getattr(b, 'names_used', set()) | wb) or wb & (getattr(a, 'names_used', set()) | wa))
        for owner, xs in elements.items():
            if any(clash(a, b) for k, a in enumerate(xs) for b in xs[k + 1:]):
                for x in xs:                     # by reference: a write through one is the other's, if they're one
                    x.ref = True

    def take_writes(self, e, me, c, io, j=0):
        """What a method assigned to self, into the caller's variables, before anything else: an exception it raised
        comes after its writes, as in Python."""
        for n, (flat, i, key) in enumerate(c.outs):
            if i != j:
                continue
            place = me.slots[key]
            theirs = c.env[flat]
            if isinstance(place, ast.Subscript):        # an element of a list of instances: its attribute's list
                t = self.env[place.value.id].elem
                if t is not theirs or t.np == 'dyn':
                    self.fail(e, f'the method makes {self.shown(place.value.id)} {theirs.name}, and here it is {t.name}')
                v = self.number_result(theirs, io, c.out_slot(n), c.out_slot(n) + 1)
                self.store_elem(ast.copy_location(ast.Subscript(place.value, place.slice, ast.Store()), place), v, theirs)
                continue
            mine = place.id
            if flat in c.assigned:
                self.assigned.add(mine)          # assigned by what it calls: so assigned here too, for a caller of this
            t = self.env[mine]
            if t is not theirs or t.np == 'dyn':
                self.fail(e, f'the method makes {self.shown(mine)} {theirs.name}, and here it is {t.name}')
            at = c.out_slot(n)
            if t.kind == 'float':
                self.emit(f'memcpy(&v_{mine}, &{io}[{at}], 8);')
            else:
                self.emit(f'v_{mine} = ({t.ctype}){io}[{at}];')

    def call_number(self, e):
        c, io, _ = self.emit_call(e)
        return self.number_result(c.ret, io, 0, 2)

    def number_result(self, r, io, at, flag_at):
        v = self.temp()
        if r.kind == 'float':
            self.emit(f'double {v}_d; memcpy(&{v}_d, &{io}[{at}], 8); {r.ctype} {v} = ({r.ctype}){v}_d;')
        else:
            self.emit(f'{r.ctype} {v} = ({r.ctype}){io}[{at}];')
        if r.np != 'dyn':
            return v, flag_of(r)
        return v, self.hold_flag(f'{io}[{flag_at}] != 0')

    def take_array(self, name, t, c, io, k, arrays):
        """Fill a variable from the record of the k-th array a call returned. A block made in the call becomes this function's."""
        at = c.record_slot(k)
        self.emit(f'v_{name} = ({t.ctype.strip()})(uintptr_t){io}[{at}];')
        for d in range(t.ndim):
            self.emit(f'{self.extent(name, d)} = {io}[{at + 10 + 2 * d}];')
            if t.strided:
                self.emit(f'v_{name}__s{d} = {io}[{at + 11 + 2 * d}];')
        which = f'{io}[{at + 1}]'
        if self.tracks:
            bs = ', '.join(f'v_{a}__b' for a in arrays) or '0'
            os_ = ', '.join(f'v_{a}__o' for a in arrays) or '0'
            self.open(f'if ({io}[{at + 3}])')
            self.emit(f'v_{name}__b = -1; v_{name}__o = (int){io}[{at + 2}];')
            self.close()
            self.open('else')
            self.emit(f'int b_[] = {{{bs}}}, o_[] = {{{os_}}}; v_{name}__b = b_[{which}]; v_{name}__o = {io}[{at + 2}] && o_[{which}];')
            self.close()
        if self.makes:
            hs = ', '.join(f'v_{a}__h' for a in arrays) or '0'
            self.open(f'if ({io}[{at + 3}])')
            self.emit(f'v_{name}__h = (sj_block *)(uintptr_t){io}[{at + 3}]; sj_adopt(&sj_h, v_{name}__h);')
            self.close()
            self.open('else')
            self.emit(f'sj_block *h_[] = {{{hs}}}; v_{name}__h = h_[{which}];')
            self.close()

    def call_array(self, e):
        c, io, arrays = self.emit_call(e)
        name = self.view_name(e)
        self.take_array(name, c.ret, c, io, 0, arrays)
        return name

    def call_results(self, e):
        """For a call returning a tuple: for each element, the variable holding its array, or (value, flag)."""
        c, io, arrays = self.emit_call(e)
        out, k = [], 0
        at = [0]                                   # the pair of slots each number came back in, nested tuples too

        def numbers(t):
            vals = []
            for et in t.elem:
                if et.kind == 'tuple':
                    vals.append(numbers(et))
                else:
                    vals.append(self.number_result(et, io, 4 + 2 * at[0], 5 + 2 * at[0]))
                    at[0] += 1
            return vals
        for i, et in enumerate(c.ret.elem):
            if et.kind == 'array':
                name = self.result_name(e, i)
                self.take_array(name, et, c, io, k, arrays)
                out.append(name)
                k += 1
                at[0] += 1
            elif et.kind == 'tuple':
                out.append(numbers(et))
            else:
                out.append(self.number_result(et, io, 4 + 2 * at[0], 5 + 2 * at[0]))
                at[0] += 1
        return out

    def call_statement(self, e):
        if self.c_function_of(e) is not None:
            self.emit(f'{self.c_call(e)};')
            return
        c = self.compiled_callee(e)
        if c.ret.kind == 'tuple':
            self.call_results(e)
        elif c.ret.kind == 'array':
            self.call_array(e)
        else:
            self.emit_call(e)

    def assign_from_call(self, targets, e):
        """a, b = f(x), where f returns arrays in its tuple."""
        results = self.call_results(e)
        for target in targets:
            for x, r, et in zip(target.elts, results, self.typeof(e).elem):
                if isinstance(r, str):
                    if isinstance(x, ast.Subscript):
                        self.store_slice_array(x, r)
                    else:
                        self.assign_view(x.id, ast.Name(r, ast.Load()))
                elif isinstance(x, ast.Name):
                    self.assign(x.id, r, et)
                elif self.typeof(x).kind == 'array':
                    self.store_slice_number(x, r, et)
                else:
                    self.store_elem(x, r, et)

    def return_from_call(self, e):
        results = self.call_results(e)
        k = 0
        for i, (r, et, rt) in enumerate(zip(results, self.typeof(e).elem, self.ret.elem)):
            if isinstance(r, str):
                self.return_array(ast.Name(r, ast.Load()), k)
                k += 1
            else:
                self.return_number(i, r[0], r[1], et, rt, None)
        self.emit('goto sj_done;')

    # ---- arrays made in the function: np.zeros(n), x.copy()
    def make_of(self, e):
        """For a call making an array: (what it is, its arguments, its keywords), else None."""
        if not isinstance(e, ast.Call):
            return None
        f = e.func
        if isinstance(f, ast.Attribute) and f.attr in ('copy', 'flatten', 'astype') and (
                (isinstance(f.value, ast.Name) and self.env.get(f.value.id) is not None and self.env[f.value.id].kind == 'array')
                or (not isinstance(f.value, ast.Name) and self.is_view(f.value))):
            return 'method_' + f.attr, [f.value] + e.args, e.keywords
        if isinstance(f, ast.Name) and f.id == 'sj_table' and f.id not in self.env:
            return 'table', e.args, []
        name = self.numpy_func(f)
        if name in NP_MAKE:
            return name, e.args, e.keywords
        return None

    def default_int(self):
        """What numpy makes of a Python int here: int32 on wasm32."""
        np = sys.modules.get('numpy')
        return T.NP_INT64 if np is None or np.dtype(int).name == 'int64' else T.NARROW[np.dtype(int).name]

    def dtype_elem(self, node, e):
        """The element type a dtype= argument names, as far as it can be known from the source."""
        np = sys.modules.get('numpy')
        if isinstance(node, ast.Name) and node.id not in self.env and node.id not in self.globals and node.id in ('float', 'int'):
            return T.NP_FLOAT64 if node.id == 'float' else self.default_int()
        if isinstance(node, ast.Attribute) and node.attr == 'dtype' and self.is_array_expr(node.value):
            return self.typeof(node.value).elem
        name = None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id not in self.env \
                and is_numpy(self.globals.get(node.value.id)) and node.attr in MAKE_DTYPES:
            name = node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and np is not None:
            try:
                d = np.dtype(node.value)
            except TypeError:
                self.fail(e, f'dtype {node.value!r}')
            name = d.name if d.isnative else None
        if name == 'float64':
            return T.NP_FLOAT64
        if name == 'int64':
            return T.NP_INT64
        if name in T.NARROW:
            return T.NARROW[name]
        self.fail(e, 'an array made with this dtype')

    def is_array_expr(self, e):
        return (isinstance(e, ast.Name) and self.env.get(e.id) is not None and self.env[e.id].kind == 'array') \
            or (not isinstance(e, ast.Name) and self.is_view(e) and self.typeof(e).kind == 'array')

    def shape_axes(self, node, e):
        """How many axes a shape argument gives. Numpy takes any integer: a bool too, but that's rare enough to leave."""
        if isinstance(node, (ast.Tuple, ast.List)):
            ts = [self.typeof(x) for x in node.elts]
        else:
            t = self.typeof(node)
            ts = t.elem if t.kind == 'tuple' else [t]
        if not all(t.kind == 'int' for t in ts):
            self.fail(e, 'an array shape that is not integers')
        if not 1 <= len(ts) <= MAX_MADE_DIMS:
            self.fail(e, f'an array made with {len(ts)} dimensions')
        return len(ts)

    def make_args(self, e, made, names):
        """The arguments by name, in the order numpy takes them. Anything else stays Python."""
        kind, args, kws = made
        if len(args) > len(names) or any(isinstance(a, ast.Starred) for a in args):
            self.fail(e, f'np.{kind}() with these arguments')
        out = dict(zip(names, args))
        for k in kws:
            if k.arg not in names or k.arg in out:
                self.fail(e, f'np.{kind}() with these arguments')
            out[k.arg] = k.value
        return out

    def check_full_bounds(self, e, value, elem):
        """np.full(2, 300, dtype=np.uint8) raises in newer numpy, and wraps in older. Whichever is here decides; wrapping stays Python."""
        if not T.is_narrow(elem) or elem.kind != 'int' or self.typeof(value).kind != 'int' or self.typeof(value).np is True:
            return
        np = sys.modules.get('numpy')
        try:
            np.full(1, 1 << 40, dtype=np.int8)
        except OverflowError:
            return
        self.fail(e, 'np.full() of a Python int into a narrower dtype, which wraps in this numpy')

    def fill_elem(self, e, value, dtype):
        """The element type of np.full(..., value): the value decides, unless dtype= does."""
        t = self.typeof(value)
        if not t.is_number or t.kind == 'bool':
            self.fail(e, f'np.full() of {t.name}')
        if dtype is not None:
            if dtype.kind == 'int' and t.kind == 'float':
                self.fail(e, 'np.full() of a float into an integer dtype')
            self.check_full_bounds(e, value, dtype)
            return dtype
        if T.is_narrow(t):
            return t
        if t.kind == 'float':
            return T.NP_FLOAT64
        if t.np is True:
            return T.NP_INT64
        if self.default_int() is not T.NP_INT64 and not (isinstance(value, ast.Constant) and abs(value.value) < 1 << 31):
            self.fail(e, 'np.full() of an integer, whose dtype depends on its size here')
        return self.default_int()

    def like_type(self, e, source, order_matters=True):
        """The array a *_like or a copy follows. Numpy keeps a layout that isn't C order, unlike what a kernel makes."""
        if not self.is_array_expr(source):
            self.fail(e, 'an array made from something other than an array')
        t = self.typeof(source)
        if order_matters and t.strided and t.ndim > 1:
            self.fail(e, 'a copy of a view with two dimensions or more: numpy keeps its memory layout')
        if t.ndim > MAX_MADE_DIMS:
            self.fail(e, f'an array made with {t.ndim} dimensions')
        return t

    def make_type(self, e, made):
        kind = made[0]
        if kind == 'asarray':
            args = self.make_args(e, made, ['a'])
            if 'a' not in args or not self.is_array_expr(args['a']):
                self.fail(e, 'np.asarray() of something other than an array')
            return self.typeof(args['a'])
        self.makes = True
        if kind in ('method_copy', 'method_flatten'):
            if made[2] or len(made[1]) > 1:
                self.fail(e, f'.{kind[7:]}() with these arguments')
            t = self.like_type(e, made[1][0], order_matters=False)
            ndim, elem = t.ndim if kind == 'method_copy' else 1, t.elem
        elif kind == 'method_astype':
            args = self.make_args(e, (kind, made[1][1:], made[2]), ['dtype'])
            if 'dtype' not in args:
                self.fail(e, '.astype() with these arguments')
            t = self.like_type(e, made[1][0])
            ndim, elem = t.ndim, self.dtype_elem(args['dtype'], e)
            if elem.kind == 'int' and t.elem.kind == 'float':
                self.fail(e, '.astype() from float to an integer dtype')
        elif kind == 'table':                        # a list of lists made in the function
            h, w, c = made[1]
            if any(self.typeof(n).kind not in ('int', 'bool') or self.typeof(n).np is not False for n in (h, w)):
                self.fail(e, 'a list of lists whose size is not a Python int')
            ndim, elem = 2, self.typeof(c)
            if elem not in (T.INT, T.FLOAT, T.BOOL):
                self.fail(e, f'a list of lists of {elem.name}')
        elif kind == 'arange':
            ndim, elem = 1, self.arange_elem(e, made)
        elif kind == 'linspace':
            ndim, elem = 1, self.linspace_elem(e, made)
        elif kind == 'array' and self.is_list_source(made[1][0] if made[1] else None):
            ndim, elem = self.array_of_list(e, made)
        elif kind in NP_MAKE_SHAPED or kind in NP_MAKE_LIKE:
            like = kind in NP_MAKE_LIKE
            full = kind.startswith('full')
            args = self.make_args(e, made, ['a' if like else 'shape'] + (['fill_value'] if full else []) + ['dtype'])
            if ('a' if like else 'shape') not in args or (full and 'fill_value' not in args):
                self.fail(e, f'np.{kind}() with these arguments')
            dtype = self.dtype_elem(args['dtype'], e) if 'dtype' in args else None
            if like:
                t = self.like_type(e, args['a'])
                ndim, elem = t.ndim, dtype or t.elem
                if full:
                    if not self.typeof(args['fill_value']).is_number or self.typeof(args['fill_value']).kind == 'bool' \
                            or (elem.kind == 'int' and self.typeof(args['fill_value']).kind == 'float'):
                        self.fail(e, 'np.full_like() of this value')
                    self.check_full_bounds(e, args['fill_value'], elem)
            else:
                ndim = self.shape_axes(args['shape'], e)
                elem = self.fill_elem(e, args['fill_value'], dtype) if full else dtype or T.NP_FLOAT64
        elif kind in ('copy', 'array'):
            args = self.make_args(e, made, ['a'] + (['dtype'] if kind == 'array' else []))
            if 'a' not in args:
                self.fail(e, f'np.{kind}() with these arguments')
            t = self.like_type(e, args['a'])
            ndim, elem = t.ndim, t.elem
            if 'dtype' in args:
                elem = self.dtype_elem(args['dtype'], e)
                if elem.kind == 'int' and t.elem.kind == 'float':
                    self.fail(e, 'np.array() of floats into an integer dtype')
        else:
            self.fail(e, f'np.{kind}() with these arguments')
        v = T.array(elem, True, ndim, False)
        self.env[self.view_name(e)] = v
        return v

    def arange_args(self, e, made):
        args = self.make_args(e, made, ['start', 'stop', 'step', 'dtype'])
        if 'stop' not in args and 'start' in args:
            args['start'], args['stop'] = None, args['start']   # np.arange(n)
        if 'stop' not in args or (args.get('start') is None and 'step' in args):
            self.fail(e, 'np.arange() with these arguments')
        return [args.get('start'), args['stop'], args.get('step')], args.get('dtype')

    def arange_elem(self, e, made):
        nums, dtype = self.arange_args(e, made)
        ts = [self.typeof(a) for a in nums if a is not None]
        if any(not t.is_number or t.kind == 'bool' or T.is_narrow(t) for t in ts):
            self.fail(e, 'np.arange() of these values')
        if any(t.kind == 'float' for t in ts):
            elem = T.NP_FLOAT64
        elif self.default_int() is T.NP_INT64:
            elem = T.NP_INT64
        elif len(ts) > 1 or ts[0].np:                 # on wasm32 the dtype depends on how big the numbers are
            self.fail(e, 'np.arange() of these integers, whose dtype depends on their size here')
        else:
            elem = self.default_int()
        if dtype is not None and self.dtype_elem(dtype, e) is not elem:
            self.fail(e, 'np.arange() with a dtype it would not have chosen')
        return elem

    def linspace_elem(self, e, made):
        args = self.make_args(e, made, ['start', 'stop', 'num', 'endpoint'])
        if 'start' not in args or 'stop' not in args:
            self.fail(e, 'np.linspace() with these arguments')
        for x in (args['start'], args['stop']):
            t = self.typeof(x)
            if not t.is_number or t.kind == 'bool' or T.is_narrow(t):
                self.fail(e, f'np.linspace() of {t.name}')
        if 'num' in args and self.typeof(args['num']).kind != 'int':
            self.fail(e, 'np.linspace() with a number of samples that is not an integer')
        if 'endpoint' in args and not (isinstance(args['endpoint'], ast.Constant) and type(args['endpoint'].value) is bool):
            self.fail(e, 'np.linspace() with endpoint that is not True or False')
        return T.NP_FLOAT64

    def is_list_source(self, a):
        return isinstance(a, ast.List) or (isinstance(a, ast.Name) and self.env.get(a.id) is not None and self.env[a.id].kind == 'list')

    def list_shape(self, e, a):
        """The shape of a list literal (a ragged one is numpy's error to raise), and its numbers in order."""
        if not isinstance(a, ast.List):
            return [], [a]
        parts = [self.list_shape(e, x) for x in a.elts]
        if any(p[0] != parts[0][0] for p in parts) or (parts and len(parts[0][0]) >= MAX_MADE_DIMS):
            self.fail(e, 'np.array() of lists of different lengths')
        return [len(parts)] + (parts[0][0] if parts else []), [x for p in parts for x in p[1]]

    def array_of_list(self, e, made):
        args = self.make_args(e, made, ['a', 'dtype'])
        a = args['a']
        dtype = self.dtype_elem(args['dtype'], e) if 'dtype' in args else None
        if isinstance(a, ast.Name):
            ts = [self.env[a.id].elem]
            ndim = 1
            if ts[0].kind == 'int' and dtype is None:     # np.array of no elements is float64, of any list
                self.fail(e, 'np.array() of a list of ints: an empty one gives floats. Pass dtype=')
        else:
            shape, nums = self.list_shape(e, a)
            ndim = len(shape)
            if 0 in shape[:-1]:
                self.fail(e, 'np.array() of empty lists')
            ts = [self.typeof(x) for x in nums]
        if any(not t.is_number or t.kind == 'bool' or T.is_narrow(t) for t in ts):
            self.fail(e, 'np.array() of these values')
        if dtype is not None:
            if dtype.kind == 'int' and any(t.kind == 'float' for t in ts):
                self.fail(e, 'np.array() of floats into an integer dtype')
            return ndim, dtype
        if any(t.kind == 'float' for t in ts) or not ts:
            return ndim, T.NP_FLOAT64
        if self.default_int() is T.NP_INT64:
            return ndim, T.NP_INT64
        self.fail(e, 'np.array() of integers, whose dtype depends on their size here')

    def make_arange(self, e, name, t):
        nums, _ = self.arange_args(e, self.make_of(e))
        line = self.line(e)
        vals, how = [], 0
        for k, a in enumerate(nums):
            if a is None:
                vals.append('0LL' if k == 0 else '1LL')
                continue
            at = self.typeof(a)
            vals.append(self.hold(self.expr(a)[0], at))
            how |= (int(at.kind == 'float') | int(at.np is True) << 1) << (2 * k)
        floats = any(self.typeof(a).kind == 'float' for a in nums if a is not None)
        a, b, c = vals
        ctype = t.elem.ctype
        bits = []
        for k, (v, x) in enumerate(zip(vals, nums)):
            if x is not None and self.typeof(x).kind == 'float':
                tmp = self.temp()
                self.emit(f'double {tmp}_d = {v}; int64_t {tmp} = 0; memcpy(&{tmp}, &{tmp}_d, 8);')
                bits.append(tmp)
            else:
                bits.append(f'(int64_t){v}')
        length = self.temp()
        self.emit(f'double {length}_x = ceil(((double){b} - (double){a}) / (double){c});')
        if not floats:
            self.emit(f'if ({c} != 0 && (uint64_t)((uint64_t){b} - (uint64_t){a}) + 9007199254740992ULL <= 18014398509481984ULL) '
                      f'{length}_x = ceil((double)({b} - {a}) / (double){c});')
        bad = f'{c} == 0 || !({length}_x > -9.2e18 && {length}_x < 9.2e18)'
        fail = f'sj_io[0] = {bits[0]}; sj_io[4] = {how}; SJ_FAIL({ARANGE_ERROR}, {line}, {bits[1]}, {bits[2]});'
        self.emit(f'if ({bad}) {{ {fail} }}')
        self.emit(f'int64_t {length} = {length}_x > 0 ? (int64_t){length}_x : 0;')
        if floats:                              # numpy: a range that isn't empty, but too small a part of the step to see
            self.emit(f'{{ double r_ = ((double){b} - (double){a}) / (double){c}; if (r_ == 0.0 && (double){b} != (double){a} && 1.0 / r_ > 0.0) {length} = 1; }}')
        self.new_block(name, t, [length], 0, line, fail)
        i = self.temp()
        if floats:
            self.emit(f'double {i}_f = (double){a}, {i}_d = ((double){a} + (double){c}) - (double){a};')
            self.emit(f'if ({length} > 0) v_{name}[0] = {i}_f;')    # as numpy: the first two, then those give the rest
            self.emit(f'if ({length} > 1) v_{name}[1] = (double){a} + (double){c};')
            self.open(f'for (int64_t {i} = 2; {i} < {length}; {i}++)')
            self.emit(f'v_{name}[{i}] = {i}_f + (double){i} * {i}_d;')
        else:
            self.open(f'for (int64_t {i} = 0; {i} < {length}; {i}++)')
            self.emit(f'v_{name}[{i}] = ({ctype})((uint64_t){a} + (uint64_t){i} * (uint64_t){c});')
        self.close()

    def make_linspace(self, e, name, t):
        args = self.make_args(e, self.make_of(e), ['start', 'stop', 'num', 'endpoint'])
        line = self.line(e)
        a = self.hold(self.coerce(self.expr(args['start'])[0], self.typeof(args['start']), T.FLOAT), T.FLOAT)
        b = self.hold(self.coerce(self.expr(args['stop'])[0], self.typeof(args['stop']), T.FLOAT), T.FLOAT)
        n = self.hold(self.coerce(self.expr(args['num'])[0], self.typeof(args['num']), T.INT), T.INT) if 'num' in args else '50LL'
        endpoint = 'endpoint' not in args or args['endpoint'].value
        self.emit(f'if ({n} < 0) SJ_FAIL({LINSPACE_ERROR}, {line}, {n}, 0);')
        self.new_block(name, t, [n], 0, line, f'SJ_FAIL({LINSPACE_ERROR}, {line}, {n}, 0);')
        div, delta, step, i = self.temp(), self.temp(), self.temp(), self.temp()
        self.emit(f'int64_t {div} = {n}{" - 1" if endpoint else ""}; double {delta} = {b} - {a}, {step} = {div} > 0 ? {delta} / (double){div} : 0.0;')
        self.open(f'for (int64_t {i} = 0; {i} < {n}; {i}++)')
        self.emit(f'double y_ = (double){i};')
        self.emit(f'if ({div} > 0) {{ if ({step} == 0.0) y_ = y_ / (double){div} * {delta}; else y_ = y_ * {step}; }} else y_ = y_ * {delta};')
        self.emit(f'v_{name}[{i}] = y_ + {a};')
        self.close()
        if endpoint:
            self.emit(f'if ({n} > 1) v_{name}[{n} - 1] = {b};')

    def make_from_list(self, e, name, t):
        args = self.make_args(e, self.make_of(e), ['a', 'dtype'])
        a = args['a']
        line = self.line(e)
        if isinstance(a, ast.Name):
            n = self.hold(f'v_{a.id}__n', T.INT)
            self.new_block(name, t, [n], 0, line)
            et = self.env[a.id].elem
            i = self.temp()
            self.open(f'for (int64_t {i} = 0; {i} < {n}; {i}++)')
            self.emit(f'v_{name}[{i}] = {self.coerce(f"v_{a.id}[{i}]", et, t.elem)};')
            self.close()
            return
        shape, nums = self.list_shape(e, a)
        held = []
        for x in nums:                                  # computed in full, as Python builds the list
            xt = self.typeof(x)
            v = self.expr(x)
            held.append(((self.hold(v[0], xt), v[1]), xt, x))
        self.new_block(name, t, [f'{k}LL' for k in shape], 0, line)
        for i, (v, xt, x) in enumerate(held):
            self.check_fits(x, t.elem, v, xt)
            self.emit(f'v_{name}[{i}] = {self.coerce(v[0], xt, t.elem)};')

    def new_block(self, name, t, shape, zero, line, fail=None):
        """Allocate a variable's array. shape: C integers already evaluated. fail: C for what to raise, if not numpy's for an empty array of this shape."""
        dims, b = self.temp(), self.temp()
        self.emit(f'int64_t {dims}[] = {{{", ".join(shape)}}};')
        self.emit(f'sj_block *{b} = sj_make(&sj_h, {len(shape)}, {dims}, sizeof({t.elem.ctype}), {zero});')
        at = [(shape[k] if k < len(shape) else f'(-{-NO_AXIS - 1}LL - 1)') for k in range(MAX_MADE_DIMS)]
        code = MAKE_ERROR + MAKE_DTYPES.index(numpy_dtype(t.elem))
        self.emit(f'if (!{b}) {{ {fail or f"sj_io[0] = {at[0]}; sj_io[4] = {at[3]}; SJ_FAIL({code}, {line}, {at[1]}, {at[2]});"} }}')
        self.emit(f'v_{name} = ({t.ctype.strip()})({b} + 1); v_{name}__h = {b};')
        if self.tracks:
            self.emit(f'v_{name}__b = -1; v_{name}__o = 1;')
        for k, n in enumerate(shape):
            self.emit(f'{self.extent(name, k)} = {n};')

    def fill_block(self, name, t, val):
        total = ' * '.join(self.extent(name, k) for k in range(t.ndim))
        i = self.temp()
        self.open(f'for (int64_t {i} = 0, n_ = {total}; {i} < n_; {i}++)')
        self.emit(f'v_{name}[{i}] = {val};')
        self.close()

    def copy_into(self, name, t, src):
        """Elements in C order, whatever the source's strides."""
        st = self.env[src]
        if not st.strided and st.elem is t.elem:
            total = ' * '.join(self.extent(src, k) for k in range(st.ndim))
            self.emit(f'memcpy(v_{name}, v_{src}, sizeof({t.elem.ctype}) * (size_t)({total}));')
            return
        shape, strides = self.view_axes(src)
        at = self.temp()
        self.emit(f'int64_t {at} = 0;')

        def body(idx):
            read = ' + '.join(f'{i} * {s}' for i, s in zip(idx, strides))
            self.emit(f'v_{name}[{at}++] = {self.coerce(f"v_{src}[{read}]", st.elem, t.elem)};')
        self.each_element(shape, strides, body)

    def shape_values(self, node):
        """Evaluate a shape argument, in order."""
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.hold(self.coerce(self.expr(x)[0], self.typeof(x), T.INT), T.INT) for x in node.elts]
        t = self.typeof(node)
        if t.kind == 'tuple':
            return [self.hold(self.coerce(c, et, T.INT), T.INT) for (c, _), et in zip(self.texpr(node), t.elem)]
        return [self.hold(self.coerce(self.expr(node)[0], t, T.INT), T.INT)]

    def make(self, e):
        made = self.make_of(e)
        kind = made[0]
        if kind == 'asarray':
            return self.materialize(self.make_args(e, made, ['a'])['a'])
        name = self.view_name(e)
        t = self.env[name]
        line = self.line(e)
        if kind == 'table':                          # range(h) first, then each row's size and value, as Python goes
            h, w, c = (self.expr(x) for x in made[1])
            shape = [self.hold(f'((int64_t)({v[0]}) > 0 ? (int64_t)({v[0]}) : 0)', T.INT) for v in (h, w)]
            val = self.hold(self.coerce(c[0], self.typeof(made[1][2]), t.elem), t.elem)
            self.new_block(name, t, shape, 0, line)
            self.fill_block(name, t, val)
            return name
        if kind == 'arange':
            self.make_arange(e, name, t)
            return name
        if kind == 'linspace':
            self.make_linspace(e, name, t)
            return name
        if kind == 'array' and self.is_list_source(made[1][0] if made[1] else None):
            self.make_from_list(e, name, t)
            return name
        if kind.startswith('method_') or kind in ('copy', 'array'):
            source = made[1][0] if kind.startswith('method_') else self.make_args(e, made, ['a', 'dtype'])['a']
            src = self.materialize(source)
            if kind == 'method_flatten':
                shape = [self.hold(' * '.join(self.extent(src, k) for k in range(self.env[src].ndim)), T.INT)]
            else:
                shape = [self.hold(self.extent(src, k), T.INT) for k in range(t.ndim)]
            self.new_block(name, t, shape, 0, line)
            self.copy_into(name, t, src)
            return name
        like = kind in NP_MAKE_LIKE
        full = kind.startswith('full')
        args = self.make_args(e, made, ['a' if like else 'shape'] + (['fill_value'] if full else []) + ['dtype'])
        if like:
            src = self.materialize(args['a'])
            shape = [self.hold(self.extent(src, k), T.INT) for k in range(t.ndim)]
        else:
            shape = self.shape_values(args['shape'])
        val = None
        if full:                                       # as numpy does: the array, then the value goes in
            vt = self.typeof(args['fill_value'])
            v = self.expr(args['fill_value'])
            val = self.hold(self.coerce(v[0], vt, t.elem), t.elem)
        self.new_block(name, t, shape, 1 if kind.startswith('zeros') else 0, line)
        if full:
            self.check_fits(e, t.elem, v, vt)
            self.fill_block(name, t, val)
        elif kind.startswith('ones'):
            self.fill_block(name, t, self.coerce('1', T.INT, t.elem))
        return name

    def numpy_func(self, func):
        """The name of the numpy function this calls, if it's one that compiles on single numbers."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id not in self.env \
                and is_numpy(self.globals.get(func.value.id)):
            return func.attr if func.attr in NP_FUNCS else None
        if isinstance(func, ast.Name) and func.id not in self.env:
            obj = self.globals.get(func.id)
            if isinstance(obj, StaticNumpy):
                return obj.name if obj.name in NP_FUNCS else None
            np = sys.modules.get('numpy')
            if np is not None and obj is not None:
                for n in NP_FUNCS:
                    if obj is getattr(np, n, None):
                        return n
        return None

    def reduction_of(self, e):
        """For np.sum(x) or x.sum(): (the operation, the array expression, the arguments after it), or None."""
        nf = self.numpy_func(e.func)
        if nf in NP_REDUCE:
            if not e.args:
                self.fail(e, f'np.{nf}() with no array')
            return NP_REDUCE[nf], e.args[0], e.args[1:]
        f = e.func
        if isinstance(f, ast.Attribute) and f.attr in ARRAY_METHODS and (
                (isinstance(f.value, ast.Name) and self.env.get(f.value.id) is not None and self.env[f.value.id].kind == 'array')
                or self.is_view(f.value)):
            return f.attr, f.value, e.args
        return None

    def reduction_array_type(self, e, op, a, rest):
        """The array being reduced, if this call gives one number that can be matched exactly. Any argument not understood here,
        such as axis= on more than one dimension, keepdims=True, dtype=, out=, where= or initial=, stays Python."""
        t = self.typeof(a)
        if t.kind != 'array':
            self.fail(e, f'np.{op}() of {t.name}: only arrays compile')
        if t.elem not in (T.NP_FLOAT64, T.NP_INT64):
            if not (T.is_narrow(t.elem) or t.elem is T.NP_UINT64) or (op in ('std', 'var') and t.elem.kind == 'float'):
                self.fail(e, f'{op}() of {t.elem.name[3:]} elements')
            if op in ('sum', 'prod') and t.elem.kind == 'int' and t.elem.ctype[0] != 'u' and self.default_int() is not T.NP_INT64:
                self.fail(e, f'{op}() of {t.elem.name[3:]} elements, which numpy adds up as its default int')
        if t.strided and t.ndim > 1:
            self.fail(e, f'{op}() of a multi-dimensional view with gaps')
        if len(rest) > 1 or any(isinstance(x, ast.Starred) for x in rest):
            self.fail(e, f'{op}() with more than an axis')
        axes = list(rest) + [k.value for k in e.keywords if k.arg == 'axis']
        for k in e.keywords:
            if k.arg == 'ddof' and op in ('std', 'var'):
                if self.typeof(k.value).kind != 'int':
                    self.fail(e, f'{op}() with a ddof that is not an integer')
                continue
            if k.arg is None or k.arg not in ('axis', 'keepdims'):
                self.fail(e, f'{op}() with {k.arg}=')
            if k.arg == 'keepdims' and not (isinstance(k.value, ast.Constant) and k.value.value is False):
                self.fail(e, f'{op}() with keepdims= (it returns an array)')
        if len(axes) > 1:
            self.fail(e, f'{op}() with two axes')
        for x in axes:
            if isinstance(x, ast.Constant) and x.value is None:
                continue
            literal = x.operand if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub) else x
            value = literal.value if isinstance(literal, ast.Constant) else None
            if type(value) is not int or t.ndim != 1 or (-value if literal is not x else value) not in (0, -1):
                self.fail(e, f'{op}() with an axis (on more than one dimension it returns an array)')
        return t

    def narrow_sum_type(self, e, el, st, start):
        """sum() where numpy's rules apply at each step: start + x0 + x1 ..., each one wrapping or rounding in the
        narrow type. From a Python start the result is that start when there's nothing to add, so it carries the flag."""
        if el.np is not True or st.np == 'dyn' or T.is_narrow_dyn(el):
            self.fail(e, f'sum() of {el.name} from {st.name}')
        r = T.narrow_result(st, el, 'Add')
        if st.np is True:
            return r
        float32_from_int = r is T.NARROW['float32'] and st.kind == 'int'
        if not T.is_narrow(r) or (r.kind != st.kind and not float32_from_int):
            self.fail(e, f'sum() of {el.name} from a Python {st.kind}, whose type would depend on the data')
        value = 0 if start is None else const_number(start)
        if not T.holds_exactly(r, value):
            self.fail(e, f'sum() of {el.name} from a start that isn\'t a literal {r.name} holds exactly')
        return T.INT_OR_F32 if float32_from_int else T.NARROW_DYN[r.name[3:]]

    def numpy_type(self, e, name):
        if name in NP_REDUCE:
            op, a, rest = self.reduction_of(e)
            return self.np_reduction_type(e, op, a, rest)
        if e.keywords or any(isinstance(a, ast.Starred) for a in e.args):
            self.fail(e, f'np.{name}() with keyword arguments')
        ts = [self.typeof(a) for a in e.args]
        if name == 'uint64':                          # np.uint64(n): a Python int must fit, numpy's ints are cast
            if len(ts) != 1 or ts[0].kind != 'int' or ts[0].np == 'dyn':
                self.fail(e, 'np.uint64() of this')
            return T.NP_UINT64
        two = name in NP_LIBM2 or name in NP_MINMAX or name == 'copysign'
        if len(ts) != NP_FLOAT_ONLY.get(name, 2 if two else 1):
            self.fail(e, f'np.{name}() with {len(ts)} arguments')
        for t in ts:
            if t.kind == 'array':
                self.fail(e, f'np.{name}() of an array: only single numbers compile; write it as a loop')
            if T.is_narrow(t):
                if name not in NP_NARROW_OK or (t.kind == 'int' and name not in NP_NARROW_INT_OK):
                    self.fail(e, f'np.{name}() of {t.name}' + (' (numpy gives a float of a size that depends on it)' if t.kind == 'int' else ''))
                continue
            if not (t in (T.INT, T.FLOAT, T.NP_INT64, T.NP_FLOAT64) or (t.np == 'dyn' and t.kind in ('int', 'float') and not T.is_narrow_dyn(t))):
                self.fail(e, f'np.{name}() of {t.name}')
        if name in T.NARROW:
            return self.narrow_convert_type(e, name, ts[0])
        if any(T.is_narrow(t) for t in ts):
            return self.narrow_numpy_type(e, name, ts)
        floats = any(t.kind == 'float' for t in ts)
        if name in NP_FLOAT_ONLY and not all(t.kind == 'float' for t in ts):
            self.fail(e, f'np.{name}() of an int (the type it gives depends on the platform)')
        if name in NP_TESTS:
            return T.NP_BOOL
        if name == 'int64':
            if ts[0].kind != 'int':
                self.fail(e, 'np.int64() of a float (it raises on nan and on values too large)')
            return T.NP_INT64
        if name in NP_ROUND and not floats:
            self.fail(e, f'np.{name}() of an int (what it gives depends on the numpy version)')
        if name in NP_FLOATS_ONLY2 and not floats:
            self.fail(e, f'np.{name}() of two ints')
        if (name in NP_KEEPS_INT or name in NP_MINMAX) and not floats:
            if any(t is not T.NP_INT64 for t in ts):
                self.fail(e, f'np.{name}() of a Python int (the type it gives depends on the platform)')
            return T.NP_INT64
        return T.NP_FLOAT64

    def narrow_convert_type(self, e, name, t):
        """np.float32(x), np.uint8(x) and the rest."""
        r = T.NARROW[name]
        if r.kind == 'int' and t.kind != 'int':
            self.fail(e, f'np.{name}() of a float (it raises on nan, and what it gives past the range depends on the platform)')
        if t.kind == 'bool' and t.np:
            self.fail(e, f'np.{name}() of a numpy bool')
        return r

    def narrow_numpy_type(self, e, name, ts):
        """A numpy function given a narrow scalar: its type, from numpy's promotion of the arguments."""
        if any(t.np == 'dyn' or (t.np and t.kind == 'bool') for t in ts):
            self.fail(e, f'np.{name}() of {", ".join(t.name for t in ts)}')
        if name in NP_TESTS:
            return T.NP_BOOL
        if name in ('float64', 'int64'):
            if name == 'int64' and ts[0].kind != 'int':
                self.fail(e, 'np.int64() of a float (it raises on nan and on values too large)')
            return T.NP_FLOAT64 if name == 'float64' else T.NP_INT64
        r = ts[0] if len(ts) == 1 else T.narrow_result(ts[0], ts[1], 'Add')
        if r.kind == 'int' and name not in NP_NARROW_INT_OK:
            self.fail(e, f'np.{name}() of ints')
        return r

    def np_reduction_type(self, e, op, a, rest):
        t = self.reduction_array_type(e, op, a, rest)
        if op in ('any', 'all'):
            return T.NP_BOOL
        if op in ('argmax', 'argmin'):
            np = sys.modules.get('numpy')
            if np is None or np.intp is np.int64:
                return T.NP_INT64
            return T.NARROW[np.dtype(np.intp).name]          # int32 on wasm32
        if op in ('sum', 'mean', 'std', 'var') and sum_chunk() is None:
            self.fail(e, f'{op}() on a numpy whose rounding isn\'t known')
        if (T.is_narrow(t.elem) or t.elem is T.NP_UINT64) and op in ('sum', 'prod', 'mean'):
            if t.elem.kind == 'float':
                return t.elem
            if op == 'mean':
                return T.NP_FLOAT64
            return T.NP_UINT64 if t.elem.ctype[0] == 'u' else T.NP_INT64     # numpy totals unsigned ints in uint64
        if op in ('mean', 'std', 'var'):
            return T.NP_FLOAT64
        return t.elem

    def np_reduction_call(self, e, op, a):
        """An element count and a stride describe any array this takes: a contiguous one is one long run, whatever its shape."""
        name = self.materialize(a)
        t = self.env[name]
        self.helpers.add('np_folds')
        n = self.hold(' * '.join(['1LL'] + [self.extent(name, k) for k in range(t.ndim)]), T.INT)
        st = self.hold(self.stride(name, t.ndim - 1), T.INT)
        p = f'v_{name}'
        floats = t.elem.kind == 'float'
        line = self.line(e)
        if T.is_narrow(t.elem) or t.elem is T.NP_UINT64:
            return self.narrow_reduction(e, op, t.elem, p, n, st, floats, line)
        if op in ('max', 'min', 'argmax', 'argmin', 'any', 'all') and not self.loops:
            self.gotchas[line + self.offset] = f'{op}() is slower here than numpy on big arrays, for now: numpy uses SIMD (see "Gotchas")'
        if op == 'sum' and floats:
            self.helpers.add('np_sum')
            return self.hold(f'sj_np_sum({p}, {n}, {st}, {sum_chunk() or 0})', T.FLOAT), '1'
        if op == 'sum':
            return self.hold(f'sj_np_sum_i({p}, {n}, {st})', T.INT), '1'
        if op == 'prod':
            return self.hold(f'sj_np_prod_{"f" if floats else "i"}({p}, {n}, {st})', t.elem), '1'
        if op == 'mean' and floats:
            self.helpers.add('np_sum')
            return self.hold(f'sj_np_sum({p}, {n}, {st}, {sum_chunk() or 0}) / (double){n}', T.FLOAT), '1'
        if op == 'mean':
            self.helpers.add('np_stats')
            return self.hold(f'sj_np_sum_as_float({p}, {n}, {st}) / (double){n}', T.FLOAT), '1'
        if op in ('std', 'var'):
            self.helpers.add('np_stats')
            ddof = next((k.value for k in e.keywords if k.arg == 'ddof'), None)
            d = '0LL' if ddof is None else self.hold(self.coerce(self.expr(ddof)[0], self.typeof(ddof), T.INT), T.INT)
            v = self.hold(f'sj_np_var_{"f" if floats else "i"}({p}, {n}, {st}, {sum_chunk() or 0}, {d})', T.FLOAT)
            return (self.hold(f'sqrt({v})', T.FLOAT) if op == 'std' else v), '1'
        suffix = 'f' if floats else 'i'
        if op in ('any', 'all'):
            return self.hold(f'sj_np_{op}_{suffix}({p}, {n}, {st})', T.BOOL), '1'
        if op in ('max', 'min', 'argmax', 'argmin'):
            words = {'max': 'zero-size array to reduction operation maximum which has no identity',
                     'min': 'zero-size array to reduction operation minimum which has no identity',
                     'argmax': 'attempt to get argmax of an empty sequence', 'argmin': 'attempt to get argmin of an empty sequence'}
            self.emit(f'if ({n} == 0) SJ_FAIL({VALUE_ERROR}, {line}, {VALUE_ERROR_MESSAGES.index(words[op])}, 0);')
        if op in ('max', 'min') and floats:
            rule = zero_rule('maximum' if op == 'max' else 'minimum')
            code = {'first': 0, 'second': 2, 'positive' if op == 'max' else 'negative': 1}.get(rule)
            if code is None:
                self.fail(e, f'{op}() on a numpy whose signed zeros go another way')
            return self.hold(f'sj_np_{op}_f({p}, {n}, {st}, {code})', T.FLOAT), '1'
        return self.hold(f'sj_np_{op}_{suffix}({p}, {n}, {st})', T.INT), '1'

    def narrow_reduction(self, e, op, elem, p, n, st, floats, line):
        """A reduction over a narrow array, by a copy of the same helper for its element type (_mathlib.typed)."""
        rt = self.typeof(e)
        pre = f'sj_np{_mathlib.TAGS[elem.ctype]}_'
        suffix = 'f' if floats else 'i'
        if op in ('max', 'min', 'argmax', 'argmin', 'any', 'all') and not self.loops:
            self.gotchas[line + self.offset] = f'{op}() is slower here than numpy on big arrays, for now: numpy uses SIMD (see "Gotchas")'
        if op in ('sum', 'mean') and floats:
            self.helpers.add(_mathlib.typed('np_sum', elem.ctype))
            s = self.hold(f'{pre}sum({p}, {n}, {st}, {sum_chunk() or 0})', rt)
            return (self.hold(f'(float)((double){s} / (double){n})', rt) if op == 'mean' else s), '1'
        if op == 'mean':
            self.helpers.add(_mathlib.typed('np_stats', elem.ctype))
            return self.hold(f'{pre}sum_as_float({p}, {n}, {st}) / (double){n}', T.FLOAT), '1'
        if op in ('std', 'var'):                      # ints: in float64, as numpy does for int64
            self.helpers.add(_mathlib.typed('np_stats', elem.ctype))
            ddof = next((k.value for k in e.keywords if k.arg == 'ddof'), None)
            d = '0LL' if ddof is None else self.hold(self.coerce(self.expr(ddof)[0], self.typeof(ddof), T.INT), T.INT)
            v = self.hold(f'{pre}var_i({p}, {n}, {st}, {sum_chunk() or 0}, {d})', T.FLOAT)
            return (self.hold(f'sqrt({v})', T.FLOAT) if op == 'std' else v), '1'
        self.helpers.add(_mathlib.typed('np_folds', elem.ctype))
        if op in ('sum', 'prod'):
            return self.hold(f'{pre}{op}_{suffix}({p}, {n}, {st})', rt), '1'
        if op in ('any', 'all'):
            return self.hold(f'{pre}{op}_{suffix}({p}, {n}, {st})', T.BOOL), '1'
        words = {'max': 'zero-size array to reduction operation maximum which has no identity',
                 'min': 'zero-size array to reduction operation minimum which has no identity',
                 'argmax': 'attempt to get argmax of an empty sequence', 'argmin': 'attempt to get argmin of an empty sequence'}
        self.emit(f'if ({n} == 0) SJ_FAIL({VALUE_ERROR}, {line}, {VALUE_ERROR_MESSAGES.index(words[op])}, 0);')
        if op in ('max', 'min') and floats:
            rule = zero_rule('maximum' if op == 'max' else 'minimum')
            code = {'first': 0, 'second': 2, 'positive' if op == 'max' else 'negative': 1}.get(rule)
            if code is None:
                self.fail(e, f'{op}() on a numpy whose signed zeros go another way')
            return self.hold(f'{pre}{op}_f({p}, {n}, {st}, {code})', rt), '1'
        return self.hold(f'{pre}{op}_{suffix}({p}, {n}, {st})', rt), '1'

    def numpy_call(self, e, name):
        if name in NP_REDUCE:
            op, a, _ = self.reduction_of(e)
            return self.np_reduction_call(e, op, a)
        ts = [self.typeof(a) for a in e.args]
        vals = [self.expr(a) for a in e.args]          # arguments first, in order
        vals = [self.hold(c, t) for (c, _), t in zip(vals, ts)]
        rt = self.typeof(e)
        if name == 'uint64':
            if ts[0].np is False and ts[0] is not T.U64:
                self.check_fits(e, T.NP_UINT64, (vals[0], '0'), ts[0])
            return f'((uint64_t)({vals[0]}))', '1'
        if T.is_narrow(rt) or any(T.is_narrow(t) for t in ts):
            for t, v in zip(ts, vals):                 # numpy: a Python int must fit the dtype
                if T.is_narrow(rt) and rt.kind == 'int' and t.kind in ('int', 'bool') and t.np is False:
                    self.check_fits(e, rt, (v, '0'), t)
            if name in T.NARROW:
                return f'({self.coerce(vals[0], ts[0], rt)})', '1'
            if T.is_narrow(rt):                        # the same function on doubles, then rounded once to the narrow type
                wide = T.NP_FLOAT64 if rt.kind == 'float' else T.NP_INT64
                ts = [wide if T.is_narrow(t) or t.np is False else t for t in ts]
                c, f = self.numpy_on(name, vals, ts, wide)
                return f'(({rt.ctype})({c}))', f
        return self.numpy_on(name, vals, ts, rt)

    def numpy_on(self, name, vals, ts, rt):
        xs = [self.hold(self.coerce(c, t, T.FLOAT), T.FLOAT) for c, t in zip(vals, ts)]
        x = xs[0]
        if name == 'int64':
            return f'(int64_t)({vals[0]})', '1'
        if name == 'float64':
            return x, '1'
        if name in NP_TESTS:
            test = {'isnan': f'{x} != {x}', 'isinf': f'isinf({x})', 'isfinite': f'isfinite({x})', 'signbit': f'signbit({x})'}[name]
            return f'(({test}) ? 1 : 0)', '1'
        if rt.kind == 'int':
            a = f'(int64_t)({vals[0]})'
            if name in ('abs', 'absolute'):
                return f'sj_abs({a})', '1'
            if name == 'sign':
                return self.hold(f'({a} > 0) - ({a} < 0)', T.INT), '1'
            if name == 'square':
                return f'sj_mul({a}, {a})', '1'
            b = f'(int64_t)({vals[1]})'
            return self.hold(f'({a} {"<" if name in ("minimum", "fmin") else ">"} {b} ? {a} : {b})', T.INT), '1'
        if name in NP_DIRECT1 or name in NP_ROUND:
            return self.hold(f'{name}({x})', T.FLOAT), '1'
        if name in ('round', 'around'):
            return self.hold(f'rint({x})', T.FLOAT), '1'
        if name in ('degrees', 'rad2deg'):
            return self.hold(f'{x} * {c_float(180.0 / math.pi)}', T.FLOAT), '1'
        if name in ('radians', 'deg2rad'):
            return self.hold(f'{x} * {c_float(math.pi / 180.0)}', T.FLOAT), '1'
        if name == 'clip':                               # a NaN anywhere wins; low above high gives high
            lo, hi = xs[1], xs[2]
            takes_lo, takes_hi = clip_ties()
            low = f'{x} <= {lo}' if takes_lo else f'{x} < {lo}'
            high = '>=' if takes_hi else '>'
            r = self.hold(f'({x} != {x} ? {x} : {lo} != {lo} ? {lo} : {hi} != {hi} ? {hi} : {low} ? {lo} : {x})', T.FLOAT)
            return self.hold(f'({r} {high} {hi} ? {hi} : {r})', T.FLOAT), '1'
        if name in ('abs', 'absolute'):
            return self.hold(f'({x} != {x} ? -{x} : fabs({x}))' if abs_flips_nan() else f'fabs({x})', T.FLOAT), '1'
        if name == 'square':
            return self.hold(f'{x} * {x}', T.FLOAT), '1'
        if name == 'sign':                               # a NaN stays that NaN; both zeros give 0.0
            return self.hold(f'({x} > 0.0 ? 1.0 : {x} < 0.0 ? -1.0 : {x} == 0.0 ? 0.0 : {x})', T.FLOAT), '1'
        if name == 'copysign':
            return self.hold(f'copysign({x}, {xs[1]})', T.FLOAT), '1'
        if name in NP_MINMAX:
            return self.numpy_minmax(name, x, xs[1]), '1'
        c = NP_LIBM1.get(name) or NP_LIBM2[name]
        args = ', '.join(xs)
        if not self.exact:
            return self.hold(f'{c}({args})', T.FLOAT), '1'
        self.uses_libm = True
        macro = 'SJ_LIBM1' if len(xs) == 1 else 'SJ_LIBM2'
        return self.hold(f'{macro}({LIBM.index(c)})({args})', T.FLOAT), '1'

    def numpy_minmax(self, name, a, b):
        op = '<' if name in ('minimum', 'fmin') else '>'
        rule = zero_rule(name)
        equal = {'negative': f'(signbit({a}) ? {a} : {b})', 'positive': f'(signbit({a}) ? {b} : {a})', 'first': a, 'second': b}[rule]
        pick = f'({a} {op} {b} ? {a} : {b} {op} {a} ? {b} : {equal})'
        if name in ('minimum', 'maximum'):       # a NaN wins
            return self.hold(f'({a} != {a} ? {a} : {b} != {b} ? {b} : {pick})', T.FLOAT)
        # a NaN loses, and which of two NaNs comes back depends on the platform
        np = sys.modules.get('numpy')
        second = np is not None and math.copysign(1.0, float(getattr(np, name)(math.nan, -math.nan))) < 0
        return self.hold(f'({a} != {a} ? ({b} != {b} ? {b if second else a} : {b}) : {b} != {b} ? {a} : {pick})', T.FLOAT)

    def isinstance_of(self, e):
        """isinstance(x, C), as a constant: x an instance (its class is in the signature), or a number against int, float
        or bool; C a class at module level or a tuple of them. None if it isn't such a call."""
        if not (isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == 'isinstance' and 'isinstance' not in self.env
                and len(e.args) == 2 and not e.keywords):
            return None
        x, c = e.args
        classes = []
        for n in (c.elts if isinstance(c, ast.Tuple) else [c]):
            t = self.env.get(n.id) if isinstance(n, ast.Name) else None
            if t is not None and t.kind in ('classref', 'madeclass'):
                classes.append(t.elem)
            elif isinstance(n, ast.Name) and n.id in ('int', 'float', 'bool') and n.id not in self.env and n.id not in self.globals:
                classes.append({'int': int, 'float': float, 'bool': bool}[n.id])
            else:
                self.fail(e, 'isinstance() with something other than classes at module level, int, float or bool')
        if isinstance(x, ClassOf):
            return any(issubclass(x.cls, k) for k in classes)
        t = self.typeof(x)
        if t.is_number and t.np != 'dyn' and not T.is_narrow_dyn(t):
            py = ({'int': int, 'float': float, 'bool': bool}[t.kind]) if not t.np else None
            if py is None:
                import numpy
                py = {'int': numpy.int64, 'float': numpy.float64, 'bool': numpy.bool_}.get(t.kind) if t.ctype in ('int64_t', 'double', 'int') else None
                if py is None or t.name not in ('np.int64', 'np.float64', 'np.bool'):
                    self.fail(e, f'isinstance() of a {t.name}')
            return any(issubclass(py, k) for k in classes)
        self.fail(e, f'isinstance() of a {t.name}')

    def call_type(self, e):
        if isinstance(e.func, ast.Name) and e.func.id in ('__sj_deref__', '__sj_room__'):
            return T.INT
        if self.isinstance_of(e) is not None:
            return T.BOOL
        if self.callee_of(e) is not None:
            return self.callee_type(e)
        if self.c_function_of(e) is not None:
            return self.c_signature(e)[0]
        nf = self.numpy_func(e.func)
        if nf:
            return self.numpy_type(e, nf)
        reduced = self.reduction_of(e)
        if reduced:
            return self.np_reduction_type(e, *reduced)
        mf = self.math_func(e.func)
        if mf:
            return self.math_type(e, mf)
        if not isinstance(e.func, ast.Name):
            self.fail(e, f'calls {ast.unparse(e.func)}()')
        name = e.func.id
        if name == 'range' or not any(self.is_builtin(name, b) for b in BUILTINS):
            self.fail(e, f'calls {name}()')
        if self.is_reduction(e):
            return self.reduction_type(e)
        if e.keywords:
            self.fail(e, f'{name}() with keyword arguments')
        if name == 'len':
            a = e.args[0] if len(e.args) == 1 else None
            if a is not None and self.is_view(a) and self.typeof(a).kind == 'array':
                return T.INT
            if not (isinstance(a, ast.Name) and a.id in self.env and self.env[a.id].kind in ('array', 'list', 'tuple')):
                self.fail(e, 'len() of something other than an array, a list or a tuple')
            return T.INT
        args = [self.typeof(a) for a in e.args]
        if not args or not all(t.is_number for t in args):
            self.fail(e, f'{name}() of a non-number')
        if name == 'abs':
            if len(args) != 1 or (args[0].np and args[0].kind == 'bool'):
                self.fail(e, 'abs() with these arguments')
            return T.INT if args[0] is T.BOOL else args[0]
        if name in ('min', 'max'):
            if len(args) < 2:
                self.fail(e, f'{name}() of an iterable')
            return self.join_all(e, args, f'{name}()')
        if len(args) != 1:
            self.fail(e, f'{name}() with {len(args)} arguments')
        if name == 'round':
            if args[0].kind != 'float' and args[0].np is not False:
                self.fail(e, 'round() of a numpy int')
            return T.INT
        if any(t is T.NP_UINT64 for t in args) and name not in ('int', 'float', 'bool', 'abs', 'min', 'max'):
            self.fail(e, f'{name}() of np.uint64 (write int(x))')
        if name == 'int' and args[0] is T.NP_UINT64:
            return T.U64
        if name == 'round' and args[0] is T.U64:
            self.fail(e, 'round() of an int past int64')
        if name == 'int' and args[0] is T.U64:
            return T.U64
        return T.INT if name == 'int' else T.FLOAT

    # ---- reductions: sum, min, max, any, all over an iterable or a comprehension, fused into
    # one loop; each follows CPython's own algorithm (bltinmodule.c)
    def reduction_type(self, e):
        name = e.func.id
        kws = {k.arg: k.value for k in e.keywords}
        allowed = {'sum': {'start'}, 'min': {'default'}, 'max': {'default'}}.get(name, set())
        npos = len(e.args)
        if set(kws) - allowed or not (1 <= npos <= (2 if name == 'sum' else 1)) \
                or (npos == 2 and 'start' in kws):
            self.fail(e, f'{name}() with these arguments')
        el = self.source_elem_type(e.args[0])
        if name in ('any', 'all'):
            return T.BOOL
        if name in ('min', 'max'):
            if 'default' in kws:
                return self.join_all(e, [el, self.typeof(kws['default'])], f'{name}() with a default')
            return el
        start = e.args[1] if npos == 2 else kws.get('start')
        st = self.typeof(start) if start is not None else T.INT
        if not st.is_number or st.kind == 'bool' or st.np == 'dyn':
            self.fail(e, 'sum() with this start')
        if el.np and el.kind == 'bool':
            self.fail(e, 'sum() of numpy bools')
        if T.has_narrow(el, st):
            return self.narrow_sum_type(e, el, st, start)
        if st.np is True:
            return T.arith_result(st, el, 'Add')
        if st.kind == 'int':
            if el.kind == 'float':
                self.fail(e, 'sum() of floats returns the int 0 when empty, so its type depends on the '
                          'data; write sum(..., 0.0)')
            return T.INT if el.np is False else T.INT_DYN
        return T.FLOAT if el.np is False else T.FLOAT_DYN

    def source_elem_type(self, src):
        if isinstance(src, (ast.GeneratorExp, ast.ListComp)):
            return self.comp_types(src)
        t = self.elem_struct(self.no_rows(self.iterable(src), src))
        if isinstance(t, list):
            self.fail(src, 'a reduction over tuples')
        return t

    def comp_types(self, e):
        saved = {}
        try:
            for g in e.generators:
                if g.is_async:
                    self.fail(e, 'async comprehensions')
                self.comp_bind_types(g.target, self.elem_struct(self.no_rows(self.iterable(g.iter), g.iter)), saved, e)
                for c in g.ifs:
                    self.check_test(c)
            t = self.typeof(e.elt)
            if not t.is_number:
                self.fail(e.elt, f'elements of type {t.name}')
            return t
        finally:
            self.restore(saved)

    def comp_bind_types(self, target, struct, saved, e):
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(struct, T.Type) and struct.kind == 'tuple':
            struct = list(struct.elem)           # a tuple element, unpacked
        if isinstance(target, ast.Name) and isinstance(struct, T.Type) and struct.kind == 'tuple':
            self.fail(e, 'a comprehension variable that is a tuple (unpack it: for a, b in ...)')
        if isinstance(target, ast.Name) and not isinstance(struct, list):
            if target.id not in saved:
                saved[target.id] = self.env.get(target.id)
            self.env[target.id] = struct
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(struct, list) \
                and len(target.elts) == len(struct):
            for t, x in zip(target.elts, struct):
                self.comp_bind_types(t, x, saved, e)
        else:
            self.fail(e, 'a comprehension target that does not match what the loop yields')

    def restore(self, saved, saved_c=None):
        for k, v in saved.items():
            if v is None:
                self.env.pop(k, None)
            else:
                self.env[k] = v
        for k, v in (saved_c or {}).items():
            if v is None:
                self.cnames.pop(k, None)
            else:
                self.cnames[k] = v

    # ---- C emission
    def emit(self, text):
        self.lines.append('    ' * self.depth + text)

    def temp(self):
        self.ntemp += 1
        return f't{self.ntemp}'

    def mark(self, node):
        # not the file's name either: cimo's diagnostics name sj (this function) or sj<k> (the k-th it calls), which
        # cimo_diagnostic maps back
        self.lines.append(f'#line {self.line(node)} "{line_tag(getattr(self, "frame", None))}"')

    def plain_arg(self, param):
        """The caller's argument a parameter that isn't a dict's key stands for."""
        if self.plain is not None:
            return self.plain[param]
        before = 0
        taken = set()
        for p in self.params[:self.params.index(param)]:
            if p in self.dict_slots:
                taken.add(self.dict_slots[p][0])
            else:
                before += 1
        i = 0
        while True:
            if i not in taken:
                if before == 0:
                    return i
                before -= 1
            i += 1

    def root_of(self, p):
        """The position in array_args of an array parameter."""
        return [q for q in self.params if self.came_as(q).kind == 'array'].index(p)

    def came_as(self, param):
        """The type a parameter has on entry, whatever else it's assigned later."""
        return self.again[0][self.params.index(param)]

    def infer_with_webs(self):
        """A type conflict may be two variables with one name. Most functions never need to find out."""
        try:
            self.infer()
        except Unsupported as u:
            if 'must keep one type' not in u.reason:
                raise
            slots = self.dict_slots
            tree = _webs.split(self.f, set(slots)) or self.f
            self.__init__(tree, *self.again)
            self.dict_slots = slots
            self.split_done = True
            self.infer()
        self.check_tagged()
        self.find_outs()
        self.rebound = self.arrays_rebound()
        self.inferred = True

    def arrays_rebound(self):
        """The arrays whose extents can change while something that read them still holds. Not a local assigned once
        outside any loop: nothing can read it before that, or see it change after."""
        stores = [n.id for n in ast.walk(self.f) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                  and self.env.get(n.id) is not None and self.env[n.id].kind == 'array']
        looped = {n.id for loop in ast.walk(self.f) if isinstance(loop, (ast.For, ast.While)) for n in ast.walk(loop)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        return {n for n in stores if n in self.params or n in looped or stores.count(n) > 1}

    def add_safepoints(self, stmts):
        """After each statement that makes an array, and at the top of a loop in which one is made."""
        def makes(node):
            return any(isinstance(n, ast.Call) and (self.make_of(n) is not None or self.callee_of(n) is not None) for n in ast.walk(node))

        def point(near):
            return ast.copy_location(ast.Expr(ast.copy_location(Safepoint(None), near)), near)
        out = []
        for s in stmts:
            nested = [f for f in ('body', 'orelse') if getattr(s, f, None)]
            for f in nested:
                setattr(s, f, self.add_safepoints(getattr(s, f)))
            if isinstance(s, (ast.For, ast.While)) and makes(s):
                s.body.insert(0, point(s))
            out.append(s)
            if not nested and not isinstance(s, ast.Return) and makes(s):
                out.append(point(s))
        return out

    def generate(self):
        if not self.inferred:
            self.infer_with_webs()
        if self.makes:
            for n in ast.walk(self.f):
                if isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)) \
                        and any(isinstance(c, ast.Call) and (self.make_of(c) is not None or self.callee_of(c) is not None) for c in ast.walk(n)):
                    self.fail(n, 'an array made inside a comprehension')
            self.f.body = self.add_safepoints(self.f.body)
        da_block(self, self.f.body, set(self.params))
        self.find_aliases()
        self.facts.update(self.mod_assigned())
        params = []
        for p in self.params:
            t = self.env[p]
            if t.kind in ('function', 'enumclass', 'classref', 'madeclass'):
                continue
            if t.kind == 'array':
                params += [f'{t.ctype}v_{p}', f'int64_t v_{p}__n'] + [f'int64_t v_{p}__n{k}' for k in range(1, t.ndim)]
                if t.strided:
                    params += [f'int64_t v_{p}__s{k}' for k in range(t.ndim)]
            elif t.kind == 'tuple':
                params += [f'{et.ctype} {cv}' for cv, et in self.tuple_leaves(p)]
            elif t.kind == 'list':                   # a list passed in: a copy of it, written back by the call
                params += [f'{t.elem.ctype} *v_{p}', f'int64_t v_{p}__n']
            elif p in self.tagged and self.came_as(p) is T.INT:
                params.append(f'int64_t v_{p}__in')
            elif t is T.U64:
                params.append(f'int64_t v_{p}__in')
            else:
                params.append(f'{t.ctype} v_{p}')
                if self.came_as(p).np == 'dyn':     # from a call: the caller's variable and its flag
                    params.append(f'int v_{p}__np')
        self.mark(self.f)
        head = len(self.lines)
        if self.makes:
            self.helpers.add('np_make')
            self.emit(f'sj_heap sj_h = {{0, 0, 1048576, {sys.maxsize}ULL}};')
        for p in sorted(self.u64_params):
            self.uses_rerun = True           # a negative one runs as Python, where it can still go negative
            self.emit(f'if ({"v_" + p + "__in" if self.env[p] is T.U64 else "v_" + p} < 0) return {RERUN};')
        for p in self.params:
            if self.env[p] is T.U64:
                self.emit(f'uint64_t v_{p} = (uint64_t)v_{p}__in;')
        for p in self.params:
            if p in self.tagged and self.came_as(p) is T.INT:
                self.uses_rerun = True       # nothing has happened yet, so Python can still take this call
                self.emit(f'if ((uint64_t)v_{p}__in + 9007199254740992ULL > 18014398509481984ULL) return {RERUN};')
                self.emit(f'double v_{p} = (double)v_{p}__in; int v_{p}__int = 1;')
            elif p in self.tagged:
                self.emit(f'int v_{p}__int = 0;')
        for name, t in self.env.items():
            if t.kind == 'list' and name in self.params:
                continue
            if t.kind == 'list':
                self.emit(f'{t.elem.ctype} *v_{name} = NULL; int64_t v_{name}__n = 0;')
                if t.elem.np == 'dyn':
                    self.emit(f'int *v_{name}__np = NULL;')
                continue
            if t.kind == 'tuple':
                for cv, et in self.tuple_leaves(name):
                    if name not in self.params:
                        self.emit(f'{et.ctype} {cv} = 0;')
                    if et.np == 'dyn':
                        self.emit(f'int {cv}__np = 0;')
                continue
            if t.kind == 'array' and self.tracks:     # which argument it's a view of, and is it that very array?
                if name in self.params:
                    self.emit(f'int v_{name}__b = {self.root_of(name)}, v_{name}__o = 1;')
                else:
                    self.emit(f'int v_{name}__b = 0, v_{name}__o = 0;')
            if t.kind == 'array' and self.makes:
                self.emit(f'sj_block *v_{name}__h = 0;')
            if t.kind == 'array':
                if name not in self.params:           # a view: filled in where it's made
                    self.emit(f'{t.ctype}v_{name} = 0; int64_t ' + ', '.join(f'{self.extent(name, k)} = 0' for k in range(t.ndim)) + ';')
                if t.strided and name not in self.params:
                    self.emit('int64_t ' + ', '.join(f'v_{name}__s{k} = 0' for k in range(t.ndim)) + ';')
                continue
            if name not in self.params:
                self.emit(f'{t.ctype} v_{name} = 0;')
                if name in self.tagged:
                    self.emit(f'int v_{name}__int = 0;')
            if t.np == 'dyn' and not (name in self.params and self.came_as(name).np == 'dyn'):
                self.emit(f'int v_{name}__np = {1 if name in self.params and self.came_as(name).np is True else 0};')
        self.emit('int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;')
        decl_at = len(self.lines)
        if self.counts:
            self.plan_counts(self.f.body)
        for s in self.f.body:
            self.stmt(s)
        marks = ' '.join(f'if (v_{n}__h) v_{n}__h->mark = 1;' for n, t in self.env.items() if t.kind == 'array' and not n.startswith('sjh__'))
        self.lines = [ln.replace('/*sj_marks*/', marks) for ln in self.lines]
        frees = [f'free(v_{n});' for n, t in self.env.items() if t.kind == 'list' and n not in self.params] + \
                [f'free(v_{n}__np);' for n, t in self.env.items() if t.kind == 'list' and t.elem.np == 'dyn' and n not in self.params] + \
                [f'free({o});' for o, _ in self.owned]
        self.lines[decl_at:decl_at] = ['    ' + f'{ct} *{o} = NULL;' for o, ct in self.owned]
        self.lines.append('sj_done:')
        self.save_outs()
        if frees:
            self.emit(' '.join(frees))
        if self.makes:
            self.emit('sj_free_blocks(&sj_h, 1);')
        self.emit('return 0;')
        self.lines.append('sj_fail:')
        self.save_outs()
        if frees:
            self.emit(' '.join(frees))
        if self.makes:
            self.emit('sj_free_blocks(&sj_h, 0);')
        self.emit('sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;')
        self.emit('return sj_st;')
        if self.uses_libm:
            params.append('const uint64_t *sj_m')
        if self.uses_stop:
            params.append('volatile int *sj_stop')
        if self.counts:
            params.append('int64_t *restrict sj_c')
        if self.cname != 'sj_kernel':
            if self.uses_rerun:
                raise Unsupported(None, 'an int argument of it becomes a float' if self.tagged else
                                  'an int argument of it must not be negative, and a call from compiled code can\'t run it as Python')
            self.unit.helpers |= self.helpers
            return '\n'.join(self.lines[:head] + [f'static int {self.cname}({", ".join(params + ["int64_t *sj_tb", "int64_t *sj_io"])}) {{']
                             + self.lines[head:] + ['}', ''])
        params.append('int64_t *sj_io')
        start = ['    *sj_stop = 0;'] if self.uses_stop else []
        self.tb_slot = self.record_slot(len(ret_arrays(self.ret)))
        if self.calls:
            start.append(f'    int64_t *sj_tb = sj_io + {self.tb_slot}; sj_tb[0] = 0;')
        if self.unit.modules and self.unit.fma is not False and (self.helpers | self.unit.helpers) & CONTRACTING:
            raise Unsupported(None, 'math that must not contract a * b + c, with C that was compiled to: pass fma=False to cimo.compile')
        return self.unit.c_source() + snippets(self.helpers | self.unit.helpers) + '\n'.join(self.unit.code + self.lines[:head] + [f'int sj_kernel({", ".join(params)}) {{'] + start
                         + self.lines[head:] + ['}', ''])

    def plan_counts(self, stmts, by_test=False):
        """Statements that always run together share one counter, added to by the first of them: a store
        per statement slows a loop far more than one per run of statements. Anything that can jump away
        ends a run. Only an exception escapes this, so lines after the one that raised can read one too many.
        An if adds its test to its body's counter, and its else is what's left: a store inside a branch
        would stop it becoming a conditional move (a random branch: 0.69 to 4.6 ns). by_test: stmts are one."""
        slot = None
        for s in stmts:
            if isinstance(s, ast.Expr) and not self.calls_out(s.value):    # a docstring: stmt() emits nothing for it
                continue
            leads = slot is None and not by_test
            if slot is None:
                slot = len(self.count_lines)
                self.count_lines.append(set())
                by_test = False
            self.count_slot[id(s)] = (slot, leads)
            self.count_lines[slot].add(self.py_line(s))
            if isinstance(s, ast.If):
                body = self.first_count(s.body, True)
                if body is not None:
                    self.count_tests[id(s)] = body
                    self.count_heads[self.py_line(s)] = ('if', body)
                    other = self.first_count(s.orelse, True)
                    if other is not None:
                        self.count_rest[other] = (slot, body)
            elif isinstance(s, (ast.For, ast.While)):
                body = self.first_count(s.body)
                if body is not None:
                    self.count_heads[self.py_line(s)] = ('loop', body)
                self.first_count(s.orelse)
            if any(isinstance(n, (ast.Break, ast.Continue, ast.Return)) for n in ast.walk(s)):
                slot = None

    def first_count(self, stmts, by_test=False):
        """Plan a nested block; returns the counter of its first statement, if it has one."""
        self.plan_counts(stmts, by_test)
        first = next((b for b in stmts if not isinstance(b, ast.Expr)), None)
        return None if first is None else self.count_slot[id(first)][0]

    def count(self, s):
        if id(s) not in self.count_slot:          # written by the compiler as it went: its own counter
            self.count_slot[id(s)] = (len(self.count_lines), True)
            self.count_lines.append({self.py_line(s)})
        slot, leads = self.count_slot[id(s)]
        if leads:
            self.emit(f'sj_c[{slot}]++;')

    def block(self, stmts):
        self.depth += 1
        for s in stmts:
            self.stmt(s)
        self.depth -= 1

    def open(self, text):
        self.emit(text + ' {')
        self.depth += 1

    def poll(self):
        self.uses_stop = True
        return f'if (*sj_stop) SJ_FAIL({INTERRUPTED}, 0, 0, 0);'

    def open_counted(self, ctype, var, start, stop):
        """for (var = start; var < stop; var++) {. So that Ctrl-C can stop it, it runs in chunks with a
        check between them: a check inside the hot loop costs far more (ema 3.08 to 5.16 ns), and a loop
        that ends within its first chunk never reads the flag. A range known to be short needs neither,
        and its setup would cost per entry (3x3 stencil loops: 28%)."""
        short, self.short_range = self.short_range, False
        if not self.polls or short:
            self.open(f'for ({ctype} {var} = {start}; {var} < {stop}; {var}++)')
            self.loops.append(None)
            self.counted[self.depth] = None
            return
        end, lim = self.temp(), self.temp()
        self.labels += 1
        label = f'sj_end{self.labels}'
        self.emit(f'{ctype} {var} = {start}, {end} = {stop};')
        self.open('for (;;)')
        self.emit(f'{ctype} {lim} = {end} > {var} && (uint64_t){end} - (uint64_t){var} > 65536 ? {var} + 65536 : {end};')
        self.open(f'for (; {var} < {lim}; {var}++)')
        self.loops.append([label, False])
        self.counted[self.depth] = (var, end)

    def close(self, text='}'):
        info = self.counted.pop(self.depth, 0)
        self.depth -= 1
        self.emit(text)
        if info == 0:
            return
        loop = self.loops.pop()
        if info is None:
            return
        var, end = info
        self.emit(f'if ({var} >= {end}) break;')
        self.emit(self.poll())
        self.depth -= 1
        self.emit('}')
        if loop[1]:
            self.lines.append(f'{loop[0]}:;')

    def assign(self, name, v, vt):
        t = self.env[name]
        self.emit(f'v_{name} = {self.coerce(v[0], vt, t)};')
        if t.np == 'dyn' and not any(name in s for s in self.sunk):
            self.emit(f'v_{name}__np = {v[1]};')
        if name in self.tagged:
            self.emit(f'v_{name}__int = {1 if vt.kind == "int" else 0};')

    def stmt(self, s):
        if isinstance(s, ast.Expr) and not self.calls_out(s.value):     # a docstring, or a place to free arrays
            if isinstance(s.value, Safepoint):
                self.emit('if (sj_h.since > sj_h.limit) { /*sj_marks*/ sj_sweep(&sj_h); }')
            return
        self.call_stmt = any(self.calls_out(n) for n in ast.walk(s))
        self.mark(s)
        if self.counts:
            self.count(s)
        if isinstance(s, ast.Expr):
            self.call_statement(s.value)
            return
        if isinstance(s, ast.Assign) and self.is_list_ctor(s.value):
            self.make_list(s.targets[0].id, s.value, s)
            return
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            if self.has_views(s.value):
                self.assign_with_views(targets, s.value)
                return
            vt = self.typeof(s.value)
            if vt.kind == 'tuple' and ret_arrays(vt):
                self.assign_from_call(targets, s.value)
                return
            if vt.kind == 'tuple':
                vals = self.texpr(s.value)
                for target in targets:
                    self.assign_tuple(target, vals, vt)
                return
            if vt.kind == 'array':
                for target in targets:
                    if isinstance(target, ast.Subscript):
                        self.store_slice_array(target, self.materialize(s.value))
                    else:
                        self.assign_view(target.id, s.value)
                return
            c, f = self.expr(s.value)
            v = (self.hold(c, vt), self.hold_flag(f)) if len(targets) > 1 else (c, f)
            for target in targets:
                if isinstance(target, ast.Name):
                    self.assign(target.id, v, vt)
                elif self.typeof(target).kind == 'array':
                    self.store_slice_number(target, v, vt)
                elif isinstance(target, ast.Subscript):
                    self.store_elem(target, v, vt)
        elif isinstance(s, ast.AugAssign):
            op = ARITH[type(s.op)]
            if isinstance(s.target, ast.Name):
                name = s.target.id
                ct = self.env[name]
                cur = (f'v_{name}', f'v_{name}__np' if ct.np == 'dyn' else flag_of(ct))
                rt = self.binop_type(ct, self.typeof(s.value), op, s)
                self.assign(name, self.arith(op, cur, ct, s.value, s), rt)
            else:
                target = self.named(s.target)
                arr, idx = self.element(target)       # checked once, before the right side
                name = target.value.id
                at = self.env[name]
                rt = self.binop_type(at.elem, self.typeof(s.value), op, s)
                r = self.arith(op, (self.hold(f'{arr}[{idx}]', at.elem), self.elem_flag(name, idx)), at.elem, s.value, s)
                self.emit(f'{arr}[{idx}] = {self.coerce(r[0], rt, at.elem)};')
                if at.kind == 'list' and at.elem.np == 'dyn':
                    self.emit(f'v_{name}__np[{idx}] = {r[1]};')
        elif isinstance(s, ast.If):
            test = self.truth(self.expr(s.test)[0], self.typeof(s.test))
            if id(s) in self.count_tests:
                t = self.temp()
                self.emit(f'int {t} = ({test}) != 0;')
                self.emit(f'sj_c[{self.count_tests[id(s)]}] += {t};')
                test = t
            self.open(f'if ({test})')
            saved = self.with_facts(self.guard_facts(s.test, s.body))
            for b in s.body:
                self.stmt(b)
            self.drop_facts(saved)
            if s.orelse:
                self.close('} else {')
                self.depth += 1
                for b in s.orelse:
                    self.stmt(b)
            self.close()
        elif isinstance(s, (ast.While, ast.For)):
            flag = self.temp() if s.orelse else None     # an else runs unless the loop was left by a break
            if flag:
                self.emit(f'int {flag} = 0;')
            self.broke.append(flag)
            sunk = self.sinkable_flags(s) if isinstance(s, ast.For) else set()
            self.sunk.append(sunk)
            self.trips.append(None)
            if isinstance(s, ast.While):
                self.while_loop(s)
            elif self.iterable(s.iter)[0] == 'range':
                self.for_range(s)
            else:
                self.for_iter(s, self.iterable(s.iter))
            self.sunk.pop()
            trip = self.trips.pop()
            if sunk:                         # each was assigned a numpy value on every iteration, if there was one
                assert trip is not None
                self.emit(f'if ({trip}) {{ ' + ' '.join(f'v_{n}__np = 1;' for n in sorted(sunk)) + ' }')
            self.broke.pop()
            if flag:
                self.open(f'if (!{flag})')
                for b in s.orelse:
                    self.stmt(b)
                self.close()
        elif isinstance(s, ast.Return) and self.ret.kind == 'list':
            name = s.value.id
            if name in self.params:          # the list passed in, which Python returns itself
                self.emit(f'sj_io[5] = {self.params.index(name) + 1};')
            else:                            # its memory is the caller's now: copied into a list, then released
                self.helpers.add('np_make')
                self.emit(f'sj_io[0] = (int64_t)(uintptr_t)v_{name}; sj_io[4] = v_{name}__n; sj_io[5] = 0; v_{name} = NULL;')
            self.emit('goto sj_done;')
        elif isinstance(s, ast.Return) and self.ret.kind == 'array':
            self.return_array(s.value, 0)
            self.emit('goto sj_done;')
        elif isinstance(s, ast.Return) and ret_arrays(self.ret) and self.callee_of(s.value) is not None:
            self.return_from_call(s.value)
        elif isinstance(s, ast.Return) and ret_arrays(self.ret):
            k = 0
            for i, (x, et, rt) in enumerate(zip(s.value.elts, self.typeof_return(s.value), self.ret.elem)):
                if rt.kind == 'array':
                    self.return_array(x, k)
                    k += 1
                    continue
                c, f = self.expr(x)
                self.return_number(i, c, f, et, rt, x)
            self.emit('goto sj_done;')
        elif isinstance(s, ast.Return):
            if s.value is not None and self.ret.kind == 'tuple':
                vals = self.texpr(s.value)
                given = s.value.elts if isinstance(s.value, ast.Tuple) and len(T.leaves(self.ret)) == len(self.ret.elem) else None
                for i, ((c, f), et, rt) in enumerate(zip(flat(vals), T.leaves(self.typeof(s.value)), T.leaves(self.ret))):
                    self.return_number(i, c, f, et, rt, given[i] if given else None)
                self.emit('goto sj_done;')
            elif s.value is not None:
                v = self.expr(s.value)
                val = self.coerce(v[0], self.typeof(s.value), self.ret)
                if self.ret.kind == 'float':
                    t = self.temp()
                    self.emit(f'{{ double {t} = {val}; memcpy(&sj_io[0], &{t}, 8); }}')
                else:
                    self.emit(f'sj_io[0] = (int64_t)({val});')
                if self.ret.np == 'dyn':
                    self.emit(f'sj_io[2] = {self.ret_flag(s.value, v[1])};')
            self.emit('goto sj_done;')
        elif isinstance(s, ast.Break):
            if self.broke[-1]:
                self.emit(f'{self.broke[-1]} = 1;')
            loop = self.loops[-1]
            if loop is None:
                self.emit('break;')
            else:                                       # out of the chunks as well
                loop[1] = True
                self.emit(f'goto {loop[0]};')
        elif isinstance(s, ast.Continue):
            self.emit('continue;')

    def typeof_return(self, v):
        return [self.typeof(x) for x in v.elts]

    def return_number(self, i, c, f, et, rt, node):
        val = self.coerce(c, et, rt)
        if rt.kind == 'float':
            t = self.temp()
            self.emit(f'{{ double {t} = {val}; memcpy(&sj_io[{4 + 2 * i}], &{t}, 8); }}')
        else:
            self.emit(f'sj_io[{4 + 2 * i}] = (int64_t)({val});')
        self.emit(f'sj_io[{5 + 2 * i}] = {self.ret_flag(node, f)};')

    def return_array(self, value, k):
        src = self.materialize(value)
        t = self.env[src]
        at = self.record_slot(k)
        self.emit(f'sj_io[{at}] = (int64_t)(uintptr_t)v_{src}; sj_io[{at + 1}] = v_{src}__b; sj_io[{at + 2}] = v_{src}__o;')
        self.emit(f'sj_io[{at + 3}] = 0;')
        if self.makes:
            self.open(f'if (v_{src}__h)')
            self.emit(f'sj_block *b_ = v_{src}__h; b_->mark = 2;')
            self.emit(f'sj_io[{at + 3}] = (int64_t)(uintptr_t)b_; sj_io[{at + 4}] = (int64_t)(uintptr_t)(b_ + 1); sj_io[{at + 5}] = b_->ndim;')
            for d in range(MAX_MADE_DIMS):
                self.emit(f'sj_io[{at + 6 + d}] = b_->shape[{d}];')
            self.close()
        for d in range(t.ndim):
            self.emit(f'sj_io[{at + 10 + 2 * d}] = {self.extent(src, d)}; sj_io[{at + 11 + 2 * d}] = {self.stride(src, d)};')

    def while_loop(self, s):
        self.open('for (;;)')
        self.loops.append(None)
        if self.polls:
            self.emit(self.poll())
        self.emit(f'if (!({self.truth(self.expr(s.test)[0], self.typeof(s.test))})) break;')
        for b in s.body:
            self.stmt(b)
        self.loops.pop()
        self.close()

    def written_lists(self):
        """The positions of the list parameters this function stores into, which the call copies back."""
        stored = {n.value.id for n in ast.walk(self.f) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                  and isinstance(n.value, ast.Name)}
        any_store = any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) for n in ast.walk(self.f))
        return {self.plain_arg(p) for p in self.params if p not in self.dict_slots and ((self.env[p].kind == 'list' and p in stored)
                or (T.is_pylist(self.env[p]) and any_store))}     # a row is a view: a store through it writes the list

    def written_columns(self):
        """For a list of instances: (argument index, attribute) of each attribute's list it stores into."""
        stored = {n.value.id for n in ast.walk(self.f) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                  and isinstance(n.value, ast.Name)}
        stored |= self.ref_written               # given by reference to a callee that writes it
        return {self.dict_slots[p][:2] for p in self.params if p in self.dict_slots and p in stored and self.env[p].kind == 'list'}

    def sinkable_flags(self, s):
        """Variables of this for loop whose numpy flag can be set once after it instead of on each iteration, which keeps a
        constant store out of the loop so cimo can vectorize it (s = 0, then s += x[i]): every assignment in the loop gives a
        numpy value, one at the top of the body runs on every iteration, and the loop never reads the flag, since every read
        meets another numpy value. No break, continue or return, so an iteration that ran made the assignment."""
        body = list(ast.walk(ast.Module(body=s.body, type_ignores=[])))
        if any(isinstance(n, (ast.Break, ast.Continue, ast.Return)) for n in body):
            return set()
        outs = {flat for flat, _, _ in self.outs} if getattr(self, 'outs', None) else set()
        parents = {id(ch): p for p in body for ch in ast.iter_child_nodes(p)}
        top = {x.id for st in s.body for x in ([st.target] if isinstance(st, ast.AugAssign) else st.targets if isinstance(st, ast.Assign) else [])
               if isinstance(x, ast.Name)}
        found = set()
        for name in top:
            t = self.env.get(name)
            if t is None or t.np != 'dyn' or name in self.tagged or name in outs or name in self.params:
                continue
            ok = True
            for n in body:
                binds = [n.target] if isinstance(n, (ast.For, ast.comprehension, ast.NamedExpr, ast.AugAssign)) else \
                    n.targets if isinstance(n, ast.Assign) else [i.optional_vars for i in n.items] if isinstance(n, ast.With) else []
                if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in n.targets):
                    ok = ok and len(n.targets) == 1 and self.typeof(n.value).np is True
                elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == name:
                    ok = ok and self.typeof(n.value).np is True
                elif any(x is not None and any(isinstance(y, ast.Name) and y.id == name for y in ast.walk(x)) for x in binds):
                    ok = False                       # a loop target, unpacking...
                elif isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load):
                    p = parents.get(id(n))
                    other = (p.right if p.left is n else p.left) if isinstance(p, ast.BinOp) else None
                    ok = ok and other is not None and self.typeof(other).np is True
            if ok:
                found.add(name)
        return found

    def ret_flag(self, e, flag):
        """A returned value's flag: 0 a Python float or int, 1 numpy, RETURNS_INT an int held as a double."""
        if isinstance(e, ast.Name) and e.id in self.tagged:
            return f'(v_{e.id}__int ? {RETURNS_INT} : {flag})'
        return flag

    def for_range(self, s):
        args = [self.coerce(self.expr(a)[0], self.typeof(a), T.INT) for a in s.iter.args]
        var = s.target.id
        start, stop = ('0LL', args[0]) if len(args) == 1 else (args[0], args[1])
        step = args[2] if len(args) == 3 else None
        ts, te = self.temp(), self.temp()
        self.emit(f'int64_t {ts} = {start}, {te} = {stop};')
        unit = step is None or (isinstance(s.iter.args[2], ast.Constant) and s.iter.args[2].value == 1)
        k = s.iter.args[2] if len(s.iter.args) == 3 else None
        if unit or (isinstance(k, ast.Constant) and type(k.value) is int and k.value > 0):
            self.trips[-1] = f'{te} > {ts}'
        bounds = [const_number(a) for a in s.iter.args]
        short = None not in bounds and (bounds[1] - bounds[0] if len(bounds) > 1 else bounds[0]) <= 65536 \
            and (len(bounds) < 3 or bounds[2] > 0)
        fact = self.range_fact(s) if unit else None
        if self.innermost(s) and (not unit or (fact is None or fact[0] is None)) and self.version_wider(s, var, ts, te, unit, short):
            return
        if unit and fact:
            # loop versioning: if no access var + c can leave its array during this loop, run a
            # copy without those checks; otherwise the checked copy raises where Python would
            extra = self.version_candidates(s, var, fact)
            if extra and self.may_version(s):
                cond = ' && '.join(self.version_test(key, c, te) for key, c in sorted(extra.items(), key=str))
                self.open(f'if ({te} > {ts} && {cond})')
                self.unit_loop(s, var, ts, te, fact + (extra,), short)
                self.close('} else {')
                self.depth += 1
                self.slow_copy += 1
                self.unit_loop(s, var, ts, te, fact, short)
                self.slow_copy -= 1
                self.close()
                return
        if unit:
            self.unit_loop(s, var, ts, te, fact, short)
            return
        tp, tn, tk = self.temp(), self.temp(), self.temp()
        self.emit(f'int64_t {tp} = {step};')
        self.emit(f'if ({tp} == 0) SJ_FAIL({VALUE_ERROR}, {self.line(s)}, 1, 0);')
        self.emit(f'uint64_t {tn} = {tp} > 0 ? ({ts} < {te} ? ((uint64_t){te} - (uint64_t){ts} - 1) / (uint64_t){tp} + 1 : 0)'
                  f' : ({ts} > {te} ? ((uint64_t){ts} - (uint64_t){te} - 1) / (0 - (uint64_t){tp}) + 1 : 0);')
        self.trips[-1] = f'{tn} > 0'
        self.short_range = short
        self.open_counted('uint64_t', tk, '0', tn)
        self.emit(f'v_{var} = (int64_t)((uint64_t){ts} + {tk} * (uint64_t){tp});')
        k = s.iter.args[2]                    # a positive literal step keeps range()'s facts: var starts at start, stays below stop
        positive = isinstance(k, ast.Constant) and type(k.value) is int and k.value > 0
        self.loop_body(s, var, self.range_fact(s) if positive else None)

    @staticmethod
    def innermost(s):
        return not any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(ast.Module(body=s.body, type_ignores=[])))

    def version_wider(self, s, var, ts, te, unit, short):
        """Versioning for an innermost loop with a positive literal step, or a start that isn't known to be nonnegative: the fast
        copy may assume var >= 0 when the test checks start >= 0, and the test uses the last value var takes, not stop
        (range(0, n, 2) with q[i + 1] is in bounds because i stops at n - 2). Emits both copies and returns True, or emits
        nothing and returns False."""
        args = s.iter.args
        step = None
        if not unit:
            k = args[2]
            if not (isinstance(k, ast.Constant) and type(k.value) is int and k.value > 0):
                return False
            step = k.value
        known = self.range_bounds(s)
        if known is None:
            return False
        lo, hi, slack = known
        fast = (lo if lo is not None and lo >= 0 else 0, hi, slack)
        extra = self.version_candidates(s, var, fast)
        if not extra or not self.may_version(s):
            return False
        tests = [] if lo is not None and lo >= 0 else [f'{ts} >= 0']
        if step is None:
            last = te
        else:                                     # one past the last value var takes
            last = self.temp()
            self.emit(f'int64_t {last} = {te} > {ts} ? (int64_t)((uint64_t){ts} + ((uint64_t){te} - (uint64_t){ts} - 1) / {step}ULL * {step}ULL) + 1 : {ts};')
        tests += [self.version_test(key, c, last) for key, c in sorted(extra.items(), key=str)]
        self.open(f'if ({te} > {ts} && {" && ".join(tests)})')
        self.stepped_loop(s, var, ts, te, step, fast + (extra,), short)
        self.close('} else {')
        self.depth += 1
        self.slow_copy += 1
        self.stepped_loop(s, var, ts, te, step, self.range_fact(s) if step is None else None, short)
        self.slow_copy -= 1
        self.close()
        return True

    def stepped_loop(self, s, var, ts, te, step, fact, short):
        if step is None:
            self.unit_loop(s, var, ts, te, fact, short)
            return
        tn, tk = self.temp(), self.temp()
        self.emit(f'uint64_t {tn} = {ts} < {te} ? ((uint64_t){te} - (uint64_t){ts} - 1) / {step}ULL + 1 : 0;')
        self.short_range = short
        self.open_counted('uint64_t', tk, '0', tn)
        self.emit(f'v_{var} = (int64_t)((uint64_t){ts} + {tk} * {step}ULL);')
        self.loop_body(s, var, fact)

    def may_version(self, s):
        """Is this loop one versioning may copy (plan_versioning)?"""
        if self.versionable is None:
            self.versionable = self.plan_versioning()
        return id(s) in self.versionable

    def plan_versioning(self):
        """The loops whose bodies versioning may copy, within a budget of syntax nodes: the most deeply nested
        first, since they run most often, and of those the first. A loop's cost includes the copies of loops chosen
        inside it, which a copy of it copies again."""
        loops = []

        def walk(node, depth):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.For):
                    loops.append((depth, len(loops), child))
                walk(child, depth + isinstance(child, (ast.For, ast.While)))
        walk(self.f, 0)
        chosen, cost = set(), {}
        budget = max(VERSION_MIN, VERSION_TOTAL - sum(1 for _ in ast.walk(self.f)))
        for depth, _, loop in sorted(loops, key=lambda x: (-x[0], x[1])):
            inner = sum(cost[id(n)] for n in ast.walk(ast.Module(body=loop.body, type_ignores=[])) if id(n) in chosen)
            size = sum(1 for _ in ast.walk(ast.Module(body=loop.body, type_ignores=[]))) + inner
            if size <= budget:
                chosen.add(id(loop))
                cost[id(loop)] = size
                budget -= size
        return chosen

    def iter_prep(self, d):
        """Evaluate an iterable's operands (in Python's order); return (trip count, elem(k)),
        where elem(k) gives (C value, type, flag) leaves, nested in lists for tuples."""
        kind = d[0]
        if kind == 'rows':
            name = self.materialize(d[2]) if d[2] is not None else d[1]
            first = self.hold(self.stride(name, 0), T.INT)
            return f'(uint64_t){self.hold(self.extent(name, 0), T.INT)}', lambda k: RowOf(name, k, first)
        if kind == 'array':
            if len(d) > 2:
                self.materialize(d[2])
            name, t = d[1], self.env[d[1]].elem
            if self.env[name].strided:
                return f'(uint64_t)v_{name}__n', lambda k: (f'v_{name}[(int64_t){k} * v_{name}__s0]', t, '1')
            return f'(uint64_t)v_{name}__n', lambda k: (f'v_{name}[{k}]', t, self.elem_flag(name, k))
        if kind == 'range':
            args = [self.coerce(self.expr(a)[0], self.typeof(a), T.INT) for a in d[1].args]
            start, stop = ('0LL', args[0]) if len(args) == 1 else (args[0], args[1])
            ts, te, tp, tn = self.temp(), self.temp(), self.temp(), self.temp()
            self.emit(f'int64_t {ts} = {start}, {te} = {stop}, {tp} = {args[2] if len(args) == 3 else "1LL"};')
            if len(args) == 3:
                self.emit(f'if ({tp} == 0) SJ_FAIL({VALUE_ERROR}, {self.line(d[1])}, 1, 0);')
            self.emit(f'uint64_t {tn} = {tp} > 0 ? ({ts} < {te} ? ((uint64_t){te} - (uint64_t){ts} - 1) / (uint64_t){tp} + 1 : 0)'
                      f' : ({ts} > {te} ? ((uint64_t){ts} - (uint64_t){te} - 1) / (0 - (uint64_t){tp}) + 1 : 0);')
            self.range_parts = (ts, tp, tn)
            return tn, lambda k: (f'(int64_t)((uint64_t){ts} + {k} * (uint64_t){tp})', T.INT, '0')
        if kind == 'enumerate':
            n, inner = self.iter_prep(d[1])
            st = self.temp()
            start = '0LL' if d[2] is None else self.coerce(self.expr(d[2])[0], self.typeof(d[2]), T.INT)
            self.emit(f'int64_t {st} = {start};')
            return n, lambda k: [(f'sj_add({st}, (int64_t){k})', T.INT, '0'), inner(k)]
        if kind == 'tuple':                          # all of it first, as Python builds it; element k by a select
            vals, t = self.texpr(d[1]), d[2]
            n = len(vals)

            def pick(k, at):
                parts = [flat([v])[at] if not isinstance(v, list) else flat(v)[at] for v in vals]
                c = parts[-1][0]
                f = parts[-1][1]
                for i in range(n - 2, -1, -1):
                    c = f'((uint64_t){k} == {i} ? {parts[i][0]} : {c})'
                    f = f'((uint64_t){k} == {i} ? {parts[i][1]} : {f})' if f != parts[i][1] else f
                return c, f

            def rebuild(et, k, at):
                if et.kind != 'tuple':
                    return pick(k, at[0]), at.__setitem__(0, at[0] + 1)
                out = []
                for x in et.elem:
                    out.append(rebuild(x, k, at)[0])
                return out, None

            def elem(k):
                if t.kind == 'tuple':
                    return TupleOf(rebuild(t, k, [0])[0], t)
                c, f = pick(k, 0)
                return c, t, f
            return f'{n}ULL', elem
        if kind == 'zip':
            parts = [self.iter_prep(x) for x in d[1]]
            n = self.temp()
            self.emit(f'uint64_t {n} = {parts[0][0]};')
            for c, _ in parts[1:]:
                self.emit(f'if ({c} < {n}) {n} = {c};')
            return n, lambda k: [f(k) for _, f in parts]
        inner = d[1]                                  # reversed
        if inner[0] == 'array':
            if len(inner) > 2:
                self.materialize(inner[2])
            name, t = inner[1], self.env[inner[1]].elem
            if self.env[name].strided:
                return f'(uint64_t)v_{name}__n', lambda k: (f'v_{name}[(v_{name}__n - 1 - (int64_t){k}) * v_{name}__s0]', t, '1')
            return f'(uint64_t)v_{name}__n', lambda k: (f'v_{name}[v_{name}__n - 1 - (int64_t){k}]', t,
                                                        self.elem_flag(name, f'(v_{name}__n - 1 - (int64_t){k})'))
        n, _ = self.iter_prep(inner)
        ts, tp, tn = self.range_parts
        return n, lambda k: (f'(int64_t)((uint64_t){ts} + ({tn} - 1 - {k}) * (uint64_t){tp})', T.INT, '0')

    def bind_values(self, target, v):
        if isinstance(v, RowOf):
            self.bind_row(target.id, v)
        elif isinstance(v, TupleOf):
            if isinstance(target, ast.Name):
                self.assign_tuple(target, v.vals, v.t)
            else:
                for x, val, et in zip(target.elts, v.vals, v.t.elem):
                    self.bind_values(x, TupleOf(val, et) if et.kind == 'tuple' else (val[0], et, val[1]))
        elif isinstance(target, ast.Name):
            c, t, f = v
            self.assign(target.id, (c, f), t)
        else:
            for t, x in zip(target.elts, v):
                self.bind_values(t, x)

    def bind_row(self, target, row):
        """A loop variable that is a view of each first-axis element in turn."""
        t = self.env[target]
        self.emit(f'v_{target} = v_{row.name} + (int64_t){row.k} * {row.first};')
        self.same_root(target, row.name, 0)
        for axis in range(t.ndim):
            self.emit(f'{self.extent(target, axis)} = {self.extent(row.name, axis + 1)};')
            if t.strided:
                self.emit(f'v_{target}__s{axis} = {self.stride(row.name, axis + 1)};')

    def for_iter(self, s, d):
        n, elem = self.iter_prep(d)
        n = self.hold(n, T.U64)                   # the trip count, also for the test after the loop
        self.trips[-1] = f'{n} > 0'
        k = self.temp()
        self.open_counted('uint64_t', k, '0', n)
        self.bind_values(s.target, elem(k))
        fact_var, fact = self.iter_fact(s, d)
        saved = self.facts.get(fact_var) if fact_var else None
        if fact:
            self.facts[fact_var] = fact
        for b in s.body:
            self.stmt(b)
        if fact:
            self.facts.pop(fact_var, None)
            if saved:
                self.facts[fact_var] = saved
        self.close()

    def iter_fact(self, s, d):
        """for i, x in enumerate(a[, c]): i is c + k, so >= c, and below len(a) when c is 0."""
        if d[0] != 'enumerate' or not isinstance(s.target, ast.Tuple) or not isinstance(s.target.elts[0], ast.Name):
            return None, None
        var = s.target.elts[0].id
        for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
            if isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store):
                return None, None
        start = d[2]
        lo = 0 if start is None else (start.value if isinstance(start, ast.Constant) and type(start.value) is int else None)
        if lo is None:
            return None, None
        inner = d[1]
        if inner[0] == 'zip':
            inner = inner[1][0]
        hi = (inner[1], 0) if inner[0] == 'array' and lo == 0 and inner[1] not in self.rebound else None
        return var, (lo, hi, 0)

    def unit_loop(self, s, var, ts, te, fact, short=False):
        ti = self.temp()
        found = None if self.no_forwarding else self.forwarded(s, var, fact)
        if found and found[2] and not self.may_version(s):
            found = None                       # no room for a second copy
        if found and found[2]:                 # other arrays are stored to: forward only where none shares y's memory
            y = found[0]
            def span(a):
                n = ' * '.join(['1LL'] + [self.extent(a, k) for k in range(self.env[a].ndim)])
                return f'(char *)v_{a}', f'(char *)v_{a} + ({n}) * (int64_t)sizeof({self.env[a].elem.ctype})'
            y0, y1 = span(y)
            apart = ' && '.join(f'({o1} <= {y0} || {y1} <= {o0})' for o0, o1 in map(span, found[2]))
            self.open(f'if ({apart})')
            self.forwarding_loop(s, var, ts, te, fact, short, ti, found)
            self.close('} else {')
            self.depth += 1
            self.no_forwarding = True
            self.unit_loop(s, var, ts, te, fact, short)
            self.no_forwarding = False
            self.close()
            return
        self.forwarding_loop(s, var, ts, te, fact, short, ti, found)

    def forwarding_loop(self, s, var, ts, te, fact, short, ti, found):
        name = found[0] if found else None
        if found:                              # the first y[i - 1] to y[i - k] really are read from memory
            temps = []
            for k in range(1, found[1] + 1):
                t = self.temp()
                self.emit(f'{self.env[name].elem.ctype} {t} = {te} > {ts} && {ts} - {k} < {self.extent(name, 0)} '
                          f'? v_{name}[{ts} - {k}] : 0;')
                temps.append(t)
            self.forwards[name] = (var, temps)
        self.short_range = short
        self.open_counted('int64_t', ti, ts, te)
        self.emit(f'v_{var} = {ti};')
        self.loop_body(s, var, fact)
        self.forwards.pop(name, None)

    def forwarded(self, s, var, fact):
        """(y, k, others) when this loop can read y[var - 1] to y[var - k] from temporaries instead of memory,
        because earlier iterations just stored them: y's one store in the loop is y[var] = ..., on every
        iteration. Another store could change those elements only through an array sharing y's memory:
        others are the arrays stored to, which a test before the loop must find apart from y (contiguous,
        and not assigned in the loop, so their memory is known there). They're read only up to y's store.
        The dtype holds exactly what the C value does. Otherwise None."""
        if not fact or fact[0] is None:
            return None
        body = list(ast.walk(ast.Module(body=s.body, type_ignores=[])))
        if any(isinstance(n, (ast.For, ast.While, ast.comprehension, ast.Continue)) or self.calls_out(n) for n in body):
            return None                        # a call can write any array it's given
        stores = [n for n in body if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)]
        if not stores or not all(isinstance(n.value, ast.Name) for n in stores):
            return None
        assigned = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        read_back = [n.value.id for n in stores if any(self.is_previous(r, n.value.id, var) for r in body)]
        if len(read_back) != 1:
            return None
        name = read_back[0]
        if sum(n.value.id == name for n in stores) != 1:
            return None
        target = next(n for n in stores if n.value.id == name)
        others = sorted({n.value.id for n in stores} - {name})
        for o in others:
            ot = self.env.get(o)
            if ot is None or o in assigned or o in self.rebound or (ot.kind == 'array' and ot.strided) or ot.kind not in ('array', 'list'):
                return None
        others = [o for o in others if self.env[o].kind == 'array']     # a list is memory of its own
        t = self.env.get(name)
        if t is None or t.kind != 'array' or t.ndim != 1 or t.strided or name in self.rebound \
                or t.elem.name not in ('np.float64', 'np.int64') \
                or not (isinstance(target.slice, ast.Name) and target.slice.id == var):
            return None
        at = [k for k, b in enumerate(s.body) if isinstance(b, ast.Assign) and b.targets == [target]]
        if not at:
            return None                        # inside an if, or not a plain assignment

        def reads(nodes):
            return [n for b in nodes for n in ast.walk(b) if self.is_previous(n, name, var)]
        if not reads(s.body[:at[0] + 1]) or reads(s.body[at[0] + 1:]):
            return None
        self.facts[var] = fact
        try:
            for r in reads(s.body):
                nonneg, inbounds = self.index_facts(name, r.slice, 0)
                if not (nonneg and (inbounds or not self.boundscheck)):
                    return None
        finally:
            self.facts.pop(var, None)
        return name, max(self.is_previous(r, name, var) for r in reads(s.body)), others

    def is_previous(self, e, name, var):
        """How far back e is, if it's the element name[var - k] for k from 1 to FORWARD_DEPTH: k, else 0."""
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id == name \
                and isinstance(e.slice, ast.BinOp) and isinstance(e.slice.op, ast.Sub) \
                and isinstance(e.slice.left, ast.Name) and e.slice.left.id == var \
                and isinstance(e.slice.right, ast.Constant) and type(e.slice.right.value) is int \
                and 1 <= e.slice.right.value <= FORWARD_DEPTH:
            return e.slice.right.value
        return 0

    def loop_body(self, s, var, fact):
        if self.env[var].np == 'dyn':
            self.emit(f'v_{var}__np = 0;')
        saved = self.facts.get(var)
        if fact:
            self.facts[var] = fact
        hoisted = self.hoist_wraps(s, var) if fact else []
        for b in s.body:
            self.stmt(b)
        for key in hoisted:
            self.hoisted.pop(key, None)
        self.facts.pop(var, None)
        if saved:
            self.facts[var] = saved
        self.close()

    def version_test(self, key, c, te):
        """The C test that makes one versioning candidate safe for a loop stopping at te."""
        n = self.extent(key[0], key[1])
        if len(key) == 2:
            return f'{te} <= {n} - ({c})'
        if key[2] == 'mod':
            return f'{self.extent(*key[3])} <= {n}'
        # base + var, with var >= 0: base >= 0 and var < te <= n - base give 0 <= base + var < n
        # with no overflow, and neither can this test overflow, since 0 <= n and 0 <= base
        b = self.hold(self.expr(c)[0], T.INT)
        return f'{b} >= 0 && {te} <= {n} - {b}'

    def index_base(self, idx, var):
        """For an index var + base, where base is +, - and * of Python ints and int literals not
        involving var: base, as an expression. Compiled ints wrap, so any order gives the same
        value, and evaluating base early can't raise. None for any other index."""
        terms = []

        def flatten(e, sign):
            if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
                flatten(e.left, sign)
                flatten(e.right, sign if isinstance(e.op, ast.Add) else -sign)
            else:
                terms.append((e, sign))

        def pure(e):
            if isinstance(e, ast.Constant):
                return type(e.value) is int and INT64_MIN <= e.value <= INT64_MAX
            if isinstance(e, ast.Name):
                return e.id != var and e.id not in self.cnames and self.env.get(e.id) is T.INT
            if isinstance(e, ast.BinOp):
                return isinstance(e.op, (ast.Add, ast.Sub, ast.Mult)) and pure(e.left) and pure(e.right)
            return False

        flatten(idx, 1)
        at = [k for k, (e, sign) in enumerate(terms) if isinstance(e, ast.Name) and e.id == var and sign > 0]
        if len(terms) < 2 or len(at) != 1:
            return None
        rest = terms[:at[0]] + terms[at[0] + 1:]
        if not all(pure(e) for e, _ in rest) or all(isinstance(e, ast.Constant) for e, _ in rest):
            return None                 # var + c has its own candidates, using the loop's start
        base = rest[0][0] if rest[0][1] > 0 else ast.BinOp(ast.Constant(0), ast.Sub(), rest[0][0])
        for e, sign in rest[1:]:
            base = ast.BinOp(base, ast.Add() if sign > 0 else ast.Sub(), e)
        ast.copy_location(base, idx)
        return ast.fix_missing_locations(base)

    def version_candidates(self, s, var, fact):
        """Indexes var + c on some axis of an array, nonnegative but not proven below that axis's
        extent: the largest c for each (array, axis). In an innermost loop whose variable starts at
        0 or above, also indexes var + base, with base unchanged in the loop: (array, axis, 'base',
        base's dump) -> base. Those drop the negative-index test too, so they apply with
        boundscheck=False as well. Only innermost loops, so copies don't multiply."""
        self.facts[var] = fact
        extra = {}
        body = list(ast.walk(ast.Module(body=s.body, type_ignores=[])))
        stored = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        innermost = fact[0] is not None and fact[0] >= 0 and not any(
            isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in body)
        try:
            for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
                if not (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                        and n.value.id in self.env and self.env[n.value.id].kind == 'array'):
                    continue
                name = n.value.id
                idxs = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
                if len(idxs) != self.env[name].ndim or name in stored or any(isinstance(k, ast.Slice) for k in idxs):
                    continue
                for axis, idx in enumerate(idxs):
                    base = self.index_base(idx, var) if innermost else None
                    if base is not None and not any(
                            isinstance(x, ast.Name) and x.id in stored for x in ast.walk(base)):
                        nonneg, inbounds = self.index_facts(name, idx, axis)
                        if not (nonneg and inbounds):
                            extra[(name, axis, 'base', ast.dump(base))] = base
                        continue
                    if not self.boundscheck:
                        continue
                    if isinstance(idx, ast.BinOp) and isinstance(idx.op, ast.Mod):
                        src = self.len_of(idx.right)          # x % n: in [0, n), n another extent
                        if src and src != (name, axis) and self.uses_var(idx.left, var):
                            extra[(name, axis, 'mod', src)] = 0
                        continue
                    base, off = idx, 0
                    if isinstance(base, ast.BinOp) and isinstance(base.op, (ast.Add, ast.Sub)) \
                            and isinstance(base.right, ast.Constant) and type(base.right.value) is int:
                        off = base.right.value if isinstance(base.op, ast.Add) else -base.right.value
                        base = base.left
                    if not (isinstance(base, ast.Name) and base.id == var):
                        continue
                    nonneg, inbounds = self.index_facts(name, idx, axis)
                    if nonneg and not inbounds:
                        extra[(name, axis)] = max(off, extra.get((name, axis), off))
        finally:
            self.facts.pop(var, None)
        return extra

    def range_fact(self, s):
        """What a unit-step range() loop proves about its variable, if the body never assigns it:
        a lower bound, and an array A with var < len(A) - slack."""
        b = self.range_bounds(s)
        return b if b is not None and (b[0] is not None or b[1]) else None

    def range_bounds(self, s):
        """(lower bound or None, array or None, slack) for a range() loop whose body never assigns its variable, else None.
        A positive step keeps both: var starts at start and stays below stop."""
        var = s.target.id
        for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
            if isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store):
                return None
        args = s.iter.args
        start, stop = (None, args[0]) if len(args) == 1 else (args[0], args[1])
        if start is None:
            lo = 0
        elif isinstance(start, ast.Constant) and type(start.value) is int:
            lo = start.value
        elif self.len_of(start):
            lo = 0
        else:
            lo = None
        hi, slack = self.len_of(stop), 0
        if hi is None and isinstance(stop, ast.BinOp) and isinstance(stop.op, ast.Sub) \
                and isinstance(stop.right, ast.Constant) and type(stop.right.value) is int and stop.right.value >= 0:
            hi, slack = self.len_of(stop.left), stop.right.value
        if hi and self.env[hi[0]].kind == 'list' and any(
                isinstance(n, ast.Name) and n.id == hi[0] and isinstance(n.ctx, ast.Store)
                for n in ast.walk(ast.Module(body=s.body, type_ignores=[]))):
            hi = None                          # a list rebound in the loop can change length
        return lo, hi, slack

    def len_of(self, e):
        """The (array, axis) whose extent e is, if that can't change: nothing is known about an array that's assigned."""
        r = self.extent_of(e)
        return None if r and r[0] in self.rebound else r

    def extent_of(self, e):
        """(array, axis) if e is that axis's extent: len(a), a.shape[k], or a variable assigned
        from one of those exactly once (h, w = a.shape)."""
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and self.is_builtin(e.func.id, 'len') \
                and len(e.args) == 1 and isinstance(e.args[0], ast.Name) and e.args[0].id in self.env \
                and self.env[e.args[0].id].kind in ('array', 'list'):
            return e.args[0].id, 0
        if isinstance(e, ast.Name):
            return self.aliases.get(e.id)
        k = self.shape_index(e)
        return k

    def shape_index(self, e):
        """(array, k) for a.shape[k] on an array parameter."""
        if isinstance(e, ast.Subscript) and self.is_shape(e.value) and isinstance(e.slice, ast.Constant) \
                and type(e.slice.value) is int:
            nd = self.env[e.value.value.id].ndim
            if -nd <= e.slice.value < nd:
                return e.value.value.id, e.slice.value % nd
        return None

    def is_shape(self, e):
        return isinstance(e, ast.Attribute) and e.attr == 'shape' and isinstance(e.value, ast.Name) \
            and e.value.id in self.env and self.env[e.value.id].kind == 'array'

    def find_aliases(self):
        """Variables assigned exactly once, from an extent: they stand for it in range facts."""
        stores = {}
        for n in ast.walk(self.f):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                stores[n.id] = stores.get(n.id, 0) + 1
        self.aliases = {}
        for n in ast.walk(self.f):
            if not isinstance(n, ast.Assign) or len(n.targets) != 1:
                continue
            t, v = n.targets[0], n.value
            if isinstance(t, ast.Name) and stores.get(t.id) == 1:
                a = self.len_of(v) if not isinstance(v, ast.Name) else None
                if a:
                    self.aliases[t.id] = a
            elif isinstance(t, ast.Tuple) and self.is_shape(v) and v.value.id not in self.rebound:
                for k, x in enumerate(t.elts):
                    if isinstance(x, ast.Name) and stores.get(x.id) == 1 and k < self.env[v.value.id].ndim:
                        self.aliases[x.id] = (v.value.id, k)

    def guard_facts(self, test, region):
        """Facts a true test proves about variables not assigned in region (statements, or None for
        the rest of an expression): v > c, v >= c give a lower bound; v < n - c an upper one."""
        out = {}
        tests = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
        stored = set() if region is None else {n.id for r in region for n in ast.walk(r)
                                               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        pairs = []                            # 0 <= k < n is 0 <= k and k < n
        for t in tests:
            if isinstance(t, ast.Compare):
                operands = [t.left] + t.comparators
                pairs += [(operands[k], type(o), operands[k + 1]) for k, o in enumerate(t.ops)]
        for left, op, right in pairs:
            if isinstance(right, ast.Name) and not isinstance(left, ast.Name):
                left, right = right, left
                op = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}.get(op, op)
            if not isinstance(left, ast.Name) or left.id in stored \
                    or self.env.get(left.id) not in (T.INT, T.NP_INT64, T.INT_DYN):
                continue
            lo, hi, slack = (out.get(left.id) or self.facts.get(left.id) or (None, None, 0))[:3]   # what the test showed so far
            if isinstance(right, ast.Constant) and type(right.value) is int and op in (ast.Gt, ast.GtE):
                c = right.value + (1 if op is ast.Gt else 0)
                lo = c if lo is None else max(lo, c)
            elif op in (ast.Lt, ast.LtE):
                base, k = right, 0
                if isinstance(base, ast.BinOp) and isinstance(base.op, ast.Sub) and isinstance(base.right, ast.Constant) \
                        and type(base.right.value) is int:
                    base, k = base.left, base.right.value
                src = self.len_of(base)
                if src:                       # v < n - k: v + k < n;  v <= n - k: v + k - 1 < n
                    hi, slack = src, k - (1 if op is ast.LtE else 0)
                else:
                    continue
            else:
                continue
            out[left.id] = (lo, hi, slack)
        return out

    def with_facts(self, facts):
        saved = {k: self.facts.get(k) for k in facts}
        self.facts.update(facts)
        return saved

    def drop_facts(self, saved):
        for k, v in saved.items():
            if v is None:
                self.facts.pop(k, None)
            else:
                self.facts[k] = v

    def mod_assigned(self):
        """Variables whose every assignment is x % n for an extent n: always in [0, n)."""
        out, bad = {}, set()
        for n in ast.walk(self.f):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.For)) else []
            for t in targets:
                for x in ast.walk(t):
                    if not (isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)):
                        continue
                    v = n.value if isinstance(n, ast.Assign) and t is x else None
                    src = self.len_of(v.right) if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mod) else None
                    if src and x.id not in bad and out.get(x.id, src) == src:
                        out[x.id] = src
                    else:
                        bad.add(x.id)
        return {k: (0, v, 0) for k, v in out.items() if k not in bad and self.env.get(k) in (T.INT, T.NP_INT64, T.INT_DYN)}

    def wrap_expr(self, node, a, b):
        """(i + c) % n as compares: with 0 <= i < n, i + 1 can only reach n, and i - 1 only -1."""
        x = node.left
        off = 0
        if isinstance(x, ast.BinOp):
            off = x.right.value if isinstance(x.op, ast.Add) else -x.right.value
        if off > 0:
            return f'({a} >= {b} ? {a} - {b} : {a})'
        if off < 0:
            return f'({a} < 0 ? {a} + {b} : {a})'
        return a

    def hoist_wraps(self, s, var):
        """Compute each distinct (var + c) % n wraparound in the loop once, at the top of the body
        (pure, and it can't raise: the loop proves n >= 1), as hand-written stencils do."""
        keys = []
        if self.rebound:
            return keys
        for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
            if not (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and self.wrap_mod(n)):
                continue
            base = n.left.left if isinstance(n.left, ast.BinOp) else n.left
            key = ast.dump(n)
            if base.id != var or key in self.hoisted or not isinstance(n.right, ast.Name) \
                    and self.shape_index(n.right) is None and self.len_of(n.right) is None:
                continue
            a = self.hold(self.coerce(self.expr(n.left)[0], self.typeof(n.left), T.INT), T.INT)
            b = self.hold(self.coerce(self.expr(n.right)[0], self.typeof(n.right), T.INT), T.INT)
            self.hoisted[key] = self.hold(self.wrap_expr(n, a, b), T.INT)
            keys.append(key)
        return keys

    def wrap_mod(self, node):
        """Whether node is (i + c) % n, where a loop proves 0 <= i < n (so n >= 1) and |c| <= 1."""
        if not isinstance(node, ast.BinOp):
            return False
        x, off = node.left, 0
        if isinstance(x, ast.BinOp) and isinstance(x.op, (ast.Add, ast.Sub)) \
                and isinstance(x.right, ast.Constant) and type(x.right.value) is int:
            off = x.right.value if isinstance(x.op, ast.Add) else -x.right.value
            x = x.left
        if abs(off) > 1 or not isinstance(x, ast.Name) or x.id not in self.facts:
            return False
        lo, hi, slack = self.facts[x.id][:3]
        return lo is not None and lo >= 0 and hi is not None and slack >= 0 and self.len_of(node.right) == hi

    def uses_var(self, e, var):
        return any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(e))

    def index_facts(self, name, idx, axis=0):
        """(never negative, always below name's extent on axis) for an index, from loop facts."""
        if isinstance(idx, ast.BinOp) and isinstance(idx.op, ast.Mod):
            src = self.len_of(idx.right)
            if src == (name, axis):
                return True, True    # x % n is in [0, n) (n == 0 raises ZeroDivisionError first)
            if src and any(len(f) > 3 and (name, axis, 'mod', src) in f[3] for f in self.facts.values()):
                return True, True    # and a versioning test proved n <= name's extent
            return False, False
        for var, f in self.facts.items():
            if len(f) > 3 and any(len(k) > 2 and k[2] == 'base' for k in f[3]):
                base = self.index_base(idx, var)
                if base is not None and (name, axis, 'base', ast.dump(base)) in f[3]:
                    return True, True    # a versioning test proved it
        off = 0
        if isinstance(idx, ast.BinOp) and isinstance(idx.op, (ast.Add, ast.Sub)) \
                and isinstance(idx.right, ast.Constant) and type(idx.right.value) is int:
            off = idx.right.value if isinstance(idx.op, ast.Add) else -idx.right.value
            idx = idx.left
        if not isinstance(idx, ast.Name) or idx.id not in self.facts:
            return False, False
        fact = self.facts[idx.id]
        lo, hi, slack = fact[:3]
        versioned = fact[3] if len(fact) > 3 else {}
        nonneg = lo is not None and lo + off >= 0
        return nonneg, nonneg and ((hi == (name, axis) and off <= slack) or off <= versioned.get((name, axis), -1 << 62))

    def element(self, sub, store=False):
        """Index an array or a list: Python's negative indices on every axis; numpy's IndexError
        (naming the axis) for arrays, Python's for lists. Returns (C array, linear index)."""
        sub = self.named(sub)
        name = sub.value.id
        is_list = self.env[name].kind == 'list' or T.is_pylist(self.env[name])
        strided = self.env[name].strided
        idxs = self.indices(sub)
        raws = [self.coerce(self.expr(k)[0], self.typeof(k), T.INT) for k in idxs]
        raws = [self.hold(r, T.INT) for r in raws]            # all indices first, in order
        for k, raw in zip(idxs, raws):                         # past int64: numpy's OverflowError, or the list's IndexError
            if self.typeof(k) in (T.U64, T.NP_UINT64):
                self.emit(f'if ({raw} < 0) SJ_FAIL({INT_BOUNDS_ERROR}, {self.line(sub)}, {raw}, {201 if is_list else 200});')
        lin = None
        for axis, (k, raw) in enumerate(zip(idxs, raws)):
            n = self.extent(name, axis)
            nonneg, inbounds = self.index_facts(name, k, axis)
            t = self.temp()
            if nonneg:                  # proven: no negative-index adjustment
                self.emit(f'int64_t {t} = {raw};')
            else:
                self.emit(f'int64_t {t} = {raw} < 0 ? {raw} + {n} : {raw};')
            if is_list and not inbounds:    # Python's lists always check
                self.emit(f'if ((uint64_t){t} >= (uint64_t){n}) SJ_FAIL({LIST_INDEX_ERROR}, {self.line(sub)}, {1 if store else 0}, 0);')
            elif not is_list and self.boundscheck and not inbounds:
                if self.slow_copy:
                    self.hoisted_lines.add(self.py_line(sub))
                else:
                    self.checks[self.py_line(sub)] = self.checks.get(self.py_line(sub), 0) + 1
                self.emit(f'if ((uint64_t){t} >= (uint64_t){n}) {{ sj_io[4] = {axis}; SJ_FAIL({INDEX_ERROR}, {self.line(sub)}, {raw}, {n}); }}')
            if strided:
                lin = f'{t} * v_{name}__s{axis}' if lin is None else f'({lin} + {t} * v_{name}__s{axis})'
            else:
                lin = t if lin is None else f'({lin} * {n} + {t})'
        if len(idxs) > 1 or strided:
            lin = self.hold(lin, T.INT)
        return f'v_{name}', lin

    def named(self, sub):
        """sub, indexing a variable: a view it indexes is made first."""
        if isinstance(sub.value, ast.Name):
            return sub
        return ast.copy_location(ast.Subscript(ast.Name(self.materialize(sub.value), ast.Load()), sub.slice, sub.ctx), sub)

    def stride(self, name, axis):
        """How many elements apart this axis's neighbours are."""
        t = self.env[name]
        if t.strided:
            return f'v_{name}__s{axis}'
        return ' * '.join(['1LL'] + [self.extent(name, k) for k in range(axis + 1, t.ndim)])

    def materialize(self, e):
        """Fill in the variable holding a view expression, and return its name. Bounds and steps are evaluated
        first, in order, as Python builds the index before numpy looks at it."""
        if isinstance(e, ast.Name):
            return e.id
        if self.callee_of(e) is not None:
            return self.call_array(e)
        if isinstance(e, ast.Call):
            return self.make(e)
        if isinstance(e, ast.Attribute):           # .T: the axes in reverse
            base = self.materialize(e.value)
            name = self.view_name(e)
            t = self.env[name]
            self.emit(f'v_{name} = v_{base};')
            self.same_root(name, base, 0)
            for k in range(t.ndim):
                self.emit(f'{self.extent(name, k)} = {self.extent(base, t.ndim - 1 - k)};')
                if t.strided:
                    self.emit(f'v_{name}__s{k} = {self.stride(base, t.ndim - 1 - k)};')
            return name
        base = self.materialize(e.value)
        name = self.view_name(e)
        t = self.env[name]
        items = e.slice.elts if isinstance(e.slice, ast.Tuple) else [e.slice]
        line = self.line(e)

        def number(part):
            return self.hold(self.coerce(self.expr(part)[0], self.typeof(part), T.INT), T.INT)
        vals = []
        for k in items:
            if isinstance(k, ast.Slice):
                vals.append(tuple(None if p is None else number(p) for p in (k.lower, k.upper, k.step)))
            else:
                vals.append(number(k))
        off = self.temp()
        self.emit(f'int64_t {off} = 0;')
        axes = []
        at = self.env[base]
        for axis, (k, v) in enumerate(zip(items, vals)):
            n = self.extent(base, axis)
            st = self.stride(base, axis)
            if not isinstance(k, ast.Slice):
                i = self.temp()
                self.emit(f'int64_t {i} = {v} < 0 ? {v} + {n} : {v};')
                if T.is_pylist(at):              # a row of a list of lists: Python's error, always checked
                    self.emit(f'if ((uint64_t){i} >= (uint64_t){n}) SJ_FAIL({LIST_INDEX_ERROR}, {line}, 0, 0);')
                elif self.boundscheck:
                    self.emit(f'if ((uint64_t){i} >= (uint64_t){n}) {{ sj_io[4] = {axis}; SJ_FAIL({INDEX_ERROR}, {line}, {v}, {n}); }}')
                self.emit(f'{off} += {i} * ({st});')
                continue
            a, b, c = v
            if c is not None:
                self.emit(f'if ({c} == 0) SJ_FAIL({VALUE_ERROR}, {line}, {VALUE_ERROR_MESSAGES.index("slice step cannot be zero")}, 0);')
            first, step, count = self.temp(), self.temp(), self.temp()
            self.emit(f'int64_t {first} = 0, {step} = 1, {count} = sj_slice({0 if a is None else 1}, {a or "0"}, {0 if b is None else 1}, '
                      f'{b or "0"}, {c or "1"}, {n}, &{first}, &{step});')
            self.emit(f'{off} += {first} * ({st});')
            axes.append((count, f'{step} * ({st})'))
        for axis in range(len(items), at.ndim):
            axes.append((self.extent(base, axis), self.stride(base, axis)))
        self.emit(f'v_{name} = v_{base} + {off};')
        self.same_root(name, base, 0)
        for k, (count, st) in enumerate(axes):
            self.emit(f'{self.extent(name, k)} = {count};')
            if t.strided:
                self.emit(f'v_{name}__s{k} = {st};')
        return name

    def same_root(self, name, base, same):
        if self.tracks:
            self.emit(f'v_{name}__b = v_{base}__b; v_{name}__o = {f"v_{base}__o" if same else 0};')
        if self.makes:
            self.emit(f'v_{name}__h = v_{base}__h;')

    def assign_view(self, target, value):
        """row = m[i]: a view is its pointer, extents and strides, so this copies those."""
        src = self.materialize(value)
        t = self.env[target]
        self.emit(f'v_{target} = v_{src};')
        self.same_root(target, src, 1)
        for k in range(t.ndim):
            self.emit(f'{self.extent(target, k)} = {self.extent(src, k)};')
            if t.strided:
                self.emit(f'v_{target}__s{k} = v_{src}__s{k};')

    def elem_flag(self, name, idx):
        t = self.env[name]
        if t.kind == 'array':
            return flag_of(t.elem) if T.is_pylist(t) else '1'
        return f'v_{name}__np[{idx}]' if t.elem.np == 'dyn' else flag_of(t.elem)

    def store_elem(self, target, v, vt):
        """target[i] = v (already computed): the index is evaluated and checked now."""
        target = self.named(target)
        name = target.value.id
        et = self.env[name].elem
        flag = self.hold_flag(v[1])
        arr, idx = self.element(target, store=True)       # numpy checks the index first
        self.check_fits(target, et, v, vt)
        val = self.hold(self.coerce(v[0], vt, et), et)
        self.emit(f'{arr}[{idx}] = {val};')
        if name in self.forwards:                  # what was y[i - 1] is y[i - 2] next time
            temps = self.forwards[name][1]
            self.emit(' '.join(f'{temps[k]} = {temps[k - 1]};' for k in range(len(temps) - 1, 0, -1)) + f' {temps[0]} = {val};')
        if self.env[name].kind == 'list' and et.np == 'dyn':
            self.emit(f'v_{name}__np[{idx}] = {flag};')

    def check_fits(self, node, et, v, vt):
        """numpy (NEP 50): a Python int that doesn't fit the dtype raises OverflowError; numpy scalars are cast (and wrap)."""
        dtype = et.name[3:]
        if vt is T.U64 and et.kind != 'float':      # numpy: past int64 it doesn't convert at all
            iv = self.hold(f'(uint64_t)({v[0]})', T.U64)
            if dtype == 'uint64':
                return
            hi = T.INT_RANGE[dtype][1] if dtype in T.INT_RANGE else INT64_MAX
            self.emit(f'if ({iv} > {hi}ULL) SJ_FAIL({INT_BOUNDS_ERROR}, {self.line(node)}, (int64_t){iv}, '
                      f'{NARROW_INT_DTYPES.index(dtype) + 100});')
            return
        if dtype == 'uint64':                        # numpy casts numpy scalars; a Python int must not be negative
            if vt.kind == 'float':
                self.fail(node, 'storing a float into uint64 (numpy\'s cast past the range depends on the platform)')
            if vt.np is not True:
                iv = self.hold(self.coerce(v[0], vt, T.INT), T.INT)
                cond = f'{iv} < 0' if vt.np is False else f'!{v[1]} && {iv} < 0'
                self.emit(f'if ({cond}) SJ_FAIL({INT_BOUNDS_ERROR}, {self.line(node)}, {iv}, {NARROW_INT_DTYPES.index("uint64")});')
            return
        if dtype in T.INT_RANGE and vt.kind in ('int', 'bool') and vt.np is not True:
            lo, hi = T.INT_RANGE[dtype]
            iv = self.hold(self.coerce(v[0], vt, T.INT), T.INT)
            cond = f'{iv} < {c_int(lo)} || {iv} > {c_int(hi)}'
            if vt.np == 'dyn':
                cond = f'!{v[1]} && ({cond})'
            self.emit(f'if ({cond}) SJ_FAIL({INT_BOUNDS_ERROR}, {self.line(node)}, {iv}, {NARROW_INT_DTYPES.index(dtype)});')

    def each_element(self, shape, strides, body):
        """Nested loops over a view's axes, calling body(indices) inside."""
        indices = []
        for n, st in zip(shape, strides):
            i = self.temp()
            self.open(f'for (int64_t {i} = 0; {i} < {n}; {i}++)')
            indices.append(i)
        body(indices)
        for _ in indices:
            self.close()

    def view_axes(self, name):
        t = self.env[name]
        return ([self.hold(self.extent(name, k), T.INT) for k in range(t.ndim)],
                [self.hold(self.stride(name, k), T.INT) for k in range(t.ndim)])

    def store_slice_number(self, target, v, vt):
        """x[a:b] = 0.0: every element. The view is made first, as numpy looks at the index before the value."""
        name = self.materialize(target)
        et = self.env[name].elem
        self.check_fits(target, et, v, vt)
        val = self.hold(self.coerce(v[0], vt, et), et)
        shape, strides = self.view_axes(name)

        def store(idx):
            at = ' + '.join(f'{i} * {st}' for i, st in zip(idx, strides))
            self.emit(f'v_{name}[{at}] = {val};')
        self.each_element(shape, strides, store)

    def store_slice_array(self, target, src):
        """x[a:b] = y: shapes broadcast, as in numpy, and if the two share memory it's as if y were copied first."""
        dst = self.materialize(target)
        dt, st = self.env[dst], self.env[src]
        line = self.line(target)
        dshape, dstr = self.view_axes(dst)
        sshape, sstr = self.view_axes(src)
        tail = dshape[dt.ndim - st.ndim:]

        def packed(shape):
            return '0' if not shape else shape[-1] if len(shape) == 1 else f'(({shape[0]} << 32) | {shape[1]})'
        if st.ndim:
            bad = ' || '.join(f'({a} != {b} && {a} != 1)' for a, b in zip(sshape, tail))
            self.open(f'if ({bad})')
            self.emit(f'sj_io[4] = {st.ndim << 8 | dt.ndim};')
            self.emit(f'SJ_FAIL({BROADCAST_ERROR}, {line}, {packed(sshape)}, {packed(dshape)});')
            self.close()
        # the bytes each covers, to see if they share any: an element less than none if a view is empty
        ct = st.elem.ctype

        def span(ptr, shape, strides, ctype):
            lo, hi = self.temp(), self.temp()
            self.emit(f'int64_t {lo} = 0, {hi} = 1;')
            for n, stride in zip(shape, strides):
                self.emit(f'if ({stride} < 0) {lo} += ({n} - 1) * {stride}; else {hi} += ({n} - 1) * {stride};')
            return f'(const char *)({ptr} + {lo})', f'(const char *)({ptr} + {hi})'
        a0, a1 = span(f'v_{src}', sshape, sstr, ct)
        b0, b1 = span(f'v_{dst}', dshape, dstr, dt.elem.ctype)
        empty = ' || '.join(f'{n} == 0' for n in sshape + dshape)
        total = ' * '.join(['1LL'] + sshape)
        overlap, tmp, read = self.temp(), self.temp(), self.temp()
        self.emit(f'int {overlap} = !({empty}) && {a0} < {b1} && {b0} < {a1};')
        self.emit(f'{ct} *{tmp} = 0;')
        self.emit(f'const {ct} *{read} = v_{src};')
        steps = list(sstr)
        self.open(f'if ({overlap})')
        self.emit(f'{tmp} = malloc(sizeof({ct}) * (size_t){total});')
        self.emit(f'if (!{tmp}) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
        n = self.temp()
        self.emit(f'int64_t {n} = 0;')

        def copy(idx):
            at = ' + '.join(f'{i} * {k}' for i, k in zip(idx, sstr)) or '0'
            self.emit(f'{tmp}[{n}++] = v_{src}[{at}];')
        self.each_element(sshape, sstr, copy)
        self.emit(f'{read} = {tmp};')
        self.close()
        rstr = []
        for k in range(st.ndim):                       # the copy is dense, in order; a single element repeats
            r = self.temp()
            dense = ' * '.join(['1LL'] + sshape[k + 1:])
            self.emit(f'int64_t {r} = {sshape[k]} == 1 ? 0 : {overlap} ? {dense} : {sstr[k]};')
            rstr.append(r)
        skip = dt.ndim - st.ndim

        def store(idx):
            to = ' + '.join(f'{i} * {k}' for i, k in zip(idx, dstr))
            at = ' + '.join(f'{i} * {k}' for i, k in zip(idx[skip:], rstr)) or '0'
            self.emit(f'v_{dst}[{to}] = {self.coerce(f"{read}[{at}]", st.elem, dt.elem)};')
        self.each_element(dshape, dstr, store)
        self.emit(f'free({tmp});')

    def assign_with_views(self, targets, value):
        """a, b = m[0], m[1]: everything on the right first, in order, so that a, b = b, a works."""
        held = []
        for e in value.elts:
            t = self.typeof(e)
            if t.kind != 'array':
                c, f = self.expr(e)
                held.append((self.hold(c, t), self.hold_flag(f), t))
                continue
            src = self.materialize(e)
            name = f'sjh__{self.temp()}'
            self.env[name] = t
            self.emit(f'{t.ctype}v_{name} = v_{src}; int64_t ' + ', '.join(f'{self.extent(name, k)} = {self.extent(src, k)}' for k in range(t.ndim)) + ';')
            if self.tracks:
                self.emit(f'int v_{name}__b = v_{src}__b, v_{name}__o = v_{src}__o;')
            if self.makes:
                self.emit(f'sj_block *v_{name}__h = v_{src}__h;')
            if t.strided:
                self.emit('int64_t ' + ', '.join(f'v_{name}__s{k} = v_{src}__s{k}' for k in range(t.ndim)) + ';')
            held.append(name)
        for target in targets:
            for x, v in zip(target.elts, held):
                if isinstance(v, str):
                    if isinstance(x, ast.Subscript):
                        self.store_slice_array(x, v)
                    else:
                        self.assign_view(x.id, ast.Name(v, ast.Load()))
                elif isinstance(x, ast.Name):
                    self.assign(x.id, (v[0], v[1]), v[2])
                elif self.typeof(x).kind == 'array':
                    self.store_slice_number(x, (v[0], v[1]), v[2])
                else:
                    self.store_elem(x, (v[0], v[1]), v[2])

    def make_list(self, name, v, s):
        """Build a list into a temporary (so an error midway leaves the old list bound, as in
        Python), then free the old one and bind the new."""
        lt = self.env[name]
        et = lt.elem
        dyn = et.np == 'dyn'
        line = self.line(s)
        p, pf, n = self.temp(), self.temp(), self.temp()
        self.owned.append((p, et.ctype))
        if dyn:
            self.owned.append((pf, 'int'))

        def alloc(count, each=1):
            # past this, the size in bytes would wrap to something small, and malloc would succeed
            self.emit(f'if ((uint64_t)({count}) > 9223372036854775807ULL / sizeof({et.ctype}) / {each}) '
                      f'SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
            count = f'({count}) * {each}'
            self.emit(f'int64_t {n} = 0;')
            self.emit(f'{p} = malloc(sizeof({et.ctype}) * (size_t)({count} > 0 ? {count} : 1));')
            self.emit(f'if (!{p}) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
            if dyn:
                self.emit(f'{pf} = malloc(sizeof(int) * (size_t)({count} > 0 ? {count} : 1));')
                self.emit(f'if (!{pf}) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')

        def push(c, f, t):
            self.emit(f'{p}[{n}] = {self.coerce(c, t, et)};' + (f' {pf}[{n}] = {f};' if dyn else '') + f' {n}++;')
        if isinstance(v, ast.ListComp) and len(v.generators) > 1:    # the size isn't known up front
            cap = self.temp()
            self.emit(f'int64_t {cap} = 16;')
            alloc('16')
            saved, saved_c, opened = {}, {}, 0
            try:
                for g in v.generators:
                    cnt, elem = self.iter_prep(self.iterable(g.iter))
                    k = self.temp()
                    self.open_counted('uint64_t', k, '0', cnt)
                    opened += 1
                    self.comp_bind_values(g.target, elem(k), saved, saved_c)
                    for c in g.ifs:
                        self.emit(f'if (!({self.truth(self.expr(c)[0], self.typeof(c))})) continue;')
                c, f = self.expr(v.elt)
                c = self.hold(c, self.typeof(v.elt))
                grow = f'{cap} *= 2; {{ {et.ctype} *g_ = realloc({p}, sizeof({et.ctype}) * (size_t){cap}); if (!g_) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0); {p} = g_; }}'
                if dyn:
                    grow += f' {{ int *h_ = realloc({pf}, sizeof(int) * (size_t){cap}); if (!h_) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0); {pf} = h_; }}'
                self.emit(f'if ({n} == {cap}) {{ {grow} }}')
                push(c, f, self.typeof(v.elt))
            finally:
                for _ in range(opened):
                    self.close()
                self.restore(saved, saved_c)
        elif isinstance(v, ast.ListComp):            # its single iterable gives the capacity
            d = self.iterable(v.generators[0].iter)
            saved, saved_c = {}, {}
            cnt, elem = self.iter_prep(d)
            alloc(cnt)
            k = self.temp()
            self.open_counted('uint64_t', k, '0', cnt)
            try:
                self.comp_bind_values(v.generators[0].target, elem(k), saved, saved_c)
                for c in v.generators[0].ifs:
                    self.emit(f'if (!({self.truth(self.expr(c)[0], self.typeof(c))})) continue;')
                c, f = self.expr(v.elt)
                push(c, f, self.typeof(v.elt))
            finally:
                self.close()
                self.restore(saved, saved_c)
        else:
            lit, k = (v, None) if isinstance(v, ast.List) else \
                ((v.left, v.right) if isinstance(v.left, ast.List) else (v.right, v.left))
            kc = None
            if k is not None and k is v.left:          # k * [..]: k first, as Python evaluates it
                kc = self.hold(self.coerce(self.expr(k)[0], self.typeof(k), T.INT), T.INT)
            vals = []
            for x in lit.elts:
                c, f = self.expr(x)
                vals.append((self.hold(c, self.typeof(x)), self.hold_flag(f), self.typeof(x)))
            if k is not None and kc is None:
                kc = self.hold(self.coerce(self.expr(k)[0], self.typeof(k), T.INT), T.INT)
            if k is None:
                alloc(str(len(vals)))
                for c, f, t in vals:
                    push(c, f, t)
            else:
                reps = self.hold(f'({kc} > 0 ? {kc} : 0)', T.INT)
                alloc(reps, len(vals))
                r = self.temp()
                self.open_counted('int64_t', r, '0', reps)
                for c, f, t in vals:
                    push(c, f, t)
                self.close()
        self.emit(f'free(v_{name}); v_{name} = {p}; v_{name}__n = {n}; {p} = NULL;')
        if dyn:
            self.emit(f'free(v_{name}__np); v_{name}__np = {pf}; {pf} = NULL;')

    def tuple_var(self, name, t=None, prefix=None):
        """A tuple variable's C variables, one per number, as nested lists of (value, flag) following its type."""
        t = self.env[name] if t is None else t
        prefix = f'v_{name}' if prefix is None else prefix
        return [self.tuple_var(name, et, f'{prefix}__{i}') if et.kind == 'tuple'
                else (f'{prefix}__{i}', f'{prefix}__{i}__np' if et.np == 'dyn' else flag_of(et))
                for i, et in enumerate(t.elem)]

    def tuple_leaves(self, name):
        """(C variable, type) for each number in a tuple variable, in order."""
        return [(c, et) for (c, _), et in zip(flat(self.tuple_var(name)), T.leaves(self.env[name]))]

    def texpr(self, e):
        """Emit a tuple-valued expression: a list of (value, flag), each computed now."""
        t = self.typeof(e)
        if isinstance(e, ast.Tuple):
            out = []
            for x, et in zip(e.elts, t.elem):
                if et.kind == 'tuple':
                    out.append(self.texpr(x))
                    continue
                c, f = self.expr(x)
                out.append((self.hold(c, et), self.hold_flag(f)))
            return out
        if isinstance(e, ast.Name):
            def held(vals, ts):
                return [held(v, et.elem) if et.kind == 'tuple' else (self.hold(v[0], et), self.hold_flag(v[1])) for v, et in zip(vals, ts)]
            return held(self.tuple_var(e.id), t.elem)
        if isinstance(e, ast.Subscript) and self.tuple_typed(e.value):
            return self.texpr(e.value)[self.tuple_index(e)]
        if self.is_shape(e):
            return [(self.extent(e.value.id, k), '0') for k in range(len(t.elem))]
        if isinstance(e, ast.Call) and self.math_func(e.func):
            return self.math_tuple(e, self.math_func(e.func))
        if self.callee_of(e) is not None:
            return self.call_results(e)
        raise AssertionError(e)

    def extent(self, name, k):
        return f'v_{name}__n' if k == 0 else f'v_{name}__n{k}'

    def assign_tuple(self, target, vals, vt):
        """Assign a tuple (already computed in full, as Python does) to a name or to targets."""
        if isinstance(target, ast.Name):
            t = self.env[target.id]
            for (c, f), et, (cv, _), tt in zip(flat(vals), T.leaves(vt), flat(self.tuple_var(target.id)), T.leaves(t)):
                self.emit(f'{cv} = {self.coerce(c, et, tt)};')
                if tt.np == 'dyn':
                    self.emit(f'{cv}__np = {f};')
            return
        for x, v, et in zip(target.elts, vals, vt.elem):
            if isinstance(x, ast.Name) and et.kind == 'tuple':
                self.assign_tuple(x, v, et)
            elif isinstance(x, ast.Name):
                self.assign(x.id, v, et)
            elif isinstance(x, ast.Subscript):
                self.store_elem(x, v, et)
            else:
                self.assign_tuple(x, v, et)

    def coerce(self, c, frm, to):
        if frm.ctype == to.ctype:
            return c
        if to.ctype == 'float':                  # a Python int goes through a double, in numpy too
            return f'(float)({c})' if T.is_narrow(frm) or frm.kind == 'float' else f'(float)(double)({c})'
        if to.kind == 'float':
            return f'(double)({c})'
        return f'({to.ctype})({c})'

    def truth(self, c, t):
        if t.kind == 'enum':                          # Enum members are always true; IntEnum's are ints
            return f'({c}) != 0' if T.is_int_enum(t) else '1'
        if t.kind == 'float':
            return f'({c}) != 0.0'
        if t.kind == 'int':
            return f'({c}) != 0'
        return c

    def hold(self, c, t):
        """Put a value in a temporary, so it is computed now and only once."""
        v = self.temp()
        self.emit(f'{t.ctype} {v} = {c};')
        return v

    def hold_flag(self, f):
        if f in ('0', '1'):
            return f
        v = self.temp()
        self.emit(f'int {v} = {f};')
        return v

    def expr(self, e):
        """Emit e; return (C value, C numpy flag). In a statement with a call in it, what's read so far is read now: the call can write arrays."""
        v = self.expr_now(e)
        if self.call_stmt and not isinstance(e, (ast.Constant, ast.Name)):
            t = self.typeof(e)
            if t.is_number:
                return self.hold(v[0], t), self.hold_flag(v[1])
        return v

    def expr_now(self, e):
        if self.callee_of(e) is not None:
            return self.call_number(e)
        if self.c_function_of(e) is not None:
            return self.c_number(e)
        if isinstance(e, ast.Constant):
            v = e.value
            if isinstance(v, bool):
                return ('1' if v else '0'), '0'
            if isinstance(v, int) and v > INT64_MAX:
                return f'{v}ULL', '0'
            return (c_int(v) if isinstance(v, int) else c_float(v)), '0'
        if isinstance(e, ast.Name):
            if e.id in self.cnames:
                return self.cnames[e.id]
            t = self.env[e.id]
            return f'v_{e.id}', (f'v_{e.id}__np' if t.np == 'dyn' else flag_of(t))
        if isinstance(e, ast.Attribute) and self.enum_attribute(e) is not None:
            if self.enum_attribute(e).kind == 'enum':
                return c_int(getattr(self.env[e.value.id].elem, e.attr)._value_), '0'
            return self.expr(e.value)[0], '0'
        if isinstance(e, ast.Attribute):
            if e.attr in ('ndim', 'size') and isinstance(e.value, ast.Name) and e.value.id in self.env:
                t = self.env[e.value.id]
                if e.attr == 'ndim':
                    return f'{t.ndim}LL', '0'
                return '(' + ' * '.join(self.extent(e.value.id, k) for k in range(t.ndim)) + ')', '0'
            return c_float(MATH_CONSTANTS[e.attr]), '0'
        if isinstance(e, ast.BinOp) and folds(e) is not None:
            v = folds(e)
            return (f'{v}ULL' if v > INT64_MAX else c_int(v)), '0'
        if isinstance(e, ast.BinOp):
            lt = self.typeof(e.left)
            return self.arith(ARITH[type(e.op)], self.expr(e.left), lt, e.right, e)
        if isinstance(e, ast.UnaryOp):
            t = self.typeof(e.operand)
            c, f = self.expr(e.operand)
            if isinstance(e.op, ast.Not):
                return f'!({self.truth(c, t)})', '0'
            cast = f'({t.ctype})' if T.is_narrow(t) or t is T.NP_UINT64 else ''
            if isinstance(e.op, ast.Invert):
                return f'({cast}(~(int64_t)({c})))', f
            if isinstance(e.op, ast.UAdd):
                return (self.coerce(c, t, T.INT) if t is T.BOOL else c), f
            if t.kind == 'float':
                return f'(-({c}))', f
            return f'{cast}sj_neg((int64_t)({c}))', f
        if isinstance(e, ast.Compare):
            return self.compare(e)
        if isinstance(e, ast.BoolOp):
            t = self.typeof(e)
            first = self.expr(e.values[0])
            r = self.hold(self.coerce(first[0], self.typeof(e.values[0]), t), t)
            rf = self.temp()
            self.emit(f'int {rf} = {first[1]};')
            closes = 0
            guards = []
            for i, v in enumerate(e.values[1:]):
                test = self.truth(r, t)
                self.open(f'if ({test if isinstance(e.op, ast.And) else "!(" + test + ")"})')
                closes += 1
                if isinstance(e.op, ast.And):         # what earlier operands proved holds here
                    guards.append(self.with_facts(self.guard_facts(e.values[i], None)))
                c, f = self.expr(v)
                self.emit(f'{r} = {self.coerce(c, self.typeof(v), t)}; {rf} = {f};')
            for g in reversed(guards):
                self.drop_facts(g)
            for _ in range(closes):
                self.close()
            return r, rf
        if isinstance(e, ast.IfExp):
            t = self.typeof(e)
            r, rf = self.temp(), self.temp()
            self.emit(f'{t.ctype} {r}; int {rf};')
            self.open(f'if ({self.truth(self.expr(e.test)[0], self.typeof(e.test))})')
            c, f = self.expr(e.body)
            self.emit(f'{r} = {self.coerce(c, self.typeof(e.body), t)}; {rf} = {f};')
            self.close('} else {')
            self.depth += 1
            c, f = self.expr(e.orelse)
            self.emit(f'{r} = {self.coerce(c, self.typeof(e.orelse), t)}; {rf} = {f};')
            self.close()
            return r, rf
        if isinstance(e, ast.Subscript) and self.is_shape(e.value):
            arr, k = self.shape_index(e)
            return self.extent(arr, k), '0'
        if isinstance(e, ast.Subscript) and self.tuple_typed(e.value):
            if isinstance(e.value, ast.Name):
                return self.tuple_var(e.value.id)[self.tuple_index(e)]
            return self.texpr(e.value)[self.tuple_index(e)]
        if isinstance(e, ast.Subscript):
            e = self.named(e)
            name = e.value.id
            if name in self.forwards and self.is_previous(e, name, self.forwards[name][0]):
                return self.forwards[name][1][self.is_previous(e, name, self.forwards[name][0]) - 1], '1'
            arr, idx = self.element(e)
            return self.hold(f'{arr}[{idx}]', self.env[name].elem), self.hold_flag(self.elem_flag(name, idx))
        if isinstance(e, ast.Call):
            return self.call(e)
        raise AssertionError(e)

    def power(self, a, b, fa, fb, rt, npf, line, right):
        """The value of a ** b. npf: 1 when numpy's rules apply."""
        out = self.temp()
        self.emit(f'{rt.ctype} {out} = 0;')
        self.uses_libm = self.uses_libm or self.exact
        table = f'SJ_LIBM2({LIBM.index("pow")})'

        def numpy():
            if rt.kind == 'int':                       # a narrow int wraps as numpy's does: the low bits of the 64-bit power
                self.helpers.add('ipow')
                self.emit(f'if ((int64_t){b} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 15, 0);')
                self.emit(f'{out} = ({rt.ctype})sj_ipow((int64_t){a}, (int64_t){b});')
            elif rt.ctype == 'float':                  # numpy calls powf
                f = f'((float (*)(float, float))(uintptr_t)sj_m[{LIBM.index("powf")}])' if self.exact else 'powf'
                self.emit(f'{out} = {f}({fa}, {fb});')
            elif self.exact:
                self.emit(f'{out} = {table}({fa}, {fb});')
            else:
                fast()

        def fast():                                    # parity=False: x * x for small powers, else C's pow
            exp = const_number(right)
            self.emit(f'{out} = ' + (f'{fa} * {fa};' if exp == 2 else f'{fa} * {fa} * {fa};' if exp == 3 else f'pow({fa}, {fb});'))

        def python():
            if rt.kind == 'int':
                self.helpers.add('ipow')
                if const_number(right) is None:
                    self.emit(f'if ((int64_t){b} < 0) {self.differs(line, 16)}')
                self.emit(f'{out} = sj_ipow((int64_t){a}, (int64_t){b});')
            elif not self.exact:
                fast()
            else:
                self.helpers.add('floatpow')
                err = self.temp()
                self.emit(f'int {err} = sj_float_pow({table}, {fa}, {fb}, &{out});')
                self.emit(f'if ({err} == 1) SJ_FAIL({ZERO_DIVISION}, {line}, 6, 0);')
                self.emit(f'if ({err} == 2) SJ_FAIL({OVERFLOW_ERROR}, {line}, 4, 0);')
                self.emit(f'if ({err} == 3) {self.differs(line, 17)}')
        if npf == '1':
            numpy()
        elif npf == '0':
            python()
        else:
            self.open(f'if ({npf})')
            numpy()
            self.close('} else {')
            self.depth += 1
            python()
            self.close()
        return out

    def arith(self, op, left, at, right, node):
        """left <op> right, where left (already emitted, of type at) is the left operand."""
        bt = self.typeof(right)
        at, bt = (T.INT if at.kind == 'enum' else at), (T.INT if bt.kind == 'enum' else bt)      # IntEnum: its value
        rt = self.binop_type(at, bt, op, node)
        a, af = self.hold(left[0], at), self.hold_flag(left[1])
        bv = self.expr(right)
        b, bf = self.hold(bv[0], bt), self.hold_flag(bv[1])
        npf = flag_or(af, bf)              # 1 when numpy's rules apply, not Python's
        line = self.line(node)
        for n, py, v in ((at, bt, b), (bt, at, a)):       # numpy: a Python int must fit the scalar's dtype, except for /
            if op != 'Div' and T.is_narrow(n) and n.kind == 'int' and py.kind != 'float' and py.np is False:
                self.check_fits(node, n, (v, '0'), py)
        if rt is T.NP_UINT64:
            return self.uint64_arith(op, a, at, b, bt, line, node), npf
        if T.U64 in (at, bt) and at.kind != 'float' and bt.kind != 'float':
            return self.u64_arith(op, a, at, b, bt, rt, npf, line, right, node), npf
        if op in ('LShift', 'RShift'):
            return self.shift(op, a, at, b, rt, npf, line, right), npf
        if not (T.is_narrow(rt) and rt.kind == 'int'):
            return self.arith_as(op, a, at, b, bt, right, rt, npf, line, node), npf
        if op in ('Add', 'Sub', 'Mult'):   # 32 bits rather than 64, so that the vectorizer keeps its lanes
            sym = {'Add': '+', 'Sub': '-', 'Mult': '*'}[op]
            return f'(({rt.ctype})((uint32_t){a} {sym} (uint32_t){b}))', npf
        return f'(({rt.ctype})({self.arith_as(op, a, at, b, bt, right, rt, npf, line, node)}))', npf

    def uint64_arith(self, op, a, at, b, bt, line, node):
        """numpy's uint64: everything wraps, // and % by zero give 0, and a shift by 64 or more gives 0."""
        for v, vt in ((a, at), (b, bt)):             # a Python int must not be negative
            if vt.np is False and vt.kind == 'int' and vt is not T.U64:
                self.check_fits(node, T.NP_UINT64, (v, '0'), vt)
        ua, ub = f'(uint64_t)({a})', f'(uint64_t)({b})'
        if op in ('Add', 'Sub', 'Mult', 'BitAnd', 'BitOr', 'BitXor'):
            r = f'{ua} {dict(Add="+", Sub="-", Mult="*", BitAnd="&", BitOr="|", BitXor="^")[op]} {ub}'
        elif op in ('FloorDiv', 'Mod'):
            r = f'{ub} == 0 ? 0 : {ua} {"/" if op == "FloorDiv" else "%"} {ub}'
        elif op in ('LShift', 'RShift'):
            sym = '<<' if op == 'LShift' else '>>'
            k = f'(uint32_t){ub}' if uint64_shift_masks() else ub
            r = f'{k} < 64 ? {ua} {sym} {k} : 0'
        else:                                        # Pow
            self.helpers.add('ipow')
            r = f'(uint64_t)sj_ipow((int64_t){ua}, (int64_t){ub})'
        return f'((uint64_t)({r}))'

    def u64_arith(self, op, a, at, b, bt, rt, npf, line, right, node):
        """Ints where one may be past int64 (u64_binop_type says which results stay unsigned). The bits are the same
        either way: only //, %, >> and what the result means differ."""
        ua, ub = f'(uint64_t)({a})', f'(uint64_t)({b})'
        if op in ('BitAnd', 'BitOr', 'BitXor'):
            r = f'({ua} {dict(BitAnd="&", BitOr="|", BitXor="^")[op]} {ub})'
        elif op in ('FloorDiv', 'Mod'):              # both are nonnegative here
            self.emit(f'if ({ub} == 0) SJ_FAIL({ZERO_DIVISION}, {line}, {2 if op == "FloorDiv" else 3}, 0);')
            r = f'({ua} {"/" if op == "FloorDiv" else "%"} {ub})'
        elif op == 'RShift' and at is T.U64:
            k = self.hold(f'(int64_t)({b})', T.INT) if bt is not T.U64 else self.hold(ub, T.U64)
            if bt is not T.U64:
                self.emit(f'if ({k} < 0) SJ_FAIL({VALUE_ERROR}, {line}, {VALUE_ERROR_MESSAGES.index("negative shift count")}, 0);')
            r = f'((uint64_t){k} < 64 ? {ua} >> {k} : 0)'
        elif op == 'LShift' and bt is not T.U64:      # wraps, as on any int
            r = self.shift(op, a, T.INT, b, T.INT, npf, line, right)
        elif op == 'LShift':                           # a count held unsigned: past 63 every bit is gone
            r = f'({ub} < 64 ? {ua} << {ub} : 0)'
        elif op == 'RShift':                           # an int64 by an unsigned count
            r = f'({ub} < 64 ? (int64_t)({a}) >> {ub} : ((int64_t)({a}) < 0 ? -1 : 0))'
        elif op == 'Pow':
            self.helpers.add('ipow')
            if const_number(right) is None or const_number(right) < 0:
                self.emit(f'if ((int64_t)({b}) < 0) {self.differs(line, 16)}')
            r = f'sj_ipow((int64_t)({a}), (int64_t)({b}))'
        else:
            r = f'sj_{op.lower()[:3]}((int64_t)({a}), (int64_t)({b}))'
        return f'(({rt.ctype})({r}))'

    def shift(self, op, a, at, b, rt, npf, line, right):
        """a << b and a >> b. Python raises on a negative count; numpy gives 0 (or -1 for >> of a negative), as it does
        for a count past the width. Python's << is unbounded: here it wraps, as * does, so past 64 bits it's 0."""
        bits = T.bits(rt)
        x = self.hold(f'(int64_t)({self.coerce(a, at, rt)})', T.INT)
        if isinstance(right, ast.Constant) and type(right.value) is int and 0 <= right.value < bits:   # nothing to check
            if op == 'LShift':
                return f'(({rt.ctype})((uint64_t){x} << {right.value}))'
            return f'(({rt.ctype})({x} >> {right.value}))'
        k = self.hold(f'(int64_t)({b})', T.INT)
        if npf != '1':
            cond = f'{k} < 0' if npf == '0' else f'!{npf} && {k} < 0'
            self.emit(f'if ({cond}) SJ_FAIL({VALUE_ERROR}, {line}, {VALUE_ERROR_MESSAGES.index("negative shift count")}, 0);')
        within = f'(uint64_t){k} < {bits}'
        if op == 'LShift':
            return f'({within} ? ({rt.ctype})((uint64_t){x} << {k}) : 0)'
        return f'({within} ? ({rt.ctype})({x} >> {k}) : ({x} < 0 ? -1 : 0))'

    def arith_as(self, op, a, at, b, bt, right, rt, npf, line, node):
        """The C for a <op> b, whose result type is rt; integers are computed in 64 bits."""
        if op in ('BitAnd', 'BitOr', 'BitXor'):
            sym = {'BitAnd': '&', 'BitOr': '|', 'BitXor': '^'}[op]
            return f'((int64_t){a} {sym} (int64_t){b})'
        isfloat = rt.kind == 'float'
        fa, fb = self.coerce(a, at, rt if isfloat else T.FLOAT), self.coerce(b, bt, rt if isfloat else T.FLOAT)
        if op == 'Pow':
            return self.power(a, b, fa, fb, rt, npf, line, right)
        if op in ('Add', 'Sub', 'Mult'):
            if op == 'Mult' and self.exact and at.kind == 'float' and bt.kind == 'int' and npf != '1':
                # CPython computes float * int(-1) as a negation, so a NaN's sign bit flips
                # (multiplying by -1.0, by any other int, or numpy scalars, leaves it alone)
                py = f'{b} == -1' if npf == '0' else f'!{npf} && {b} == -1'
                return f'({py} ? -{a} : {a} * (double){b})'
            if isfloat:
                sym = {'Add': '+', 'Sub': '-', 'Mult': '*'}[op]
                return f'({fa} {sym} {fb})'
            return f'sj_{op.lower()[:3]}((int64_t){a}, (int64_t){b})'
        both_int = at.kind != 'float' and bt.kind != 'float'
        wrap = self.wrap_mod(node) if op == 'Mod' and npf == '0' else None
        if wrap and ast.dump(node) in self.hoisted:
            return self.hoisted[ast.dump(node)]
        if wrap:                                  # (i + c) % n with 0 <= i < n, |c| <= 1: n >= 1, so
            return self.wrap_expr(node, a, b)         # no division
        if both_int and op in ('FloorDiv', 'Mod') and isinstance(right, ast.Constant) \
                and type(right.value) is int and right.value > 0 and right.value & (right.value - 1) == 0:
            k = right.value.bit_length() - 1      # x // 2**k is an arithmetic shift, x % 2**k a mask:
            if op == 'FloorDiv':                  # exactly Python's floor semantics, for every sign
                return f'((int64_t){a} >> {k})'
            return f'((int64_t){a} & {right.value - 1}LL)'
        zero = f'{b} == 0' if both_int else f'{fb} == 0.0'
        msg = {('Div', True): 0, ('Div', False): 1, ('FloorDiv', True): 2, ('FloorDiv', False): 4,
               ('Mod', True): 3, ('Mod', False): 5}[(op, both_int if op == 'Div' else not isfloat)]
        # Python raises on a zero divisor; numpy doesn't. Without exact, only an int // or % is
        # checked (in C it would trap); / and float // and % give IEEE results.
        if npf != '1' and (self.exact or (both_int and op != 'Div')):
            cond = zero if npf == '0' else f'!{npf} && {zero}'
            self.emit(f'if ({cond}) SJ_FAIL({ZERO_DIVISION}, {line}, {msg}, 0);')
        if op == 'Div':
            if not both_int or npf == '1' or not self.exact:
                return f'({fa} / {fb})'
            exact = f'sj_truediv((int64_t){a}, (int64_t){b})'
            return exact if npf == '0' else f'({npf} ? {fa} / {fb} : {exact})'
        if not isfloat and op in ('FloorDiv', 'Mod'):
            # one division and one remainder (cimo doesn't merge repeated expressions), then
            # Python's floor: adjust when the remainder is nonzero and its sign differs from b's.
            # b == -1 is set apart (INT64_MIN / -1 traps in C); b == 0 is numpy's 0 here.
            q, r, adj = self.temp(), self.temp(), self.temp()
            self.emit(f'int64_t {q} = 0, {r} = 0;')
            d = f'(int64_t){b}'                   # a narrow unsigned one would never equal -1
            self.emit(f'if ({d} == -1) {q} = sj_neg((int64_t){a}); else if ({d} != 0) {{ {q} = (int64_t){a} / {d}; {r} = (int64_t){a} % {d}; }}')
            self.emit(f'int {adj} = {r} != 0 && (({r} < 0) != ({d} < 0));')
            if op == 'FloorDiv':
                return f'({q} - {adj})'
            return f'({r} + ({adj} ? {d} : 0))'
        if rt.ctype == 'float':                   # numpy's algorithm, in the type's own precision
            if op == 'FloorDiv':
                return f'({fb} == 0.0f ? {fa} / {fb} : sj_ffloordiv32({fa}, {fb}))'
            return f'({fb} == 0.0f ? (float)fmod({fa}, {fb}) : sj_fmod_py32({fa}, {fb}))'
        if op == 'FloorDiv':
            return f'({fb} == 0.0 ? {fa} / {fb} : sj_ffloordiv({fa}, {fb}))'
        return f'({fb} == 0.0 ? fmod({fa}, {fb}) : sj_fmod_py({fa}, {fb}))'

    def cmp(self, a, at, b, bt, op):
        if at.kind == 'enum' or bt.kind == 'enum':
            return self.enum_cmp(a, at, b, bt, op)
        af, bf = at.kind == 'float', bt.kind == 'float'
        if T.U64 in (at, bt) or T.NP_UINT64 in (at, bt):
            return self.u64_cmp(a, at, b, bt, op)
        ct = T.compare_in(at, bt)
        if ct == 'float':
            return f'({self.coerce(a, at, T.NARROW["float32"])} {op} {self.coerce(b, bt, T.NARROW["float32"])})'
        if ct == 'double':
            return f'((double){a} {op} (double){b})'
        if af == bf:
            if af:
                return f'({a} {op} {b})'
            return f'((int64_t){a} {op} (int64_t){b})'
        if not self.exact:                       # C: the int is converted to double
            return f'((double){a} {op} (double){b})'
        if af:                                   # float <op> int: compare int <swapped op> float
            a, b, op = b, a, SWAP[op]
        # |int| <= 2**53 converts to double exactly: compare directly. Beyond, compare exactly.
        code = {'<': 0, '<=': 1, '>': 2, '>=': 3, '==': 4, '!=': 5}[op]
        return (f'((int64_t){a} >= -9007199254740992LL && (int64_t){a} <= 9007199254740992LL ? '
                f'(double){a} {op} {b} : sj_cmp_test(sj_cmp_if((int64_t){a}, {b}), {code}))')

    def enum_cmp(self, a, at, b, bt, op):
        """Members are their values. Two of the same class are the same member exactly when their values are equal (an
        alias is the member it names); of different classes, or with a number, an Enum is never equal."""
        if op in ('is', 'is not'):
            same = at is bt
            return f'(({a}) {"==" if op == "is" else "!="} ({b}))' if same else ('0' if op == 'is' else '1')
        if T.is_int_enum(at) and T.is_int_enum(bt) or (T.is_int_enum(at) or T.is_int_enum(bt)) and at.kind != 'float' and bt.kind != 'float' \
                and not (at.kind == 'enum' and bt.kind == 'enum' and not (T.is_int_enum(at) and T.is_int_enum(bt))):
            return f'((int64_t)({a}) {op} (int64_t)({b}))'        # IntEnum: an int
        if at is bt:
            return f'(({a}) {op} ({b}))'
        if T.is_int_enum(at) or T.is_int_enum(bt):                # IntEnum with a float
            return self.cmp(a, T.INT if at.kind == 'enum' else at, b, T.INT if bt.kind == 'enum' else bt, op)
        return '0' if op == '==' else '1'

    def u64_cmp(self, a, at, b, bt, op):
        """Exact, as Python's are, and as numpy's are with ints. With np.float64 numpy rounds the int to a double first."""
        if at not in (T.U64, T.NP_UINT64):
            a, at, b, bt, op = b, bt, a, at, SWAP[op]
        if bt in (T.U64, T.NP_UINT64):
            return f'((uint64_t)({a}) {op} (uint64_t)({b}))'
        if bt.kind == 'float':
            if bt.np or at is T.NP_UINT64:            # numpy makes both doubles first
                return f'((double)({a}) {op} (double)({b}))'
            self.helpers.add('cmp_uf')
            code = {'<': 0, '<=': 1, '>': 2, '>=': 3, '==': 4, '!=': 5}[op]
            return f'sj_cmp_test(sj_cmp_uf((uint64_t)({a}), (double)({b})), {code})'
        # a in [0, 2**64) against an int64: a negative one is below everything
        below = {'<': 0, '<=': 0, '>': 1, '>=': 1, '==': 0, '!=': 1}[op]
        return f'((int64_t)({b}) < 0 ? {below} : (uint64_t)({a}) {op} (uint64_t)(int64_t)({b}))'

    def compare(self, e):
        operands = [e.left] + e.comparators
        types = [self.typeof(x) for x in operands]
        first = self.expr(operands[0])
        a, af = self.hold(first[0], types[0]), first[1]
        if len(e.ops) == 1:
            second = self.expr(operands[1])
            b = self.hold(second[0], types[1])
            return self.cmp(a, types[0], b, types[1], CMP.get(type(e.ops[0])) or IS[type(e.ops[0])]), flag_or(af, second[1])
        r, rf = self.temp(), self.temp()
        self.emit(f'int {r} = 1, {rf} = 0;')
        closes = 0
        for i, op in enumerate(e.ops):
            if i:
                self.open(f'if ({r})')
                closes += 1
            nxt = self.expr(operands[i + 1])
            b = self.hold(nxt[0], types[i + 1])
            self.emit(f'{r} = {self.cmp(a, types[i], b, types[i + 1], CMP.get(type(op)) or IS[type(op)])}; {rf} = {flag_or(af, nxt[1])};')
            a, af = b, nxt[1]
        for _ in range(closes):
            self.close()
        return r, rf

    def call(self, e):
        if isinstance(e.func, ast.Name) and e.func.id == '__sj_room__':      # a new place in a pool, if there's room left
            v = self.hold(self.expr(e.args[0])[0], T.INT)
            self.emit(f'if ({v} >= v_{e.args[1].id}__n) SJ_FAIL({POOL_FULL}, {self.line(e)}, 0, 0);')
            return v, flag_of(T.INT)
        if isinstance(e.func, ast.Name) and e.func.id == '__sj_deref__':     # a reference read through: None raises
            v = self.hold(self.expr(e.args[0])[0], T.INT)
            self.emit(f'if ({v} < 0) SJ_FAIL({ATTRIBUTE_NONE}, {self.line(e)}, {e.args[1].value}, 0);')
            return v, flag_of(T.INT)
        known = self.isinstance_of(e)
        if known is not None:
            return ('1' if known else '0'), flag_of(T.BOOL)
        nf = self.numpy_func(e.func)
        if nf:
            return self.numpy_call(e, nf)
        reduced = self.reduction_of(e)
        if reduced:
            return self.np_reduction_call(e, reduced[0], reduced[1])
        mf = self.math_func(e.func)
        if mf:
            return self.math_call(e, mf)
        name = e.func.id
        line = self.line(e)
        if self.is_reduction(e):
            return self.reduction(e)
        if name == 'len':
            if self.is_view(e.args[0]):
                return f'v_{self.materialize(e.args[0])}__n', '0'
            t = self.env[e.args[0].id]
            if t.kind == 'tuple':
                return f'{len(t.elem)}LL', '0'
            return f'v_{e.args[0].id}__n', '0'
        ts = [self.typeof(a) for a in e.args]
        if name == 'abs':
            c, f = self.expr(e.args[0])
            if ts[0] in (T.U64, T.NP_UINT64):
                return c, f
            return (f'fabs({c})' if ts[0].kind == 'float' else f'sj_abs((int64_t)({c}))'), f
        if name in ('min', 'max'):
            t = self.typeof(e)
            first = self.expr(e.args[0])
            r = self.hold(self.coerce(first[0], ts[0], t), t)
            rf = self.temp()
            self.emit(f'int {rf} = {first[1]};')
            for a, at in zip(e.args[1:], ts[1:]):
                c, f = self.expr(a)
                b = self.hold(self.coerce(c, at, t), t)
                self.emit(f'if ({b} {">" if name == "max" else "<"} {r}) {{ {r} = {b}; {rf} = {f}; }}')
            return r, rf
        c, _ = self.expr(e.args[0])
        t = ts[0]
        if name == 'float':
            return self.coerce(c, t, T.FLOAT), '0'
        if t.kind != 'float':
            return self.coerce(c, t, T.INT), '0'
        if name == 'round':                 # halves to even: C's rint in the default rounding mode
            c = f'rint((double)({c}))'
        v = self.hold(c, T.FLOAT)
        r = self.temp()
        # one test on the way that matters: a cast truncates toward zero, as trunc; NaN, inf and the rest go the long way
        self.emit(f'int64_t {r};')
        self.emit(f'if (fabs({v}) < 9223372036854775808.0) {r} = (int64_t){v}; else {{ '
                  f'if ({v} != {v}) SJ_FAIL({VALUE_ERROR}, {line}, 0, 0); '
                  f'if (isinf({v})) SJ_FAIL({OVERFLOW_ERROR}, {line}, 0, 0); {r} = sj_float_to_int({v}); }}')
        return r, '0'


    def source_loops(self, src, after_first, item):
        """Emit the loops over a reduction's iterable. after_first() runs once the first iterable is
        evaluated (Python evaluates a reduction's other arguments then); item((c, flag), type) per
        element."""
        if not isinstance(src, (ast.GeneratorExp, ast.ListComp)):
            n, elem = self.iter_prep(self.iterable(src))
            after_first()
            k = self.temp()
            self.open_counted('uint64_t', k, '0', n)
            c, t, f = elem(k)
            item((self.hold(c, t), f), t)
            self.close()
            return
        saved, saved_c, opened = {}, {}, 0
        try:
            for i, g in enumerate(src.generators):
                n, elem = self.iter_prep(self.iterable(g.iter))
                if i == 0:
                    after_first()
                k = self.temp()
                self.open_counted('uint64_t', k, '0', n)
                opened += 1
                self.comp_bind_values(g.target, elem(k), saved, saved_c)
                for c in g.ifs:
                    self.emit(f'if (!({self.truth(self.expr(c)[0], self.typeof(c))})) continue;')
            item(self.expr(src.elt), self.typeof(src.elt))
        finally:
            for _ in range(opened):
                self.close()
            self.restore(saved, saved_c)

    def comp_bind_values(self, target, v, saved, saved_c):
        if isinstance(v, TupleOf):
            v = [TupleOf(val, et) if et.kind == 'tuple' else (val[0], et, val[1]) for val, et in zip(v.vals, v.t.elem)]
        if isinstance(target, ast.Name):
            c, t, f = v
            cn = self.temp()
            self.emit(f'{t.ctype} {cn} = {c};')
            if target.id not in saved:
                saved[target.id] = self.env.get(target.id)
                saved_c[target.id] = self.cnames.get(target.id)
            self.env[target.id] = t
            self.cnames[target.id] = (cn, f)
        else:
            for t, x in zip(target.elts, v):
                self.comp_bind_values(t, x, saved, saved_c)

    def reduction(self, e):
        name = e.func.id
        line = self.line(e)
        rt = self.typeof(e)
        kws = {k.arg: k.value for k in e.keywords}
        src = e.args[0]
        el = self.source_elem_type(src)
        if name in ('any', 'all'):
            r = self.temp()
            self.emit(f'int {r} = {0 if name == "any" else 1};')
            short = not isinstance(src, ast.ListComp)   # a list is built in full first
            label = f'{r}_done'

            def item(v, t):
                test = self.truth(v[0], t)
                hit, val = (test, 1) if name == 'any' else (f'!({test})', 0)
                self.emit(f'if ({hit}) {{ {r} = {val};{f" goto {label};" if short else ""} }}')
            self.source_loops(src, lambda: None, item)
            if short:
                self.lines.append(f'{label}:;')
            return r, '0'
        if name in ('min', 'max'):
            r, rf, got = self.temp(), self.temp(), self.temp()
            self.emit(f'{rt.ctype} {r} = 0; int {rf} = 0, {got} = 0;')
            default = {}

            def after_first():
                if 'default' in kws:
                    c, f = self.expr(kws['default'])
                    default['v'] = (self.hold(self.coerce(c, self.typeof(kws['default']), rt), rt), self.hold_flag(f))

            def item(v, t):
                x = self.hold(self.coerce(v[0], t, rt), rt)
                self.emit(f'if (!{got} || {x} {"<" if name == "min" else ">"} {r}) {{ {r} = {x}; {rf} = {v[1]}; {got} = 1; }}')
            self.source_loops(src, after_first, item)
            if 'v' in default:
                dv, df = default['v']
                self.emit(f'if (!{got}) {{ {r} = {dv}; {rf} = {df}; }}')
            else:
                self.emit(f'if (!{got}) SJ_FAIL({VALUE_ERROR}, {line}, {2 if name == "min" else 3}, 0);')
            return r, rf
        # sum
        start = e.args[1] if len(e.args) == 2 else kws.get('start')
        st = self.typeof(start) if start is not None else T.INT
        if T.is_narrow(T.as_narrow(rt)):              # numpy's rules: each addition wraps or rounds in the narrow type
            r = T.as_narrow(rt)
            acc, rf = self.temp(), self.temp()
            self.emit(f'{r.ctype} {acc} = 0; int {rf} = {1 if st.np is True else 0};')

            def after_first():
                if start is not None:
                    self.emit(f'{acc} = {self.coerce(self.expr(start)[0], st, r)};')

            def item(v, t):
                x = self.coerce(v[0], t, r)
                if r.kind == 'int':
                    self.emit(f'{acc} = ({r.ctype})((uint32_t){acc} + (uint32_t)({x})); {rf} = 1;')
                else:
                    self.emit(f'{acc} = {acc} + {x}; {rf} = 1;')
            self.source_loops(src, after_first, item)
            return acc, ('1' if st.np is True else rf)
        if rt.kind == 'int':
            acc, rf = self.temp(), self.temp()
            self.emit(f'int64_t {acc} = 0; int {rf} = 0;')

            def after_first():
                if start is not None:
                    c, f = self.expr(start)
                    self.emit(f'{acc} = (int64_t)({c}); {rf} = {f};')

            def item(v, t):
                self.emit(f'{acc} = sj_add({acc}, (int64_t){v[0]}); {rf} |= {v[1]};')
            self.source_loops(src, after_first, item)
            return acc, rf
        f_, c_, gen = self.temp(), self.temp(), self.temp()
        self.emit(f'double {f_} = 0.0, {c_} = 0.0; int {gen} = 0;')
        generic = st.np is True                       # a numpy start: plain additions throughout

        def after_first():
            c, _ = self.expr(start)
            self.emit(f'{f_} = (double)({c});{f" {gen} = 1;" if generic else ""}')

        def item(v, t):
            x = self.hold(self.coerce(v[0], t, T.FLOAT), T.FLOAT)
            plain = f'{f_} = {f_} + {x};'
            if t.kind == 'float':                     # CPython's Neumaier compensated summation
                fast = (f'{{ double tt = {f_} + {x}; if (fabs({f_}) >= fabs({x})) {c_} += ({f_} - tt) + {x}; '
                        f'else {c_} += ({x} - tt) + {f_}; {f_} = tt; }}')
            else:                                     # a Python int: added, not compensated
                fast = f'{f_} += {x};'
            settle = f'if ({c_} != 0.0 && isfinite({c_})) {f_} += {c_};'
            if v[1] == '0':
                self.emit(f'if (!{gen}) {fast} else {plain}')
            elif v[1] == '1':
                self.emit(f'if (!{gen}) {{ {settle} {gen} = 1; }} {plain}')
            else:
                self.emit(f'if (!{gen}) {{ if (!{v[1]}) {fast} else {{ {settle} {plain} {gen} = 1; }} }} else {plain}')
        self.source_loops(src, after_first, item)
        self.emit(f'if (!{gen} && {c_} != 0.0 && isfinite({c_})) {f_} += {c_};')
        return f_, ('1' if generic else gen if rt.np == 'dyn' else '0')


    def math_iter(self, e, name):
        line = self.line(e)
        kws = {k.arg: k.value for k in e.keywords}
        if name == 'fsum':
            self.helpers.add('fsum')
            s, r, err = self.temp(), self.temp(), self.temp()
            self.emit(f'sj_fsum_t {s}; {s}.n = 0; {s}.special = 0.0; {s}.inf = 0.0;')

            def item(v, t):
                x = self.hold(self.coerce(v[0], t, T.FLOAT), T.FLOAT)
                self.emit(f'if (sj_fsum_add(&{s}, {x})) SJ_FAIL({OVERFLOW_ERROR}, {line}, 3, 0);')
            self.seq_loops(e.args[0], lambda: None, item)
            self.emit(f'double {r} = 0.0; int {err} = sj_fsum_end(&{s}, &{r});')
            self.emit(f'if ({err}) SJ_FAIL({VALUE_ERROR}, {line}, 14, 0);')
            return r, '0'
        if name == 'prod':
            rt = self.typeof(e)
            acc, rf = self.temp(), self.temp()
            self.emit(f'{rt.ctype} {acc} = 1; int {rf} = 0;')

            def after_first():
                if 'start' in kws:
                    c, _ = self.expr(kws['start'])
                    self.emit(f'{acc} = ({rt.ctype})({c});')

            def item(v, t):
                x = self.hold(self.coerce(v[0], t, rt), rt)
                self.emit(f'{acc} = ' + (f'sj_mul({acc}, {x});' if rt.kind == 'int' else f'{acc} * {x};') + f' {rf} |= {v[1]};')
            self.seq_loops(e.args[0], after_first, item)
            return acc, rf
        # dist, sumprod: both sequences read in full first, as CPython does
        rt = self.typeof(e)
        ct = 'double' if name == 'dist' else 'int64_t'
        bufs, counts, flags = [], [], []
        for a in e.args:
            n = (str(len(a.elts)) if isinstance(a, ast.List) else str(len(self.typeof(a).elem)) if self.is_tuple_expr(a)
                 else self.iter_prep(self.iterable(a))[0])
            buf, cnt, fl = self.temp(), self.temp(), self.temp()
            self.owned.append((buf, ct))
            self.emit(f'{buf} = malloc(sizeof({ct}) * (size_t)({n} + 1)); if (!{buf}) SJ_FAIL({MEMORY_ERROR}, {line}, 0, 0);')
            self.emit(f'int64_t {cnt} = 0; int {fl} = 0;')

            def item(v, t, buf=buf, cnt=cnt, fl=fl):
                to = T.FLOAT if name == 'dist' else T.INT
                self.emit(f'{buf}[{cnt}++] = {self.coerce(v[0], t, to)}; {fl} |= {v[1]};')
            self.seq_loops(a, lambda: None, item)
            bufs.append(buf)
            counts.append(cnt)
            flags.append(fl)
        if name == 'dist':
            self.emit(f'if ({counts[0]} != {counts[1]}) SJ_FAIL({VALUE_ERROR}, {line}, 10, 0);')
            mx, fn = self.temp(), self.temp()
            self.emit(f'double {mx} = 0.0; int {fn} = 0;')
            self.emit(f'for (int64_t k = 0; k < {counts[0]}; k++) {{ {bufs[0]}[k] = fabs({bufs[0]}[k] - {bufs[1]}[k]); '
                      f'{fn} |= {bufs[0]}[k] != {bufs[0]}[k]; if ({bufs[0]}[k] > {mx}) {mx} = {bufs[0]}[k]; }}')
            return self.hold(f'sj_vector_norm({counts[0]}, {bufs[0]}, {mx}, {fn})', T.FLOAT), '0'
        self.emit(f'if ({counts[0]} != {counts[1]}) SJ_FAIL({VALUE_ERROR}, {line}, 11, 0);')
        acc = self.temp()
        self.emit(f'int64_t {acc} = 0;')
        self.emit(f'for (int64_t k = 0; k < {counts[0]}; k++) {acc} = sj_add({acc}, sj_mul({bufs[0]}[k], {bufs[1]}[k]));')
        return acc, f'({flags[0]} | {flags[1]})'

    def math_tuple(self, e, name):
        """frexp and modf: a list of (value, flag)."""
        t = self.typeof(e.args[0])
        c, _ = self.expr(e.args[0])
        x = self.hold(self.coerce(c, t, T.FLOAT), T.FLOAT)
        if name == 'frexp':
            self.helpers.add('frexp')
            m, ex = self.temp(), self.temp()
            self.emit(f'double {m} = {x}; int64_t {ex} = 0;')
            self.emit(f'if (isfinite({x}) && {x} != 0.0) {m} = sj_frexp({x}, &{ex});')
            return [(m, '0'), (ex, '0')]
        fr, ip = self.temp(), self.temp()
        self.emit(f'double {ip} = trunc({x}), {fr} = isinf({x}) ? copysign(0.0, {x}) : copysign({x} - {ip}, {x});')
        return [(fr, '0'), (ip, '0')]

    def math_ported(self, e, name, kind, ts, vals, xs):
        """The functions ported from mathmodule.c that take numbers."""
        line = self.line(e)
        dom = f'SJ_FAIL({VALUE_ERROR}, {line}, 4, 0)'
        rng = f'SJ_FAIL({OVERFLOW_ERROR}, {line}, 1, 0)'
        kws = {k.arg: k.value for k in e.keywords}
        if kind == 'gamma':
            self.uses_libm = True
            self.helpers.add(name)
            r, err = self.temp(), self.temp()
            fn = 'sj_m_tgamma' if name == 'gamma' else 'sj_m_lgamma'
            x = self.hold(xs[0], T.FLOAT)
            self.emit(f'double {r}; int {err} = {fn}(sj_m, {x}, &{r});')
            if self.exact:
                self.emit(f'if ({err} == 1) {dom};')
                self.emit(f'if ({err} == 2) {rng};')
            return r, '0'
        if kind == 'remainder':
            self.helpers.add('remainder')
            a, b = self.hold(xs[0], T.FLOAT), self.hold(xs[1], T.FLOAT)
            r = self.hold(f'sj_m_remainder({a}, {b})', T.FLOAT)
            if self.exact:
                self.emit(f'if ({r} != {r} && {a} == {a} && {b} == {b}) {dom};')
            return r, '0'
        if kind == 'nextafter':
            self.helpers.add('nextafter')
            if 'steps' in kws:
                c, _ = self.expr(kws['steps'])
                st = self.hold(f'(int64_t)({c})', T.INT)
                self.emit(f'if ({st} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 12, 0);')
                return f'sj_nextafter({xs[0]}, {xs[1]}, (uint64_t){st})', '0'
            self.uses_libm = True                       # without steps, CPython calls the C library's
            return f'SJ_LIBM2({LIBM.index("nextafter")})({xs[0]}, {xs[1]})', '0'
        if kind == 'ulp':
            self.helpers.add('ulp')
            return f'sj_ulp({xs[0]})', '0'
        if kind == 'fma':
            a, b, c = (self.hold(x, T.FLOAT) for x in xs)
            r = self.hold(f'fma({a}, {b}, {c})', T.FLOAT)
            self.emit(f'if ({r} != {r} && {a} == {a} && {b} == {b} && {c} == {c}) SJ_FAIL({VALUE_ERROR}, {line}, 13, 0);')
            self.emit(f'if (isinf({r}) && isfinite({a}) && isfinite({b}) && isfinite({c})) SJ_FAIL({OVERFLOW_ERROR}, {line}, 2, 0);')
            return r, '0'
        if kind == 'ldexp':
            self.uses_libm = True
            x = self.hold(xs[0], T.FLOAT)
            i = self.hold(f'(int64_t){vals[1][0]}', T.INT)
            r = self.temp()
            self.emit(f'double {r} = {x};')
            self.emit(f'if ({x} != 0.0 && isfinite({x})) {{ if ({i} > 2147483647LL) {r} = copysign(INFINITY, {x}); '
                      f'else if ({i} < -2147483648LL) {r} = copysign(0.0, {x}); '
                      f'else {r} = ((double (*)(double, int))(uintptr_t)sj_m[{LIBM.index("ldexp")}])({x}, (int){i}); '
                      f'if (isinf({r})) {rng}; }}')
            return r, '0'
        # isclose
        a, b = self.hold(xs[0], T.FLOAT), self.hold(xs[1], T.FLOAT)
        tol = {}
        for k in e.keywords:                            # in the order written, after the arguments
            c, _ = self.expr(k.value)
            tol[k.arg] = self.hold(self.coerce(c, self.typeof(k.value), T.FLOAT), T.FLOAT)
        rel, ab = tol.get('rel_tol', '1e-09'), tol.get('abs_tol', '0.0')
        self.emit(f'if ({rel} < 0.0 || {ab} < 0.0) SJ_FAIL({VALUE_ERROR}, {line}, 9, 0);')
        d = self.hold(f'fabs({b} - {a})', T.FLOAT)
        return (f'({a} == {b} || (!isinf({a}) && !isinf({b}) && ({d} <= fabs({rel} * {b}) || {d} <= fabs({rel} * {a}) '
                f'|| {d} <= {ab})))'), '0'

    def math_int(self, e, name, vals):
        line = self.line(e)
        xs = [self.hold(f'(int64_t){c}', T.INT) for c, _ in vals]
        if name in ('gcd', 'lcm'):
            self.helpers.add(name)
            fn = 'sj_gcd2' if name == 'gcd' else 'sj_lcm2'
            if not xs:
                return ('0LL' if name == 'gcd' else '1LL'), '0'
            acc = self.temp()
            self.emit(f'int64_t {acc} = sj_abs({xs[0]});')
            for x in xs[1:]:
                self.emit(f'{acc} = {fn}({acc}, {x});')
            return acc, '0'
        if name == 'isqrt':
            self.helpers.add('isqrt')
            self.emit(f'if ({xs[0]} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 6, 0);')
            return f'sj_isqrt({xs[0]})', '0'
        if name == 'factorial':
            self.helpers.add('perm')
            self.emit(f'if ({xs[0]} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 5, 0);')
            return f'sj_perm({xs[0]}, {xs[0]})', '0'
        self.helpers.add(name)
        if len(xs) == 1:                                # perm(n) is factorial(n)
            self.emit(f'if ({xs[0]} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 5, 0);')
            return f'sj_perm({xs[0]}, {xs[0]})', '0'
        n, k = xs
        self.emit(f'if ({n} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 7, 0);')
        self.emit(f'if ({k} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 8, 0);')
        return f'sj_{name}({n}, {k})', '0'

    def math_call(self, e, name):
        kind, can_overflow = MATH_FUNCS[name]
        if kind == 'iter':
            return self.math_iter(e, name)
        line = self.line(e)
        ts = [self.typeof(a) for a in e.args]
        vals = [self.expr(a) for a in e.args]          # arguments first, in order, as Python does
        vals = [(self.hold(c, t), self.hold_flag(f)) for (c, f), t in zip(vals, ts)]
        if kind == 'intf':
            return self.math_int(e, name, vals)
        xs = [self.coerce(c, t, T.FLOAT) for (c, _), t in zip(vals, ts)]
        if kind in ('gamma', 'remainder', 'nextafter', 'ulp', 'fma', 'ldexp', 'isclose'):
            return self.math_ported(e, name, kind, ts, vals, xs)
        dom = f'SJ_FAIL({VALUE_ERROR}, {line}, 4, 0)'
        rng = f'SJ_FAIL({OVERFLOW_ERROR}, {line}, 1, 0)'

        def libm(n):
            self.uses_libm = True
            return LIBM.index(n)
        if not self.exact and (kind in ('f1', '1a', 'log', 'atan2', 'pow', 'fmod') or (kind == 'hypot' and len(xs) == 2)):
            # parity=False: cimo's C library, IEEE results (nan, inf) instead of exceptions
            if kind == 'log' and len(xs) == 2:
                return f'(log({xs[0]}) / log({xs[1]}))', '0'
            return f'{name}({", ".join(xs)})', '0'
        if kind in ('f1', '1a'):
            x = self.hold(xs[0], T.FLOAT)
            if name == 'sqrt':
                call = f'sqrt({x})'
            elif name == 'fabs':
                call = f'fabs({x})'
            elif name == 'log1p':                       # CPython's _Py_log1p keeps -0.0
                call = f'({x} == 0.0 ? {x} : SJ_LIBM1({libm(name)})({x}))'
            else:
                call = f'SJ_LIBM1({libm(name)})({x})'
            r = self.hold(call, T.FLOAT)
            if kind == 'f1':                            # math_1
                self.emit(f'if ({r} != {r} && {x} == {x}) {dom};')
                self.emit(f'if (isinf({r}) && isfinite({x})) {rng if can_overflow else dom};')
            return r, '0'
        if kind == 'log':
            def loghelper(i, fn):
                t = ts[i]
                if t.kind in ('int', 'bool') and t.np is False:     # a Python int: checked directly
                    self.emit(f'if ((int64_t){vals[i][0]} <= 0) {dom};')
                    return self.hold(f'sj_m_log(SJ_LIBM1({libm(fn)}), (double){vals[i][0]})', T.FLOAT)
                x = self.hold(xs[i], T.FLOAT)
                r = self.hold(f'sj_m_log(SJ_LIBM1({libm(fn)}), {x})', T.FLOAT)
                self.emit(f'if ({r} != {r} && {x} == {x}) {dom};')
                self.emit(f'if (isinf({r}) && isfinite({x})) {dom};')
                return r
            num = loghelper(0, name)
            if len(e.args) == 1:
                return num, '0'
            den = loghelper(1, 'log')
            self.emit(f'if ({den} == 0.0) SJ_FAIL({ZERO_DIVISION}, {line}, 1, 0);')
            return f'({num} / {den})', '0'
        if kind == 'atan2':
            y, x = self.hold(xs[0], T.FLOAT), self.hold(xs[1], T.FLOAT)
            r = self.hold(f'sj_m_atan2(SJ_LIBM2({libm("atan2")}), {y}, {x})', T.FLOAT)
            self.emit(f'if ({r} != {r} && {y} == {y} && {x} == {x}) {dom};')
            self.emit(f'if (isinf({r}) && isfinite({y}) && isfinite({x})) {rng};')
            return r, '0'
        if kind in ('pow', 'fmod'):
            r, err = self.temp(), self.temp()
            self.emit(f'double {r}; int {err} = ' + (f'sj_math_pow(SJ_LIBM2({libm("pow")}), {xs[0]}, {xs[1]}, &{r});'
                                                    if kind == 'pow' else f'sj_math_fmod({xs[0]}, {xs[1]}, &{r});'))
            self.emit(f'if ({err} == 1) {dom};')
            if kind == 'pow':
                self.emit(f'if ({err} == 2) {rng};')
            return r, '0'
        if kind == 'copysign':
            return f'copysign({xs[0]}, {xs[1]})', '0'
        if kind == 'hypot':
            if not xs:
                return '0.0', '0'
            v, mx, fn = self.temp(), self.temp(), self.temp()
            self.emit(f'double {v}[{len(xs)}] = {{{", ".join(f"fabs({x})" for x in xs)}}}; double {mx} = 0.0; int {fn} = 0;')
            self.emit(f'for (int k = 0; k < {len(xs)}; k++) {{ {fn} |= {v}[k] != {v}[k]; if ({v}[k] > {mx}) {mx} = {v}[k]; }}')
            return self.hold(f'sj_vector_norm({len(xs)}, {v}, {mx}, {fn})', T.FLOAT), '0'
        if kind == 'scale':
            k = 180.0 / math.pi if name == 'degrees' else math.pi / 180.0
            return f'({xs[0]} * {c_float(k)})', '0'
        if kind == 'test':
            x = self.hold(xs[0], T.FLOAT)
            return {'isnan': f'({x} != {x})', 'isinf': f'isinf({x})', 'isfinite': f'isfinite({x})'}[name], '0'
        # floor, ceil, trunc: a Python int is returned as it is (its __floor__ etc.); a float, or a
        # numpy int (no __floor__: converted to float first), is rounded and converted to int
        t, (c, f) = ts[0], vals[0]
        fn = {'floor': 'floor', 'ceil': 'ceil', 'trunc': 'trunc'}[name]

        def rounded():
            x = self.hold(f'{fn}({xs[0]})', T.FLOAT)
            self.emit(f'if ({x} != {x}) SJ_FAIL({VALUE_ERROR}, {line}, 0, 0);')
            self.emit(f'if (isinf({x})) SJ_FAIL({OVERFLOW_ERROR}, {line}, 0, 0);')
            return f'sj_float_to_int({x})'
        if t.kind == 'float':
            return rounded(), '0'
        if t.np is False:
            return self.coerce(c, t, T.INT), '0'
        if t.np is True:
            return rounded(), '0'
        r = self.temp()
        self.emit(f'int64_t {r};')
        self.open(f'if ({f})')
        self.emit(f'{r} = {rounded()};')
        self.close('} else {')
        self.depth += 1
        self.emit(f'{r} = (int64_t)({c});')
        self.close()
        return r, '0'


def ends_with_return(stmts):
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, ast.Return):
        return True
    if isinstance(last, ast.If):
        return ends_with_return(last.body) and ends_with_return(last.orelse)
    if isinstance(last, ast.While) and isinstance(last.test, ast.Constant) and last.test.value is True:
        return not any(isinstance(n, ast.Break) for n in ast.walk(last))
    if isinstance(last, (ast.For, ast.While)) and ends_with_return(last.orelse):
        return not has_own_break(last.body)      # the else is the only way past it
    return False


def has_own_break(stmts):
    """A break in these statements that leaves the loop they're the body of."""
    for s in stmts:
        if isinstance(s, ast.Break):
            return True
        if isinstance(s, (ast.For, ast.While)):
            if has_own_break(s.orelse):
                return True
        elif isinstance(s, ast.If) and (has_own_break(s.body) or has_own_break(s.orelse)):
            return True
    return False


# ---- definite assignment: every name is assigned on every path before it is read

ALL = None   # "every name": the set after a statement that never completes (return, break)


def da_reads(c, node, assigned):
    local = {n.id for comp in ast.walk(node) if isinstance(comp, (ast.GeneratorExp, ast.ListComp))
             for g in comp.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in local:
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in c.env \
                and assigned is not ALL and n.id not in assigned:
            c.fail(n, f'{n.id} may be read before it is assigned')


def da_block(c, stmts, assigned):
    for s in stmts:
        if assigned is ALL:
            return ALL
        assigned = da_stmt(c, s, assigned)
    return assigned


def da_stmt(c, s, assigned):
    if isinstance(s, ast.Assign) and len(s.targets) > 1:
        da_reads(c, s.value, assigned)
        for t in s.targets:
            assigned = da_stmt(c, ast.Assign(targets=[t], value=ast.Constant(0)), assigned)
        return assigned
    if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        da_reads(c, s.value, assigned)
        target = s.targets[0] if isinstance(s, ast.Assign) else s.target
        if isinstance(target, ast.Subscript):
            da_reads(c, target.value, assigned)
            da_reads(c, target.slice, assigned)
        elif isinstance(s, ast.AugAssign):
            da_reads(c, target, assigned)
        if isinstance(target, (ast.Tuple, ast.List)):
            for x in ast.walk(target):
                if isinstance(x, ast.Subscript):
                    da_reads(c, x.slice, assigned)
            return assigned | {x.id for x in ast.walk(target) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        return assigned | {target.id} if isinstance(target, ast.Name) else assigned
    if isinstance(s, ast.If):
        da_reads(c, s.test, assigned)
        a = da_block(c, s.body, set(assigned))
        b = da_block(c, s.orelse, set(assigned))
        if a is ALL:
            return b
        if b is ALL:
            return a
        return a & b
    if isinstance(s, (ast.While, ast.For)):
        inside = set(assigned)
        if isinstance(s, ast.While):
            da_reads(c, s.test, assigned)
        else:
            da_reads(c, s.iter, assigned)
            inside |= {n.id for n in ast.walk(s.target) if isinstance(n, ast.Name)}
        c.da_breaks.append([])
        da_block(c, s.body, inside)
        breaks = c.da_breaks.pop()
        if isinstance(s, ast.While) and isinstance(s.test, ast.Constant) and s.test.value is True:
            out = ALL                             # while True: only the breaks lead past it, and none means never
            for b in breaks:
                out = b if out is ALL else out & b
            return out
        if not s.orelse:
            return assigned
        out = da_block(c, s.orelse, set(assigned))    # only the else and the breaks lead past this loop
        for b in breaks:
            out = b if out is ALL else out & b
        return out
    if isinstance(s, ast.Return):
        if s.value is not None:
            da_reads(c, s.value, assigned)
        return ALL
    if isinstance(s, ast.Break):
        c.da_breaks[-1].append(set(assigned))
        return ALL
    if isinstance(s, ast.Continue):
        return ALL
    return assigned
