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
import math

from . import _types as T
from ._errors import Unsupported
import sys

from ._mathlib import snippets
from ._runtime import (INDEX_ERROR, INTERRUPTED, INT_BOUNDS_ERROR, LIBM, LIST_INDEX_ERROR, MEMORY_ERROR, NARROW_INT_DTYPES,
                       OVERFLOW_ERROR, VALUE_ERROR, ZERO_DIVISION)

BUILTINS = ('range', 'len', 'abs', 'min', 'max', 'int', 'float', 'round', 'sum', 'any', 'all')
MATH_CONSTANTS = {'pi': math.pi, 'e': math.e, 'tau': math.tau, 'inf': math.inf, 'nan': math.nan}
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
ARITH = {ast.Add: 'Add', ast.Sub: 'Sub', ast.Mult: 'Mult', ast.Div: 'Div', ast.FloorDiv: 'FloorDiv',
         ast.Mod: 'Mod', ast.BitAnd: 'BitAnd', ast.BitOr: 'BitOr', ast.BitXor: 'BitXor', ast.Pow: 'Pow'}
CMP = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
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


class Compiler:
    def __init__(self, fdef, arg_types, globals_, filename, line_offset, boundscheck=True, parity=True):
        local = {p.arg for p in fdef.args.posonlyargs + fdef.args.args}
        local |= {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.f = Lazy(local, set(globals_)).visit(copy.deepcopy(fdef))
        self.exact = parity    # False: C's float semantics where CPython parity costs speed
        self.globals = globals_
        self.filename = filename
        self.offset = line_offset
        self.boundscheck = boundscheck
        a = fdef.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.kw_defaults:
            raise Unsupported(fdef, 'only plain positional parameters are compiled')
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
        return node.lineno + self.offset

    def fail(self, node, reason):
        raise Unsupported(node, reason)

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
            j = t if ret is None else T.join(ret, t)
            if j is None:
                self.fail(node, f'returns {ret.name} in one place and {t.name} in another')
            ret = j
        self.ret = ret or T.NONE
        if self.ret is not T.NONE and not ends_with_return(self.f.body):
            self.fail(self.f, 'can fall off the end (returning None) and also return a value')

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
        if t.kind == 'array':
            self.fail(node, f'{name}: arrays can only be parameters')
        if t is T.NONE:
            self.fail(node, f'{name} would be None')
        old = self.env.get(name)
        if old is not None and old.kind == 'array':
            self.fail(node, f'{name} is reassigned')
        if old is not None and old is not t:
            j = T.join(old, t)
            if j is None:
                self.fail(node, f'{name} is {old.name} and later {t.name}; a variable must keep one type')
            t = j
        self.env[name] = t

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
            vt = self.typeof(s.value)
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
            if isinstance(s, ast.While) and s.orelse:
                self.fail(s, 'while ... else')
            self.infer_block(s.orelse)
        elif isinstance(s, ast.For):
            if s.orelse:
                self.fail(s, 'for ... else')
            self.bind_types(s.target, self.elem_struct(self.iterable(s.iter)), s)
            lists = {n.id for n in ast.walk(s.iter) if isinstance(n, ast.Name)
                     and self.env.get(n.id) is not None and self.env[n.id].kind == 'list'}
            for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
                if isinstance(n, ast.Name) and n.id in lists and isinstance(n.ctx, ast.Store):
                    self.fail(n, f'rebinds {n.id} inside a loop over it (Python keeps iterating the old list)')
            self.infer_block(s.body)
        elif isinstance(s, ast.Return):
            self.returns.append((s, self.typeof(s.value) if s.value is not None else T.NONE))
        elif isinstance(s, (ast.Break, ast.Continue, ast.Pass)):
            pass
        elif isinstance(s, ast.Expr) and docstring_ok and isinstance(s.value, ast.Constant) \
                and isinstance(s.value.value, str):
            pass
        elif isinstance(s, ast.Expr):
            self.fail(s, 'an expression statement (calls with side effects are not compiled yet)')
        else:
            self.fail(s, f'the {type(s).__name__} statement')

    def infer_target(self, target, t, s):
        if isinstance(target, ast.Name):
            self.assign_type(s, target.id, t)
        elif isinstance(target, ast.Subscript):
            at = self.array_of(target)
            self.index_type(target)
            if at.kind == 'list':             # a list element takes the stored value's type
                e = T.join(at.elem, t) if t.is_number else None
                if e is None:
                    self.fail(s, f'stores {t.name} into a {at.name}')
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
        if not t.is_number:
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
                self.fail(it, 'iterating over a multi-dimensional array (rows are views, not compiled yet)')
            return ('array', it.id)
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
                if inner[0] not in ('range', 'array'):
                    self.fail(it, 'reversed() of something other than range() or an array')
                return ('reversed', inner)
        self.fail(it, 'loops over something other than range(), an array, enumerate(), zip() or reversed()')

    def elem_struct(self, d):
        kind = d[0]
        if kind == 'range':
            return T.INT
        if kind == 'array':
            return self.env[d[1]].elem
        if kind == 'enumerate':
            return [T.INT, self.elem_struct(d[1])]
        if kind == 'zip':
            return [self.elem_struct(x) for x in d[1]]
        return self.elem_struct(d[1])

    def bind_types(self, target, struct, s):
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

    def array_of(self, sub):
        if not isinstance(sub.value, ast.Name) or sub.value.id not in self.env \
                or self.env[sub.value.id].kind not in ('array', 'list'):
            self.fail(sub, 'indexing something other than an array parameter or a list')
        return self.env[sub.value.id]

    def tuple_index(self, sub):
        t = self.env[sub.value.id]
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
        t = self.env[sub.value.id]
        nd = t.ndim if t.kind == 'array' else 1
        idx = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        if len(idx) != nd:
            self.fail(sub, f'{len(idx)} indices into {t.name} (slices and views are not compiled yet)')
        return idx

    def index_type(self, sub):
        for k in self.indices(sub):
            if isinstance(k, ast.Slice):
                self.fail(sub, 'slices (views are not compiled yet)')
            t = self.typeof(k)
            if t.kind != 'int':
                self.fail(sub, f'an index of type {t.name}')

    def binop_type(self, a, b, op, node):
        if not (a.is_number and b.is_number):
            self.fail(node, f'{op} on {a.name} and {b.name}')
        if op == 'Pow':
            return self.pow_type(a, b, node)
        if op in ('BitAnd', 'BitOr', 'BitXor'):
            if a.kind == 'float' or b.kind == 'float':
                self.fail(node, 'a bitwise operator on a float')
            if a is T.BOOL and b is T.BOOL:
                return T.BOOL
        r = T.arith_result(a, b, op)
        if r is None:
            self.fail(node, 'arithmetic on numpy bools')
        return r

    def pow_type(self, a, b, node):
        """a ** b. numpy's rules give one type whatever the values. Python's give a float for an int to
        a negative power, and a complex for a negative float to a fractional power. Compiled code has one
        type per expression, so it assumes neither happens and raises if one does: a documented
        difference from Python. A constant exponent or base that rules it out has no such case."""
        r = T.arith_result(a, b, 'Pow')
        if r is None:
            self.fail(node, 'arithmetic on numpy bools')
        if r.np is True:
            return r
        exp = const_number(node.right if isinstance(node, ast.BinOp) else node.value)
        if r.kind == 'int':
            if exp is None:
                self.gotcha(node, 'raises ValueError if the exponent is negative, where Python gives a float')
                return r
            if exp >= 0:
                return r
            if r.np:
                self.fail(node, 'int ** a negative int on a value that may be a numpy int')
            return T.FLOAT
        base = const_number(node.left) if isinstance(node, ast.BinOp) else None
        integral = b.kind != 'float' or (exp is not None and (exp != exp or abs(exp) == math.inf or exp == math.floor(exp)))
        if self.exact and not integral and not (base is not None and base >= 0):
            self.gotcha(node, 'raises ValueError if the base is negative, where Python gives a complex')
        return r

    def gotcha(self, node, text):
        """Note that this line can differ from Python, for preflight."""
        line, have = self.line(node), self.gotchas.get(self.line(node))
        self.gotchas[line] = text if have is None or text in have else f'{have}; {text}'

    def join_all(self, node, ts, what):
        r = ts[0]
        for t in ts[1:]:
            j = T.join(r, t)
            if j is None:
                self.fail(node, f'{what} of values of different types (the result type would vary)')
            r = j
        return r

    def typeof(self, e):
        if isinstance(e, ast.Constant):
            v = e.value
            if isinstance(v, bool):
                return T.BOOL
            if isinstance(v, int):
                if not INT64_MIN <= v <= INT64_MAX:
                    self.fail(e, 'an int literal outside int64')
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
            if isinstance(e.value, ast.Name) and e.value.id not in self.env \
                    and self.globals.get(e.value.id) is math and e.attr in MATH_CONSTANTS:
                return T.FLOAT
            if isinstance(e.value, ast.Name) and e.value.id in self.env and self.env[e.value.id].kind == 'array':
                t = self.env[e.value.id]
                if e.attr == 'shape':
                    return T.tuple_of([T.INT] * t.ndim)
                if e.attr in ('ndim', 'size'):
                    return T.INT
            self.fail(e, f'the attribute .{e.attr}')
        if isinstance(e, ast.BinOp):
            op = ARITH.get(type(e.op))
            if op is None:
                self.fail(e, OPERATOR_NAMES.get(type(e.op).__name__, f'the {type(e.op).__name__} operator'))
            return self.binop_type(self.typeof(e.left), self.typeof(e.right), op, e)
        if isinstance(e, ast.UnaryOp):
            t = self.typeof(e.operand)
            if not t.is_number:
                self.fail(e, f'a unary operator on {t.name}')
            if isinstance(e.op, ast.Not):
                return T.BOOL
            if t.np and t.kind == 'bool':
                self.fail(e, 'arithmetic on numpy bools')
            if isinstance(e.op, ast.Invert) and t.kind != 'int':
                self.fail(e, '~ on a non-int')
            return T.INT if t is T.BOOL else t
        if isinstance(e, ast.Compare):
            if not all(type(o) in CMP for o in e.ops):
                self.fail(e, 'in, not in, is and is not')
            ts = [self.typeof(x) for x in [e.left] + e.comparators]
            if not all(t.is_number for t in ts):
                self.fail(e, 'comparisons of non-numbers')
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
            if not t.is_number:
                self.fail(e, f'a conditional expression of type {t.name}')
            return t
        if isinstance(e, ast.Subscript):
            if self.is_shape(e.value):
                if self.shape_index(e) is None:
                    self.fail(e, 'a.shape indexed with something other than a constant in range')
                return T.INT
            if isinstance(e.value, ast.Name) and e.value.id in self.env and self.env[e.value.id].kind == 'tuple':
                return self.env[e.value.id].elem[self.tuple_index(e)]
            at = self.array_of(e)
            self.index_type(e)
            return at.elem
        if isinstance(e, ast.Tuple):
            ts = [self.typeof(x) for x in e.elts]
            if not all(t.is_number for t in ts):
                self.fail(e, 'tuples of non-numbers (nested tuples, arrays)')
            return T.tuple_of(ts)
        if isinstance(e, ast.Call):
            return self.call_type(e)
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

    def call_type(self, e):
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
        t = self.elem_struct(self.iterable(src))
        if isinstance(t, list):
            self.fail(src, 'a reduction over tuples')
        return t

    def comp_types(self, e):
        saved = {}
        try:
            for g in e.generators:
                if g.is_async:
                    self.fail(e, 'async comprehensions')
                self.comp_bind_types(g.target, self.elem_struct(self.iterable(g.iter)), saved, e)
                for c in g.ifs:
                    self.check_test(c)
            t = self.typeof(e.elt)
            if not t.is_number:
                self.fail(e.elt, f'elements of type {t.name}')
            return t
        finally:
            self.restore(saved)

    def comp_bind_types(self, target, struct, saved, e):
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
        name = self.filename.replace('\\', '\\\\').replace('"', '\\"')
        self.lines.append(f'#line {self.line(node)} "{name}"')

    def generate(self):
        self.infer()
        da_block(self, self.f.body, set(self.params))
        self.find_aliases()
        self.facts.update(self.mod_assigned())
        params = []
        for p in self.params:
            t = self.env[p]
            if t.kind == 'array':
                params += [f'{t.ctype}v_{p}', f'int64_t v_{p}__n'] + [f'int64_t v_{p}__n{k}' for k in range(1, t.ndim)]
            elif t.kind == 'tuple':
                params += [f'{et.ctype} v_{p}__{i}' for i, et in enumerate(t.elem)]
            else:
                params.append(f'{t.ctype} v_{p}')
        self.mark(self.f)
        head = len(self.lines)
        for name, t in self.env.items():
            if t.kind == 'list':
                self.emit(f'{t.elem.ctype} *v_{name} = NULL; int64_t v_{name}__n = 0;')
                if t.elem.np == 'dyn':
                    self.emit(f'int *v_{name}__np = NULL;')
                continue
            if t.kind == 'tuple':
                for i, et in enumerate(t.elem):
                    if name not in self.params:
                        self.emit(f'{et.ctype} v_{name}__{i} = 0;')
                    if et.np == 'dyn':
                        self.emit(f'int v_{name}__{i}__np = 0;')
                continue
            if name not in self.params:
                self.emit(f'{t.ctype} v_{name} = 0;')
            if t.np == 'dyn':
                self.emit(f'int v_{name}__np = 0;')
        self.emit('int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;')
        decl_at = len(self.lines)
        for s in self.f.body:
            self.stmt(s)
        frees = [f'free(v_{n});' for n, t in self.env.items() if t.kind == 'list'] + \
                [f'free(v_{n}__np);' for n, t in self.env.items() if t.kind == 'list' and t.elem.np == 'dyn'] + \
                [f'free({o});' for o, _ in self.owned]
        self.lines[decl_at:decl_at] = ['    ' + f'{ct} *{o} = NULL;' for o, ct in self.owned]
        self.lines.append('sj_done:')
        if frees:
            self.emit(' '.join(frees))
        self.emit('return 0;')
        self.lines.append('sj_fail:')
        if frees:
            self.emit(' '.join(frees))
        self.emit('sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;')
        self.emit('return sj_st;')
        if self.uses_libm:
            params.append('const uint64_t *sj_m')
        if self.uses_stop:
            params.append('volatile int *sj_stop')
        params.append('int64_t *sj_io')
        start = ['    *sj_stop = 0;'] if self.uses_stop else []
        return snippets(self.helpers) + '\n'.join(self.lines[:head] + [f'int sj_kernel({", ".join(params)}) {{'] + start
                         + self.lines[head:] + ['}', ''])

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
        if t.np == 'dyn':
            self.emit(f'v_{name}__np = {v[1]};')

    def stmt(self, s):
        if isinstance(s, ast.Expr):           # a docstring
            return
        self.mark(s)
        if isinstance(s, ast.Assign) and self.is_list_ctor(s.value):
            self.make_list(s.targets[0].id, s.value, s)
            return
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            vt = self.typeof(s.value)
            if vt.kind == 'tuple':
                vals = self.texpr(s.value)
                for target in targets:
                    self.assign_tuple(target, vals, vt)
                return
            c, f = self.expr(s.value)
            v = (self.hold(c, vt), self.hold_flag(f)) if len(targets) > 1 else (c, f)
            for target in targets:
                if isinstance(target, ast.Name):
                    self.assign(target.id, v, vt)
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
                arr, idx = self.element(s.target)     # checked once, before the right side
                name = s.target.value.id
                at = self.env[name]
                rt = self.binop_type(at.elem, self.typeof(s.value), op, s)
                r = self.arith(op, (self.hold(f'{arr}[{idx}]', at.elem), self.elem_flag(name, idx)), at.elem, s.value, s)
                self.emit(f'{arr}[{idx}] = {self.coerce(r[0], rt, at.elem)};')
                if at.kind == 'list' and at.elem.np == 'dyn':
                    self.emit(f'v_{name}__np[{idx}] = {r[1]};')
        elif isinstance(s, ast.If):
            self.open(f'if ({self.truth(self.expr(s.test)[0], self.typeof(s.test))})')
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
        elif isinstance(s, ast.While):
            self.open('for (;;)')
            self.loops.append(None)
            if self.polls:
                self.emit(self.poll())
            self.emit(f'if (!({self.truth(self.expr(s.test)[0], self.typeof(s.test))})) break;')
            for b in s.body:
                self.stmt(b)
            self.loops.pop()
            self.close()
        elif isinstance(s, ast.For):
            d = self.iterable(s.iter)
            if d[0] == 'range':
                self.for_range(s)
            else:
                self.for_iter(s, d)
        elif isinstance(s, ast.Return):
            if s.value is not None and self.ret.kind == 'tuple':
                vals = self.texpr(s.value)
                for i, ((c, f), et, rt) in enumerate(zip(vals, self.typeof(s.value).elem, self.ret.elem)):
                    val = self.coerce(c, et, rt)
                    if rt.kind == 'float':
                        t = self.temp()
                        self.emit(f'{{ double {t} = {val}; memcpy(&sj_io[{4 + 2 * i}], &{t}, 8); }}')
                    else:
                        self.emit(f'sj_io[{4 + 2 * i}] = (int64_t)({val});')
                    self.emit(f'sj_io[{5 + 2 * i}] = {f};')
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
                    self.emit(f'sj_io[2] = {v[1]};')
            self.emit('goto sj_done;')
        elif isinstance(s, ast.Break):
            loop = self.loops[-1]
            if loop is None:
                self.emit('break;')
            else:                                       # out of the chunks as well
                loop[1] = True
                self.emit(f'goto {loop[0]};')
        elif isinstance(s, ast.Continue):
            self.emit('continue;')

    def for_range(self, s):
        args = [self.coerce(self.expr(a)[0], self.typeof(a), T.INT) for a in s.iter.args]
        var = s.target.id
        start, stop = ('0LL', args[0]) if len(args) == 1 else (args[0], args[1])
        step = args[2] if len(args) == 3 else None
        ts, te = self.temp(), self.temp()
        self.emit(f'int64_t {ts} = {start}, {te} = {stop};')
        unit = step is None or (isinstance(s.iter.args[2], ast.Constant) and s.iter.args[2].value == 1)
        bounds = [const_number(a) for a in s.iter.args]
        short = None not in bounds and (bounds[1] - bounds[0] if len(bounds) > 1 else bounds[0]) <= 65536 \
            and (len(bounds) < 3 or bounds[2] > 0)
        fact = self.range_fact(s) if unit else None
        if unit and fact and self.boundscheck:
            # loop versioning: if no access var + c can leave its array during this loop, run a
            # copy without those checks; otherwise the checked copy raises where Python would
            extra = self.version_candidates(s, var, fact)
            if extra:
                cond = ' && '.join(
                    f'{self.extent(*key[3])} <= {self.extent(key[0], key[1])}' if len(key) > 2 else
                    f'{te} <= {self.extent(key[0], key[1])} - ({c})' for key, c in sorted(extra.items(), key=str))
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
        self.short_range = short
        self.open_counted('uint64_t', tk, '0', tn)
        self.emit(f'v_{var} = (int64_t)((uint64_t){ts} + {tk} * (uint64_t){tp});')
        self.loop_body(s, var, None)

    def iter_prep(self, d):
        """Evaluate an iterable's operands (in Python's order); return (trip count, elem(k)),
        where elem(k) gives (C value, type, flag) leaves, nested in lists for tuples."""
        kind = d[0]
        if kind == 'array':
            name, t = d[1], self.env[d[1]].elem
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
        if kind == 'zip':
            parts = [self.iter_prep(x) for x in d[1]]
            n = self.temp()
            self.emit(f'uint64_t {n} = {parts[0][0]};')
            for c, _ in parts[1:]:
                self.emit(f'if ({c} < {n}) {n} = {c};')
            return n, lambda k: [f(k) for _, f in parts]
        inner = d[1]                                  # reversed
        if inner[0] == 'array':
            name, t = inner[1], self.env[inner[1]].elem
            return f'(uint64_t)v_{name}__n', lambda k: (f'v_{name}[v_{name}__n - 1 - (int64_t){k}]', t,
                                                        self.elem_flag(name, f'(v_{name}__n - 1 - (int64_t){k})'))
        n, _ = self.iter_prep(inner)
        ts, tp, tn = self.range_parts
        return n, lambda k: (f'(int64_t)((uint64_t){ts} + ({tn} - 1 - {k}) * (uint64_t){tp})', T.INT, '0')

    def bind_values(self, target, v):
        if isinstance(target, ast.Name):
            c, t, f = v
            self.assign(target.id, (c, f), t)
        else:
            for t, x in zip(target.elts, v):
                self.bind_values(t, x)

    def for_iter(self, s, d):
        n, elem = self.iter_prep(d)
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
        hi = (inner[1], 0) if inner[0] == 'array' and lo == 0 else None
        return var, (lo, hi, 0)

    def unit_loop(self, s, var, ts, te, fact, short=False):
        ti = self.temp()
        self.short_range = short
        self.open_counted('int64_t', ti, ts, te)
        self.emit(f'v_{var} = {ti};')
        self.loop_body(s, var, fact)

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

    def version_candidates(self, s, var, fact):
        """Indexes var + c on some axis of an array, nonnegative but not proven below that axis's
        extent: the largest c for each (array, axis)."""
        self.facts[var] = fact
        extra = {}
        try:
            for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
                if not (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                        and n.value.id in self.env and self.env[n.value.id].kind == 'array'):
                    continue
                name = n.value.id
                idxs = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
                if len(idxs) != self.env[name].ndim:
                    continue
                for axis, idx in enumerate(idxs):
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
        return (lo, hi, slack) if lo is not None or hi else None

    def len_of(self, e):
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
            elif isinstance(t, ast.Tuple) and self.is_shape(v):
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
        for t in tests:
            if not (isinstance(t, ast.Compare) and len(t.ops) == 1):
                continue
            left, op, right = t.left, type(t.ops[0]), t.comparators[0]
            if isinstance(right, ast.Name) and not isinstance(left, ast.Name):
                left, right = right, left
                op = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}.get(op, op)
            if not isinstance(left, ast.Name) or left.id in stored \
                    or self.env.get(left.id) not in (T.INT, T.NP_INT64, T.INT_DYN):
                continue
            lo, hi, slack = self.facts.get(left.id, (None, None, 0))[:3]
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
        name = sub.value.id
        is_list = self.env[name].kind == 'list'
        idxs = self.indices(sub)
        raws = [self.coerce(self.expr(k)[0], self.typeof(k), T.INT) for k in idxs]
        raws = [self.hold(r, T.INT) for r in raws]            # all indices first, in order
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
                    self.hoisted_lines.add(self.line(sub))
                else:
                    self.checks[self.line(sub)] = self.checks.get(self.line(sub), 0) + 1
                self.emit(f'if ((uint64_t){t} >= (uint64_t){n}) {{ sj_io[4] = {axis}; SJ_FAIL({INDEX_ERROR}, {self.line(sub)}, {raw}, {n}); }}')
            lin = t if lin is None else f'({lin} * {n} + {t})'
        if len(idxs) > 1:
            lin = self.hold(lin, T.INT)
        return f'v_{name}', lin

    def elem_flag(self, name, idx):
        t = self.env[name]
        if t.kind == 'array':
            return '1'
        return f'v_{name}__np[{idx}]' if t.elem.np == 'dyn' else flag_of(t.elem)

    def store_elem(self, target, v, vt):
        """target[i] = v (already computed): the index is evaluated and checked now."""
        name = target.value.id
        et = self.env[name].elem
        dtype = et.name[3:]
        flag = self.hold_flag(v[1])
        arr, idx = self.element(target, store=True)       # numpy checks the index first
        if dtype in T.INT_RANGE and vt.kind in ('int', 'bool') and vt.np is not True:
            # numpy (NEP 50): a Python int that doesn't fit the dtype raises OverflowError; numpy
            # scalars are cast (and wrap)
            lo, hi = T.INT_RANGE[dtype]
            iv = self.hold(self.coerce(v[0], vt, T.INT), T.INT)
            cond = f'{iv} < {c_int(lo)} || {iv} > {c_int(hi)}'
            if vt.np == 'dyn':
                cond = f'!{v[1]} && ({cond})'
            self.emit(f'if ({cond}) SJ_FAIL({INT_BOUNDS_ERROR}, {self.line(target)}, {iv}, {NARROW_INT_DTYPES.index(dtype)});')
        val = self.hold(self.coerce(v[0], vt, et), et)
        self.emit(f'{arr}[{idx}] = {val};')
        if self.env[name].kind == 'list' and et.np == 'dyn':
            self.emit(f'v_{name}__np[{idx}] = {flag};')

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

        def alloc(count):
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
                alloc(f'{reps} * {len(vals)}')
                r = self.temp()
                self.open_counted('int64_t', r, '0', reps)
                for c, f, t in vals:
                    push(c, f, t)
                self.close()
        self.emit(f'free(v_{name}); v_{name} = {p}; v_{name}__n = {n}; {p} = NULL;')
        if dyn:
            self.emit(f'free(v_{name}__np); v_{name}__np = {pf}; {pf} = NULL;')

    def tuple_var(self, name):
        t = self.env[name]
        return [(f'v_{name}__{i}', f'v_{name}__{i}__np' if et.np == 'dyn' else flag_of(et))
                for i, et in enumerate(t.elem)]

    def texpr(self, e):
        """Emit a tuple-valued expression: a list of (value, flag), each computed now."""
        t = self.typeof(e)
        if isinstance(e, ast.Tuple):
            out = []
            for x, et in zip(e.elts, t.elem):
                c, f = self.expr(x)
                out.append((self.hold(c, et), self.hold_flag(f)))
            return out
        if isinstance(e, ast.Name):
            return [(self.hold(c, et), self.hold_flag(f)) for (c, f), et in zip(self.tuple_var(e.id), t.elem)]
        if self.is_shape(e):
            return [(self.extent(e.value.id, k), '0') for k in range(len(t.elem))]
        if isinstance(e, ast.Call) and self.math_func(e.func):
            return self.math_tuple(e, self.math_func(e.func))
        raise AssertionError(e)

    def extent(self, name, k):
        return f'v_{name}__n' if k == 0 else f'v_{name}__n{k}'

    def assign_tuple(self, target, vals, vt):
        """Assign a tuple (already computed in full, as Python does) to a name or to targets."""
        if isinstance(target, ast.Name):
            t = self.env[target.id]
            for i, ((c, f), et, tt) in enumerate(zip(vals, vt.elem, t.elem)):
                self.emit(f'v_{target.id}__{i} = {self.coerce(c, et, tt)};')
                if tt.np == 'dyn':
                    self.emit(f'v_{target.id}__{i}__np = {f};')
            return
        for x, v, et in zip(target.elts, vals, vt.elem):
            if isinstance(x, ast.Name):
                self.assign(x.id, v, et)
            elif isinstance(x, ast.Subscript):
                self.store_elem(x, v, et)
            else:
                self.assign_tuple(x, v, et)

    def coerce(self, c, frm, to):
        if frm.ctype == to.ctype:
            return c
        if to.kind == 'float':
            return f'(double)({c})'
        return f'({to.ctype})({c})'

    def truth(self, c, t):
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
        """Emit e; return (C value, C numpy flag)."""
        if isinstance(e, ast.Constant):
            v = e.value
            if isinstance(v, bool):
                return ('1' if v else '0'), '0'
            return (c_int(v) if isinstance(v, int) else c_float(v)), '0'
        if isinstance(e, ast.Name):
            if e.id in self.cnames:
                return self.cnames[e.id]
            t = self.env[e.id]
            return f'v_{e.id}', (f'v_{e.id}__np' if t.np == 'dyn' else flag_of(t))
        if isinstance(e, ast.Attribute):
            if e.attr in ('ndim', 'size') and isinstance(e.value, ast.Name) and e.value.id in self.env:
                t = self.env[e.value.id]
                if e.attr == 'ndim':
                    return f'{t.ndim}LL', '0'
                return '(' + ' * '.join(self.extent(e.value.id, k) for k in range(t.ndim)) + ')', '0'
            return c_float(MATH_CONSTANTS[e.attr]), '0'
        if isinstance(e, ast.BinOp):
            lt = self.typeof(e.left)
            return self.arith(ARITH[type(e.op)], self.expr(e.left), lt, e.right, e)
        if isinstance(e, ast.UnaryOp):
            t = self.typeof(e.operand)
            c, f = self.expr(e.operand)
            if isinstance(e.op, ast.Not):
                return f'!({self.truth(c, t)})', '0'
            if isinstance(e.op, ast.Invert):
                return f'(~(int64_t)({c}))', f
            if isinstance(e.op, ast.UAdd):
                return (self.coerce(c, t, T.INT) if t is T.BOOL else c), f
            if t.kind == 'float':
                return f'(-({c}))', f
            return f'sj_neg((int64_t)({c}))', f
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
        if isinstance(e, ast.Subscript) and self.env.get(getattr(e.value, 'id', None)) is not None \
                and self.env[e.value.id].kind == 'tuple':
            return self.tuple_var(e.value.id)[self.tuple_index(e)]
        if isinstance(e, ast.Subscript):
            arr, idx = self.element(e)
            name = e.value.id
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
            if rt.kind == 'int':
                self.helpers.add('ipow')
                self.emit(f'if ((int64_t){b} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 15, 0);')
                self.emit(f'{out} = sj_ipow((int64_t){a}, (int64_t){b});')
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
                    self.emit(f'if ((int64_t){b} < 0) SJ_FAIL({VALUE_ERROR}, {line}, 16, 0);')
                self.emit(f'{out} = sj_ipow((int64_t){a}, (int64_t){b});')
            elif not self.exact:
                fast()
            else:
                self.helpers.add('floatpow')
                err = self.temp()
                self.emit(f'int {err} = sj_float_pow({table}, {fa}, {fb}, &{out});')
                self.emit(f'if ({err} == 1) SJ_FAIL({ZERO_DIVISION}, {line}, 6, 0);')
                self.emit(f'if ({err} == 2) SJ_FAIL({OVERFLOW_ERROR}, {line}, 4, 0);')
                self.emit(f'if ({err} == 3) SJ_FAIL({VALUE_ERROR}, {line}, 17, 0);')
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
        rt = self.binop_type(at, bt, op, node)
        a, af = self.hold(left[0], at), self.hold_flag(left[1])
        bv = self.expr(right)
        b, bf = self.hold(bv[0], bt), self.hold_flag(bv[1])
        npf = flag_or(af, bf)              # 1 when numpy's rules apply, not Python's
        line = self.line(node)
        if op in ('BitAnd', 'BitOr', 'BitXor'):
            sym = {'BitAnd': '&', 'BitOr': '|', 'BitXor': '^'}[op]
            return f'((int64_t){a} {sym} (int64_t){b})', npf
        isfloat = rt.kind == 'float'
        fa, fb = self.coerce(a, at, T.FLOAT), self.coerce(b, bt, T.FLOAT)
        if op == 'Pow':
            return self.power(a, b, fa, fb, rt, npf, line, right), npf
        if op in ('Add', 'Sub', 'Mult'):
            if op == 'Mult' and self.exact and at.kind == 'float' and bt.kind == 'int' and npf != '1':
                # CPython computes float * int(-1) as a negation, so a NaN's sign bit flips
                # (multiplying by -1.0, by any other int, or numpy scalars, leaves it alone)
                py = f'{b} == -1' if npf == '0' else f'!{npf} && {b} == -1'
                return f'({py} ? -{a} : {a} * (double){b})', npf
            if isfloat:
                sym = {'Add': '+', 'Sub': '-', 'Mult': '*'}[op]
                return f'({fa} {sym} {fb})', npf
            return f'sj_{op.lower()[:3]}((int64_t){a}, (int64_t){b})', npf
        both_int = at.kind != 'float' and bt.kind != 'float'
        wrap = self.wrap_mod(node) if op == 'Mod' and npf == '0' else None
        if wrap and ast.dump(node) in self.hoisted:
            return self.hoisted[ast.dump(node)], npf
        if wrap:                                  # (i + c) % n with 0 <= i < n, |c| <= 1: n >= 1, so
            return self.wrap_expr(node, a, b), npf    # no division
        if both_int and op in ('FloorDiv', 'Mod') and isinstance(right, ast.Constant) \
                and type(right.value) is int and right.value > 0 and right.value & (right.value - 1) == 0:
            k = right.value.bit_length() - 1      # x // 2**k is an arithmetic shift, x % 2**k a mask:
            if op == 'FloorDiv':                  # exactly Python's floor semantics, for every sign
                return f'((int64_t){a} >> {k})', npf
            return f'((int64_t){a} & {right.value - 1}LL)', npf
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
                return f'({fa} / {fb})', npf
            exact = f'sj_truediv((int64_t){a}, (int64_t){b})'
            return (exact if npf == '0' else f'({npf} ? {fa} / {fb} : {exact})'), npf
        if not isfloat and op in ('FloorDiv', 'Mod'):
            # one division and one remainder (cimo doesn't merge repeated expressions), then
            # Python's floor: adjust when the remainder is nonzero and its sign differs from b's.
            # b == -1 is set apart (INT64_MIN / -1 traps in C); b == 0 is numpy's 0 here.
            q, r, adj = self.temp(), self.temp(), self.temp()
            self.emit(f'int64_t {q} = 0, {r} = 0;')
            self.emit(f'if ({b} == -1) {q} = sj_neg((int64_t){a}); else if ({b} != 0) {{ {q} = (int64_t){a} / {b}; {r} = (int64_t){a} % {b}; }}')
            self.emit(f'int {adj} = {r} != 0 && (({r} < 0) != ({b} < 0));')
            if op == 'FloorDiv':
                return f'({q} - {adj})', npf
            return f'({r} + ({adj} ? {b} : 0))', npf
        if op == 'FloorDiv':
            return f'({fb} == 0.0 ? {fa} / {fb} : sj_ffloordiv({fa}, {fb}))', npf
        return f'({fb} == 0.0 ? fmod({fa}, {fb}) : sj_fmod_py({fa}, {fb}))', npf

    def cmp(self, a, at, b, bt, op):
        af, bf = at.kind == 'float', bt.kind == 'float'
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

    def compare(self, e):
        operands = [e.left] + e.comparators
        types = [self.typeof(x) for x in operands]
        first = self.expr(operands[0])
        a, af = self.hold(first[0], types[0]), first[1]
        if len(e.ops) == 1:
            second = self.expr(operands[1])
            b = self.hold(second[0], types[1])
            return self.cmp(a, types[0], b, types[1], CMP[type(e.ops[0])]), flag_or(af, second[1])
        r, rf = self.temp(), self.temp()
        self.emit(f'int {r} = 1, {rf} = 0;')
        closes = 0
        for i, op in enumerate(e.ops):
            if i:
                self.open(f'if ({r})')
                closes += 1
            nxt = self.expr(operands[i + 1])
            b = self.hold(nxt[0], types[i + 1])
            self.emit(f'{r} = {self.cmp(a, types[i], b, types[i + 1], CMP[type(op)])}; {rf} = {flag_or(af, nxt[1])};')
            a, af = b, nxt[1]
        for _ in range(closes):
            self.close()
        return r, rf

    def call(self, e):
        mf = self.math_func(e.func)
        if mf:
            return self.math_call(e, mf)
        name = e.func.id
        line = self.line(e)
        if self.is_reduction(e):
            return self.reduction(e)
        if name == 'len':
            t = self.env[e.args[0].id]
            if t.kind == 'tuple':
                return f'{len(t.elem)}LL', '0'
            return f'v_{e.args[0].id}__n', '0'
        ts = [self.typeof(a) for a in e.args]
        if name == 'abs':
            c, f = self.expr(e.args[0])
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
        self.emit(f'if ({v} != {v}) SJ_FAIL({VALUE_ERROR}, {line}, 0, 0);')
        self.emit(f'if (isinf({v})) SJ_FAIL({OVERFLOW_ERROR}, {line}, 0, 0);')
        return f'sj_float_to_int({v})', '0'


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
    if isinstance(s, ast.While):
        da_reads(c, s.test, assigned)
        da_block(c, s.body, set(assigned))
        return assigned
    if isinstance(s, ast.For):
        da_reads(c, s.iter, assigned)
        names = {n.id for n in ast.walk(s.target) if isinstance(n, ast.Name)}
        da_block(c, s.body, assigned | names)
        return assigned
    if isinstance(s, ast.Return):
        if s.value is not None:
            da_reads(c, s.value, assigned)
        return ALL
    if isinstance(s, (ast.Break, ast.Continue)):
        return ALL
    return assigned
