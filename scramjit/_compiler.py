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
import math

from . import _types as T
from ._errors import Unsupported
from ._runtime import INDEX_ERROR, OVERFLOW_ERROR, VALUE_ERROR, ZERO_DIVISION

BUILTINS = ('range', 'len', 'abs', 'min', 'max', 'int', 'float')
MATH_CONSTANTS = {'pi': math.pi, 'e': math.e, 'tau': math.tau, 'inf': math.inf, 'nan': math.nan}
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
ARITH = {ast.Add: 'Add', ast.Sub: 'Sub', ast.Mult: 'Mult', ast.Div: 'Div', ast.FloorDiv: 'FloorDiv',
         ast.Mod: 'Mod', ast.BitAnd: 'BitAnd', ast.BitOr: 'BitOr', ast.BitXor: 'BitXor'}
CMP = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
SWAP = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==', '!=': '!='}


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


class Compiler:
    def __init__(self, fdef, arg_types, globals_, filename, line_offset, boundscheck=True, exact=True):
        self.f = fdef
        self.exact = exact     # False: C's float semantics where CPython parity costs speed
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
            if len(s.targets) != 1:
                self.fail(s, 'chained assignment')
            self.infer_target(s.targets[0], self.typeof(s.value), s)
        elif isinstance(s, ast.AnnAssign):
            if s.value is None:
                self.fail(s, 'annotation without a value')
            self.infer_target(s.target, self.typeof(s.value), s)
        elif isinstance(s, ast.AugAssign):
            op = ARITH.get(type(s.op))
            if op is None:
                self.fail(s, f'the {type(s.op).__name__} operator')
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
            if not isinstance(s.target, ast.Name):
                self.fail(s, 'a for loop target that is not a single name')
            self.range_args(s.iter)
            self.assign_type(s, s.target.id, T.INT)
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
            if not at.writable:
                self.fail(s, 'writes a read-only array')
            self.index_type(target)
            if not t.is_number or (at.elem.kind == 'int' and t.kind == 'float'):
                self.fail(s, f'stores {t.name} into a {at.name} array')
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

    def array_of(self, sub):
        if not isinstance(sub.value, ast.Name) or sub.value.id not in self.env \
                or self.env[sub.value.id].kind != 'array':
            self.fail(sub, 'indexing something other than an array parameter')
        return self.env[sub.value.id]

    def index_type(self, sub):
        if isinstance(sub.slice, (ast.Slice, ast.Tuple)):
            self.fail(sub, 'slices and multi-dimensional indexing')
        t = self.typeof(sub.slice)
        if t.kind != 'int':
            self.fail(sub, f'an index of type {t.name}')

    def binop_type(self, a, b, op, node):
        if not (a.is_number and b.is_number):
            self.fail(node, f'{op} on {a.name} and {b.name}')
        if op in ('BitAnd', 'BitOr', 'BitXor'):
            if a.kind == 'float' or b.kind == 'float':
                self.fail(node, 'a bitwise operator on a float')
            if a is T.BOOL and b is T.BOOL:
                return T.BOOL
        r = T.arith_result(a, b, op)
        if r is None:
            self.fail(node, 'arithmetic on numpy bools')
        return r

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
                return self.env[e.id]
            self.fail(e, f'uses the global or undefined name {e.id!r}')
        if isinstance(e, ast.Attribute):
            if isinstance(e.value, ast.Name) and e.value.id not in self.env \
                    and self.globals.get(e.value.id) is math and e.attr in MATH_CONSTANTS:
                return T.FLOAT
            self.fail(e, 'attribute access')
        if isinstance(e, ast.BinOp):
            op = ARITH.get(type(e.op))
            if op is None:
                self.fail(e, f'the {type(e.op).__name__} operator')
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
            return self.join_all(e, [self.typeof(e.body), self.typeof(e.orelse)], 'a conditional expression')
        if isinstance(e, ast.Subscript):
            at = self.array_of(e)
            self.index_type(e)
            return at.elem
        if isinstance(e, ast.Call):
            return self.call_type(e)
        self.fail(e, f'the {type(e).__name__} expression')

    def call_type(self, e):
        if not isinstance(e.func, ast.Name) or e.keywords:
            self.fail(e, 'this call')
        name = e.func.id
        if name == 'range' or not any(self.is_builtin(name, b) for b in BUILTINS):
            self.fail(e, f'calls {name}()')
        args = [self.typeof(a) for a in e.args]
        if name == 'len':
            if len(args) != 1 or args[0].kind != 'array' or not isinstance(e.args[0], ast.Name):
                self.fail(e, 'len() of a non-array')
            return T.INT
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
        return T.INT if name == 'int' else T.FLOAT

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
        params = []
        for p in self.params:
            t = self.env[p]
            if t.kind == 'array':
                params += [f'{t.ctype}v_{p}', f'int64_t v_{p}__n']
            else:
                params.append(f'{t.ctype} v_{p}')
        params.append('int64_t *sj_io')
        self.mark(self.f)
        head = len(self.lines)
        for name, t in self.env.items():
            if name not in self.params:
                self.emit(f'{t.ctype} v_{name} = 0;')
            if t.np == 'dyn':
                self.emit(f'int v_{name}__np = 0;')
        self.emit('int sj_st = 0; int64_t sj_ln = 0, sj_x = 0, sj_y = 0;')
        for s in self.f.body:
            self.stmt(s)
        self.emit('return 0;')
        self.lines.append('sj_fail:')
        self.emit('sj_io[1] = sj_ln; sj_io[2] = sj_x; sj_io[3] = sj_y;')
        self.emit('return sj_st;')
        return '\n'.join(self.lines[:head] + [f'int sj_kernel({", ".join(params)}) {{']
                         + self.lines[head:] + ['}', ''])

    def block(self, stmts):
        self.depth += 1
        for s in stmts:
            self.stmt(s)
        self.depth -= 1

    def open(self, text):
        self.emit(text + ' {')
        self.depth += 1

    def close(self, text='}'):
        self.depth -= 1
        self.emit(text)

    def assign(self, name, v, vt):
        t = self.env[name]
        self.emit(f'v_{name} = {self.coerce(v[0], vt, t)};')
        if t.np == 'dyn':
            self.emit(f'v_{name}__np = {v[1]};')

    def stmt(self, s):
        if isinstance(s, ast.Expr):           # a docstring
            return
        self.mark(s)
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            target = s.targets[0] if isinstance(s, ast.Assign) else s.target
            v, vt = self.expr(s.value), self.typeof(s.value)
            if isinstance(target, ast.Name):
                self.assign(target.id, v, vt)
            else:
                at = self.env[target.value.id]
                val = self.hold(self.coerce(v[0], vt, at.elem), at.elem)
                arr, idx = self.element(target)
                self.emit(f'{arr}[{idx}] = {val};')
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
                at = self.env[s.target.value.id]
                rt = self.binop_type(at.elem, self.typeof(s.value), op, s)
                r = self.arith(op, (self.hold(f'{arr}[{idx}]', at.elem), '1'), at.elem, s.value, s)
                self.emit(f'{arr}[{idx}] = {self.coerce(r[0], rt, at.elem)};')
        elif isinstance(s, ast.If):
            self.open(f'if ({self.truth(self.expr(s.test)[0], self.typeof(s.test))})')
            for b in s.body:
                self.stmt(b)
            if s.orelse:
                self.close('} else {')
                self.depth += 1
                for b in s.orelse:
                    self.stmt(b)
            self.close()
        elif isinstance(s, ast.While):
            self.open('for (;;)')
            self.emit(f'if (!({self.truth(self.expr(s.test)[0], self.typeof(s.test))})) break;')
            for b in s.body:
                self.stmt(b)
            self.close()
        elif isinstance(s, ast.For):
            self.for_range(s)
        elif isinstance(s, ast.Return):
            if s.value is not None:
                v = self.expr(s.value)
                val = self.coerce(v[0], self.typeof(s.value), self.ret)
                if self.ret.kind == 'float':
                    t = self.temp()
                    self.emit(f'{{ double {t} = {val}; memcpy(&sj_io[0], &{t}, 8); }}')
                else:
                    self.emit(f'sj_io[0] = (int64_t)({val});')
                if self.ret.np == 'dyn':
                    self.emit(f'sj_io[2] = {v[1]};')
            self.emit('return 0;')
        elif isinstance(s, ast.Break):
            self.emit('break;')
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
        fact = self.range_fact(s) if unit else None
        if unit and fact and self.boundscheck:
            # loop versioning: if no access var + c can leave its array during this loop, run a
            # copy without those checks; otherwise the checked copy raises where Python would
            extra = self.version_candidates(s, var, fact)
            if extra:
                cond = ' && '.join(f'{te} <= v_{arr}__n - ({c})' for arr, c in sorted(extra.items()))
                self.open(f'if ({te} > {ts} && {cond})')
                self.unit_loop(s, var, ts, te, fact + (extra,))
                self.close('} else {')
                self.depth += 1
                self.unit_loop(s, var, ts, te, fact)
                self.close()
                return
        if unit:
            self.unit_loop(s, var, ts, te, fact)
            return
        tp, tn, tk = self.temp(), self.temp(), self.temp()
        self.emit(f'int64_t {tp} = {step};')
        self.emit(f'if ({tp} == 0) SJ_FAIL({VALUE_ERROR}, {self.line(s)}, 1, 0);')
        self.emit(f'uint64_t {tn} = {tp} > 0 ? ({ts} < {te} ? ((uint64_t){te} - (uint64_t){ts} - 1) / (uint64_t){tp} + 1 : 0)'
                  f' : ({ts} > {te} ? ((uint64_t){ts} - (uint64_t){te} - 1) / (0 - (uint64_t){tp}) + 1 : 0);')
        self.open(f'for (uint64_t {tk} = 0; {tk} < {tn}; {tk}++)')
        self.emit(f'v_{var} = (int64_t)((uint64_t){ts} + {tk} * (uint64_t){tp});')
        self.loop_body(s, var, None)

    def unit_loop(self, s, var, ts, te, fact):
        ti = self.temp()
        self.open(f'for (int64_t {ti} = {ts}; {ti} < {te}; {ti}++)')
        self.emit(f'v_{var} = {ti};')
        self.loop_body(s, var, fact)

    def loop_body(self, s, var, fact):
        if self.env[var].np == 'dyn':
            self.emit(f'v_{var}__np = 0;')
        saved = self.facts.get(var)
        if fact:
            self.facts[var] = fact
        for b in s.body:
            self.stmt(b)
        self.facts.pop(var, None)
        if saved:
            self.facts[var] = saved
        self.close()

    def version_candidates(self, s, var, fact):
        """Arrays indexed as var + c in the loop, nonnegative but not proven below len(): the
        largest c for each."""
        self.facts[var] = fact
        extra = {}
        try:
            for n in ast.walk(ast.Module(body=s.body, type_ignores=[])):
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) \
                        and n.value.id in self.env and self.env[n.value.id].kind == 'array':
                    idx, off = n.slice, 0
                    if isinstance(idx, ast.BinOp) and isinstance(idx.op, (ast.Add, ast.Sub)) \
                            and isinstance(idx.right, ast.Constant) and type(idx.right.value) is int:
                        off = idx.right.value if isinstance(idx.op, ast.Add) else -idx.right.value
                        idx = idx.left
                    if not (isinstance(idx, ast.Name) and idx.id == var):
                        continue
                    nonneg, inbounds = self.index_facts(n.value.id, n.slice)
                    if nonneg and not inbounds:
                        extra[n.value.id] = max(off, extra.get(n.value.id, off))
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
        return (lo, hi, slack) if lo is not None or hi else None

    def len_of(self, e):
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and self.is_builtin(e.func.id, 'len') \
                and len(e.args) == 1 and isinstance(e.args[0], ast.Name):
            return e.args[0].id
        return None

    def index_facts(self, name, idx):
        """(never negative, always below len(name)) for an index expression, from loop facts."""
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
        return nonneg, nonneg and ((hi == name and off <= slack) or off <= versioned.get(name, -1 << 62))

    def element(self, sub):
        """Index an array: Python's negative indices, numpy's IndexError. Returns (array, index)."""
        name = sub.value.id
        raw = self.coerce(self.expr(sub.slice)[0], self.typeof(sub.slice), T.INT)
        n = f'v_{name}__n'
        nonneg, inbounds = self.index_facts(name, sub.slice)
        t0, t1 = self.temp(), self.temp()
        if nonneg:                      # proven: no negative-index adjustment
            self.emit(f'int64_t {t0} = {raw}, {t1} = {t0};')
        else:
            self.emit(f'int64_t {t0} = {raw}, {t1} = {t0} < 0 ? {t0} + {n} : {t0};')
        if self.boundscheck and not inbounds:
            self.emit(f'if ((uint64_t){t1} >= (uint64_t){n}) SJ_FAIL({INDEX_ERROR}, {self.line(sub)}, {t0}, {n});')
        return f'v_{name}', t1

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
            t = self.env[e.id]
            return f'v_{e.id}', (f'v_{e.id}__np' if t.np == 'dyn' else flag_of(t))
        if isinstance(e, ast.Attribute):
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
            for v in e.values[1:]:
                test = self.truth(r, t)
                self.open(f'if ({test if isinstance(e.op, ast.And) else "!(" + test + ")"})')
                closes += 1
                c, f = self.expr(v)
                self.emit(f'{r} = {self.coerce(c, self.typeof(v), t)}; {rf} = {f};')
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
        if isinstance(e, ast.Subscript):
            arr, idx = self.element(e)
            return self.hold(f'{arr}[{idx}]', self.env[e.value.id].elem), '1'
        if isinstance(e, ast.Call):
            return self.call(e)
        raise AssertionError(e)

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
        if op == 'FloorDiv':
            if not isfloat:
                return f'({b} == 0 ? 0 : sj_floordiv((int64_t){a}, (int64_t){b}))', npf
            return f'({fb} == 0.0 ? {fa} / {fb} : sj_ffloordiv({fa}, {fb}))', npf
        if not isfloat:
            return f'({b} == 0 ? 0 : sj_mod((int64_t){a}, (int64_t){b}))', npf
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
        name = e.func.id
        line = self.line(e)
        if name == 'len':
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
        v = self.hold(c, T.FLOAT)
        self.emit(f'if ({v} != {v}) SJ_FAIL({VALUE_ERROR}, {line}, 0, 0);')
        self.emit(f'if (isinf({v})) SJ_FAIL({OVERFLOW_ERROR}, {line}, 0, 0);')
        return f'sj_float_to_int({v})', '0'


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
    for n in ast.walk(node):
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
    if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        da_reads(c, s.value, assigned)
        target = s.targets[0] if isinstance(s, ast.Assign) else s.target
        if isinstance(target, ast.Subscript):
            da_reads(c, target.slice, assigned)
        elif isinstance(s, ast.AugAssign):
            da_reads(c, target, assigned)
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
        da_block(c, s.body, assigned | {s.target.id})
        return assigned
    if isinstance(s, ast.Return):
        if s.value is not None:
            da_reads(c, s.value, assigned)
        return ALL
    if isinstance(s, (ast.Break, ast.Continue)):
        return ALL
    return assigned
