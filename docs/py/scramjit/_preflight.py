"""Preflight: which functions compile, and which lines keep them from it.

Static analysis reads source without importing or running it. A run reports what happened with the
real argument types. A function that doesn't compile whole may still have loops that do: those are
"partly compiled". Rendering is in cli/report.py.
"""

import ast
import copy
import math
import os
import re

from . import _lift
from . import _types as T
from ._compiler import Callee, Compiler, StaticNumpy, const_int
from ._errors import Unsupported

_OTHER = object()     # a module-level name whose value static analysis can't know
REDUCTIONS = {'sum', 'prod', 'mean', 'max', 'min', 'any', 'all', 'argmax', 'argmin', 'std', 'var'}
SKIP_DIRS = {'__pycache__', 'node_modules', 'venv', 'env', 'site-packages', 'build', 'dist'}

# What to do about a reason: the first match wins. Hints must be something to act on.
HINTS = [
    (r'^dict', 'if the keys are small integers, use an array indexed by key; otherwise do this outside the compiled function'),
    (r'^sets|^set comp', 'if the values are small integers, use a boolean array indexed by value'),
    (r'^f-strings|calls (str|print|repr|format)\(\)', 'keep text and printing out of the function: return numbers and format them after the call'),
    (r'slice|view', 'read and write through views, or loop over the indices you want: for i in range(lo, hi): a[i] = 0.0'),
    (r'^calls [\w.]*\.\w+\(\)', 'write it as a loop; methods and numpy calls on whole arrays aren\'t compiled'),
    (r'^calls \w+\(\): ', 'make that function compile first: this one can call it once it does'),
    (r'^calls \w+\(\)', 'decorate it with @scram: a compiled function can call compiled functions, not Python ones'),
    (r'empty list|a list assigned', 'make a numpy array of the final size: np.zeros(n)'),
    (r'must keep one type', 'use a new name for the new type'),
    (r'global or undefined name', 'pass it in as an argument'),
    (r'keeps? one type|would be None', 'make it the same type on every path'),
    (r'multi-dimensional array', 'index the rows: for i in range(rows): a[i, j]'),
    (r'closures', 'pass what it reads as arguments, and define it at module level'),
    (r'expression statement', 'assign the result, or move the call out of the function'),
    (r'is not available', 'define it in a file, not in exec() or a one-line REPL session'),
]


def hint_for(reason):
    for pattern, hint in HINTS:
        if re.search(pattern, reason):
            return hint
    return None


class Problem:
    """Why one line keeps a function in Python."""

    __slots__ = ('line', 'col', 'width', 'reason', 'hint')

    def __init__(self, line, col, width, reason):
        self.line, self.col, self.width, self.reason = line, col, width, reason
        self.hint = hint_for(reason)

    def as_dict(self):
        return {'line': self.line, 'column': self.col, 'reason': self.reason, 'hint': self.hint}


class Analysis:
    """One function under one signature. compiler is None when it doesn't compile."""

    __slots__ = ('signature', 'compiler', 'problems', 'loops')

    def __init__(self, signature, compiler, problems):
        self.signature, self.compiler, self.problems = signature, compiler, problems
        self.loops = []       # (first line, last line) of each loop that compiles on its own

    @property
    def state(self):
        return 'compiled' if self.compiler is not None else 'partly' if self.loops else 'python'

    def native_lines(self):
        return {n for a, b in self.loops for n in range(a, b + 1)}


class StaticCallee:
    """A @scram function of the module being read, as what a call to it compiles against."""

    def __init__(self, fdef, globals_):
        self.fdef, self.globals = fdef, globals_

    def _callee(self):
        tree, _ = with_globals(self.fdef, self.globals)
        count = len(self.fdef.args.posonlyargs + self.fdef.args.args)
        hidden = [(0, p.arg) for p in (tree.args.posonlyargs + tree.args.args)[count:]]
        opts = options(self.fdef)
        return Callee(self, tree, self.globals, '', 0, opts['boundscheck'], opts['parity'], hidden, None, self.fdef.name)


def _c_module(e):
    """cimo.compile('...') of a literal, as that's the only way to know what its functions take. Nothing runs but the compiler."""
    if not (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == 'compile' and isinstance(e.func.value, ast.Name)
            and e.func.value.id == 'cimo' and len(e.args) == 1 and not e.keywords and isinstance(e.args[0], ast.Constant) and isinstance(e.args[0].value, str)):
        return None
    import cimo
    try:
        return cimo.compile(e.args[0].value)
    except cimo.CompileError:
        return None


def module_globals(tree):
    """What compiling needs from a module, without running it: math, and which names are taken."""
    out = {}
    for s in tree.body:
        if isinstance(s, ast.Import):
            for a in s.names:
                out[(a.asname or a.name).split('.')[0]] = math if a.name == 'math' else StaticNumpy() if a.name == 'numpy' else _OTHER
        elif isinstance(s, ast.ImportFrom):
            for a in s.names:
                out[a.asname or a.name] = getattr(math, a.name, _OTHER) if s.module == 'math' \
                    else StaticNumpy(a.name) if s.module == 'numpy' else _OTHER
        elif isinstance(s, ast.FunctionDef) and any(is_scram(d) for d in s.decorator_list):
            out[s.name] = StaticCallee(s, out)
        elif isinstance(s, ast.ClassDef) and _static_enum(s) is not None:
            out[s.name] = _static_enum(s)
        elif isinstance(s, (ast.FunctionDef, ast.ClassDef)):
            out[s.name] = _OTHER
        elif isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = _c_module(s.value) or ast.literal_eval(s.value)
                    except ValueError:
                        v = const_int(s.value)                    # (1 << 64) - 1
                        out[t.id] = _OTHER if v is None else v
    return out


def _static_enum(cls):
    """The enum a class statement makes, built from source, when it's an Enum or IntEnum of int literals and auto():
    the same members a run would have. None for any other class."""
    import enum
    bases = [ast.unparse(b) for b in cls.bases]
    if len(bases) != 1 or bases[0] not in ('Enum', 'IntEnum', 'enum.Enum', 'enum.IntEnum') or cls.keywords or cls.decorator_list:
        return None
    members, last = [], 0
    for st in cls.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            continue                                   # a docstring
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
            return None
        v = st.value
        if isinstance(v, ast.Call) and ast.unparse(v.func) in ('auto', 'enum.auto') and not v.args and not v.keywords:
            value = last + 1
        else:
            try:
                value = ast.literal_eval(v)
            except ValueError:
                return None
        if type(value) is not int:
            return None
        members.append((st.targets[0].id, value))
        last = value
    if not members:
        return None
    return (enum.IntEnum if bases[0].endswith('IntEnum') else enum.Enum)(cls.name, members)


def is_scram(d):
    d = d.func if isinstance(d, ast.Call) else d
    return (isinstance(d, ast.Name) and d.id == 'scram') or (isinstance(d, ast.Attribute) and d.attr == 'scram')


def options(fdef):
    opts = {'boundscheck': True, 'parity': True}
    for d in fdef.decorator_list:
        if isinstance(d, ast.Call) and is_scram(d):
            for k in d.keywords:
                if k.arg in opts and isinstance(k.value, ast.Constant):
                    opts[k.arg] = bool(k.value.value)
    return opts


def assumed_signature(fdef, globals_=None, seen=()):
    """Argument types from annotations; the rest are assumed to be numbers or float64 arrays, by how
    the function uses them, or by how a function it passes them to does."""
    ann = {'float': T.FLOAT, 'int': T.INT, 'bool': T.BOOL, 'list[float]': T.list_of(T.FLOAT), 'list[int]': T.list_of(T.INT),
           'list[bool]': T.list_of(T.BOOL)}
    ndim = {}
    passed_ints = set()
    passed_tuples = {}
    for n in ast.walk(fdef):
        f = (globals_ or {}).get(n.func.id) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) else None
        c = (globals_ or {}).get(n.func.value.id) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) else None
        if T.is_c_module(c) and n.func.attr in dir(c) and hasattr(c, 'signature'):
            for a, p in zip(n.args, c.signature(n.func.attr)[1]):
                if isinstance(a, ast.Name) and p.endswith('*'):
                    ndim.setdefault(a.id, 1)
                elif isinstance(a, ast.Name) and p.endswith('_t'):
                    passed_ints.add(a.id)
        if isinstance(f, StaticCallee) and f.fdef not in seen:
            for a, t in zip(n.args, assumed_signature(f.fdef, globals_, seen + (fdef,))):
                if isinstance(a, ast.Name) and t.kind == 'array':
                    ndim[a.id] = max(ndim.get(a.id, 0), t.ndim)
                elif isinstance(a, ast.Name) and t is T.INT:
                    passed_ints.add(a.id)
                elif isinstance(a, ast.Name) and t.kind == 'tuple':
                    passed_tuples.setdefault(a.id, t)
    for n in ast.walk(fdef):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            k = len(n.slice.elts) if isinstance(n.slice, ast.Tuple) else 1
            ndim[n.value.id] = max(ndim.get(n.value.id, 0), k)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.attr in ('shape', 'size', 'ndim'):
            ndim.setdefault(n.value.id, 1)
        elif isinstance(n, ast.For) and isinstance(n.iter, ast.Name):
            ndim.setdefault(n.iter.id, 1)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.attr in REDUCTIONS:
            ndim.setdefault(n.value.id, 1)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in REDUCTIONS and n.args \
                and isinstance(n.args[0], ast.Name):
            ndim.setdefault(n.args[0].id, 1)
        elif isinstance(n, ast.comprehension) and isinstance(n.iter, ast.Name):
            ndim.setdefault(n.iter.id, 1)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('len', 'enumerate', 'zip', 'reversed'):
            for a in n.args:
                if isinstance(a, ast.Name):
                    ndim.setdefault(a.id, 1)
    # row = m[i], with row indexed in turn, says m has more dimensions than one index shows
    for n in ast.walk(fdef):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ndim \
                and isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Name):
            items = n.value.slice.elts if isinstance(n.value.slice, ast.Tuple) else [n.value.slice]
            ints = sum(not isinstance(k, ast.Slice) for k in items)
            ndim[n.value.value.id] = max(ndim.get(n.value.value.id, 0), ndim[n.targets[0].id] + ints)
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id in ndim and isinstance(n.iter, ast.Name):
            ndim[n.iter.id] = max(ndim.get(n.iter.id, 0), ndim[n.target.id] + 1)
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Name):
            items = n.value.slice.elts if isinstance(n.value.slice, ast.Tuple) else [n.value.slice]
            inner = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            ints = sum(not isinstance(k, ast.Slice) for k in items)
            ndim[n.value.value.id] = max(ndim.get(n.value.value.id, 0), ints + len(inner))
    # rows, cols = a.shape says how many dimensions a has
    for n in ast.walk(fdef):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Attribute) \
                and n.value.attr == 'shape' and isinstance(n.value.value, ast.Name):
            ndim[n.value.value.id] = max(ndim.get(n.value.value.id, 1), len(n.targets[0].elts))
    keys = _assumed_dicts(fdef)
    objects = _assumed_objects(fdef, ndim)
    ints = _assumed_ints(fdef) | passed_ints
    sig = []
    a = fdef.args
    params = a.posonlyargs + a.args
    shapes = {}                        # (x0, y0), (x1, y1) = box says box is a tuple of that shape

    def shape(target):
        return T.tuple_of([shape(x) for x in target.elts]) if isinstance(target, (ast.Tuple, ast.List)) else T.FLOAT
    for n in ast.walk(fdef):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], (ast.Tuple, ast.List)) \
                and isinstance(n.value, ast.Name):
            shapes.setdefault(n.value.id, shape(n.targets[0]))
    for k, v in passed_tuples.items():
        shapes.setdefault(k, v)
    literal = {}                       # a literal default says what the parameter usually is
    for p, d in zip(params[len(params) - len(a.defaults):], a.defaults):
        if isinstance(d, ast.Constant) and type(d.value) in (int, float, bool):
            literal[p.arg] = {int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(d.value)]
    for p in params:
        name = ast.unparse(p.annotation) if p.annotation is not None else None
        if p.arg in objects and name not in ann:
            sig.append(T.object_of('object', sorted(objects[p.arg].items())))
        elif p.arg in literal and name is None and p.arg not in ndim and p.arg not in keys:
            sig.append(literal[p.arg])
        elif p.arg in shapes and name is None and p.arg not in ndim and p.arg not in keys:
            sig.append(shapes[p.arg])
        elif p.arg in keys:
            sig.append(T.dict_of(sorted(keys[p.arg].items(), key=lambda kv: repr(kv[0]))))
        elif name in ann:
            sig.append(ann[name])
        elif p.arg in ndim:
            sig.append(T.array(T.NP_FLOAT64, True, ndim[p.arg]))
        elif p.arg in ints:
            sig.append(T.INT)
        else:
            sig.append(T.FLOAT)
    return sig


def _assumed_ints(fdef):
    """Names that must be ints: range(n), np.zeros(n), np.zeros((h, w))."""
    found = set()
    for n in ast.walk(fdef):
        if not isinstance(n, ast.Call):
            continue
        args = []
        if isinstance(n.func, ast.Name) and n.func.id == 'range':
            args = n.args
        elif isinstance(n.func, ast.Attribute) and n.func.attr in ('zeros', 'ones', 'empty', 'full') and isinstance(n.func.value, ast.Name):
            shape = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == 'shape'), None)
            args = shape.elts if isinstance(shape, (ast.Tuple, ast.List)) else [shape]
        found |= {x.id for x in args if isinstance(x, ast.Name)}
    return found


def _assumed_dicts(fdef):
    """Parameters read with a string key are dicts: {name: {key: its assumed type}}. A value indexed in turn is an
    array, one used as a range bound or an index is an int, and the rest are floats."""
    def key_of(n):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and isinstance(n.slice, ast.Constant) \
                and type(n.slice.value) is str:
            return n.value.id, n.slice.value
        return None
    return _assumed_fields(fdef, key_of)


ARRAY_ATTRIBUTES = {'T', 'dtype', 'shape', 'size', 'ndim', 'copy', 'flatten', 'astype', 'item', 'tolist', 'reshape', 'real', 'imag', 'flat'} | set(REDUCTIONS)


def _assumed_objects(fdef, ndim):
    """Parameters read as p.a, where that isn't what an array has and p isn't used as an array, are instances, as for dicts
    above. p.a() is a method, which runs as Python: no type here says so, so the compile that follows finds it."""
    calls = {id(n.func) for n in ast.walk(fdef) if isinstance(n, ast.Call)}
    found = {n.value.id for n in ast.walk(fdef) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and id(n) not in calls
             and n.attr not in ARRAY_ATTRIBUTES and n.value.id not in ndim}

    def key_of(n):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in found and id(n) not in calls:
            return n.value.id, n.attr
        return None
    return _assumed_fields(fdef, key_of)


def _assumed_fields(fdef, key_of):
    a = fdef.args
    names = {p.arg for p in a.posonlyargs + a.args}
    found = {}
    for n in ast.walk(fdef):
        k = key_of(n)
        if k and k[0] in names:
            found.setdefault(k[0], {}).setdefault(k[1], T.FLOAT)
    if not found:
        return found

    def mark(e, t):
        for x in ast.walk(e):
            k = key_of(x)
            if k and k[0] in found:
                found[k[0]][k[1]] = t
    for n in ast.walk(fdef):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'range':
            for x in n.args:
                mark(x, T.INT)
        elif isinstance(n, ast.Subscript) and not key_of(n):
            mark(n.slice, T.INT)
    counters = {n.target.id for n in ast.walk(fdef) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and isinstance(n.iter, ast.Call) and getattr(n.iter.func, 'id', None) == 'range'}
    for n in ast.walk(fdef):                 # p['day'] = day, with day a range's variable
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in counters:
            for t in n.targets:
                mark(t, T.INT)
    for n in ast.walk(fdef):                 # after the indices: p['a'][p['i']] has an array and an int
        if isinstance(n, ast.Subscript) and not key_of(n):
            k = key_of(n.value)
            if k and k[0] in found:
                d = len(n.slice.elts) if isinstance(n.slice, ast.Tuple) else 1
                found[k[0]][k[1]] = T.array(T.NP_FLOAT64, True, d)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'len' and n.args:
            k = key_of(n.args[0])
            if k and k[0] in found:
                found[k[0]][k[1]] = T.array(T.NP_FLOAT64, True, 1)
    return found


def _innermost_stmt(fdef, line):
    """The list holding the innermost statement on this line, and its index."""
    found = None
    for node in ast.walk(fdef):
        for field in ('body', 'orelse', 'finalbody'):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                for i, s in enumerate(stmts):
                    if isinstance(s, ast.stmt) and s.lineno <= line <= (s.end_lineno or s.lineno):
                        if found is None or s.lineno >= found[0][found[1]].lineno:
                            found = (stmts, i)
    return found


def _bound_names(stmt):
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def analyze(fdef, sig, globals_, filename, offset, opts, first_only=False):
    """Compile as far as possible. A failing statement is set aside and compiling retried, so every
    line that keeps the function in Python is found, not only the first. A failure that only reads a
    name a set-aside statement would have bound follows from that one, so it isn't reported."""
    tree = copy.deepcopy(fdef)
    problems = []
    gone = set()
    for _ in range(25):
        try:
            c = Compiler(tree, list(sig), globals_, filename, offset, opts['boundscheck'], opts['parity'])
            c.generate()
            return Analysis(sig, c if not problems else None, problems)
        except Unsupported as u:
            node, line = u.node, u.lineno
            reads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} if isinstance(node, ast.AST) else set()
            if not (reads & gone) and not any(f"'{g}'" in u.reason for g in gone):
                col = getattr(node, 'col_offset', None)
                same = getattr(node, 'end_lineno', None) == line
                width = (node.end_col_offset - col) if col is not None and same else 1
                problems.append(Problem((line or 1) + offset, col, width, u.reason))
            where = _innermost_stmt(tree, line) if line and line > tree.lineno and not first_only else None
            if where is None:
                if not problems:
                    problems.append(Problem((line or 1) + offset, None, 1, u.reason))
                return Analysis(sig, None, problems)
            stmts, i = where
            gone |= _bound_names(stmts[i])
            stmts[i] = ast.copy_location(ast.Pass(), stmts[i])
    return Analysis(sig, None, problems)


def _number_tree(v):
    """The type of a number, or of a tuple of numbers and such tuples, as a call would see it; else None."""
    if type(v) in (int, float, bool):
        return {int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(v)] if type(v) is not int or -(1 << 63) <= v < 1 << 63 else None
    if type(v) is tuple and v:
        elems = [_number_tree(x) for x in v]
        return None if None in elems else T.tuple_of(elems)
    return None


def with_globals(fdef, globals_):
    """fdef with what it reads from module level as parameters, as a call would pass them: annotated if they're plain numbers,
    so their types are known. A name whose value can't be known counts if the function indexes it, as an array."""
    local = _lift.local_names(fdef)
    loads = set()
    for st in fdef.body:
        loads |= _lift._split(st)[0]
    called = {n.func.id for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    indexed = {n.value.id for n in ast.walk(fdef) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)}
    seen, stack = {fdef.name}, [globals_[n] for n in called - local if isinstance(globals_.get(n), StaticCallee)]
    while stack:                       # a global read by a function this calls is passed through it
        f = stack.pop()
        if f.fdef.name in seen:
            continue
        seen.add(f.fdef.name)
        inner = _lift.local_names(f.fdef)
        for st in f.fdef.body:
            loads |= _lift._split(st)[0] - inner
        for n in ast.walk(f.fdef):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id not in inner:
                called.add(n.func.id)
                if isinstance(globals_.get(n.func.id), StaticCallee):
                    stack.append(globals_[n.func.id])
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                indexed.add(n.value.id)
    added = []
    tuples = {}
    for name in sorted(loads - local - called):
        if name not in globals_:
            continue
        v = globals_[name]
        if isinstance(v, type) and T.enum_ok(v):                # an enum class: what Level.HIGH needs
            added.append(ast.arg(name))
            tuples[name] = T.enum_class_of(v)
        elif type(v) is int and (1 << 63) <= v < (1 << 64):      # held unsigned, as a call would pass it
            added.append(ast.arg(name))
            tuples[name] = T.U64
        elif type(v) in (int, float, bool):
            added.append(ast.arg(name, ast.Name(type(v).__name__, ast.Load())))
        elif _number_tree(v) is not None and type(v) is tuple:
            added.append(ast.arg(name))
            tuples[name] = _number_tree(v)
        elif type(v) is dict or (v is _OTHER and name in indexed):
            added.append(ast.arg(name))
    if not added:
        return fdef, tuples
    tree = copy.deepcopy(fdef)
    tree.args.args = tree.args.args + [ast.copy_location(a, fdef) for a in added]
    return tree, tuples


def best_analysis(fdef, globals_, filename, offset, opts, first_only=False):
    """Analyze under the assumed signature. An unannotated array whose values are used as indices
    isn't float64, so before calling a function uncompilable, try its arrays as int64 too."""
    fdef, tuples = with_globals(fdef, globals_)
    sig = assumed_signature(fdef, globals_)
    params = fdef.args.posonlyargs + fdef.args.args
    sig = [tuples.get(p.arg, t) for p, t in zip(params, sig)]
    first = None
    arrays = [i for i, t in enumerate(sig) if t.kind == 'array']
    for mask in range(1 << min(len(arrays), 4)):
        trial = list(sig)
        for bit, i in enumerate(arrays[:4]):
            if mask >> bit & 1:
                trial[i] = T.array(T.NP_INT64, True, sig[i].ndim)
        a = analyze(fdef, trial, globals_, filename, offset, opts, first_only)
        if a.compiler is not None:
            return a
        first = first or a
    if arrays:                         # bytes: int() of them is never negative, which a 64-bit hash needs
        trial = [T.array(T.NARROW['uint8'], True, t.ndim) if t.kind == 'array' else t for t in sig]
        a = analyze(fdef, trial, globals_, filename, offset, opts, first_only)
        if a.compiler is not None:
            return a
    if not first_only:
        first.loops = lifted_loops(fdef, sig, globals_, filename, offset, opts)
        inside = first.native_lines()
        first.problems = [p for p in first.problems if p.line not in inside]
    return first


_NOT_NUMBERS = (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.JoinedStr, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _assumed_local(fdef, name):
    """A local's type as far as its assignments show: a type, None if it can't be passed to a compiled loop,
    or ... if they don't say."""
    seen = ...
    for n in ast.walk(fdef):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            v = n.value
            if isinstance(v, _NOT_NUMBERS) or (isinstance(v, ast.Constant) and not isinstance(v.value, (bool, int, float))):
                return None
            if isinstance(v, ast.Constant) and seen is ...:
                seen = T.BOOL if isinstance(v.value, bool) else T.INT if isinstance(v.value, int) else T.FLOAT
    return seen


def _loop_compiles(c, loop_def, fdef, sig, globals_, filename, offset, opts):
    """Does this lifted loop compile, under what the function's source says its variables are?"""
    args = fdef.args.posonlyargs + fdef.args.args
    by_arg = {p.arg: (t, p.annotation is not None) for p, t in zip(args, sig)}
    usage = assumed_signature(loop_def)
    fixed, vary = [], []
    for k, name in enumerate(c.params):
        t, annotated = by_arg.get(name, (None, False))
        if t is None:
            t = _assumed_local(fdef, name)
            if t is None:
                return False
            annotated = t is not ...
            if t is ...:
                t = usage[k]
        fixed.append(t)
        if not annotated:
            vary.append(k)
    for mask in range(1 << min(len(vary), 4)):
        trial = list(fixed)
        for bit, k in enumerate(vary[:4]):
            t = usage[k] if usage[k].kind == 'array' else fixed[k]
            if mask >> bit & 1:
                t = T.array(T.NP_INT64, True, t.ndim) if t.kind == 'array' else T.INT
            trial[k] = t
        if analyze(loop_def, trial, globals_, filename, offset, opts, True).compiler is not None:
            return True
    return False


def lifted_loops(fdef, sig, globals_, filename, offset, opts, root=None):
    """(first line, last line) of the outermost loops that would compile on their own."""
    if not _lift.can_lift(fdef):
        return []
    out = []
    for c in _lift.candidates(fdef, _lift.local_names(fdef), root):
        loop_def = _lift.loop_function(c, fdef.name)
        if _loop_compiles(c, loop_def, fdef, sig, globals_, filename, offset, opts):
            out.append((c.loop.lineno + offset, c.loop.end_lineno + offset))
        else:
            out += lifted_loops(loop_def, _inner_signature(c, fdef, sig), globals_, filename, offset, opts, c.loop)
    return out


def _inner_signature(c, fdef, sig):
    """What a lifted loop's parameters are known to be, for the loops inside it."""
    args = fdef.args.posonlyargs + fdef.args.args
    by_arg = {p.arg: t for p, t in zip(args, sig)}
    usage = assumed_signature(_lift.loop_function(c, fdef.name))
    return [by_arg.get(name, usage[k]) for k, name in enumerate(c.params)]


class FileReport:
    __slots__ = ('path', 'source', 'functions', 'error', 'candidates')

    def __init__(self, path, source, functions, error=None, candidates=()):
        self.path, self.source, self.functions, self.error = path, source, functions, error
        self.candidates = candidates


def _is_candidate(fdef, path):
    """Worth suggesting @scram for: an undecorated module-level function with a loop to speed up, and
    not a test."""
    base = os.path.basename(path)
    if fdef.decorator_list or fdef.name.startswith('test_') or base.startswith('test_') or base == 'conftest.py':
        return False
    a = fdef.args
    if not (a.posonlyargs or a.args):
        return False                                    # nothing comes in, so nothing to specialize on
    return any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(fdef))


def read_file(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def scan_file(path, suggest=True):
    """A FileReport for every @scram function in a file, and the undecorated functions that would
    compile, without importing it; None if it has neither."""
    source = read_file(path)
    try:
        tree = ast.parse(source, path)
    except SyntaxError as e:
        return FileReport(path, source, [], e)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any(is_scram(d) for d in n.decorator_list)]
    g = module_globals(tree)
    candidates = []
    if suggest:
        for f in tree.body:
            if isinstance(f, ast.FunctionDef) and _is_candidate(f, path):
                a = best_analysis(f, g, path, 0, options(f), first_only=True)
                if a.compiler is not None:
                    candidates.append((f, a))
    if not funcs and not candidates:
        return None
    return FileReport(path, source, [(f, best_analysis(f, g, path, 0, options(f))) for f in funcs], candidates=candidates)


def python_files(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS)
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith('.py')]
    return out


def has_loop(fdef):
    return any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(fdef))


def subscript_lines(fdef, offset):
    return {n.lineno + offset for n in ast.walk(fdef) if isinstance(n, ast.Subscript)}


def loop_lines(fdef, offset):
    return {n + offset for s in ast.walk(fdef) if isinstance(s, (ast.For, ast.While))
            for b in s.body for n in range(b.lineno, (b.end_lineno or b.lineno) + 1)}
