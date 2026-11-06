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
import types

from . import _lift
from . import _types as T
from ._compiler import OPERATORS, Callee, Compiler, StaticNumpy, const_int, returns_made
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
        self._tree = fdef

    @property
    def class_globals(self):
        """What returns_made reads: the module's classes, as the stand-in types their class statements are here."""
        out = dict(self.globals or {})
        out.update({c.name: _stand_in(c) for c in (self.globals or {}).get(CLASSES, [])})
        return out

    def _uses_of(self, idx, cls, stack=()):
        """(attributes used, assigned, methods called) of parameter idx, through the functions of this module it's passed to."""
        params = self.fdef.args.posonlyargs + self.fdef.args.args
        if idx >= len(params) or (self.fdef.name, idx) in stack:
            return set(), set(), set()
        me = params[idx].arg
        called = {id(n.func) for n in ast.walk(self.fdef) if isinstance(n, ast.Call)}
        names, writes, calls = set(), set(), set()
        for n in ast.walk(self.fdef):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == me:
                if id(n) in called:
                    calls.add(n.attr)
                else:
                    names.add(n.attr)
                    if not isinstance(n.ctx, ast.Load):
                        writes.add(n.attr)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and isinstance(self.globals.get(n.func.id), StaticCallee):
                for pos, a in enumerate(n.args):
                    if isinstance(a, ast.Name) and a.id == me:
                        u = self.globals[n.func.id]._uses_of(pos, cls, stack + ((self.fdef.name, idx),))
                        names, writes, calls = names | u[0], writes | u[1], calls | u[2]
        return names, writes, calls

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
        if isinstance(s, ast.ClassDef):
            out.setdefault(CLASSES, []).append(s)
            for d in s.body:
                if isinstance(d, ast.FunctionDef):
                    d.sj_class = s           # self.m() in it can be found in the class
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
    passed_objects = {}                # name -> {attribute: type}, from the functions it's passed to
    passed_methods = {}                # name -> what those types say of their methods
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
                elif isinstance(a, (ast.Name, ast.Subscript)) and t.kind == 'object':
                    name = a.id if isinstance(a, ast.Name) else a.value.id if isinstance(a.value, ast.Name) else None
                    if name is not None:
                        passed_objects.setdefault(name, {}).update(t.elem)
                        passed_methods.setdefault(name, {}).update(t.methods or {})
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
    alias = _element_names(fdef)
    lists = _assumed_fields(fdef, lambda n: (alias[n.value.id], n.attr) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                            and n.value.id in alias and n.attr not in ARRAY_ATTRIBUTES else None)
    for name, fields in passed_objects.items():          # an element passed on: what the callee uses is the list's too
        if name in alias:
            lists.setdefault(alias[name], {}).update(fields)
    for name in lists:
        ndim.pop(name, None)
    objects = _assumed_objects(fdef, ndim)
    for name, fields in passed_objects.items():
        if name not in alias and name not in lists:
            objects.setdefault(name, {}).update(fields)
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
        if p.arg in lists and name not in ann:
            methods = {}
            for k in [p.arg] + [k for k, v in alias.items() if v == p.arg]:
                methods.update(passed_methods.get(k, {}))
            sig.append(T.objlist_of(T.object_of('object', sorted(lists[p.arg].items()), methods)))
        elif p.arg in objects and name not in ann:
            sig.append(T.object_of('object', sorted(objects[p.arg].items()), passed_methods.get(p.arg)))
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
    for i, p in enumerate(params):        # methods called on a parameter, or on an element of one: from its class statement
        names = {p.arg} | {k for k, v in alias.items() if v == p.arg}
        calls = {n.func.attr for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id in names}
        calls |= {OPERATORS[type(n.op)] for n in ast.walk(fdef) if isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS
                  and isinstance(n.left if isinstance(n, ast.BinOp) else n.operand, ast.Name)
                  and (n.left if isinstance(n, ast.BinOp) else n.operand).id in names}      # a + b calls a.__add__
        if not calls:
            continue
        own = getattr(fdef, 'sj_class', None)
        if i == 0 and own is not None:
            cls = own
        else:                              # the one class in the module with all these methods, if there's one
            found = [c for c in (globals_ or {}).get(CLASSES, []) if calls <= {d.name for d in c.body if isinstance(d, ast.FunctionDef)}]
            if len(found) != 1:
                continue
            cls = found[0]
        t = sig[i].elem if sig[i].kind == 'objlist' else sig[i]
        t = _static_self(fdef, cls, t if t.kind == 'object' else T.object_of('object', []), calls, globals_, seen)
        if t is not None:
            sig[i] = T.objlist_of(t) if sig[i].kind == 'objlist' or p.arg in lists else t
    classes = {c.name: c for c in (globals_ or {}).get(CLASSES, [])}
    made = {n.targets[0].id: classes[n.value.func.id] for n in ast.walk(fdef) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id in classes}
    index = {p.arg: k for k, p in enumerate(params)}
    for n in ast.walk(fdef):             # a parameter passed to a method of an instance made here or of a known class
        if isinstance(n, ast.BinOp) and type(n.op) in OPERATORS and isinstance(n.left, ast.Name):    # a + b: b given to a.__add__
            n = ast.Call(ast.Attribute(n.left, OPERATORS[type(n.op)], ast.Load()), [n.right], [])
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)):
            continue
        owner = n.func.value.id
        cls = made.get(owner)
        if cls is None and owner in index and sig[index[owner]].kind == 'object' and sig[index[owner]].cls is not None:
            cls = next((c for c in classes.values() if c.name == sig[index[owner]].py), None)
        d = next((x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == n.func.attr), None) if cls is not None else None
        if d is None or d is fdef or d in seen:
            continue
        csig = assumed_signature(d, globals_, seen + (fdef,))
        for pos, a in enumerate(n.args):
            if isinstance(a, ast.Name) and a.id in index and pos + 1 < len(csig) and csig[pos + 1].kind == 'object':
                k = index[a.id]
                sig[k] = _merge_static(sig[k] if sig[k].kind == 'object' else None, csig[pos + 1])
    for i, p in enumerate(params):        # p.a.b, p.a.m(), p.a passed to a method: a is an object of its own
        if sig[i].kind == 'object':
            inner = _static_inner(fdef, p.arg, sig, params, globals_, seen)
            if inner:
                fields = dict(sig[i].elem)
                for k, t in inner.items():
                    fields[k] = _merge_static(fields.get(k), t)
                sig[i] = T.object_of(sig[i].py or 'object', sorted(fields.items()), sig[i].methods, sig[i].cls, sig[i].writes)
    return sig


def _merge_static(a, b):
    """Two assumed object types as one: their attributes and methods together."""
    if a is None or a.kind != 'object':
        return b
    if b is None or b.kind != 'object':
        return a
    fields = dict(a.elem)
    for k, t in b.elem:
        fields[k] = _merge_static(fields.get(k), t) if t.kind == 'object' else fields.get(k, t)
    methods = dict(a.methods or {})
    methods.update(b.methods or {})
    return T.object_of(a.py if a.py != 'object' else b.py or 'object', sorted(fields.items()), methods, a.cls or b.cls, a.writes | b.writes)


def _static_inner(fdef, me, sig, params, globals_, seen):
    """{attribute: object type} for the attributes of parameter me used as objects: their attributes read, their methods
    called (from the one class in the module with them all), and what a method they're passed to uses of them."""
    reads, calls, passed = {}, {}, {}
    called = {id(n.func) for n in ast.walk(fdef) if isinstance(n, ast.Call)}
    for n in ast.walk(fdef):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name) and n.value.value.id == me:
            (calls if id(n) in called else reads).setdefault(n.value.attr, set()).add(n.attr)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            for pos, a in enumerate(n.args):
                if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == me:
                    passed.setdefault(a.attr, []).append((n, pos))
    out = {}
    index = {p.arg: k for k, p in enumerate(params)}
    for a in set(reads) | set(calls) | set(passed):
        fields = {b: T.FLOAT for b in reads.get(a, ())}
        for call, pos in passed.get(a, ()):      # x.m(p.a): what m makes of its parameter there
            owner = call.func.value.id if isinstance(call.func.value, ast.Name) else None
            t = sig[index[owner]] if owner in index else None
            fn = dict(t.elem).get(call.func.attr) if t is not None and t.kind == 'object' else None
            callee = fn.elem if fn is not None and fn.kind == 'function' else None
            if callee is not None and hasattr(callee, 'fdef'):
                csig = assumed_signature(callee.fdef, globals_, seen + (fdef,))
                if pos + 1 < len(csig) and csig[pos + 1].kind == 'object':
                    fields.update(csig[pos + 1].elem)
        t = T.object_of('object', sorted(fields.items()))
        if calls.get(a):
            found = [c for c in (globals_ or {}).get(CLASSES, []) if calls[a] <= {d.name for d in c.body if isinstance(d, ast.FunctionDef)}]
            if len(found) == 1:
                t = _static_self(fdef, found[0], t, calls[a], globals_, seen) or t
        out[a] = t
    return out


def _static_self(fdef, cls, own, calls, globals_, seen):
    """self, in a method that calls the class's other methods: what they use too, and each of them, as a call would find
    them on the class statement. None if one isn't a plain method there."""
    defs = {s.name: s for s in cls.body if isinstance(s, ast.FunctionDef)}
    fields = dict(own.elem) if own.kind == 'object' else {}
    methods = dict(own.methods or {}) if own.kind == 'object' else {}
    for m in sorted(calls):
        d = defs.get(m)
        if d is None or d is fdef or d in seen or any(isinstance(x, ast.Name) and x.id in ('staticmethod', 'classmethod', 'property') for x in d.decorator_list):
            return None
        sig = assumed_signature(d, globals_, seen + (fdef,))
        mt = sig[0] if sig else None
        inner = list(mt.elem) if mt is not None and mt.kind == 'object' else []
        for k, ft in inner:
            fields[k] = _merge_static(fields.get(k), ft) if ft.kind == 'object' else fields.get(k, ft)
        mine = d.args.posonlyargs + d.args.args
        stores = {n.attr for n in ast.walk(d) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and mine
                  and n.value.id == mine[0].arg and not isinstance(n.ctx, ast.Load)}
        nested = (mt.methods or {}) if inner else {}
        methods.update(nested)
        methods[m] = ({k for k, ft in inner if ft.kind != 'function'}, stores.union(*(w for _, w in nested.values())))
        fields[m] = T.function_of(StaticCallee(d, globals_))
    return T.object_of(cls.name, sorted(fields.items(), key=lambda kv: kv[0]), methods, _stand_in(cls))


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


CLASSES = ' classes'          # in module_globals: the module's class statements (not a name anything can have)

ARRAY_ATTRIBUTES = {'T', 'dtype', 'shape', 'size', 'ndim', 'copy', 'flatten', 'astype', 'item', 'tolist', 'reshape', 'real', 'imag', 'flat'} | set(REDUCTIONS)


def _element_names(fdef):
    """{name: list}: names bound to an element by for p in ps, for i, p in enumerate(ps) or p = ps[i], ps a parameter."""
    params = {p.arg for p in fdef.args.posonlyargs + fdef.args.args}
    out = {}
    for n in ast.walk(fdef):
        target, it = (n.target, n.iter) if isinstance(n, (ast.For, ast.comprehension)) else (None, None)
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'enumerate' and it.args and isinstance(target, ast.Tuple) \
                and len(target.elts) == 2:
            it, target = it.args[0], target.elts[1]
        if isinstance(target, ast.Name) and isinstance(it, ast.Name) and it.id in params:
            out[target.id] = it.id
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Subscript) \
                and isinstance(n.value.value, ast.Name) and n.value.value.id in params and not isinstance(n.value.slice, ast.Slice):
            out[n.targets[0].id] = n.value.value.id
    calls = {id(n.func) for n in ast.walk(fdef) if isinstance(n, ast.Call)}
    used = {n.value.id for n in ast.walk(fdef) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and id(n) not in calls
            and n.attr not in ARRAY_ATTRIBUTES}
    passed = {a.id for n in ast.walk(fdef) if isinstance(n, ast.Call) for a in n.args if isinstance(a, ast.Name)}
    return {k: v for k, v in out.items() if k in used or k in passed}    # an element of an array is a number


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


def _methods_named(fdef, globals_):
    """The module's methods that x.m() in fdef may call, by name: what they read from module level fdef passes them."""
    names = {n.func.attr for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    return [StaticCallee(d, globals_) for c in (globals_ or {}).get(CLASSES, []) for d in c.body
            if isinstance(d, ast.FunctionDef) and d.name in names and d is not fdef]


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
    stack += _methods_named(fdef, globals_)
    while stack:                       # a global read by a function this calls is passed through it
        f = stack.pop()
        if f.fdef.name in seen:
            continue
        seen.add(f.fdef.name)
        inner = _lift.local_names(f.fdef)
        for st in f.fdef.body:
            loads |= _lift._split(st)[0] - inner
        stack += _methods_named(f.fdef, globals_)
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
    classes = {c.name: c for c in (globals_ or {}).get(CLASSES, [])}
    makes = {n.func.id for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in classes and n.func.id not in local}
    named = {n.func.attr for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    named |= {OPERATORS[type(n.op)] for n in ast.walk(fdef) if isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS}
    stand = {c.name: _stand_in(c) for c in classes.values()}
    for c in classes.values():                   # a method it calls returning a new instance makes one here too
        for d in c.body:
            if isinstance(d, ast.FunctionDef) and d.name in named and d is not fdef:
                made_cls = returns_made(d, stand)
                if made_cls is not None:
                    makes.add(made_cls.__name__)
    for name in sorted(makes):                  # a class it makes instances of: built from the class statement
        t = _static_made(classes[name], fdef, globals_)
        if t is not None:
            added.append(ast.arg(name))
            tuples[name] = t
    pool = _static_pool(fdef, globals_)
    if pool is not None:                         # references chosen by data: the pool a call would make
        added.append(ast.arg('sjpool'))
        tuples['sjpool'] = pool[1]
        tuples.update(pool[0])
    if not added:
        return fdef, tuples
    tree = copy.deepcopy(fdef)
    tree.args.args = tree.args.args + [ast.copy_location(a, fdef) for a in added]
    return tree, tuples


_STATIC_CLASSES = {}


def _stand_in(cdef):
    """The type a class statement stands for here: one per statement, so its instances are one class throughout."""
    return _STATIC_CLASSES.setdefault(id(cdef), type(cdef.name, (), {}))


def _fields_of(cdef):
    """{attribute: annotation text or None} of a class statement: a dataclass's fields, or what __init__ assigns to self."""
    out = {}
    for s_ in cdef.body:
        if isinstance(s_, ast.AnnAssign) and isinstance(s_.target, ast.Name):
            out[s_.target.id] = ast.unparse(s_.annotation)
        if isinstance(s_, ast.FunctionDef) and s_.name == '__init__' and s_.args.args:
            me = s_.args.args[0].arg
            for n in ast.walk(s_):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == me and isinstance(n.ctx, ast.Store):
                    out.setdefault(n.attr, None)
    return out


def _static_pool(fdef, globals_):
    """For a function holding references chosen by data, read from source: ({parameter: ref type}, the pool's type), the
    pool's class the one in the module with every attribute used through references, its numbers by annotation and its
    other candidate attributes references to it. None if no class fits."""
    from ._dispatch import _ref_candidates
    attrs, names = _ref_candidates(fdef)
    if not attrs and not names:
        return None
    params = [p.arg for p in fdef.args.posonlyargs + fdef.args.args]
    used = {n.attr for n in ast.walk(fdef) if isinstance(n, ast.Attribute) and (isinstance(n.value, ast.Name) and n.value.id in names
            or isinstance(n.value, ast.Attribute) and n.value.attr in attrs)}
    found = [c for c in (globals_ or {}).get(CLASSES, []) if used and used <= set(_fields_of(c)) | {d.name for d in c.body if isinstance(d, ast.FunctionDef)}]
    if len(found) != 1:
        return None
    cdef = found[0]
    fields = _fields_of(cdef)
    kinds = {'int': T.INT, 'float': T.FLOAT, 'bool': T.BOOL}
    numbers = [(a, kinds[fields[a]]) for a in sorted(used) if a in fields and fields[a] in kinds and a not in attrs]
    numbers += [(a, T.FLOAT) for a in sorted(used) if a in fields and fields[a] is None and a not in attrs]
    cls = _stand_in(cdef)
    refs = [(a, cls) for a in sorted(attrs) if a in fields and fields[a] not in kinds]
    pooled = {p: T.ref_of(cls) for p in params if p in names}
    return pooled, T.pool_of([(cls, numbers, refs)], [(params.index(p), '') for p in pooled])


def _static_made(cdef, fdef, globals_):
    """What scramjit would make of cdef(...) in fdef, from source: a dataclass's fields, or an __init__ that only assigns."""
    is_dc = any((isinstance(d, ast.Name) and d.id == 'dataclass') or (isinstance(d, ast.Attribute) and d.attr == 'dataclass')
                or (isinstance(d, ast.Call) and getattr(d.func, 'attr', getattr(d.func, 'id', None)) == 'dataclass') for d in cdef.decorator_list)
    defs = {d.name: d for d in cdef.body if isinstance(d, ast.FunctionDef)}
    params, assigns = [], []
    if is_dc:
        for s in cdef.body:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and 'ClassVar' not in ast.unparse(s.annotation):
                default = None
                if s.value is not None:
                    try:
                        default = ('const', ast.literal_eval(s.value))
                    except ValueError:
                        return None              # field(default_factory=...) and the like
                params.append((s.target.id, default, False))
                assigns.append((s.target.id, ('param', s.target.id)))
    elif '__init__' in defs:
        init = defs['__init__']
        me = init.args.args[0].arg if init.args.args else None
        names = [a.arg for a in init.args.args[1:]]
        ds = [None] * (len(names) - len(init.args.defaults)) + [('const', ast.literal_eval(d)) if isinstance(d, ast.Constant) else None for d in init.args.defaults]
        params = [(n, d, False) for n, d in zip(names, ds)]
        for s in init.body:
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
                continue
            if not (isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Attribute) and isinstance(s.targets[0].value, ast.Name)
                    and s.targets[0].value.id == me):
                return None
            assigns.append((s.targets[0].attr, ('expr', s.value)))
    local = {n.targets[0].id for n in ast.walk(fdef) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name) and n.value.func.id == cdef.name and isinstance(n.targets[0], ast.Name)}
    calls = {n.func.attr for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id in local}
    named = {n.func.attr for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    named |= {OPERATORS[type(n.op)] for n in ast.walk(fdef) if isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS}
    calls |= {m for m in named & set(defs) if defs[m] is not fdef}     # new instances from calls and operators: what's called on them
    if '__post_init__' in defs:
        calls.add('__post_init__')
    own = T.object_of('object', [])
    t = _static_self(fdef, cdef, own, calls, globals_, ()) if calls else own
    if t is None:
        return None
    cls = _STATIC_CLASSES.setdefault(id(cdef), type(cdef.name, (), {}))
    writes = set().union(*(w for _, w in (t.methods or {}).values()))
    return T.madeclass_of(cls, {'params': params, 'assigns': assigns, 'post_init': '__post_init__' in defs, 'frozen': False},
                          t.methods or {}, writes, [(k, ft) for k, ft in t.elem if ft.kind == 'function'])


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
