"""The @scram decorator: signatures, the compile cache, the call boundary and fallback."""

import array
import ast
import collections
import itertools
import copy
import enum
import math
import functools
import operator
import inspect
import struct
import sys
import textwrap
import ctypes
import os
import threading
import time
import types
import warnings
import weakref

from . import _types as T
from ._compiler import OPERATORS, Callee, Compiler, Unit, array_record_slots, cimo_diagnostic, leaf_types, ret_arrays, ret_pairs, returns_made
from ._errors import CompileError, FallbackWarning, Unsupported
from . import _interrupt
from . import _lift
from ._runtime import LIBM, PRELUDE, RERUN, RETURNS_INT, exception

_cimo = None
_cimo_lock = threading.Lock()


def _get_cimo():
    """cimo, imported with fused multiply-add contraction off (Python never contracts)."""
    global _cimo
    if _cimo is None:
        with _cimo_lock:
            if _cimo is None:
                import cimo
                _cimo = cimo
    return _cimo


_libm = None


def _libm_table():
    """Addresses of the C library math functions CPython itself calls (LIBM's order), packed for
    compiled code, which calls them through this table: the same functions, the same results."""
    global _libm
    if _libm is None:
        import ctypes
        import ctypes.util
        import struct as st
        if sys.platform == 'emscripten':         # Pyodide: libm is in the main module; addresses are table indices
            lib = ctypes.CDLL(None)
        elif sys.platform == 'win32':
            lib = ctypes.CDLL('ucrtbase')
        elif sys.platform == 'darwin':
            lib = ctypes.CDLL('libSystem.dylib')
        else:
            lib = ctypes.CDLL(ctypes.util.find_library('m') or 'libm.so.6')
        _libm = st.pack(f'<{len(LIBM)}Q', *(ctypes.cast(getattr(lib, n), ctypes.c_void_p).value for n in LIBM))
    return _libm


def _arg_type(x):
    """The compiled type of an argument, or None if it isn't compiled."""
    t = type(x)
    if t is Function or T.is_c_module(x) or T.is_c_function(x):
        return T.function_of(x)
    if t is int:                             # past int64 but below 2**64: held unsigned, passed as its bits
        return T.INT if -(1 << 63) <= x < (1 << 63) else T.U64 if 0 <= x < 1 << 64 else None
    if t is float:
        return T.FLOAT
    if t is bool:
        return T.BOOL
    if t is list and x and type(x[0]) is list:   # a list of lists: rectangular, every list its own object
        shape = _nested_shape(x)
        if shape is None:
            return None
        flat = _flatten(x, len(shape))
        elem = _arg_type(flat)
        return None if elem is None else T.array(elem.elem, True, len(shape))
    if t is list:                            # copied into an array at the call, and back if written
        kinds = set(map(type, x))                 # in C: a list of 200,000 in about 2 ms
        if len(kinds) != 1 or kinds.pop() not in (int, float, bool):
            return None
        if type(x[0]) is int and not (-(1 << 63) <= min(x) and max(x) < 1 << 63):
            return None
        return T.list_of({int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(x[0])])
    if t is tuple:
        elems = [_arg_type(v) for v in x]
        if not elems or any(e is None or not T.is_number_tree(e) or e is T.U64 or (e.kind == 'tuple' and T.U64 in T.leaves(e)) for e in elems):
            return None
        return T.tuple_of(elems)
    if isinstance(x, enum.Enum) and T.enum_ok(t):       # a member: passed as its value
        return T.enum_member_of(t)
    if isinstance(x, type) and issubclass(x, enum.Enum):
        return T.enum_class_of(x) if T.enum_ok(x) else None
    if isinstance(x, type):
        return T.classref_of(x)
    np = sys.modules.get('numpy')
    if np is not None:
        if t is np.ndarray:
            if x.ndim >= 1 and x.dtype.isnative:
                name = x.dtype.name
                elem = T.NP_FLOAT64 if name == 'float64' else T.NP_INT64 if name == 'int64' else T.NP_UINT64 if name == 'uint64' else T.NARROW.get(name)
                if elem is not None:
                    if x.flags.c_contiguous:
                        return T.array(elem, x.flags.writeable, x.ndim)
                    if x.flags.aligned and all(s % x.itemsize == 0 for s in x.strides):      # a view, as it is
                        return T.array(elem, x.flags.writeable, x.ndim, True)
            return None
        if t is np.float64:
            return T.NP_FLOAT64
        if t is np.int64:
            return T.NP_INT64
        if t is np.bool_:
            return T.NP_BOOL
        if t is np.uint64:
            return T.NP_UINT64
        if issubclass(t, np.generic) and x.dtype.isnative and x.dtype.name in T.NARROW:     # np.intc too
            return T.NARROW[x.dtype.name]
    return None


def _describe(x, t):
    if t is not None:
        return t.name
    np = sys.modules.get('numpy')
    if np is not None and isinstance(x, np.ndarray):
        return f'{x.dtype}{"[:]" * x.ndim}' + ('' if x.flags.c_contiguous else ' (strided)')
    return type(x).__name__


_ANNOTATIONS = {int: 'int', float: 'float', bool: 'bool'}


def _contract_ok(x, ann):
    np = sys.modules.get('numpy')
    if ann is bool:
        return isinstance(x, bool) or (np is not None and isinstance(x, np.bool_))
    ints = (int,) + ((np.integer,) if np is not None else ())
    if ann is int:
        return isinstance(x, ints)
    return isinstance(x, ints + (float,) + ((np.floating,) if np is not None else ()))


def _raise_at(exc, filename, line, name):
    """Raise exc so the traceback's last frame is the user's file and line."""
    if not name.isidentifier():
        name = 'compiled'
    ns = {'__sj_exc__': exc}
    if line and line >= 2:
        exec(compile('\n' * (line - 2) + f'def {name}():\n raise __sj_exc__\n', filename, 'exec'), ns)
        ns[name]()
    raise exc


def _raise_chain(exc, frames):
    """Raise exc through a frame for each of [(file, line, function name)], outermost first, so the traceback reads as Python's would."""
    def run(i):
        if i == len(frames) - 1:
            _raise_at(exc, *frames[i])
        filename, line, name = frames[i]
        if not name.isidentifier():
            name = 'compiled'
        ns = {'__sj_next__': lambda: run(i + 1)}
        if line and line >= 2:
            exec(compile('\n' * (line - 2) + f'def {name}():\n __sj_next__()\n', filename, 'exec'), ns)
            ns[name]()
        run(i + 1)
    run(0)


def _mangle(fdef, qualname):
    """A function defined in a class: __x in it means _Class__x, as Python compiles it. Done on the tree once, so
    everything after sees the names Python uses."""
    parts = qualname.split('.')
    if len(parts) < 2 or parts[-2] == '<locals>' or not parts[-2].lstrip('_'):
        return
    cls = parts[-2].lstrip('_')

    def mangled(name):
        return f'_{cls}{name}' if name.startswith('__') and not name.endswith('__') else name
    for n in ast.walk(fdef):
        if isinstance(n, ast.Attribute):
            n.attr = mangled(n.attr)
        elif isinstance(n, ast.Name):
            n.id = mangled(n.id)
        elif isinstance(n, ast.arg):
            n.arg = mangled(n.arg)


def _dict_keys(fdef):
    """For each parameter subscripted by a constant string or int: the keys, if it turns out to be a dict."""
    if fdef is None:
        return {}
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    found = {}
    for n in ast.walk(fdef):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in index \
                and isinstance(n.slice, ast.Constant) and type(n.slice.value) in (str, int):
            found.setdefault(index[n.value.id], set()).add(n.slice.value)
    return {i: sorted(keys, key=repr) for i, keys in found.items()}


def _attrs_used(fdef):
    """For each parameter used as p.a: (the attributes read or assigned, sorted; those assigned; the methods called, as
    p.m(...)), if it turns out to be an instance. Not p.__a: in a class that means another name, which the source doesn't say."""
    if fdef is None:
        return {}
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    index.update({q: index[p] for q, p in _object_aliases(fdef).items()})
    found = {}
    for name, attr, how in _attribute_uses(fdef):
        if name in index:
            _add_use(found.setdefault(index[name], (set(), set(), set())), attr, how)
    return {i: (sorted(names), stores, calls) for i, (names, stores, calls) in found.items()}


def _made_locals(fdef, classes):
    """{name: class's name} for v = C(...) statements, C one of the names given."""
    out = {}
    for n in ast.walk(fdef):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Name) and n.value.func.id in classes:
            out[n.targets[0].id] = n.value.func.id
        elif isinstance(n, ast.Return) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id in classes:
            out[f'sjret__{n.lineno}_{n.col_offset}'] = n.value.func.id      # return C(...): made, and handed back
    return out


_DATACLASS_INIT_FILES = ('<string>',)


def _init_recipe(cls):
    """How cls(...) fills a new instance, if it only stores into it: a dataclass's own __init__, an __init__ of
    nothing but self.a = expression of its parameters, or none. A reason, as a string, if not."""
    import dataclasses
    if type(cls).__call__ is not type.__call__:
        return f'{cls.__name__} has a metaclass with its own __call__'
    if _in_class(cls, '__new__') is not object.__dict__['__new__']:
        return f'{cls.__name__} defines __new__'
    if _in_class(cls, '__getattribute__') is not object.__dict__['__getattribute__'] or _in_class(cls, '__getattr__') is not _MISSING:
        return f'{cls.__name__} defines __getattribute__ or __getattr__'
    init = _in_class(cls, '__init__')
    params, assigns = [], []
    frozen = dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
    if dataclasses.is_dataclass(cls) and cls.__dataclass_params__.init and getattr(getattr(init, '__code__', None), 'co_filename', '') in _DATACLASS_INIT_FILES:
        for f in cls.__dataclass_fields__.values():
            if f._field_type is not dataclasses._FIELD:
                if f._field_type is dataclasses._FIELD_INITVAR:
                    return f'{cls.__name__} has an InitVar'
                continue
            if f.default_factory is not dataclasses.MISSING:
                return f'{cls.__name__}.{f.name} has a default_factory'
            default = None if f.default is dataclasses.MISSING else ('const', f.default)
            if f.init:
                params.append((f.name, default, bool(getattr(f, 'kw_only', False))))
                assigns.append((f.name, ('param', f.name)))
            elif default is not None:
                assigns.append((f.name, default))
    elif init is object.__dict__['__init__']:
        pass
    elif type(init) is types.FunctionType:
        w = _method(init)
        tree = w._tree
        if tree is None or not tree.args.args or tree.args.vararg or tree.args.kwarg or tree.args.posonlyargs:
            return f'{cls.__name__}.__init__ takes what can\'t be bound here'
        me = tree.args.args[0].arg
        names = [a.arg for a in tree.args.args[1:w._npos]]
        defaults = list(init.__defaults__ or ())
        defaults = [None] * (len(names) - len(defaults)) + [('const', d) for d in defaults]
        kwd = init.__kwdefaults__ or {}
        params = [(n, d, False) for n, d in zip(names, defaults)]
        params += [(a.arg, ('const', kwd[a.arg]) if a.arg in kwd else None, True) for a in w._kwonly_args]
        allowed = set(names) | {a.arg for a in w._kwonly_args}
        for k, s in enumerate(tree.body):
            if k == 0 and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue
            if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Attribute)
                    and isinstance(s.targets[0].value, ast.Name) and s.targets[0].value.id == me):
                return f'{cls.__name__}.__init__ does more than assign attributes of {me}'
            used = {n.id for n in ast.walk(s.value) if isinstance(n, ast.Name)}
            if not used <= allowed or any(isinstance(n, (ast.Call, ast.Lambda, ast.NamedExpr)) for n in ast.walk(s.value)):
                return f'{cls.__name__}.__init__ assigns something other than an expression of its parameters'
            assigns.append((s.targets[0].attr, ('expr', s.value)))
    else:
        return f'{cls.__name__}.__init__ is not a plain function'
    for a, _ in assigns:
        c = _in_class(cls, a)
        if c is not _MISSING and (type(c) is not types.MemberDescriptorType and any(hasattr(type(c), m) for m in ('__get__', '__set__', '__delete__'))):
            return f'{cls.__name__}.{a} is a descriptor'
    if not frozen and _in_class(cls, '__setattr__') is not _DEFAULT_SET:
        return f'{cls.__name__} defines __setattr__'
    return {'params': params, 'assigns': assigns, 'post_init': _in_class(cls, '__post_init__') is not _MISSING, 'frozen': bool(frozen)}


def _local_classes(f, owners):
    """{name: class} for the names in f holding an instance it makes: v = C(...), and v = g(...), v = x.m(...) or v = x + y
    when what's called returns a new instance. owners: the classes of names known already (its instance parameters)."""
    tree = f._ctree or f._tree
    g = f._fn.__globals__
    known, out = dict(owners or {}), {}
    for _ in range(8):
        grew = False
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id not in out:
                cls = _made_by(n.value, g, known)
                if cls is not None:
                    out[n.targets[0].id] = known[n.targets[0].id] = cls
                    grew = True
        if not grew:
            break
    return out


def _made_by(e, g, known):
    """The class of the new instance an expression makes: C(...), or a call of a function or method returning one."""
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
        x = g.get(e.func.id)
        if isinstance(x, type) and not issubclass(x, (enum.Enum, BaseException)):
            return x
        if type(x) is Function and x._tree is not None:
            return returns_made(x._tree, x._fn.__globals__)
        return None
    w = None
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute):
        c = _class_of(e.func.value, g, known)
        w = _in_class(c, e.func.attr) if c is not None else None
    elif isinstance(e, (ast.BinOp, ast.UnaryOp)) and type(e.op) in OPERATORS:
        c = _class_of(e.left if isinstance(e, ast.BinOp) else e.operand, g, known)
        w = _in_class(c, OPERATORS[type(e.op)]) if c is not None else None
    if type(w) is Function or type(w) is types.FunctionType:
        w = _method(w)
        if w._tree is not None:
            return returns_made(w._tree, w._fn.__globals__)
    return None


def _class_of(e, g, known):
    """The class of the instance an expression is: a name whose class is known, or a new instance."""
    if isinstance(e, ast.Name):
        return known.get(e.id)
    return _made_by(e, g, known)


def _unnamed_uses(tree, g, known, cls):
    """(attribute, 'load' or 'call') used directly on a new instance of cls that has no name: (a - b).norm(), (a + b).x."""
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and not isinstance(n.value, ast.Name) and _made_by(n.value, g, known) is cls:
            yield n.attr, 'call' if id(n) in called else 'load'
        elif isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS:
            x = n.left if isinstance(n, ast.BinOp) else n.operand
            if not isinstance(x, ast.Name) and _made_by(x, g, known) is cls:
                yield OPERATORS[type(n.op)], 'call'


def _callees_of(f, known):
    """[(compiled function, {its parameter: class})] for what f calls: functions at module level by name, and methods of
    names whose class is known, operators too."""
    tree = f._ctree or f._tree
    g = f._fn.__globals__
    out = []
    for n in ast.walk(tree):
        w, me = None, None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and type(g.get(n.func.id)) is Function:
            out.append((g[n.func.id], {}))
            continue
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            me = _class_of(n.func.value, g, known)
            w = _in_class(me, n.func.attr) if me is not None else None
        elif isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in OPERATORS:
            me = _class_of(n.left if isinstance(n, ast.BinOp) else n.operand, g, known)
            w = _in_class(me, OPERATORS[type(n.op)]) if me is not None else None
        if type(w) is Function or type(w) is types.FunctionType:
            w = _method(w)
            ps = w._tree.args.posonlyargs + w._tree.args.args if w._tree is not None else []
            if ps:
                out.append((w, {ps[0].arg: me}))
    return out


def _made_uses(f, cls, owners, seen=None):
    """(attributes used, assigned, methods called) of the instances of cls that f, or anything it calls, makes; None if
    nothing does. owners: {name: class} of f's instance parameters."""
    seen = set() if seen is None else seen
    if id(f) in seen or f._tree is None:
        return None
    seen.add(id(f))
    tree = f._ctree or f._tree
    g = f._fn.__globals__
    mine = _local_classes(f, owners)
    known = dict(owners or {})
    known.update(mine)
    local = {v for v, c in mine.items() if c is cls}
    handed = any(isinstance(n, ast.Return) and n.value is not None and _made_by(n.value, g, known) is cls
                 or isinstance(n, (ast.Call, ast.BinOp, ast.UnaryOp)) and _made_by(n, g, known) is cls for n in ast.walk(tree))
    uses = None
    if local or handed:
        uses = (set(), set(), set())
        for v, attr, how in _attribute_uses(tree):
            if v in local:
                _add_use(uses, attr, how)
        for attr, how in _unnamed_uses(tree, g, known, cls):
            _add_use(uses, attr, how)
        for n in ast.walk(tree):                 # passed on: what the callee makes of it
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                h = g.get(n.func.id)
                for pos, a in enumerate(n.args):
                    if isinstance(a, ast.Name) and a.id in local and type(h) is Function and h._tree is not None:
                        u = _param_uses(h, pos, cls, h._attrs.get(pos, ((), set(), set())))
                        for k in range(3):
                            uses[k].update(u[k])
    for h, theirs in _callees_of(f, known):      # what they make and hand back, or make for themselves
        u = _made_uses(h, cls, theirs, seen)
        if u is not None:
            uses = uses or (set(), set(), set())
            for k in range(3):
                uses[k].update(u[k])
    return uses


def _made_class_type(f, cls, name, owners=None):
    """The type of a class the function, or what it calls, makes instances of, under this name; None if they can't be
    compiled, or nothing makes any."""
    found = _made_uses(f, cls, owners)
    if found is None:
        return None
    recipe = _init_recipe(cls)
    if type(recipe) is str:
        f._made_why = recipe
        return None
    recipe = dict(recipe)
    uses = found
    if recipe['post_init']:
        uses[2].add('__post_init__')
    assigned = {a for a, _ in recipe['assigns']}
    found = _methods_called(None, uses[2], cls) if uses[2] else {}
    if type(found) is str:
        f._made_why = found
        return None
    names = set(uses[0]).union(*(n for n, _, _ in found.values()))
    writes = set(uses[1]).union(*(w for _, w, _ in found.values()))
    if not names <= assigned:
        f._made_why = f'{cls.__name__}.__init__ doesn\'t assign {sorted(names - assigned)[0]}'
        return None
    if writes and recipe['frozen']:
        f._made_why = f'{cls.__name__} is frozen'
        return None
    methods = {m: (n, w) for m, (n, w, _) in found.items()}
    return T.madeclass_of(cls, recipe, methods, writes, [(m, T.function_of(w)) for m, (_, _, w) in sorted(found.items())])


def _merge_nested(a, b):
    """Two {attribute: [used, assigned, called, deeper]} as one."""
    if not b:
        return a
    out = {k: [set(v[0]), set(v[1]), set(v[2]), dict(v[3])] for k, v in (a or {}).items()}
    for k, v in b.items():
        e = out.setdefault(k, [set(), set(), set(), {}])
        e[0] |= v[0]
        e[1] |= v[1]
        e[2] |= v[2]
        e[3] = _merge_nested(e[3], v[3])
    return out


def _chain(n):
    """(name, [attributes]) for name.a.b..., else (None, None)."""
    path = []
    while isinstance(n, ast.Attribute):
        path.append(n.attr)
        n = n.value
    return (n.id, path[::-1]) if isinstance(n, ast.Name) else (None, None)


def _nested_uses(fdef):
    """For each parameter used as p.a.b: {a: [attributes of p.a used, assigned, methods called, {deeper}]}."""
    if fdef is None:
        return {}
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    index.update({q: index[p] for q, p in _object_aliases(fdef).items()})
    called = {id(n.func) for n in ast.walk(fdef) if isinstance(n, ast.Call)}
    out = {}
    for n in ast.walk(fdef):
        if not isinstance(n, ast.Attribute):
            continue
        base, path = _chain(n)
        if base not in index or path is None or len(path) < 2 or any(a.startswith('__') and not a.endswith('__') for a in path):
            continue
        node = out.setdefault(index[base], {})
        for a in path[:-2]:
            node = node.setdefault(a, [set(), set(), set(), {}])[3]
        entry = node.setdefault(path[-2], [set(), set(), set(), {}])
        how = 'call' if id(n) in called else 'load' if isinstance(n.ctx, ast.Load) else 'store'
        _add_use(entry, path[-1], how)
        for a in path[:-2]:                      # the objects on the way are read
            pass
    return out


def _object_aliases(fdef):
    """{q: p} for q = p at the top of the body, p a parameter: the compiler takes q as p where that's safe."""
    params = {p.arg for p in fdef.args.posonlyargs + fdef.args.args}
    return {s.targets[0].id: s.value.id for s in fdef.body if isinstance(s, ast.Assign) and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Name) and s.value.id in params and s.targets[0].id not in params}


def _returned(fdef):
    """The parameters returned as they are, by name or through q = p, or only asked isinstance() of."""
    if fdef is None:
        return set()
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    index.update({q: index[p] for q, p in _object_aliases(fdef).items()})
    out = {index[n.value.id] for n in ast.walk(fdef) if isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id in index}
    return out | {index[n.args[0].id] for n in ast.walk(fdef) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)       # isinstance(p, C)
                  and n.func.id == 'isinstance' and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id in index}


def _element_nested(fdef):
    """For each parameter that may be a list of instances: {attribute: [used, assigned, called, {deeper}]} of the objects in
    its elements' attributes, through the names _element_attrs follows: b.pos.x for b in bs."""
    if fdef is None:
        return {}
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    alias = _element_alias(fdef, index)
    called = {id(n.func) for n in ast.walk(fdef) if isinstance(n, ast.Call)}
    out = {}
    for n in ast.walk(fdef):
        if not isinstance(n, ast.Attribute):
            continue
        path, e = [], n
        while isinstance(e, ast.Attribute):
            path.append(e.attr)
            e = e.value
        path.reverse()
        ps = alias.get(e.id) if isinstance(e, ast.Name) else e.value.id if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) \
            and e.value.id in index else None
        if ps is None or len(path) < 2 or any(a.startswith('__') and not a.endswith('__') for a in path):
            continue
        node = out.setdefault(index[ps], {})
        for a in path[:-2]:
            node = node.setdefault(a, [set(), set(), set(), {}])[3]
        _add_use(node.setdefault(path[-2], [set(), set(), set(), {}]), path[-1],
                 'call' if id(n) in called else 'load' if isinstance(n.ctx, ast.Load) else 'store')
    return out


def _element_alias(fdef, index):
    """{name: list parameter} for names bound to an element: for p in ps, for i, p in enumerate(ps), p = ps[i]."""
    alias = {}
    for n in ast.walk(fdef):
        target, it = (n.target, n.iter) if isinstance(n, (ast.For, ast.comprehension)) else (None, None)
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'enumerate' and it.args and isinstance(target, ast.Tuple) \
                and len(target.elts) == 2:
            it, target = it.args[0], target.elts[1]
        if isinstance(target, ast.Name) and isinstance(it, ast.Name) and it.id in index:
            alias[target.id] = it.id
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Subscript) \
                and isinstance(n.value.value, ast.Name) and n.value.value.id in index:
            alias[n.targets[0].id] = n.value.value.id
    return alias


def _element_attrs(fdef):
    """For each parameter that may be a list of instances: the attributes its elements are used with, as _attrs_used
    gives them, through for p in ps, for i, p in enumerate(ps), p = ps[i], and ps[i].a."""
    if fdef is None:
        return {}
    index = {p.arg: i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args)}
    alias = _element_alias(fdef, index)
    called = {id(n.func) for n in ast.walk(fdef) if isinstance(n, ast.Call)}
    found = {}
    for n in ast.walk(fdef):
        if not isinstance(n, ast.Attribute) or (n.attr.startswith('__') and not n.attr.endswith('__')):
            continue
        v = n.value
        ps = alias.get(v.id) if isinstance(v, ast.Name) else v.value.id if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) \
            and v.value.id in index else None
        if ps is not None:
            how = 'call' if id(n) in called else 'load' if isinstance(n.ctx, ast.Load) else 'store'
            _add_use(found.setdefault(index[ps], (set(), set(), set())), n.attr, how)
    return {i: (sorted(names), stores, calls) for i, (names, stores, calls) in found.items()}


def _objlist_type(xs, attrs, seen, dicts=False, nested=None):
    """A list of instances, all one class whose lookups run no code, holding numbers in every attribute used, or objects
    of one class each doing the same (each its own object), and not passed otherwise. Its numbers are copied into a list
    each, at the call, by path (pos.x)."""
    if not xs:
        return None
    cls = type(xs[0])
    if set(map(type, xs)) != {cls}:
        return None
    methods = {}
    fields = _object_fields(xs[0], *attrs, methods=methods, nested=nested)
    if type(fields) is not list:
        return None
    ids = list(map(id, xs))
    shared = len(set(ids)) != len(ids)           # an instance twice in it: each once in a pool, and an index list
    if shared:
        xs = list({id(x): x for x in xs}.values())
        ids = list(map(id, xs))
    if seen.intersection(ids):
        return None                              # an instance here and in another argument: not yet
    seen.update(ids)
    written = set(attrs[1]).union(*(w for _, w in methods.values()))
    has_dict = hasattr(xs[0], '__dict__')
    levels, pooled = {}, set()
    if not _inner_ok(xs, fields, levels, pooled):
        return None
    inner = [id(o) for objs in levels.values() for o in objs]
    if len(set(inner)) != len(inner) or seen.intersection(inner) or set(inner) & set(ids):
        return None                              # an object at two paths, or here and elsewhere: not yet
    seen.update(inner)
    if shared:
        pooled.add('')
    for k, ft in fields:
        if ft.kind == 'function':
            if has_dict and any(k in x.__dict__ for x in xs):
                return None
            continue
        if ft.kind == 'object':
            continue
        if ft not in (T.INT, T.FLOAT, T.BOOL):
            return None
        want = {T.INT: int, T.FLOAT: float, T.BOOL: bool}[ft]
        try:
            values = list(map(operator.attrgetter(k), xs))      # in C: the class was checked, so this runs none of its code
        except AttributeError:
            return None
        if set(map(type, values)) != {want} or (want is int and not (-(1 << 63) <= min(values) and max(values) < 1 << 63)):
            return None
        if k in written and has_dict and type(_in_class(cls, k)) is not types.MemberDescriptorType and not all(k in d for d in map(vars, xs)):
            return None                          # assigning would add it to an instance that only had the class's
    if dicts and has_dict:                       # an instance's own __dict__, passed as a dict too, is one thing under two names
        held = list(map(id, map(vars, xs)))
        if seen.intersection(held):
            return None
        seen.update(held)
    return T.objlist_of(T.object_of(cls.__qualname__, fields, methods, cls, written), pooled)


def _inner_ok(xs, fields, levels, pooled, prefix=''):
    """Every element's objects in attributes are of the class the first's are, holding the same types. levels: path -> the
    objects there, each once; pooled gets the paths where one object is at several places."""
    for k, ft in fields:
        if ft.kind != 'object':
            continue
        try:
            objs = list(map(operator.attrgetter(k), xs))
        except AttributeError:
            return False
        if set(map(type, objs)) != {ft.cls}:
            return False
        once = list({id(o): o for o in objs}.values())
        if len(once) != len(objs):
            pooled.add(prefix + k)
        objs = levels[prefix + k] = once
        has_dict = hasattr(objs[0], '__dict__')
        for k2, ft2 in ft.elem:
            if ft2.kind == 'function':
                if has_dict and any(k2 in o.__dict__ for o in objs):
                    return False
            elif ft2.kind != 'object':
                want = {T.INT: int, T.FLOAT: float, T.BOOL: bool}.get(ft2)
                vals = list(map(operator.attrgetter(k2), objs))
                if want is None or set(map(type, vals)) != {want} or (want is int and not (-(1 << 63) <= min(vals) and max(vals) < 1 << 63)):
                    return False
        if not _inner_ok(objs, ft.elem, levels, pooled, prefix + k + '.'):
            return False
    return True


def _pool_levels(pool, t):
    """For a list of instances' type: [(path, the objects there each once, or at each place if not pooled; the index list
    or None)], breadth first as its parameters come, the list itself first."""
    out = [('', pool, None)]
    objs_at = {'': pool}
    todo = [('', t.elem.elem)]
    while todo:
        prefix, fields = todo.pop(0)
        for key, ft in fields:
            if ft.kind != 'object':
                continue
            path = prefix + key
            objs = list(map(operator.attrgetter(key), objs_at[prefix[:-1]]))
            ix = None
            if path in t.strided:
                where = {}
                ix = array.array('q', (where.setdefault(id(o), len(where)) for o in objs))
                objs = list({id(o): o for o in objs}.values())
            objs_at[path] = objs
            out.append((path, objs, ix))
            todo.append((path + '.', ft.elem))
    return out


def _copy_back_attr(xs, key, a, et):
    """Into the instances, the attribute's values the kernel changed: the others stay the very objects they were. key may
    be a path (pos.x)."""
    if '.' in key:
        *path, key = key.split('.')
        xs = [_member(x, '.'.join(path)) for x in xs]
    for k, v in enumerate(a):
        old = getattr(xs[k], key)
        if et is T.FLOAT:
            if old == v and math.copysign(1.0, old) == math.copysign(1.0, v) or (old != old and v != v):
                continue
        elif old == v and type(old) is type(bool(v) if et is T.BOOL else v):
            continue
        setattr(xs[k], key, bool(v) if et is T.BOOL else v)


def _passes(fdef):
    """{parameter index: [(target, position)]}: a parameter, or an element of one (for p in ps, p = ps[i]), passed as a
    positional argument to a function at module level, ('f', name), or to a method, ('m', its owner's name, method)."""
    if fdef is None:
        return {}
    names = [p.arg for p in fdef.args.posonlyargs + fdef.args.args]
    index = {n: i for i, n in enumerate(names)}
    index.update({q: index[p] for q, p in _object_aliases(fdef).items()})
    alias = {}
    for n in ast.walk(fdef):
        target, it = (n.target, n.iter) if isinstance(n, (ast.For, ast.comprehension)) else (None, None)
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'enumerate' and it.args and isinstance(target, ast.Tuple) \
                and len(target.elts) == 2:
            it, target = it.args[0], target.elts[1]
        if isinstance(target, ast.Name) and isinstance(it, ast.Name) and it.id in index:
            alias[target.id] = it.id
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Subscript) \
                and isinstance(n.value.value, ast.Name) and n.value.value.id in index:
            alias[n.targets[0].id] = n.value.value.id
    found = {}
    for n in ast.walk(fdef):
        if isinstance(n, ast.BinOp) and isinstance(n.left, ast.Name) and type(n.op) in OPERATORS:
            target, given = ('m', n.left.id, OPERATORS[type(n.op)]), [n.right]       # a + b passes b to a.__add__
        elif isinstance(n, ast.Call):
            f = n.func
            target = ('f', f.id) if isinstance(f, ast.Name) else ('m', f.value.id, f.attr) \
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else None
            given = n.args
        else:
            continue
        if target is None:
            continue
        for pos, a in enumerate(given):
            if isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name) and a.value.id in index:
                a = a.value                       # ps[i]: an element
            if isinstance(a, ast.Name) and (a.id in index or a.id in alias):
                found.setdefault(index[alias.get(a.id, a.id)], []).append((target, pos))
            elif isinstance(a, ast.Attribute):     # p.pos: an object in an attribute, of a parameter or an element
                base, path = _chain(a)
                if base in index or base in alias:
                    found.setdefault(index[alias.get(base, base)], []).append((target, pos, tuple(path)))
    return found


def _unused(fdef):
    """The parameters the function never names: anything can be passed there."""
    if fdef is None:
        return set()
    named = {n.id for s in fdef.body for n in ast.walk(s) if isinstance(n, ast.Name)}
    return {i for i, p in enumerate(fdef.args.posonlyargs + fdef.args.args) if p.arg not in named}


def _owner_classes(f, args):
    """{name: class} for the parameters that are instances, and for names bound to an element of a list of them."""
    names = [p.arg for p in f._tree.args.posonlyargs + f._tree.args.args]
    out = {n: type(x) for n, x in zip(names, args) if type(x) is not list}
    out.update({q: out[p] for q, p in _object_aliases(f._tree).items() if p in out})
    made = getattr(f, '_made', None)
    if made:                                     # v = C(...): v's class is the global C
        out.update({v: f._fn.__globals__[c] for v, c in _made_locals(f._ctree or f._tree, set(made)).items() if c in f._fn.__globals__})
    lists = {n: type(x[0]) for n, x in zip(names, args) if type(x) is list and x}
    for n in ast.walk(f._tree):
        target, it = (n.target, n.iter) if isinstance(n, (ast.For, ast.comprehension)) else (None, None)
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'enumerate' and it.args and isinstance(target, ast.Tuple) \
                and len(target.elts) == 2:
            it, target = it.args[0], target.elts[1]
        if isinstance(target, ast.Name) and isinstance(it, ast.Name) and it.id in lists:
            out[target.id] = lists[it.id]
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Subscript) \
                and isinstance(n.value.value, ast.Name) and n.value.value.id in lists:
            out[n.targets[0].id] = lists[n.value.value.id]
    for v, c in _local_classes(f, out).items():         # v = a + b, v = a.add(b): new instances, of the class the call returns
        out.setdefault(v, c)
    return out


def _param_uses(f, idx, cls, base, owners=None, stack=(), nested_out=None):
    """(attributes used, those assigned, methods called) of an instance of cls given to f as parameter idx, through the
    functions and methods f passes it to. base: f's own uses of it. A target that can't be found adds nothing: the call
    to it won't compile either."""
    names, writes, calls = set(base[0]), set(base[1]), set(base[2])
    if (id(f), idx) in stack:
        return names, writes, calls
    stack = stack + ((id(f), idx),)
    for entry in f._passes.get(idx, ()):
        target, pos = entry[:2]
        path = entry[2] if len(entry) > 2 else ()
        g, at = None, pos
        if target[0] == 'f':
            g = f._fn.__globals__.get(target[1])
        elif owners is not None and target[1] in owners:
            c = _in_class(owners[target[1]], target[2])
            g, at = (c if type(c) is Function or type(c) is types.FunctionType else None), pos + 1
            g = None if g is None else _method(g)
        if type(g) is not Function or g._tree is None:
            continue
        deeper = {}
        inner = _param_uses(g, at, cls, g._attrs.get(at, ((), set(), set())), None, stack, deeper)
        deeper = _merge_nested(deeper, g._nested.get(at, {}))
        if not path:
            names |= inner[0]
            writes |= inner[1]
            calls |= inner[2]
            if nested_out is not None:
                nested_out.update(_merge_nested(dict(nested_out), deeper))
        elif nested_out is not None:         # given p.pos: what it does with that is p's pos's
            node = nested_out
            for a in path[:-1]:
                node = node.setdefault(a, [set(), set(), set(), {}])[3]
            e = node.setdefault(path[-1], [set(), set(), set(), {}])
            e[0] |= inner[0]
            e[1] |= inner[1]
            e[2] |= inner[2]
            e[3] = _merge_nested(e[3], deeper)
            names.add(path[0])
    return names, writes, calls


def _add_use(entry, attr, how):
    names, stores, calls = entry[:3]
    if how == 'call':
        calls.add(attr)
    else:
        names.add(attr)
        if how == 'store':
            stores.add(attr)


def _attribute_uses(fdef):
    """(name, attribute, 'load', 'store' or 'call') for each name.attribute; name + x calls name.__add__, if it's an instance."""
    called = {id(n.func) for n in ast.walk(fdef) if isinstance(n, ast.Call)}
    for n in ast.walk(fdef):
        if isinstance(n, ast.BinOp) and isinstance(n.left, ast.Name) and type(n.op) in OPERATORS:
            yield n.left.id, OPERATORS[type(n.op)], 'call'
        elif isinstance(n, ast.UnaryOp) and isinstance(n.operand, ast.Name) and type(n.op) in OPERATORS:
            yield n.operand.id, OPERATORS[type(n.op)], 'call'
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and not (n.attr.startswith('__') and not n.attr.endswith('__')):
            yield n.value.id, n.attr, 'call' if id(n) in called else 'load' if isinstance(n.ctx, ast.Load) else 'store'


_methods = weakref.WeakKeyDictionary()   # a plain function on a class -> the Function that compiles it as a callee


def _method(fn):
    """The Function a method on a class compiles as, when compiled code calls it: itself if it's @scram."""
    if type(fn) is Function:
        return fn
    w = _methods.get(fn)
    if w is None:
        w = _methods[fn] = Function(fn, cache=True)
        _functions.discard(w)               # not a decorated function, for preflight
    return w


def _in_class(cls, name):
    """What the class holds under a name, as Python's attribute lookup finds it. _MISSING if nothing."""
    for k in cls.__mro__:
        if name in k.__dict__:
            return k.__dict__[name]
    return _MISSING


_DEFAULT_GET = (object.__dict__['__getattribute__'], tuple.__dict__['__getattribute__'])    # both Python's own lookup
_DEFAULT_SET = object.__dict__['__setattr__']
_TUPLE_FIELD = type(collections.namedtuple('_', 'a').a)     # a namedtuple's field; property if its C version is missing


def _in_slots(cls, name):
    for k in cls.__mro__:
        slots = k.__dict__.get('__slots__', ())
        if name == slots or (not isinstance(slots, str) and name in slots):
            return True
    return False


def _object_fields(x, names, written, calls=(), methods=None, nested=None, ids=None, inner=None, refs=None):
    """[(attribute, type)] if reading these attributes runs no code of the object's, and assigning those written only
    stores; if not, the reason as a string. This holds only for now, since a class can change: it's asked at every call.
    A method called is a plain function on the class, or a @scram one; what it uses joins these, and methods (a dict, if
    given) gets what each call assigns. inner(object, attribute, its uses): the type of an object in an attribute, or a
    reason; by default it's found by these same rules."""
    cls = type(x)
    if calls:
        found = _methods_called(x, calls)
        if type(found) is str:
            return found
        names = sorted(set(names).union(*(n for n, _, _ in found.values())))
        written = set(written).union(*(w for _, w, _ in found.values()))
        for _, _, w in found.values():          # what the methods do with objects in attributes
            nested = _merge_nested(nested, w._nested.get(0, {}))
            more = {}
            _param_uses(w, 0, cls, ((), set(), set()), None, (), more)
            nested = _merge_nested(nested, more)
        if methods is not None:
            methods.update({m: (n, w) for m, (n, w, _) in found.items()})
    if not any(_in_class(cls, '__getattribute__') is d for d in _DEFAULT_GET):
        return f'{cls.__name__} defines __getattribute__'
    if _in_class(cls, '__getattr__') is not _MISSING:
        return f'{cls.__name__} defines __getattr__'
    if written and _in_class(cls, '__setattr__') is not _DEFAULT_SET:
        return f'{cls.__name__} defines __setattr__ (a frozen dataclass does), and {sorted(written)[0]} is assigned'
    held = getattr(x, '__dict__', None)
    if type(held) is not dict:
        held = {}
    fields = []
    for name in names:
        c = _in_class(cls, name)
        if c is _MISSING:
            if name not in held:
                return f'{cls.__name__} has no attribute {name}'     # assigning a new one, or Python's AttributeError
        elif type(c) is _TUPLE_FIELD and _TUPLE_FIELD is not property and isinstance(x, tuple):
            if name in written:
                return f'{name} is a field of a namedtuple, and is assigned'
        elif type(c) is types.MemberDescriptorType and _in_slots(cls, name):
            pass
        elif any(hasattr(type(c), m) for m in ('__get__', '__set__', '__delete__')):
            return f'{cls.__name__}.{name} is a method, property or other descriptor, which runs Python'
        elif name in written and name not in held:
            return f'{name} is on the class and not yet on the instance, and is assigned'
        v = getattr(x, name, _MISSING)
        if v is _MISSING:
            return f'{cls.__name__} has no attribute {name}'
        if refs is not None and name in refs[0]:  # a reference chosen by data: its index in the pool
            fields.append((name, T.ref_of(refs[1])))
            continue
        if nested and name in nested and _arg_type(v) is None:     # an object in an attribute: its own attributes, by the same rules
            if name in written:
                return f'{name}, an object, is assigned'
            sub = nested[name]
            if inner is not None:
                got = inner(v, name, sub)
                if type(got) is str:
                    return f'{name}: {got}'
                fields.append((name, got))
                continue
            inner_methods = {}
            got = _object_fields(v, sorted(sub[0]), sub[1], sub[2], inner_methods, sub[3], ids)
            if type(got) is str:
                return f'{name}: {got}'
            if ids is not None:
                ids.append(id(v))
            w = set(sub[1]).union(*(w for _, w in inner_methods.values()))
            fields.append((name, T.object_of(type(v).__qualname__, got, inner_methods, type(v), w)))
            continue
        t = _arg_type(v)
        if t is None or t is T.U64 or t.kind in ('list', 'function', 'enumclass', 'classref') or T.is_pylist(t):
            return f'{name} is a {type(v).__name__}'
        fields.append((name, t))
    if calls:
        fields += [(m, T.function_of(w)) for m, (_, _, w) in sorted(found.items())]
    return fields


def _methods_called(x, calls, cls=None):
    """{method: (attributes it uses, those a call assigns, its Function)}, through the methods it calls; or a reason.
    x None: for instances of cls the function makes, which hold nothing of their own but what __init__ assigns."""
    cls = type(x) if x is not None else cls
    held = getattr(x, '__dict__', None) if x is not None and type(x).__getattribute__ is object.__getattribute__ else None
    found, todo, called = {}, list(calls), {}
    while todo:
        m = todo.pop()
        if m in found:
            continue
        c = _in_class(cls, m)
        if type(c) is not Function and type(c) is not types.FunctionType:
            return f'{cls.__name__}.{m} is {"not there" if c is _MISSING else "not a plain method"}'
        if type(held) is dict and m in held:
            return f'{m} is on the instance, so calling it calls what is held there'
        w = _method(c)
        if w._tree is None:
            return f'{cls.__name__}.{m}: {w._why_not}'
        if not (w._tree.args.posonlyargs + w._tree.args.args):
            return f'{cls.__name__}.{m} takes no self'
        names, stores, inner = _param_uses(w, 0, cls, w._attrs.get(0, ((), set(), set())))
        found[m] = (set(names), set(stores), w)
        called[m] = set(inner)
        todo += sorted(inner)
    for _ in found:                      # what a call assigns, through the methods it calls
        for m in found:
            for k in called[m]:
                found[m][0].update(found[k][0])
                found[m][1].update(found[k][1])
    return found


def _construction_uses(f):
    """{parameter index: use tree} for parameters given to v = C(...), C a class at module level whose __init__ only stores:
    what f does through the attribute each becomes (v.pos.x is the parameter's x)."""
    tree = f._ctree or f._tree
    index = {p.arg: i for i, p in enumerate(tree.args.posonlyargs + tree.args.args)}
    g = f._fn.__globals__
    out = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and isinstance(g.get(n.value.func.id), type)):
            continue
        given = {}
        recipe = _init_recipe(g[n.value.func.id])
        if type(recipe) is str:
            continue
        positional = [p for p, _, kw in recipe['params'] if not kw]
        for p, a in list(zip(positional, n.value.args)) + [(k.arg, k.value) for k in n.value.keywords]:
            if isinstance(a, ast.Name) and a.id in index:
                given[p] = index[a.id]
        attrs = {a: given[src[1]] for a, src in recipe['assigns'] if src[0] == 'param' and src[1] in given}
        v = n.targets[0].id
        cls = g[n.value.func.id]
        for c in ast.walk(tree):                 # through v's methods, and what v is given to: what they do with those attributes
            w, at = None, None
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) and c.func.value.id == v:
                w, at = _in_class(cls, c.func.attr), 0
                w = _method(w) if type(w) in (Function, types.FunctionType) else None
            elif isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and type(g.get(c.func.id)) is Function:
                at = next((k for k, x in enumerate(c.args) if isinstance(x, ast.Name) and x.id == v), None)
                w = g[c.func.id] if at is not None else None
            if w is None or w._tree is None:
                continue
            more = {}
            _param_uses(w, at, cls, w._attrs.get(at, ((), set(), set())), None, (), more)
            more = _merge_nested(more, w._nested.get(at, {}))
            for a, sub in more.items():
                if a in attrs:
                    _node_merge(out.setdefault(attrs[a], [set(), set(), set(), {}]), [sub[0], sub[1], sub[2], sub[3]])
        called = {id(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
        for m in ast.walk(tree):
            if not isinstance(m, ast.Attribute):
                continue
            base, path = _chain(m)
            if base != v or not path or len(path) < 2 or path[0] not in attrs:
                continue
            node = out.setdefault(attrs[path[0]], [set(), set(), set(), {}])
            for a in path[1:-1]:
                node = node[3].setdefault(a, [set(), set(), set(), {}])
            _add_use(node, path[-1], 'call' if id(m) in called else 'load' if isinstance(m.ctx, ast.Load) else 'store')
    return out


_POOLDICT = {'sjpool': object()}     # the pool argument's place among the globals: a value no type has, typed by _pool_plan


def _ref_candidates(fdef):
    """(attributes, names) that may hold references chosen by data, from the tree alone: compared with None, given None
    or an object, or read into a name that's rebound (n = n.next). The values at the call decide which really are."""
    if fdef is None:
        return set(), set()
    params = {p.arg for p in fdef.args.posonlyargs + fdef.args.args}
    stores, looped = {}, set()
    for n in ast.walk(fdef):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            stores[n.id] = stores.get(n.id, 0) + 1
        elif isinstance(n, (ast.For, ast.While, ast.If)):
            looped.update(m.id for b in n.body + n.orelse for m in ast.walk(b) if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store))

    def none(e):
        return isinstance(e, ast.Constant) and e.value is None

    def parts(e):                                # what a value may be: both sides of a if c else b
        return parts(e.body) + parts(e.orelse) if isinstance(e, ast.IfExp) else [e]
    attrs, names = set(), set()
    for n in ast.walk(fdef):
        if isinstance(n, ast.Compare):
            sides = [n.left] + n.comparators
            for k, op in enumerate(n.ops):
                if isinstance(op, (ast.Is, ast.IsNot)):
                    for x, y in ((sides[k], sides[k + 1]), (sides[k + 1], sides[k])):
                        if none(y) or isinstance(op, ast.Is) or isinstance(op, ast.IsNot):
                            if isinstance(x, ast.Attribute):
                                attrs.add(x.attr)
                            elif isinstance(x, ast.Name):
                                names.add(x.id)
        elif isinstance(n, ast.Assign) and len(n.targets) == 1:
            t, v = n.targets[0], n.value
            vs = parts(v)
            if isinstance(t, ast.Attribute) and all(isinstance(x, (ast.Name, ast.Attribute)) or none(x) for x in vs):
                attrs.add(t.attr)
            elif isinstance(t, ast.Name) and all(none(x) for x in vs):
                names.add(t.id)
            elif isinstance(t, ast.Name) and all(isinstance(x, ast.Attribute) or none(x) for x in vs) and \
                    (stores.get(t.id, 0) > 1 or t.id in looped or t.id in params):
                attrs.update(x.attr for x in vs if isinstance(x, ast.Attribute))
                names.add(t.id)
    for _ in range(8):
        grew = False
        for n in ast.walk(fdef):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                t, v = n.targets[0].id, n.value
                if isinstance(v, ast.Name) and (v.id in names) != (t in names):    # cur = head: both name one
                    names |= {t, v.id}
                    grew = True
                for x in parts(v):
                    if t in names and isinstance(x, ast.Attribute) and x.attr not in attrs:
                        attrs.add(x.attr)
                        grew = True
        if not grew:
            break
    return attrs, names


def _plain_get(x, a):
    """x.a if reading it runs none of x's code: a default lookup, and a an instance attribute or slot. Else _MISSING."""
    cls = type(x)
    if not any(_in_class(cls, '__getattribute__') is d for d in _DEFAULT_GET) or _in_class(cls, '__getattr__') is not _MISSING:
        return _MISSING
    c = _in_class(cls, a)
    if c is not _MISSING and not (type(c) is types.MemberDescriptorType and _in_slots(cls, a)) \
            and any(hasattr(type(c), m) for m in ('__get__', '__set__', '__delete__')):
        return _MISSING                          # a method, property or other descriptor: its code would run
    return getattr(x, a, _MISSING)


def _is_instance(x):
    return x is not None and _arg_type(x) is None and type(x).__module__ != 'builtins' and not isinstance(x, (type, enum.Enum))


def _pool_plan(f, args):
    """For a function whose references are chosen by data: ({parameter index: ref type}, the pool's type, the reference
    attributes, its class), the pool being every object reached from the arguments through those attributes, each once;
    or (None, a reason) if they can't be held so, or ({}, NOPOOL) if nothing is a reference."""
    tree = f._ctree or f._tree
    attrs, names = _ref_candidates(tree)
    params = [p.arg for p in tree.args.posonlyargs + tree.args.args]
    pooled = [i for i, n in enumerate(params) if n in names and (args[i] is None or _is_instance(args[i]))]
    roots, todo, fixed, refs = [], [], set(), set()      # fixed: references in a fixed place (a.pos); refs: in pooled objects
    for i, x in enumerate(args):
        if i in pooled:
            roots.append((i, ''))
            if x is not None:
                todo.append(x)
        elif _is_instance(x):
            for a in sorted(attrs):
                v = _plain_get(x, a)
                if v is None or _is_instance(v):
                    fixed.add(a)
                    roots.append((i, a))
                    if v is not None:
                        todo.append(v)
    seen, objs = set(), []
    while todo:
        o = todo.pop(0)
        if id(o) in seen:
            continue
        seen.add(id(o))
        objs.append(o)
        for a in sorted(attrs):
            v = _plain_get(o, a)
            if v is None or _is_instance(v):
                refs.add(a)
                if v is not None:
                    todo.append(v)
    if not roots:
        return {}, T.NOPOOL, set(), None
    classes = set(map(type, objs))
    if len(classes) > 1:
        return None, f'references reach objects of {len(classes)} classes ({", ".join(sorted(c.__name__ for c in classes))}): one pool, of one class, for now'
    cls = classes.pop() if classes else None
    for o in objs:
        for a in refs:
            v = _plain_get(o, a)
            if not (v is None or type(v) is cls):
                return None, f'{type(o).__name__}.{a} holds {type(v).__name__}, not a reference to one class or None'
    refnames = names | {params[i] for i in pooled}
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr not in refs:
            v = n.value
            if isinstance(v, ast.Name) and v.id in refnames or isinstance(v, ast.Attribute) and v.attr in refs | fixed:
                used.add(n.attr)
    numbers = []
    if not objs:                                 # only None reached: nothing says otherwise, and nothing is read
        refs = {a for a in attrs if a in used} - fixed
        return {i: T.ref_of(None) for i in pooled}, T.pool_of(None, [(a, T.FLOAT) for a in sorted(used - refs - fixed)], sorted(refs), roots), \
            refs | fixed, None
    for a in sorted(used):
        vals = [_plain_get(o, a) for o in objs]
        kinds = set(map(type, vals))
        if not vals:
            continue
        if len(kinds) != 1 or kinds.pop() not in (int, float, bool) or (type(vals[0]) is int and not all(-(1 << 63) <= v < 1 << 63 for v in vals)):
            if all(v is _MISSING for v in vals) and _in_class(cls, a) is not _MISSING:
                continue                         # a method: not a number of the pool's
            return None, f'{cls.__name__}.{a} isn\'t a number of one type in every object references reach'
        numbers.append((a, {int: T.INT, float: T.FLOAT, bool: T.BOOL}[type(vals[0])]))
    if cls is not None and (_in_class(cls, '__bool__') is not _MISSING or _in_class(cls, '__len__') is not _MISSING):
        return None, f'{cls.__name__} defines __bool__ or __len__, so a reference\'s truth is its code\'s'
    return {i: T.ref_of(cls) for i in pooled}, T.pool_of(cls, numbers, sorted(refs), roots), refs | fixed, cls


def _node_merge(into, node):
    """Add a use tree [attributes used, assigned, methods called, {attribute: tree}] into another. True if it grew."""
    grew = False
    for k in range(3):
        if not set(node[k]) <= into[k]:
            into[k] |= set(node[k])
            grew = True
    for a, sub in node[3].items():
        if a not in into[3]:
            into[3][a] = [set(), set(), set(), {}]
        grew = _node_merge(into[3][a], sub) or grew
    return grew


def _object_graph(roots, refs=None):
    """The types of instance arguments, roots [(index, object, use tree)], with their alias pattern: every path the function
    names is walked from the arguments, and an object met a second time is same_as where it was first met, its type made
    for what all its names use. ({index: type, or None if it can't compile}, the ids of the objects and their __dict__s.)
    Objects in attributes are walked only through attributes the function names: the cost is the code's, not the data's."""
    merged = {}
    for _ in range(16):                  # each round, an object's tree only grows: it settles
        first, grew, held, why = {}, [False], set(), {}

        def walk(x, node, at):
            k = id(x)
            m = merged.get(k)
            if m is None:
                merged[k] = m = [set(), set(), set(), {}]
            if _node_merge(m, node) and k in first:
                grew[0] = True          # what this name uses wasn't in the type made at the first
            if k in first:
                i0, p0 = first[k]
                return T.same_as(None, p0) if i0 == at[0] else T.same_as(i0, p0)
            first[k] = at
            if not any(m[:3]) and not m[3] and not at[1]:     # nothing of it reaches the C
                return T.object_of(type(x).__qualname__, [], None, type(x))
            methods = {}
            fields = _object_fields(x, sorted(m[0]), set(m[1]), set(m[2]), methods, m[3],
                                    inner=lambda v, a, sub: walk(v, sub, (at[0], at[1] + (a,))), refs=refs)
            if type(fields) is not list:
                return fields
            d = getattr(x, '__dict__', None)
            if d is not None:
                held.add(id(d))
            written = set(m[1]).union(*(w for _, w in methods.values()))
            return T.object_of(type(x).__qualname__, fields, methods, type(x), written)
        out = {}
        for i, x, node in roots:
            t = walk(x, node, (i, ()))
            out[i] = t if isinstance(t, T.Type) else None
            if type(t) is str:
                why[i] = t
        if not grew[0]:
            break
    if set(first) & held:                # an instance and its own __dict__ are two names for one thing as well
        return {i: None for i, _, _ in roots}, set(first) | held
    return out, set(first) | held


def _field_type(t, path):
    """The type at a path in an instance's type, or a type of kind 'none' if it isn't one."""
    for a in path.split('.') if isinstance(path, str) else [path]:
        if t is None or t.kind != 'object':
            return T.NONE
        t = dict(t.elem).get(a)
    return t if t is not None else T.NONE


def _member(x, key):
    """What a dict holds under a key, or an instance under an attribute, or a path of them (p.pos.x is 'pos.x')."""
    if type(x) is dict:
        return x[key]
    for a in key.split('.'):
        x = getattr(x, a)
    return x


def _set_member(x, key, value):
    if type(x) is dict:
        x[key] = value
        return
    *path, last = key.split('.')
    for a in path:
        x = getattr(x, a)
    setattr(x, last, value)


def _push(cargs, x, t, where=None):
    """The C arguments for one Python argument. where: id -> index in the pool, for references."""
    if t.kind in ('function', 'enumclass', 'classref', 'madeclass', 'same', 'nopool'):
        return
    if t.kind == 'ref':
        cargs.append(-1 if x is None else where[id(x)])
        return
    if t.kind == 'pool':
        raise AssertionError('a pool is passed by _run, which writes it back')
    if t.kind == 'enum':
        cargs.append(x._value_)
        return
    if t.kind in ('list', 'objlist'):
        raise AssertionError('lists are passed by _run, which copies them back')
    if t.kind == 'array' and t.strided:
        cargs.append(x.ctypes.data)
        cargs.extend(x.shape)
        cargs.extend(s // x.itemsize for s in x.strides)
    elif t.kind == 'array':
        cargs.append(x)
        cargs.extend(x.shape)
    elif t.kind == 'tuple':
        for v, et in zip(x, t.elem):
            _push(cargs, v, et, where)
    elif t.kind == 'dict':
        for key, ft in t.elem:
            _push(cargs, x[key], ft)
    elif t.kind == 'object':              # in the order its parameters come: breadth first through inner objects
        for path, ft in leaf_types(t):
            _push(cargs, _member(x, path), ft, where)
    elif t.np:
        cargs.append(x.item())
    elif t is T.U64:
        cargs.append(x - (1 << 64))
    else:
        cargs.append(x)


class _Interface:
    """Memory numpy can wrap as an array without copying: it keeps the array it belongs to alive."""

    __slots__ = ('base', '__array_interface__')

    def __init__(self, base, address, shape, strides, typestr=None):
        self.base = base
        self.__array_interface__ = {'data': (address, base is not None and not base.flags.writeable), 'shape': tuple(shape),
                                    'strides': None if strides is None else tuple(strides),
                                    'typestr': typestr or base.dtype.str, 'version': 3}


_MISSING = object()               # a global that isn't there: nothing compiled can read it, so that call runs as Python


class _Python:
    """Marker: this signature runs as the original Python function."""


PYTHON = _Python()


_LIST_CODES = {'int': 'q', 'float': 'd', 'bool': 'i'}      # a list's element type -> its array, as the kernel's C type


def _nested_shape(x):
    """The shape of a rectangular list of lists, or None: ragged, empty, or one list in two places (a write would show
    through both, which a copy can't do)."""
    shape, level, seen = [], [x], set()
    while level and type(level[0]) is list:
        n = len(level[0])
        if n == 0 or any(type(r) is not list or len(r) != n for r in level):
            return None
        for r in level:
            if id(r) in seen:
                return None
            seen.add(id(r))
        shape.append(n)
        level = list(itertools.chain.from_iterable(level))
    return shape                               # lists among the numbers fail the element check that follows


def _flatten(x, depth):
    for _ in range(depth - 1):
        x = list(itertools.chain.from_iterable(x))
    return x


def _lists_from(address, shape, et):
    """Nested Python lists of the numbers at address, of this shape, read in C."""
    import ctypes
    code = _LIST_CODES[et.name]
    n = 1
    for s in shape:
        n *= s
    flat = memoryview((ctypes.c_char * max(1, n * struct.calcsize(code))).from_address(address)).cast('B')[:n * struct.calcsize(code)].cast(code).tolist() if n else []
    if et is T.BOOL:
        flat = [bool(v) for v in flat]

    def nest(values, dims):
        if len(dims) == 1:
            return values
        size = 1
        for s in dims[1:]:
            size *= s
        return [nest(values[k * size:(k + 1) * size], dims[1:]) for k in range(dims[0])]
    return nest(flat, list(shape))


def _copy_back_nested(x, a, et, depth):
    """Into a list of lists passed in, the elements the kernel changed."""
    rows = [x]
    for _ in range(depth - 1):
        rows = [r for row in rows for r in row]
    n = len(rows[0])
    code = a.typecode
    for k, row in enumerate(rows):            # a row that changed is replaced in one go, in C
        new = a[k * n:(k + 1) * n]
        if array.array(code, row).tobytes() != new.tobytes():
            row[:] = [bool(v) for v in new] if et is T.BOOL else new.tolist()


def _copy_back(lst, a, et):
    """Into a list passed in, the elements the kernel changed: the others stay the very objects they were."""
    for k, v in enumerate(a):
        old = lst[k]
        if et is T.FLOAT:
            if old == v and math.copysign(1.0, old) == math.copysign(1.0, v) or (old != old and v != v):
                continue
        elif old == v:
            continue
        lst[k] = bool(v) if et is T.BOOL else v


class Compiled:
    __slots__ = ('kernel', 'ret', 'types', 'source', 'libm', 'locals', 'checks', 'hoisted', 'gotchas', 'stop', 'rerun', 'outs', 'lists',
                 'counts', 'count_lines', 'count_heads', 'count_rest', 'array_args', 'records', 'io_size', 'free', 'frames', 'columns', 'instance_ret', 'tb_at')

    def __init__(self, kernel, ret, types_, source, libm, locals_=None, checks=None, hoisted=(), gotchas=None, stop=False, rerun=False):
        self.rerun = rerun
        self.outs = []                        # written dict keys: (argument index, key, byte in io, type)
        self.lists = set()                    # the arguments that are lists the kernel writes
        self.columns = set()                  # (argument, attribute) of lists of instances the kernel writes
        self.instance_ret = None              # returns an instance: see Compiler.instance_ret
        self.counts = None                    # preflight --counts: the int64 counters the kernel adds to
        self.count_lines, self.count_heads, self.count_rest = (), {}, {}
        self.checks, self.hoisted, self.gotchas, self.stop = checks or {}, hoisted, gotchas or {}, stop
        self.locals = locals_ or {}           # variable -> inferred type name
        self.kernel = kernel
        self.ret = ret
        self.types = types_
        self.source = source
        self.libm = libm

    def read_counts(self):
        """(line -> times it ran, line -> ('if' or 'loop', times its body ran)). A kernel adds to its counters
        with the GIL released, so this works while one runs, from any thread."""
        values = list(struct.unpack(f'<{len(self.counts) // 8}q', self.counts))
        for slot in sorted(self.count_rest):      # an if comes before its else, so a chain of elifs resolves in order
            whole, taken = self.count_rest[slot]
            values[slot] = max(values[whole] - values[taken], 0)
        runs = {}
        for lines, v in zip(self.count_lines, values):
            for n in lines:
                runs[n] = max(runs.get(n, 0), v)    # a line can hold several statements
        return runs, {n: (kind, values[slot]) for n, (kind, slot) in self.count_heads.items()}


_functions = weakref.WeakSet()    # every decorated function, for preflight
_profile = False                  # preflight --run: time each call
_counts = None                    # preflight --counts: True, or the names of the functions to count
_running = None                   # preflight --live: thread id -> (the function, native or not, when it started)
_log = bool(os.environ.get('SCRAMJIT_LOG'))


def enable_profile():
    global _profile
    _profile = True


def enable_counts(names=None):
    """Compile a copy of each function, or of those named, that counts how often each line runs."""
    global _counts
    enable_profile()
    _counts = set(names) if names else True


def enable_live():
    global _running
    enable_profile()
    _running = {}


def running():
    return list(_running.values()) if _running else []


def functions():
    return list(_functions)


def _class_guards(env, i, cls, names, written, again):
    """That cls still looks up these names as it did, for a plan: a class can change, and only those written in Python can."""
    env[f'kcls{i}'], env[f'kmro{i}'], env['MISSING'] = cls, cls.__mro__, _MISSING
    out = [f'    if kcls{i}.__mro__ is not kmro{i}:\n        return slow({again})']
    asked = list(names) + ['__getattribute__', '__getattr__'] + (['__setattr__'] if written else [])
    for c, k in enumerate(k for k in cls.__mro__ if k.__flags__ & (1 << 9)):
        env[f'kheld{i}_{c}'] = k.__dict__
        for a, name in enumerate(asked):
            env[f'kwas{i}_{c}_{a}'] = k.__dict__.get(name, _MISSING)
            out.append(f'    if kheld{i}_{c}.get({name!r}, MISSING) is not kwas{i}_{c}_{a}:\n        return slow({again})')
    return out


def _object_guards(env, i, x, t, n, again):
    """What a plan checks of an instance: that its class looks up these names as it did (a class can change, and only those
    written in Python can), then what _object_fields checks of the object. The first fetched value of each attribute is
    f{i}_{k}. Nothing here runs the object's code, once the class has been checked."""
    written = t.writes
    cls = type(x)
    env[f'cls{i}'], env[f'mro{i}'], env['MISSING'] = cls, cls.__mro__, _MISSING
    out = [f'    if cls{i}.__mro__ is not mro{i}:\n        return slow({again})']
    asked = [k for k, _ in t.elem] + ['__getattribute__', '__getattr__'] + (['__setattr__'] if written else [])
    for c, k in enumerate(k for k in cls.__mro__ if k.__flags__ & (1 << 9)):     # 1 << 9: a class made at run time
        env[f'held{i}_{c}'] = k.__dict__
        for a, name in enumerate(asked):
            env[f'was{i}_{c}_{a}'] = k.__dict__.get(name, _MISSING)
            out.append(f'    if held{i}_{c}.get({name!r}, MISSING) is not was{i}_{c}_{a}:\n        return slow({again})')
    for k, (key, ft) in enumerate(t.elem):
        if ft.kind == 'function':           # a method: the class's, checked above, unless the instance holds its own
            if hasattr(x, '__dict__'):
                out.append(f'    if {key!r} in {n}.__dict__:\n        return slow({again})')
            continue
        env[f'type{i}_{k}'] = type(getattr(x, key))
        out.append(f'    f{i}_{k} = getattr({n}, {key!r}, MISSING)\n    if type(f{i}_{k}) is not type{i}_{k}:\n        return slow({again})')
        if ft is T.INT:
            out.append(f'    if not -9223372036854775808 <= f{i}_{k} <= 9223372036854775807:\n        return slow({again})')
        if key in written and type(_in_class(cls, key)) is not types.MemberDescriptorType:      # not on the class only: assigning would add it, whatever ran
            out.append(f'    if {key!r} not in {n}.__dict__:\n        return slow({again})')
    return out


def _make_plan(f, entry, args):
    """A small function for calls with these argument types, or f._slow_call if they need the general
    path. It's generated because a call is where the time goes: what's known from the types is decided
    here, once. Anything it can't vouch for at call time (an int past 64 bits, an array of another
    dtype, dimension or layout, an empty one) goes to f._slow, so plans change only speed."""
    if entry.counts is not None or entry.ret.kind == 'list' or entry.instance_ret or any(t.kind in ('pool', 'ref') for t in entry.types):
        return f._slow_call
    env = {'kernel': entry.kernel, 'slow': f._slow, 'failed': f._failed, 'box': f._box, 'ret': entry.ret, 'pool': [], 'compiled': entry,
           'addressof': ctypes.addressof, 'from_buffer': ctypes.c_char.from_buffer, 'bytearray': bytearray, 'memoryview': memoryview}
    names = [f'a{i}' for i in range(len(args))]
    real = names[:len(args) - len(f._hidden or ())]
    again = '(' + ''.join(n + ', ' for n in real) + ')'
    hidden = []                        # globals the function reads: read now, and turned down if their type has changed
    for k, (g, gname) in enumerate(f._hidden_from or ()):
        i = len(real) + k
        env[f'gname{k}'], env[f'gtype{k}'], env[f'G{k}'] = gname, type(args[i]), g
        hidden.append(f'    a{i} = G{k}.get(gname{k}, MISSING)\n    if type(a{i}) is not gtype{k}:\n        return slow({again})')
        if entry.types[i].kind in ('function', 'enumclass', 'classref', 'madeclass'):     # a new function or class is a new signature
            env[f'gval{k}'] = args[i]
            hidden.append(f'    if a{i} is not gval{k}:\n        return slow({again})')
    if hidden:
        env.update(MISSING=_MISSING)
    cargs = []
    for i, (x, t, n) in enumerate(zip(args, entry.types, names)):
        if t.kind == 'enum':
            cargs.append(f'{n}._value_')
            continue
        if t.kind in ('function', 'enumclass', 'classref', 'nopool'):
            continue
        if t.kind == 'same':                # one object under two names: the plan checks it still is
            if t.elem[1] or t.elem[0] is None:
                return f._slow_call
            hidden.append(f'    if {n} is not a{t.elem[0]}:\n        return slow({again})')
            continue
        if t.kind == 'madeclass':           # what its instances need of it: as for an instance's class
            hidden.extend(_class_guards(env, i, t.cls, [a for a, _ in t.recipe['assigns']] + [m for m, _ in t.recipe['methods']]
                                            + ['__init__', '__new__', '__post_init__'], t.writes, again))
            continue
        if t.kind == 'array' and (t.strided or T.is_pylist(t)):
            return f._slow_call
        if t.kind == 'array':
            env[f'd{i}'] = x.dtype
            cargs.append(f'p{i}' if x.flags.writeable else n)
            cargs.extend(f's{i}[{k}]' for k in range(x.ndim))
        elif t.kind in ('tuple', 'list', 'objlist'):
            return f._slow_call
        elif t.kind == 'object':
            if not all(ft.is_number or ft.kind == 'function' for _, ft in t.elem) or any(y.kind == 'dict' for y in entry.types):   # x, vars(x)
                return f._slow_call
            for k, (key, ft) in enumerate(t.elem):
                if ft.kind != 'function':
                    cargs.append(f'f{i}_{k}.item()' if ft.np else f'f{i}_{k}')
        elif t.kind == 'dict':
            if not all(ft.is_number for _, ft in t.elem):
                return f._slow_call
            for k, (key, ft) in enumerate(t.elem):
                cargs.append(f'f{i}_{k}.item()' if ft.np else f'f{i}_{k}')
        elif t.np:
            cargs.append(f'{n}.item()')
        elif t is T.U64:
            cargs.append(f'({n} - 18446744073709551616)')
        else:
            cargs.append(n)
    if entry.libm is not None:
        pointer = ctypes.c_char_p(entry.libm)
        env['libm_keep'] = (entry.libm, pointer)        # alive as long as the plan
        env['libm'] = ctypes.cast(pointer, ctypes.c_void_p).value
        cargs.append('libm')
    if entry.stop:
        env.update(get_ident=threading.get_ident, FLAG=_interrupt.FLAG, NEVER=_interrupt.NEVER, interrupt=_interrupt)
        cargs.append('(FLAG if get_ident() == interrupt._main else NEVER)')
    # an array's checks go one at a time, each read after the last passed, then its address and shape
    arrays = [i for i, t in enumerate(entry.types) if t.kind == 'array']
    src_guards = hidden
    for i, (t, n) in enumerate(zip(entry.types, names)):
        if t.kind == 'array':
            x = args[i]
            src_guards.append(f'    fl = {n}.flags\n    if {n}.dtype is not d{i} or {n}.ndim != {x.ndim} or not fl.c_contiguous or fl.writeable is not {bool(x.flags.writeable)}:\n        return slow({again})\n    s{i} = {n}.shape')
        elif t.kind == 'dict':
            # only what this function reads is checked: a missing key or another type goes the general way
            for k, (key, ft) in enumerate(t.elem):
                env[f'key{i}_{k}'], env[f'type{i}_{k}'] = key, type(args[i][key])
                src_guards.append(f'    f{i}_{k} = {n}.get(key{i}_{k})\n    if type(f{i}_{k}) is not type{i}_{k}:\n        '
                                  f'return slow({again})')
                if ft is T.INT:
                    src_guards.append(f'    if not -9223372036854775808 <= f{i}_{k} <= 9223372036854775807:\n        '
                                      f'return slow({again})')
        elif t.kind == 'object':
            src_guards.extend(_object_guards(env, i, args[i], t, n, again))
        elif t is T.U64:
            src_guards.append(f'    if not 9223372036854775808 <= {n} <= 18446744073709551615:\n        return slow({again})')
        elif t.kind == 'int' and not t.np:
            src_guards.append(f'    if not -9223372036854775808 <= {n} <= 9223372036854775807:\n        return slow({again})')
    addresses = ''.join(f'        p{i} = addressof(from_buffer({names[i]}))\n' for i in arrays if args[i].flags.writeable)
    take = (f'    try:\n{addresses}    except (ValueError, TypeError, BufferError):      # empty: no buffer to take the address of\n'
            f'        return slow({again})\n') if addresses else ''
    size = entry.io_size
    r = entry.ret
    if entry.records:
        env.update(arrays=f._array_result)
        result = f"arrays(compiled, io, ({''.join(n + ', ' for n in names)}))"
    elif r is T.NONE:
        result = 'None'
    elif r is T.INT:
        result = 'q[0]'
    elif r is T.FLOAT:
        result = 'd[0]'
    elif r.kind == 'tuple':
        env['box_tuple'] = f._box_tuple
        result = 'box_tuple(io, ret)'
    else:
        result = 'box(io, 0, 16, ret)'
    rerun = ''
    if entry.rerun:
        env.update(f=f, RERUN=RERUN)
        rerun = f"    if status == RERUN:\n        pool.append(entry)\n        return f._py({', '.join(real)})\n"
    saves = ''
    for n, (i, key, at, ot) in enumerate(entry.outs):
        env[f'okey{n}'], env[f'otype{n}'] = key, ot
        if entry.types[i].kind == 'object':
            saves += f'    {names[i]}.{key} = box(io, {at}, {at + 8}, otype{n})\n'
        else:
            saves += f'    {names[i]}[okey{n}] = box(io, {at}, {at + 8}, otype{n})\n'
    dicts = [names[i] for i, t in enumerate(entry.types) if t.kind == 'dict']
    if len(dicts) > 1:                 # two names for one dict: a write must show through both
        pairs = ' or '.join(f'{a} is {b}' for k, a in enumerate(dicts) for b in dicts[k + 1:])
        src_guards.append(f'    if {pairs}:\n        return slow({again})')
    objects = [names[i] for i, t in enumerate(entry.types) if t.kind == 'object']
    if len(objects) > 1:
        pairs = ' or '.join(f'{a} is {b}' for k, a in enumerate(objects) for b in objects[k + 1:])
        src_guards.append(f'    if {pairs}:\n        return slow({again})')
    src = f"""def plan({', '.join(real)}):
{chr(10).join(src_guards)}
{take}    if pool:
        io, ioaddr, q, d = entry = pool.pop()
    else:
        io = bytearray({size})
        ioaddr = addressof(from_buffer(io))
        view = memoryview(io)
        q, d = view.cast('q'), view.cast('d')
        entry = (io, ioaddr, q, d)
    status = kernel({''.join(c + ', ' for c in cargs)}ioaddr)
{rerun}{saves}    if status:
        pool.append(entry)
        failed(compiled, status, io)
    value = {result}
    pool.append(entry)
    return value
"""
    exec(src, env)  # noqa: S102
    return env['plan']


class Function:
    def __init__(self, fn, boundscheck=True, cache=True, fallback=True, parity=True, _loop=None, counts=False):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._py = fn             # what runs when a signature doesn't compile: fn, or fn with its loops lifted
        self._lifted = None       # the Functions of its lifted loops, once that has been tried
        self._loop = _loop        # a lifted loop: (its function's tree, the first line, the file)
        self._owner = fn.__name__ # the decorated function this is, or that this loop was lifted from
        self._counting = counts   # @scram(counts=True)
        self._boundscheck = boundscheck
        self._fallback = fallback
        self._parity = parity
        self._cache = {}
        self._use_disk = cache
        self._plans = {}          # the arguments' Python types -> a function that makes the call (see _make_plan)
        self._compile_s = {}      # signature -> seconds spent compiling it
        self._why = {}            # signature -> (line, reason) it runs as Python
        self._stats = {}          # signature -> [calls, native seconds, Python seconds], under preflight --run
        self._lock = threading.Lock()
        self._hidden = None       # the globals this function reads, found at the first call: they may not exist before it
        self._hidden_from = None  # for each of those: (the dict it's in, its name there), which a called function's may differ in
        self._ctree = None        # the tree with those as parameters, which is what's compiled
        code = fn.__code__
        if _loop is None:
            _functions.add(self)
            self._first = code.co_firstlineno
        else:
            self._first = _loop[1]
        self._filename = code.co_filename
        self._contracts = [(i, p, a) for i, (p, a) in enumerate(
            (p, fn.__annotations__.get(p)) for p in code.co_varnames[:code.co_argcount]) if a in _ANNOTATIONS]
        self._tree, self._why_not = self._parse(fn) if _loop is None else (_loop[0], None)
        self._source_tree = self._tree      # as written: for lifting loops, which makes Python of it
        self._kwonly_args = list(self._tree.args.kwonlyargs) if self._tree is not None else []
        self._kwonly = [a.arg for a in self._kwonly_args]
        self._npos = len(self._tree.args.posonlyargs + self._tree.args.args) if self._tree is not None else 0
        if self._kwonly:                    # compiled as parameters after the others, bound by keyword (_bind)
            self._tree = copy.deepcopy(self._tree)
            self._tree.args.args = self._tree.args.args + self._tree.args.kwonlyargs
            self._tree.args.kwonlyargs, self._tree.args.kw_defaults = [], []
            self._py = self._by_keyword(fn)
        self._dict_keys = _dict_keys(self._tree)
        self._attrs = _attrs_used(self._tree)
        self._elems = _element_attrs(self._tree)
        self._passes = _passes(self._tree)
        self._unused = _unused(self._tree)
        self._returned = _returned(self._tree)
        self._nested = _nested_uses(self._tree)
        self._replans = 8         # how many times more a dict's new value types get a call plan of their own
        self._np = None

    @staticmethod
    def _parse(fn):
        if hasattr(fn, '__wrapped__'):
            return None, 'the function is already wrapped by another decorator'
        if fn.__code__.co_freevars:
            return None, 'closures (the function reads variables of an enclosing function)'
        try:
            src = textwrap.dedent(inspect.getsource(fn))
            tree = ast.parse(src)
        except (OSError, TypeError, SyntaxError, IndentationError):
            return None, 'the source code is not available'
        fdef = tree.body[0] if tree.body else None
        if not isinstance(fdef, ast.FunctionDef):
            return None, 'not a plain def function'
        _mangle(fdef, fn.__qualname__)
        return fdef, None

    def _uses_of(self, idx, cls):
        """What this function makes of an instance of cls given as parameter idx: for a caller passing one."""
        return _param_uses(self, idx, cls, self._attrs.get(idx, ((), set(), set())))

    # descriptor: work as a method too (self is then an argument)
    def __get__(self, obj, cls=None):
        return self if obj is None else types.MethodType(self, obj)

    @property
    def signatures(self):
        """The argument types of each compiled specialization."""
        return [k for k, v in self._cache.items() if isinstance(v, Compiled)]

    def c(self, signature=None):
        """The C scramjit generated for a signature (the most recent one by default)."""
        compiled = [v for v in self._cache.values() if isinstance(v, Compiled)]
        if signature is None:
            if not compiled:
                raise ValueError(f'{self.__name__} has no compiled signatures yet')
            return compiled[-1].source
        v = self._cache.get(tuple(signature))
        if not isinstance(v, Compiled):
            raise KeyError(signature)
        return v.source

    def _by_keyword(self, fn):
        """fn, called with its keyword-only parameters' values after the others, as this function's calls carry them."""
        npos, names = self._npos, self._kwonly

        def py(*full):
            return fn(*full[:npos], **dict(zip(names, full[npos:])))
        return py

    def _bind(self, args, kwargs):
        """args and kwargs as one positional tuple, with the defaults filled in, as Python binds them; None for what
        isn't a plain call of plain parameters (Python then binds it, or raises its TypeError)."""
        code = self._fn.__code__
        if code.co_flags & 0x0C:      # *args or **kwargs
            return None
        n = code.co_argcount
        if code.co_kwonlyargcount:
            kw = {k: v for k, v in kwargs.items() if k in self._kwonly}
            rest = {k: v for k, v in kwargs.items() if k not in self._kwonly}
            head = self._bind_positional(args, rest, code, n)
            if head is None:
                return None
            defaults = self._fn.__kwdefaults__ or {}
            if any(k not in kw and k not in defaults for k in self._kwonly):
                return None                       # Python's TypeError
            return head + tuple(kw[k] if k in kw else defaults[k] for k in self._kwonly)
        return self._bind_positional(args, kwargs, code, n)

    def _bind_positional(self, args, kwargs, code, n):
        if len(args) > n:
            return None
        names = code.co_varnames[:n]
        bound = list(args) + [_MISSING] * (n - len(args))
        for k, v in kwargs.items():
            if k not in names or names.index(k) < code.co_posonlyargcount or bound[names.index(k)] is not _MISSING:
                return None
            bound[names.index(k)] = v
        defaults = self._fn.__defaults__ or ()
        for i in range(n):
            if bound[i] is _MISSING:
                if i < n - len(defaults):
                    return None
                bound[i] = defaults[i - (n - len(defaults))]
        return tuple(bound)

    def __call__(self, *args, **kwargs):
        if kwargs or len(args) != self._fn.__code__.co_argcount or self._kwonly:     # a default or a keyword: bound as Python binds it
            full = self._bind(args, kwargs)
            if full is None:
                return self._fn(*args, **kwargs)
            args, kwargs = full, {}
        plan = self._plans.get(tuple(map(type, args)))
        if plan is not None and not kwargs:             # these types were checked when the plan was made
            return plan(*args)
        for i, p, ann in self._contracts:
            if i < len(args) and not _contract_ok(args[i], ann):
                raise TypeError(f'{self.__name__}() argument {p!r} must be {_ANNOTATIONS[ann]}, '
                                f'not {type(args[i]).__name__}')
        real = args
        args, key = self._args_and_key(args)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        if _profile:
            return self._timed(key, entry, args)
        if entry is not PYTHON:
            self._plans[tuple(map(type, real))] = _make_plan(self, entry, args)
        elif None not in key and not any(t.kind in ('array', 'object', 'objlist') for t in key) and dict not in map(type, args) and not self._hidden:
            self._plans[tuple(map(type, real))] = self._py   # these types always run as Python
        if entry is PYTHON:
            return self._py(*real)
        return self._run(entry, args)

    def _args_and_key(self, real):
        """The arguments with the globals after them, and the call's signature. A method found in it may read globals
        this function doesn't know of yet: they become more of its globals, and the arguments are taken again."""
        args = self._with_globals(real)
        key = self._key(args)
        if self._tree is not None and self._add_method_globals(key):
            args = self._with_globals(real)
            key = self._key(args)
        return args, key

    def _add_method_globals(self, key):
        """Take the globals of the methods in these types, the instances' and the lists'. True if any were new."""
        if self._hidden is None:
            self._find_globals()
        new = []
        for t in key:
            if t is None or t.kind not in ('object', 'objlist', 'madeclass', 'elemref'):
                continue
            for _, ft in (t.elem.elem if t.kind in ('objlist', 'elemref') else t.recipe['methods'] if t.kind == 'madeclass' else t.elem):
                w = ft.elem if ft.kind == 'function' else None
                if type(w) is not Function or w._tree is None:
                    continue
                if w._hidden is None:
                    w._find_globals()
                for og, on in w._hidden_from:
                    if og is _POOLDICT:
                        continue
                    if not any(og is a and on == b for a, b in self._hidden_from + new):
                        new.append((og, on))
        if not new:
            return False
        tree = self._ctree or self._tree
        for og, on in new:
            name = f'sjg{len(self._hidden)}_{on}'
            self._hidden.append(name)
            self._hidden_from.append((og, on))
        self._ctree = copy.deepcopy(self._tree)
        self._ctree.args.args = self._ctree.args.args + [ast.copy_location(ast.arg(n), tree) for n in self._hidden]
        self._dict_keys, self._attrs, self._elems = _dict_keys(self._ctree), _attrs_used(self._ctree), _element_attrs(self._ctree)
        self._passes = _passes(self._ctree)
        self._plans.clear()             # a plan reads the globals it was made with
        return True

    def _with_globals(self, args):
        """The arguments, and after them the globals the function reads, as they are now."""
        if self._hidden is None:
            self._find_globals()
        if not self._hidden:
            return args
        return args + tuple(g.get(n, _MISSING) for g, n in self._hidden_from)

    def _find_globals(self):
        """A number, tuple, array or dict at module level is one more input: it's read at each call and passed like the others, so
        loops can use it, a new value needs no recompile, and a new type is a new signature. Anything else stays a name."""
        self._hidden, self._hidden_from = [], []
        tree = self._tree
        if tree is None:
            return
        local = _lift.local_names(tree)
        loads = set()
        for st in tree.body:
            loads |= _lift._split(st)[0]
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        g = self._fn.__globals__
        attrs = {}
        for name, attr, how in _attribute_uses(tree):
            _add_use(attrs.setdefault(name, (set(), set(), set())), attr, how)
        self._hidden = sorted(n for n in loads - local - called if n in g and (
            _arg_type(g[n]) is not None or type(g[n]) is dict
            or (n in attrs and type(_object_fields(g[n], sorted(attrs[n][0]), attrs[n][1], attrs[n][2])) is list)))
        callees = sorted(n for n in called - local if type(g.get(n)) is Function)
        self._hidden += callees + sorted(n for n in called - local if T.is_c_function(g.get(n)))
        self._made = sorted(n for n in called - local if isinstance(g.get(n), type) and not issubclass(g[n], (enum.Enum, BaseException)))      # classes it makes instances of: in the signature, by identity
        self._hidden += self._made
        self._hidden_from = [(g, n) for n in self._hidden]
        for n in callees:                # what they read is read at each call too, and may be in another module
            f = g[n]
            if f._hidden is None:
                f._find_globals()        # recursion finds this one's list unfinished, and won't compile anyway
            for og, on in f._hidden_from:
                if og is _POOLDICT:              # its own: a caller doesn't make it
                    continue
                if not any(og is a and on == b for a, b in self._hidden_from):
                    self._hidden.append(on if og is g and on not in self._hidden else f'sjg{len(self._hidden)}_{on}')
                    self._hidden_from.append((og, on))
        if any(_ref_candidates(tree)):           # references chosen by data: their pool, made at each call
            self._hidden.append('sjpool')
            self._hidden_from.append((_POOLDICT, 'sjpool'))
        if self._hidden:
            self._ctree = copy.deepcopy(tree)
            self._ctree.args.args = self._ctree.args.args + [ast.copy_location(ast.arg(n), tree) for n in self._hidden]
            self._dict_keys = _dict_keys(self._ctree)
            self._attrs = _attrs_used(self._ctree)
            self._elems = _element_attrs(self._ctree)
            self._passes = _passes(self._ctree)
            self._nested = _nested_uses(self._ctree)
            self._unused = _unused(self._tree)

    def _hidden_names_read(self):
        """The globals this function itself reads, not those it reads only through what it calls."""
        return {n for n in self._hidden if not n.startswith('sjg')}

    def _real(self, args):
        """The caller's arguments, without the globals added to them."""
        return args[:len(args) - len(self._hidden)] if self._hidden else args

    def _slow(self, args):
        """A call a plan turned down: an int past 64 bits, an array of another shape of type."""
        real = args
        args, key = self._args_and_key(args)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        if entry is PYTHON:
            return self._py(*real)
        if self._replans and any(t.kind in ('dict', 'object') for t in entry.types):    # writing p['s'] can change what the plan checks for
            self._replans -= 1
            self._plans[tuple(map(type, real))] = _make_plan(self, entry, args)
        return self._run(entry, args)

    def _failed(self, entry, status, io):
        slots = [int.from_bytes(io[i:i + 8], 'little', signed=True) for i in range(0, 40, 8)]
        first = self._first - 1                    # the kernel's lines count from the def
        frames = [(self._filename, slots[1] + first if slots[1] else 0, self.__name__)]
        if entry.frames:                 # those a call passed through, innermost first
            n = int.from_bytes(io[entry.tb_at:entry.tb_at + 8], 'little', signed=True)
            for k in range(n, 0, -1):
                v = int.from_bytes(io[entry.tb_at + 8 * k:entry.tb_at + 8 * k + 8], 'little', signed=True)
                filename, name, offset = entry.frames[v >> 32]
                line = v & 0xffffffff
                frames.append((filename, line + offset if line else 0, name))
        _raise_chain(exception(status, slots), frames)

    def _callee(self):
        """What a compiled function needs to call this one."""
        if self._tree is None:
            raise Unsupported(None, self._why_not)
        if self._hidden is None:
            self._find_globals()
        return Callee(self, self._ctree or self._tree, self._fn.__globals__, self._filename, self._first - 1, self._boundscheck, self._parity,
                      [(id(g), n) for g, n in self._hidden_from], self._hidden_map(), self.__name__, npos=self._npos)

    def _hidden_map(self):
        return {(id(g), on): p for (g, on), p in zip(self._hidden_from, self._hidden)}

    def _slow_call(self, *args):
        return self._slow(args)

    def _timed(self, key, entry, args):
        s = self._stats.setdefault(key, [0, 0.0, 0.0])
        s[0] += 1
        t = time.perf_counter()
        if _running is not None:
            me = threading.get_ident()
            outer = _running.get(me)
            _running[me] = (self, entry is not PYTHON, t)
        try:
            return self._py(*self._real(args)) if entry is PYTHON else self._run(entry, args)
        finally:
            s[2 if entry is PYTHON else 1] += time.perf_counter() - t
            if _running is not None:
                if outer is None:
                    del _running[me]
                else:
                    _running[me] = outer

    def line_counts(self):
        """With counting on, over all signatures and lifted loops: (line -> times it ran, line -> (kind, times
        its body ran)), kind being 'if' or 'loop'. Lines are the file's. A kernel adds to the counts with the GIL
        released, so another thread can read them during a call."""
        runs, heads = {}, {}
        for f in [self] + self._all_loops():
            for entry in list(f._cache.values()):
                if isinstance(entry, Compiled) and entry.counts is not None:
                    r, h = entry.read_counts()
                    for n, v in r.items():
                        runs[n] = runs.get(n, 0) + v
                    for n, (kind, v) in h.items():
                        heads[n] = (kind, heads.get(n, (kind, 0))[1] + v)
        return runs, heads

    def preflight(self, signature=None):
        """Print this function line by line: native or Python and why, types, bounds checks."""
        from .cli.report import function_report
        function_report(self, signature, sys.stdout)

    def _specialize(self, key, args):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                return entry
            sig = ', '.join(_describe(x, t) for x, t in zip(self._real(args), key))
            t0 = time.perf_counter()
            try:
                entry = self._compile(key, args)
                self._compile_s[key] = time.perf_counter() - t0
                if _log:
                    print(f'scramjit: compiled {self.__name__}({sig}) in {self._compile_s[key]:.2f} s', file=sys.stderr)
            except Unsupported as u:
                line = (u.lineno + self._first - 1) if u.lineno else self._first
                self._why[key] = (line, u.reason)
                if _log:
                    print(f'scramjit: {self.__name__}({sig}) runs as Python: line {line}: {u.reason}', file=sys.stderr)
                if not self._fallback:
                    raise CompileError(f'{self._filename}:{line}: {self.__name__}({sig}) is not compiled: '
                                       f'{u.reason}') from None
                self._lift()
                if self._loop is None:
                    apart = ', apart from the loops that compile' if self._lifted else ''
                    warnings.warn_explicit(f'{self.__name__}({sig}) runs as Python{apart}: {u.reason}',
                                           FallbackWarning, self._filename, line)
                entry = PYTHON
            self._cache[key] = entry
            return entry

    def _native_loops(self):
        """The lifted loops that have compiled for some signature, however deep."""
        out = []
        for f in self._lifted or []:
            if any(isinstance(v, Compiled) for v in f._cache.values()):
                out.append(f)
            out += f._native_loops()
        return out

    def _all_loops(self):
        out = []
        for f in self._lifted or []:
            out.append(f)
            out += f._all_loops()
        return out

    def _lift(self):
        """Python runs from now on for some signature: lift the loops that may still compile."""
        if self._lifted is not None or self._tree is None:
            return
        self._lifted = []

        def native(python, tree, loop):
            f = Function(python, self._boundscheck, self._use_disk, True, self._parity, (tree, self._first), self._counting)
            f._owner = self._owner
            return f

        source = self._source_tree if self._source_tree is not None else self._tree
        self._py, self._lifted = _lift.lift(source, self._fn, self._filename, self._first, native,
                                            source.body[0] if self._loop else None)
        if self._kwonly:
            self._py = self._by_keyword(self._py)

    def _compile(self, key, args):
        if self._tree is None:
            raise Unsupported(None, self._why_not)
        real = len(args) - len(self._hidden or ())
        if self._hidden and 'sjpool' in self._hidden and key[real + self._hidden.index('sjpool')] is None:
            raise Unsupported(None, getattr(self, '_pool_why', 'its references can\'t be pooled'))
        for i, (x, t) in enumerate(zip(args, key)):
            if t is None and i >= real and self._hidden[i - real] == 'sjpool':
                raise Unsupported(None, getattr(self, '_pool_why', 'its references can\'t be pooled'))
            if t is None and i >= real and self._hidden[i - real] not in self._hidden_names_read():
                raise Unsupported(None, f'a function it calls reads the global {self._hidden_from[i - real][1]}, which is {_describe(x, t)}')
            if t is None and i >= real and getattr(self, '_made', None) and self._hidden[i - real] in self._made:
                raise Unsupported(None, f'makes instances of {self._hidden[i - real]}: {getattr(self, "_made_why", "it can\'t be compiled")}')
            if t is None:
                fields = _object_fields(x, *self._attrs[i], nested=self._nested.get(i)) if i in self._attrs and type(x) not in (dict, list) else None
                if type(fields) is str:
                    raise Unsupported(None, f'an argument of type {_describe(x, t)} is not compiled: {fields}')
                raise Unsupported(None, f'an argument of type {_describe(x, t)} is not compiled')
            if t.kind == 'function' and i < real:
                raise Unsupported(None, 'a function passed as an argument')
        counts = self._counting or _counts is True or bool(_counts and self._owner in _counts)
        c = Compiler(self._ctree or self._tree, list(key), self._fn.__globals__, self._filename, self._first - 1,
                     self._boundscheck, self._parity, counts, Unit(self), self._hidden_map() if self._hidden else None)
        source = c.generate()
        cimo = _get_cimo()
        try:
            if c.unit.modules:           # as those modules were compiled, or their results would change
                module = cimo.compile(PRELUDE + source, cache=self._use_disk, fma=c.unit.fma, wrapv=c.unit.wrapv)
            else:
                module = cimo.compile(PRELUDE + source, cache=self._use_disk, fma=not self._parity)   # Python never contracts
            kernel = module.sj_kernel
        except cimo.CompileError as e:
            e = cimo_diagnostic(str(e), self._filename, self._first - 1, c.unit.frames)
            if c.unit.modules:
                raise Unsupported(None, f'the C it calls doesn\'t compile in one file with the C generated here: {e}')
            raise Unsupported(None, f'internal error: cimo rejected the generated C ({e}); please report it')
        if c.uses_stop:
            _interrupt.ensure()
        if any(t.np for t in key) or c.ret.np:
            import numpy
            self._np = numpy
        entry = Compiled(kernel, c.ret, key, source, _libm_table() if c.uses_libm else None,
                         c.shown_types(), c.checks, c.hoisted_lines, c.gotchas, c.uses_stop, c.uses_rerun)
        entry.outs = [(i, k, 8 * c.out_slot(n), c.env[flat]) for n, (flat, i, k) in enumerate(c.outs)]
        entry.lists = c.written_lists()
        entry.columns = c.written_columns()
        entry.instance_ret = c.instance_ret
        entry.array_args = c.array_args
        entry.records = [(8 * c.record_slot(k), struct.Struct(f'<{array_record_slots(t)}q'), t.elem.name[3:] if not T.is_pylist(t) else t)
                         for k, t in enumerate(ret_arrays(c.ret))]
        if any(not isinstance(name, T.Type) for _, _, name in entry.records):
            import numpy
            self._np = numpy
            entry.records = [(at, layout, name if isinstance(name, T.Type) else numpy.dtype(name).str) for at, layout, name in entry.records]
        entry.io_size = 8 * c.record_slot(len(entry.records))
        entry.frames = c.unit.frames if c.calls else None
        if c.calls:                      # room for the trace of a failed call
            entry.tb_at = entry.io_size
            entry.io_size += 8 * (1 + len(c.unit.frames))
        entry.free = getattr(module, 'sj_release', None) if c.makes or c.ret.kind == 'list' else None
        if counts:
            entry.counts = bytearray(8 * max(len(c.count_lines), 1))
            entry.count_lines, entry.count_heads, entry.count_rest = c.count_lines, c.count_heads, c.count_rest
        return entry

    def _key(self, args):
        """The signature of a call. A dict's type is the keys this function uses: nothing else in it matters."""
        key = tuple(map(_arg_type, args))
        lists = [id(y) for x in args if type(x) is list for y in [x] + [r for r in x if type(r) is list]]
        if len(set(lists)) != len(lists):        # one list under two names: a copy each would hide the writes
            key = tuple(None if type(x) is list else t for x, t in zip(args, key))
        if self._hidden and any(t is not None and t.kind == 'classref' for t in key):
            out = list(key)
            real = len(args) - len(self._hidden)
            owners = None
            for k, name in enumerate(self._hidden):
                x = args[real + k]
                if out[real + k] is not None and out[real + k].kind == 'classref' and not issubclass(x, enum.Enum):
                    if owners is None:
                        owners = _owner_classes(self, args)
                    t = _made_class_type(self, x, name, owners)
                    if t is not None or name in getattr(self, '_made', ()):
                        out[real + k] = t          # made here, or by what it calls: how, in the signature
            key = tuple(out)
        refs = None
        if self._hidden and 'sjpool' in self._hidden:         # references chosen by data: one pool of their objects
            out = list(key)
            at = len(args) - len(self._hidden) + self._hidden.index('sjpool')
            plan = _pool_plan(self, args)
            if plan[0] is None:
                out[at], self._pool_why = None, plan[1]
            else:
                out[at] = plan[1]
                for i, t in plan[0].items():
                    out[i] = t
                refs = (plan[2], plan[3]) if plan[1].kind == 'pool' else None
            key = tuple(out)
        if not ((self._attrs or self._elems or self._passes or self._unused or self._returned) and None in key) \
                and (not self._dict_keys or dict not in map(type, args)):
            return key
        seen, out = set(), list(key)
        owners = _owner_classes(self, args) if self._passes else None
        empty = ((), set(), set())
        roots = []                               # instances: their types together, as one graph with its alias pattern
        for i, x in enumerate(args):
            if out[i] is not None or type(x) in (dict, list) or not (i in self._attrs or i in self._passes or i in self._unused
                                                                      or i in self._returned):
                continue
            uses, nested = self._attrs.get(i, empty), self._nested.get(i)
            if i in self._passes:
                more = {}
                uses = _param_uses(self, i, type(x), uses, owners, (), more)
                nested = _merge_nested(nested, more)
            node = [set(uses[0]), set(uses[1]), set(uses[2]), {}]
            _node_merge(node, [(), (), (), nested or {}])
            name = (self._ctree or self._tree).args.args[i].arg if i < len((self._ctree or self._tree).args.args) else None
            if name is not None and any(isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)
                                        for n in ast.walk(self._ctree or self._tree)):
                recipe = _init_recipe(type(x))   # given a new instance: it starts as a copy of all __init__ assigns
                if type(recipe) is dict:
                    node[0] |= {a for a, _ in recipe['assigns']}
            built = _construction_uses(self).get(i)
            if built is not None:                # given to C(...): what's done through the attribute it becomes
                _node_merge(node, built)
            roots.append((i, x, node))
        if roots:
            found, ids = _object_graph(roots, refs)
            for i, t in found.items():
                out[i] = t
            seen |= ids
        for i, x in enumerate(args):
            if type(x) is dict:
                out[i] = None if id(x) in seen else self._dict_type(x, self._dict_keys.get(i))
                seen.add(id(x))          # two names for one dict: a write must show through both
            elif out[i] is None and type(x) is list and (i in self._elems or i in self._passes) and x:
                uses = self._elems.get(i, empty)
                tree = self._ctree or self._tree
                if getattr(self, '_enested_for', None) is not tree:
                    self._enested, self._enested_for = _element_nested(tree), tree
                nested = self._enested.get(i)
                if i in self._passes:
                    more = {}
                    uses = _param_uses(self, i, type(x[0]), uses, owners, (), more)
                    nested = _merge_nested(nested, more)
                out[i] = _objlist_type(x, (sorted(uses[0]),) + tuple(uses[1:]), seen, dict in map(type, args), nested)
        return tuple(out)

    @staticmethod
    def _dict_type(d, keys):
        if not keys:
            return None
        fields = []
        for k in keys:
            if k not in d:
                return None              # Python raises KeyError, at the right moment
            t = _arg_type(d[k])
            if t is None or t is T.U64 or t.kind == 'list' or T.is_pylist(t):     # a list in a dict: no copy back yet
                return None
            fields.append((k, t))
        return T.dict_of(fields)

    def _run(self, entry, args):
        cargs = []
        copies = []
        columns = []
        pool_t = next((t for t in entry.types if t.kind == 'pool'), None)
        objs, where, refcols = [], None, []
        if pool_t is not None:                   # every object references reach, each once, walked as _pool_plan walked it
            _, refs, roots = pool_t.elem
            where, todo = {}, []
            for i, path in roots:
                v = args[i] if not path else _member(args[i], path)
                if v is not None:
                    todo.append(v)
            while todo:
                o = todo.pop(0)
                if id(o) in where:
                    continue
                where[id(o)] = len(objs)
                objs.append(o)
                todo += [v for v in (getattr(o, a) for a in refs) if v is not None]
        for i, (x, t) in enumerate(zip(args, entry.types)):
            if t.kind == 'pool':                    # a list per number, then an index list per reference
                numbers, refs, _ = t.elem
                for key, ft in numbers:
                    a = array.array(_LIST_CODES[ft.name], (getattr(o, key) for o in objs))
                    columns.append((i, key, a, ft, objs, key))
                    cargs += [a.buffer_info()[0], len(a)]
                for key in refs:
                    a = array.array('q', (-1 if getattr(o, key) is None else where[id(getattr(o, key))] for o in objs))
                    refcols.append((i, key, a))
                    cargs += [a.buffer_info()[0], len(a)]
                continue
            if t.kind == 'objlist':                 # a list of numbers per attribute, written back into the instances
                pool, ixs = x, []
                if '' in t.strided:                 # repeats: each instance once, and which is at each place
                    where = {}
                    ixs.append(array.array('q', (where.setdefault(id(y), len(where)) for y in x)))
                    pool = list({id(y): y for y in x}.values())
                levels = _pool_levels(pool, t)
                at = {path: objs for path, objs, _ in levels}
                ixs += [ix for _, _, ix in levels if ix is not None]
                for key, ft in leaf_types(t.elem):     # breadth first through objects in attributes, as flattened
                    path, _, last = key.rpartition('.')
                    a = array.array(_LIST_CODES[ft.name], map(operator.attrgetter(last), at[path]))
                    columns.append((i, key, a, ft, at[path], last))
                    cargs += [a.buffer_info()[0], len(a)]
                for ix in ixs:
                    copies.append((None, ix))       # kept alive for the call
                    cargs += [ix.buffer_info()[0], len(ix)]
                continue
            if t.kind == 'list':
                a = array.array(_LIST_CODES[t.elem.name], x)
                copies.append((i, a))
                cargs += [a.buffer_info()[0], len(a)]
                continue
            if T.is_pylist(t):                  # a list of lists: one array, row after row, and its extents
                a = array.array(_LIST_CODES[t.elem.name], _flatten(x, t.ndim))
                copies.append((i, a))
                cargs.append(a.buffer_info()[0])
                level = x
                for _ in range(t.ndim):
                    cargs.append(len(level))
                    level = level[0]
                continue
            _push(cargs, x, t, where)
        if entry.libm is not None:
            cargs.append(entry.libm)
        if entry.stop:
            cargs.append(_interrupt.address())
        if entry.counts is not None:
            cargs.append(entry.counts)
        io = bytearray(entry.io_size)
        status = entry.kernel(*cargs, io)
        for i, key, a, ft, held, last in columns:
            if (i, key) in entry.columns:
                _copy_back_attr(held, last, a, ft)
        for i, key, a in refcols:                # references the kernel changed: the objects at those indexes
            if (i, key) in entry.columns:
                for o, k in zip(objs, a):
                    target = None if k < 0 else objs[k]
                    if getattr(o, key) is not target:
                        setattr(o, key, target)
        for i, a in copies:                       # what was written, even before an exception, as Python would have
            if i is not None and i in entry.lists:
                t = entry.types[i]
                if T.is_pylist(t):
                    _copy_back_nested(args[i], a, t.elem, t.ndim)
                else:
                    _copy_back(args[i], a, t.elem)
        if status == RERUN and entry.rerun:
            return self._py(*self._real(args))
        for i, key, at, ot in entry.outs:
            v = self._box(io, at, at + 8, ot)
            if _field_type(entry.types[i], key).kind == 'ref':     # an index in the pool: the object, or None
                v = None if v < 0 else objs[v]
            _set_member(args[i], key, v)
        if status:
            self._failed(entry, status, io)
        r = entry.ret
        if entry.instance_ret and entry.instance_ret[0] == 'made':     # a new instance, filled as its __init__ fills it
            _, cls, attrs = entry.instance_ret
            values = iter(self._box_tuple(io, r) if not entry.records else self._array_result(entry, io, args))

            def build(cls, attrs):
                obj = cls.__new__(cls)
                for a in attrs:
                    if type(a) is tuple:         # an object it holds: made too, or an argument's, as it is
                        a, held = a
                        v = build(held[1], held[2]) if held[0] == 'made' else _member(args[held[1]], held[2]) if held[2] else args[held[1]]
                    else:
                        v = next(values)
                    object.__setattr__(obj, a, v)
                return obj
            return build(cls, attrs)
        if r is T.NONE:
            return None
        if entry.records:
            return self._array_result(entry, io, args)
        if r.kind == 'tuple':
            return self._box_tuple(io, r)
        if r.kind == 'list':
            return self._list_result(entry, io, args)
        if entry.instance_ret and entry.instance_ret[0] == 'pool':     # a reference: the object at that index, or None
            k = self._box(io, 0, 16, r)
            return None if k < 0 else objs[k]
        if entry.instance_ret:                  # the instance itself: an argument, an element of one, or an object in one
            k = self._box(io, 0, 16, r)
            if entry.instance_ret[0] == 'path':
                return _member(args[entry.instance_ret[1]], entry.instance_ret[2])
            return args[k] if entry.instance_ret[0] == 'arg' else args[entry.instance_ret[1]][k]
        return self._box(io, 0, 16, r)

    def _list_result(self, entry, io, args):
        """A returned list: the argument itself, or a new list of what the kernel built, whose memory is then released."""
        q = struct.unpack_from('<6q', io, 0)
        if q[5]:
            return args[q[5] - 1]
        address, n = q[0], q[4]
        et = entry.ret.elem
        try:
            if not n:
                return []
            import ctypes
            code = _LIST_CODES[et.name]
            values = memoryview((ctypes.c_char * (n * struct.calcsize(code))).from_address(address)).cast('B').cast(code).tolist()
            return [bool(v) for v in values] if et is T.BOOL else values
        finally:
            if address:
                entry.free(address)

    def _box_tuple(self, io, r, at=None):
        """A returned tuple of numbers, nested ones too: each number has its pair of slots, in order."""
        at = at if at is not None else [0]
        out = []
        for et in r.elem:
            if et.kind == 'tuple':
                out.append(self._box_tuple(io, et, at))
            else:
                out.append(self._box(io, 32 + 16 * at[0], 40 + 16 * at[0], et))
                at[0] += 1
        return tuple(out)

    def _array_result(self, entry, io, args):
        arrays = self._arrays(entry, io, args)
        r = entry.ret
        if r.kind == 'array':
            return arrays[0]
        arrays = iter(arrays)
        return tuple([next(arrays) if et.kind == 'array' else self._box(io, 32 + 16 * i, 40 + 16 * i, et) for i, et in enumerate(r.elem)])

    def _arrays(self, entry, io, args):
        """Returned arrays. One made in the kernel is copied into a numpy array of its own, once however many results
        are in it, and its block freed; so it's what numpy would have made. The argument itself, or the array made, if
        it's that; else a real numpy view of its memory."""
        out, made = [], None
        numpy = self._np
        try:
            for at, layout, typestr in entry.records:
                value = layout.unpack_from(io, at)
                address, which, same, block, data, ndim = value[:6]
                if isinstance(typestr, T.Type):          # a list of lists: the one passed in, or new lists of what was made
                    if block:
                        out.append(_lists_from(address, value[10::2][:typestr.ndim], typestr.elem))
                    else:
                        i, key = entry.array_args[which]
                        out.append(args[i] if key is None else _member(args[i], key))
                    continue
                if block:
                    if made is None:
                        made = {}
                    root = made.get(block)
                    if root is None:
                        root = made[block] = numpy.array(_Interface(None, data, value[6:6 + ndim], None, typestr))
                    if not same:
                        address += root.__array_interface__['data'][0] - data
                else:
                    i, key = entry.array_args[which]
                    root = args[i] if key is None else _member(args[i], key)
                if same:
                    out.append(root)
                else:
                    size = root.itemsize
                    out.append(numpy.asarray(_Interface(root, address, value[10::2], [s * size for s in value[11::2]])))
        finally:
            if entry.free is not None:
                blocks = {layout.unpack_from(io, at)[3] for at, layout, _ in entry.records}
                for block in blocks:
                    if block:
                        entry.free(block)
        return out

    def _box(self, io, at, flag_at, r):
        if r.kind == 'enum':                      # its value, back as the member
            return r.elem(int.from_bytes(io[at:at + 8], 'little', signed=True))
        if r.np == 'dyn' and r.kind == 'float' and io[flag_at] == RETURNS_INT:
            return int(struct.unpack_from('<d', io, at)[0])
        np = r.np if r.np != 'dyn' else io[flag_at] != 0
        if np and self._np is None:
            import numpy
            self._np = numpy
        if r.kind == 'float':
            v = struct.unpack_from('<d', io, at)[0]
            if not np and r.py == 'int':           # still the int it started as
                return int(v)
        else:
            v = int.from_bytes(io[at:at + 8], 'little', signed=r is not T.U64 and r is not T.NP_UINT64)
        if np:
            name = T.as_narrow(r).name[3:] if T.is_narrow_dyn(r) or r.np is True else {'int': 'int64', 'float': 'float64', 'bool': 'bool_'}[r.kind]
            return getattr(self._np, 'bool_' if name == 'bool' else name)(v)
        return bool(v) if r.kind == 'bool' else v
