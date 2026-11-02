"""The @scram decorator: signatures, the compile cache, the call boundary and fallback."""

import ast
import copy
import functools
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
from ._compiler import Compiler, array_record_slots, ret_arrays, ret_pairs
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
    if t is int:
        return T.INT if -(1 << 63) <= x < (1 << 63) else None
    if t is float:
        return T.FLOAT
    if t is bool:
        return T.BOOL
    if t is tuple:
        elems = [_arg_type(v) for v in x]
        if not elems or any(e is None or not e.is_number for e in elems):
            return None
        return T.tuple_of(elems)
    np = sys.modules.get('numpy')
    if np is not None:
        if t is np.ndarray:
            if x.ndim >= 1 and x.dtype.isnative:
                name = x.dtype.name
                elem = T.NP_FLOAT64 if name == 'float64' else T.NP_INT64 if name == 'int64' else T.NARROW.get(name)
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


def _push(cargs, x, t):
    """The C arguments for one Python argument."""
    if t.kind == 'array' and t.strided:
        cargs.append(x.ctypes.data)
        cargs.extend(x.shape)
        cargs.extend(s // x.itemsize for s in x.strides)
    elif t.kind == 'array':
        cargs.append(x)
        cargs.extend(x.shape)
    elif t.kind == 'tuple':
        cargs.extend(v.item() if et.np else v for v, et in zip(x, t.elem))
    elif t.kind == 'dict':
        for key, ft in t.elem:
            _push(cargs, x[key], ft)
    elif t.np:
        cargs.append(x.item())
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


class Compiled:
    __slots__ = ('kernel', 'ret', 'types', 'source', 'libm', 'locals', 'checks', 'hoisted', 'gotchas', 'stop', 'rerun', 'outs',
                 'counts', 'count_lines', 'count_heads', 'count_rest', 'array_args', 'records', 'io_size', 'free')

    def __init__(self, kernel, ret, types_, source, libm, locals_=None, checks=None, hoisted=(), gotchas=None, stop=False, rerun=False):
        self.rerun = rerun
        self.outs = []                        # written dict keys: (argument index, key, byte in io, type)
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


def _make_plan(f, entry, args):
    """A small function for calls with these argument types, or f._slow_call if they need the general
    path. It's generated because a call is where the time goes: what's known from the types is decided
    here, once. Anything it can't vouch for at call time (an int past 64 bits, an array of another
    dtype, dimension or layout, an empty one) goes to f._slow, so plans change only speed."""
    if entry.counts is not None:
        return f._slow_call
    env = {'kernel': entry.kernel, 'slow': f._slow, 'failed': f._failed, 'box': f._box, 'ret': entry.ret, 'pool': [],
           'addressof': ctypes.addressof, 'from_buffer': ctypes.c_char.from_buffer, 'bytearray': bytearray, 'memoryview': memoryview}
    names = [f'a{i}' for i in range(len(args))]
    real = names[:len(args) - len(f._hidden or ())]
    again = '(' + ''.join(n + ', ' for n in real) + ')'
    hidden = []                        # globals the function reads: read now, and turned down if their type has changed
    for k, g in enumerate(f._hidden or ()):
        i = len(real) + k
        env[f'gname{k}'], env[f'gtype{k}'] = g, type(args[i])
        hidden.append(f'    a{i} = G.get(gname{k}, MISSING)\n    if type(a{i}) is not gtype{k}:\n        return slow({again})')
    if hidden:
        env.update(G=f._fn.__globals__, MISSING=_MISSING)
    cargs = []
    for i, (x, t, n) in enumerate(zip(args, entry.types, names)):
        if t.kind == 'array' and t.strided:
            return f._slow_call
        if t.kind == 'array':
            env[f'd{i}'] = x.dtype
            cargs.append(f'p{i}' if x.flags.writeable else n)
            cargs.extend(f's{i}[{k}]' for k in range(x.ndim))
        elif t.kind == 'tuple':
            return f._slow_call
        elif t.kind == 'dict':
            if not all(ft.is_number for _, ft in t.elem):
                return f._slow_call
            for k, (key, ft) in enumerate(t.elem):
                cargs.append(f'f{i}_{k}.item()' if ft.np else f'f{i}_{k}')
        elif t.np:
            cargs.append(f'{n}.item()')
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
        elif t.kind == 'int' and not t.np:
            src_guards.append(f'    if not -9223372036854775808 <= {n} <= 9223372036854775807:\n        return slow({again})')
    addresses = ''.join(f'        p{i} = addressof(from_buffer({names[i]}))\n' for i in arrays if args[i].flags.writeable)
    take = (f'    try:\n{addresses}    except (ValueError, TypeError, BufferError):      # empty: no buffer to take the address of\n'
            f'        return slow({again})\n') if addresses else ''
    size = entry.io_size
    r = entry.ret
    if entry.records:
        env.update(arrays=f._array_result, compiled=entry)
        result = f"arrays(compiled, io, ({''.join(n + ', ' for n in names)}))"
    elif r is T.NONE:
        result = 'None'
    elif r is T.INT:
        result = 'q[0]'
    elif r is T.FLOAT:
        result = 'd[0]'
    elif r.kind == 'tuple':
        env['boxes'] = [(32 + 16 * i, 40 + 16 * i, et) for i, et in enumerate(r.elem)]
        result = 'tuple([box(io, at, flag, et) for at, flag, et in boxes])'
    else:
        result = 'box(io, 0, 16, ret)'
    rerun = ''
    if entry.rerun:
        env.update(f=f, RERUN=RERUN)
        rerun = f"    if status == RERUN:\n        pool.append(entry)\n        return f._py({', '.join(real)})\n"
    saves = ''
    for n, (i, key, at, ot) in enumerate(entry.outs):
        env[f'okey{n}'], env[f'otype{n}'] = key, ot
        saves += f'    {names[i]}[okey{n}] = box(io, {at}, {at + 8}, otype{n})\n'
    dicts = [names[i] for i, t in enumerate(entry.types) if t.kind == 'dict']
    if len(dicts) > 1:                 # two names for one dict: a write must show through both
        pairs = ' or '.join(f'{a} is {b}' for k, a in enumerate(dicts) for b in dicts[k + 1:])
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
        failed(status, io)
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
        self._dict_keys = _dict_keys(self._tree)
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
        return fdef, None

    # descriptor: work as a method too (self is then an argument, which falls back)
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

    def __call__(self, *args, **kwargs):
        plan = self._plans.get(tuple(map(type, args)))
        if plan is not None and not kwargs:             # these types were checked when the plan was made
            return plan(*args)
        for i, p, ann in self._contracts:
            if i < len(args) and not _contract_ok(args[i], ann):
                raise TypeError(f'{self.__name__}() argument {p!r} must be {_ANNOTATIONS[ann]}, '
                                f'not {type(args[i]).__name__}')
        if kwargs:
            return self._fn(*args, **kwargs)
        real = args
        args = self._with_globals(args)
        key = self._key(args)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        if _profile:
            return self._timed(key, entry, args)
        if entry is not PYTHON:
            self._plans[tuple(map(type, real))] = _make_plan(self, entry, args)
        elif None not in key and not any(t.kind == 'array' for t in key) and dict not in map(type, args) and not self._hidden:
            self._plans[tuple(map(type, real))] = self._py   # these types always run as Python
        if entry is PYTHON:
            return self._py(*real)
        return self._run(entry, args)

    def _with_globals(self, args):
        """The arguments, and after them the globals the function reads, as they are now."""
        if self._hidden is None:
            self._find_globals()
        if not self._hidden:
            return args
        g = self._fn.__globals__
        return args + tuple(g.get(n, _MISSING) for n in self._hidden)

    def _find_globals(self):
        """A number, tuple, array or dict at module level is one more input: it's read at each call and passed like the others, so
        loops can use it, a new value needs no recompile, and a new type is a new signature. Anything else stays a name."""
        self._hidden = []
        tree = self._tree
        if tree is None:
            return
        local = _lift.local_names(tree)
        loads = set()
        for st in tree.body:
            loads |= _lift._split(st)[0]
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        g = self._fn.__globals__
        self._hidden = sorted(n for n in loads - local - called if n in g and (_arg_type(g[n]) is not None or type(g[n]) is dict))
        if self._hidden:
            self._ctree = copy.deepcopy(tree)
            self._ctree.args.args = self._ctree.args.args + [ast.copy_location(ast.arg(n), tree) for n in self._hidden]
            self._dict_keys = _dict_keys(self._ctree)

    def _real(self, args):
        """The caller's arguments, without the globals added to them."""
        return args[:len(args) - len(self._hidden)] if self._hidden else args

    def _slow(self, args):
        """A call a plan turned down: an int past 64 bits, an array of another shape of type."""
        real = args
        args = self._with_globals(args)
        key = self._key(args)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        if entry is PYTHON:
            return self._py(*real)
        if self._replans and dict in map(type, args):    # writing p['s'] can change what the plan checks for
            self._replans -= 1
            self._plans[tuple(map(type, args))] = _make_plan(self, entry, args)
        return self._run(entry, args)

    def _failed(self, status, io):
        slots = [int.from_bytes(io[i:i + 8], 'little', signed=True) for i in range(0, 40, 8)]
        _raise_at(exception(status, slots), self._filename, slots[1], self.__name__)

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

        self._py, self._lifted = _lift.lift(self._tree, self._fn, self._filename, self._first, native,
                                            self._tree.body[0] if self._loop else None)

    def _compile(self, key, args):
        if self._tree is None:
            raise Unsupported(None, self._why_not)
        for x, t in zip(args, key):
            if t is None:
                raise Unsupported(None, f'an argument of type {_describe(x, t)} is not compiled')
        counts = self._counting or _counts is True or bool(_counts and self._owner in _counts)
        c = Compiler(self._ctree or self._tree, list(key), self._fn.__globals__, self._filename, self._first - 1,
                     self._boundscheck, self._parity, counts)
        source = c.generate()
        cimo = _get_cimo()
        try:
            module = cimo.compile(PRELUDE + source, cache=self._use_disk, fma=not self._parity)   # Python never contracts
            kernel = module.sj_kernel
        except cimo.CompileError as e:
            raise Unsupported(None, f'internal error: cimo rejected the generated C ({e}); please report it')
        if c.uses_stop:
            _interrupt.ensure()
        if any(t.np for t in key) or c.ret.np:
            import numpy
            self._np = numpy
        entry = Compiled(kernel, c.ret, key, source, _libm_table() if c.uses_libm else None,
                         c.shown_types(), c.checks, c.hoisted_lines, c.gotchas, c.uses_stop, c.uses_rerun)
        entry.outs = [(i, k, 8 * c.out_slot(n), c.env[flat]) for n, (flat, i, k) in enumerate(c.outs)]
        entry.array_args = c.array_args
        entry.records = [(8 * c.record_slot(k), struct.Struct(f'<{array_record_slots(t)}q'), t.elem.name[3:]) for k, t in enumerate(ret_arrays(c.ret))]
        if entry.records:
            import numpy
            self._np = numpy
            entry.records = [(at, layout, numpy.dtype(name).str) for at, layout, name in entry.records]
        entry.io_size = 8 * c.record_slot(len(entry.records))
        entry.free = module.sj_release if c.makes else None
        if counts:
            entry.counts = bytearray(8 * max(len(c.count_lines), 1))
            entry.count_lines, entry.count_heads, entry.count_rest = c.count_lines, c.count_heads, c.count_rest
        return entry

    def _key(self, args):
        """The signature of a call. A dict's type is the keys this function uses: nothing else in it matters."""
        key = tuple(map(_arg_type, args))
        if not self._dict_keys or dict not in map(type, args):
            return key
        seen, out = set(), list(key)
        for i, x in enumerate(args):
            if type(x) is dict:
                out[i] = None if id(x) in seen else self._dict_type(x, self._dict_keys.get(i))
                seen.add(id(x))          # two names for one dict: a write must show through both
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
            if t is None:
                return None
            fields.append((k, t))
        return T.dict_of(fields)

    def _run(self, entry, args):
        cargs = []
        for x, t in zip(args, entry.types):
            _push(cargs, x, t)
        if entry.libm is not None:
            cargs.append(entry.libm)
        if entry.stop:
            cargs.append(_interrupt.address())
        if entry.counts is not None:
            cargs.append(entry.counts)
        io = bytearray(entry.io_size)
        status = entry.kernel(*cargs, io)
        if status == RERUN and entry.rerun:
            return self._py(*self._real(args))
        for i, key, at, ot in entry.outs:
            args[i][key] = self._box(io, at, at + 8, ot)
        if status:
            slots = [int.from_bytes(io[i:i + 8], 'little', signed=True) for i in range(0, 40, 8)]
            _raise_at(exception(status, slots), self._filename, slots[1], self.__name__)
        r = entry.ret
        if r is T.NONE:
            return None
        if entry.records:
            return self._array_result(entry, io, args)
        if r.kind == 'tuple':
            return tuple(self._box(io, 32 + 16 * i, 40 + 16 * i, et) for i, et in enumerate(r.elem))
        return self._box(io, 0, 16, r)

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
                    root = args[i] if key is None else args[i][key]
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
        if r.np == 'dyn' and r.kind == 'float' and io[flag_at] == RETURNS_INT:
            return int(struct.unpack_from('<d', io, at)[0])
        np = r.np if r.np != 'dyn' else io[flag_at] != 0
        if np and self._np is None:
            import numpy
            self._np = numpy
        if r.kind == 'float':
            v = struct.unpack_from('<d', io, at)[0]
        else:
            v = int.from_bytes(io[at:at + 8], 'little', signed=True)
        if np:
            name = r.name[3:] if r.np is True else {'int': 'int64', 'float': 'float64', 'bool': 'bool_'}[r.kind]
            return getattr(self._np, 'bool_' if name == 'bool' else name)(v)
        return bool(v) if r.kind == 'bool' else v
