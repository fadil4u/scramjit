"""The @scram decorator: signatures, the compile cache, the call boundary and fallback."""

import ast
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
from ._compiler import Compiler
from ._errors import CompileError, FallbackWarning, Unsupported
from . import _interrupt
from ._runtime import LIBM, PRELUDE, exception

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
            if x.ndim >= 1 and x.flags.c_contiguous and x.dtype.isnative:
                name = x.dtype.name
                elem = T.NP_FLOAT64 if name == 'float64' else T.NP_INT64 if name == 'int64' else T.NARROW.get(name)
                if elem is not None:
                    return T.array(elem, x.flags.writeable, x.ndim)
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


class _Python:
    """Marker: this signature runs as the original Python function."""


PYTHON = _Python()


class Compiled:
    __slots__ = ('kernel', 'ret', 'types', 'source', 'libm', 'locals', 'checks', 'hoisted', 'gotchas', 'stop')

    def __init__(self, kernel, ret, types_, source, libm, locals_=None, checks=None, hoisted=(), gotchas=None, stop=False):
        self.checks, self.hoisted, self.gotchas, self.stop = checks or {}, hoisted, gotchas or {}, stop
        self.locals = locals_ or {}           # variable -> inferred type name
        self.kernel = kernel
        self.ret = ret
        self.types = types_
        self.source = source
        self.libm = libm


_functions = weakref.WeakSet()    # every decorated function, for preflight
_profile = False                  # preflight --run: time each call
_log = bool(os.environ.get('SCRAMJIT_LOG'))


def enable_profile():
    global _profile
    _profile = True


def functions():
    return list(_functions)


def _make_plan(f, entry, args):
    """A small function for calls with these argument types, or f._slow_call if they need the general
    path. It's generated because a call is where the time goes: what's known from the types is decided
    here, once. Anything it can't vouch for at call time (an int past 64 bits, an array of another
    dtype, dimension or layout, an empty one) goes to f._slow, so plans change only speed."""
    env = {'kernel': entry.kernel, 'slow': f._slow, 'failed': f._failed, 'box': f._box, 'ret': entry.ret, 'pool': [],
           'addressof': ctypes.addressof, 'from_buffer': ctypes.c_char.from_buffer, 'bytearray': bytearray, 'memoryview': memoryview}
    names = [f'a{i}' for i in range(len(args))]
    cargs = []
    for i, (x, t, n) in enumerate(zip(args, entry.types, names)):
        if t.kind == 'array':
            env[f'd{i}'] = x.dtype
            cargs.append(f'p{i}' if x.flags.writeable else n)
            cargs.extend(f's{i}[{k}]' for k in range(x.ndim))
        elif t.kind == 'tuple':
            return f._slow_call
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
    src_guards = []
    for i, (t, n) in enumerate(zip(entry.types, names)):
        if t.kind == 'array':
            x = args[i]
            src_guards.append(f'    fl = {n}.flags\n    if {n}.dtype is not d{i} or {n}.ndim != {x.ndim} or not fl.c_contiguous or fl.writeable is not {bool(x.flags.writeable)}:\n        return slow(({", ".join(names)},))\n    s{i} = {n}.shape')
        elif t.kind == 'int' and not t.np:
            src_guards.append(f'    if not -9223372036854775808 <= {n} <= 9223372036854775807:\n        return slow(({", ".join(names)},))')
    addresses = ''.join(f'        p{i} = addressof(from_buffer({names[i]}))\n' for i in arrays if args[i].flags.writeable)
    take = (f'    try:\n{addresses}    except (ValueError, TypeError, BufferError):      # empty: no buffer to take the address of\n'
            f'        return slow(({", ".join(names)},))\n') if addresses else ''
    size = 40 + (16 * len(entry.ret.elem) if entry.ret.kind == 'tuple' else 0)
    r = entry.ret
    if r is T.NONE:
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
    src = f"""def plan({', '.join(names)}):
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
    if status:
        pool.append(entry)
        failed(status, io)
    value = {result}
    pool.append(entry)
    return value
"""
    exec(src, env)  # noqa: S102
    return env['plan']


class Function:
    def __init__(self, fn, boundscheck=True, cache=True, fallback=True, parity=True):
        functools.update_wrapper(self, fn)
        self._fn = fn
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
        _functions.add(self)
        code = fn.__code__
        self._filename = code.co_filename
        self._first = code.co_firstlineno
        self._contracts = [(i, p, a) for i, (p, a) in enumerate(
            (p, fn.__annotations__.get(p)) for p in code.co_varnames[:code.co_argcount]) if a in _ANNOTATIONS]
        self._tree, self._why_not = self._parse(fn)
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
        key = tuple(map(_arg_type, args))
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        if _profile:
            return self._timed(key, entry, args)
        if entry is not PYTHON:
            self._plans[tuple(map(type, args))] = _make_plan(self, entry, args)
        elif None not in key and not any(t.kind == 'array' for t in key):
            self._plans[tuple(map(type, args))] = self._fn   # these types always run as Python
        if entry is PYTHON:
            return self._fn(*args)
        return self._run(entry, args)

    def _slow(self, args):
        """A call a plan turned down: an int past 64 bits, an array of another shape of type."""
        key = tuple(map(_arg_type, args))
        entry = self._cache.get(key)
        if entry is None:
            entry = self._specialize(key, args)
        return self._fn(*args) if entry is PYTHON else self._run(entry, args)

    def _failed(self, status, io):
        slots = [int.from_bytes(io[i:i + 8], 'little', signed=True) for i in range(0, 40, 8)]
        _raise_at(exception(status, slots), self._filename, slots[1], self.__name__)

    def _slow_call(self, *args):
        return self._slow(args)

    def _timed(self, key, entry, args):
        s = self._stats.setdefault(key, [0, 0.0, 0.0])
        s[0] += 1
        t = time.perf_counter()
        try:
            return self._fn(*args) if entry is PYTHON else self._run(entry, args)
        finally:
            s[2 if entry is PYTHON else 1] += time.perf_counter() - t

    def preflight(self, signature=None):
        """Print this function line by line: native or Python and why, types, bounds checks."""
        from .cli.report import function_report
        function_report(self, signature, sys.stdout)

    def _specialize(self, key, args):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                return entry
            sig = ', '.join(_describe(x, t) for x, t in zip(args, key))
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
                warnings.warn_explicit(f'{self.__name__}({sig}) runs as Python: {u.reason}',
                                       FallbackWarning, self._filename, line)
                entry = PYTHON
            self._cache[key] = entry
            return entry

    def _compile(self, key, args):
        if self._tree is None:
            raise Unsupported(None, self._why_not)
        for x, t in zip(args, key):
            if t is None:
                raise Unsupported(None, f'an argument of type {_describe(x, t)} is not compiled')
        c = Compiler(self._tree, list(key), self._fn.__globals__, self._filename, self._first - 1,
                     self._boundscheck, self._parity)
        source = c.generate()
        cimo = _get_cimo()
        try:
            kernel = cimo.compile(PRELUDE + source, cache=self._use_disk, fma=not self._parity).sj_kernel   # Python never contracts
        except cimo.CompileError as e:
            raise Unsupported(None, f'internal error: cimo rejected the generated C ({e}); please report it')
        if c.uses_stop:
            _interrupt.ensure()
        if any(t.np for t in key) or c.ret.np:
            import numpy
            self._np = numpy
        return Compiled(kernel, c.ret, key, source, _libm_table() if c.uses_libm else None,
                        {k: t.name for k, t in c.env.items()}, c.checks, c.hoisted_lines, c.gotchas, c.uses_stop)

    def _run(self, entry, args):
        cargs = []
        for x, t in zip(args, entry.types):
            if t.kind == 'array':
                cargs.append(x)
                cargs.extend(x.shape)
            elif t.kind == 'tuple':
                cargs.extend(v.item() if et.np else v for v, et in zip(x, t.elem))
            elif t.np:
                cargs.append(x.item())
            else:
                cargs.append(x)
        if entry.libm is not None:
            cargs.append(entry.libm)
        if entry.stop:
            cargs.append(_interrupt.address())
        io = bytearray(40 + (16 * len(entry.ret.elem) if entry.ret.kind == 'tuple' else 0))
        status = entry.kernel(*cargs, io)
        if status:
            slots = [int.from_bytes(io[i:i + 8], 'little', signed=True) for i in range(0, 40, 8)]
            _raise_at(exception(status, slots), self._filename, slots[1], self.__name__)
        r = entry.ret
        if r is T.NONE:
            return None
        if r.kind == 'tuple':
            return tuple(self._box(io, 32 + 16 * i, 40 + 16 * i, et) for i, et in enumerate(r.elem))
        return self._box(io, 0, 16, r)

    def _box(self, io, at, flag_at, r):
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
