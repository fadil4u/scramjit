"""The @scram decorator: signatures, the compile cache, the call boundary and fallback."""

import ast
import functools
import inspect
import os
import struct
import sys
import textwrap
import threading
import types
import warnings

from . import _types as T
from ._compiler import Compiler
from ._errors import CompileError, FallbackWarning, Unsupported
from ._runtime import PRELUDE, exception

_cimo = None
_cimo_lock = threading.Lock()


def _get_cimo():
    """cimo, imported with fused multiply-add contraction off (Python never contracts)."""
    global _cimo
    if _cimo is None:
        with _cimo_lock:
            if _cimo is None:
                already = 'cimo' in sys.modules
                if not already:
                    os.environ.setdefault('CIMO_FMA', '0')
                import cimo
                if os.environ.get('CIMO_FMA', '1') != '0':
                    warnings.warn('cimo contracts multiply-adds (CIMO_FMA is not 0); compiled float '
                                  'results can differ from Python\'s. Set CIMO_FMA=0 before importing cimo.',
                                  RuntimeWarning, stacklevel=4)
                _cimo = cimo
    return _cimo


def _arg_type(x):
    """The compiled type of an argument, or None if it isn't compiled."""
    t = type(x)
    if t is int:
        return T.INT if -(1 << 63) <= x < (1 << 63) else None
    if t is float:
        return T.FLOAT
    if t is bool:
        return T.BOOL
    np = sys.modules.get('numpy')
    if np is not None:
        if t is np.ndarray:
            if x.ndim == 1 and x.flags.c_contiguous:
                if x.dtype == np.float64:
                    return T.array(T.NP_FLOAT64, x.flags.writeable)
                if x.dtype == np.int64:
                    return T.array(T.NP_INT64, x.flags.writeable)
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
    __slots__ = ('kernel', 'ret', 'types', 'source')

    def __init__(self, kernel, ret, types_, source):
        self.kernel = kernel
        self.ret = ret
        self.types = types_
        self.source = source


class Function:
    def __init__(self, fn, boundscheck=True, cache=True, fallback=True):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._boundscheck = boundscheck
        self._fallback = fallback
        self._cache = {}
        self._lock = threading.Lock()
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
        if entry is PYTHON:
            return self._fn(*args)
        return self._run(entry, args)

    def _specialize(self, key, args):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                return entry
            sig = ', '.join(_describe(x, t) for x, t in zip(args, key))
            try:
                entry = self._compile(key, args)
            except Unsupported as u:
                line = (u.lineno + self._first - 1) if u.lineno else self._first
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
                     self._boundscheck)
        source = c.generate()
        cimo = _get_cimo()
        try:
            kernel = cimo.compile(PRELUDE + source).sj_kernel
        except cimo.CompileError as e:
            raise Unsupported(None, f'internal error: cimo rejected the generated C ({e}); please report it')
        if any(t.np for t in key) or c.ret.np:
            import numpy
            self._np = numpy
        return Compiled(kernel, c.ret, key, source)

    def _run(self, entry, args):
        cargs = []
        for x, t in zip(args, entry.types):
            if t.kind == 'array':
                cargs.append(x)
                cargs.append(x.shape[0])
            elif t.np:
                cargs.append(x.item())
            else:
                cargs.append(x)
        io = bytearray(32)
        status = entry.kernel(*cargs, io)
        if status:
            slots = [int.from_bytes(io[i:i + 8], 'little', signed=True) for i in range(0, 32, 8)]
            _raise_at(exception(status, slots), self._filename, slots[1], self.__name__)
        r = entry.ret
        if r is T.NONE:
            return None
        np = r.np if r.np != 'dyn' else io[16] != 0     # dynamic: the flag is in slot 2
        if r.kind == 'float':
            v = struct.unpack_from('<d', io)[0]
            return self._np.float64(v) if np else v
        v = int.from_bytes(io[0:8], 'little', signed=True)
        if r.kind == 'bool':
            return self._np.bool_(v) if np else bool(v)
        return self._np.int64(v) if np else v
