"""scramjit: a JIT for Python's hot loops, written in pure Python. Compiles through cimo.

    from scramjit import scram

    @scram
    def ema(x, y, a: float):
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = a * x[i] + (1 - a) * y[i - 1]
"""

from ._dispatch import Function
from ._errors import CompileError, FallbackWarning

__version__ = '0.1.0.dev0'
__all__ = ['scram', 'CompileError', 'FallbackWarning']


def scram(fn=None, *, boundscheck=True, cache=True, fallback=True):
    """Compile fn on its first call for each combination of argument types.

    boundscheck: check every index and raise IndexError, as Python does (default True).
    cache: keep compiled code on disk for other processes (accepted; the disk cache is not built yet).
    fallback: run what can't be compiled as Python, with a FallbackWarning (default True); with
        False, raise CompileError instead.
    """
    if fn is None:
        return lambda f: Function(f, boundscheck=boundscheck, cache=cache, fallback=fallback)
    return Function(fn, boundscheck=boundscheck, cache=cache, fallback=fallback)
