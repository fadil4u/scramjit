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


def scram(fn=None, *, boundscheck=True, cache=True, fallback=True, parity=True):
    """Compile fn on its first call for each combination of argument types.

    boundscheck: check every index and raise IndexError, as Python does (default True).
    cache: keep compiled code on disk, so other processes reuse it (default True; False writes nothing).
    fallback: run what can't be compiled as Python, with a FallbackWarning (default True); with
        False, raise CompileError instead.
    parity: give results identical to CPython, bit for bit (default True). False trades that
        for speed where parity costs it: float / // % by zero and math functions give IEEE inf
        and nan instead of raising, math functions are cimo's C library, int / int and int-float
        comparisons use plain double arithmetic.
    """
    if fn is None:
        return lambda f: Function(f, boundscheck=boundscheck, cache=cache, fallback=fallback, parity=parity)
    return Function(fn, boundscheck=boundscheck, cache=cache, fallback=fallback, parity=parity)
