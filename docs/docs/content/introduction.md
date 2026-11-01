# Introduction

scramjit is a just-in-time compiler for Python, written in pure Python, for the loops the interpreter is slowest at. Put a decorator on a function with a hot loop; on the first call scramjit translates it to C, compiles that with [cimo](https://www.cimo.dev) inside your process, and from then on the call runs native code, without holding the GIL.

```python
import numpy as np
import scramjit

@scramjit.jit
def saxpy(a: float, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]

x = np.random.rand(1_000_000)
y = np.zeros_like(x)
saxpy(2.0, x, y)   # compiled on this call, native from then on
```

There is no LLVM, no C compiler and no binary wheel involved: scramjit and cimo are both pure Python, so it works wherever Python runs, including Pyodide in the browser.

## Where it shines

Code that doesn't vectorize and is held back by the interpreter and the GIL:

- **Recurrences,** where each step depends on the one before (filters, integrators, running statistics).
- **Data-dependent branches and searches:** flood fills, path finding, dynamic programming, scans over byte buffers.
- **Simulations** that step state forward one cell, particle or agent at a time.

numpy is fast when the work fits one array expression; these don't. In CPython every iteration pays the interpreter, and threads take turns on the GIL. Compiled, the loop runs as C, and the GIL is released while it runs, so Python threads calling it run in parallel.

**Also a fit:** vectorizable array math written as plain loops, where only pure Python is allowed or numpy isn't available.

## Wherever Python runs

- **Python in the browser.** Pyodide, JupyterLite, marimo and PyScript have no Python compiler today, so a slow loop has no fix.
- **Plugins in embedded Pythons.** QGIS, Blender, Houdini, Maya, Nuke, FreeCAD and KiCad make compiled dependencies banned or painful.
- **Platforms other JITs haven't reached yet:** new CPython releases, free-threaded builds, Windows on arm64.
- **Fast hot paths in a pure-Python package,** with no wheel matrix to build and release.

## When not to

- Machine learning: use PyTorch or JAX.
- Code that is already one vectorized numpy or pandas expression, where numpy runs: it is already running native loops.
- If an existing JIT already works for you on a standard desktop or server, it has a head start, notably SIMD vectorization.

## It is still Python

The decorated function is plain Python. Without scramjit it still runs, slowly, in CPython, and stays debuggable and testable as Python. scramjit's first rule is that it never changes results: a compiled function returns what CPython returns for the same function undecorated, within the documented [semantics](semantics.md).

## How these docs are organized

The first three parts are a walkthrough; read them in order the first time. **Next** at the bottom of each page takes you on. The **Reference** part has platforms and the FAQ. Press <kbd>/</kbd> anywhere to search.
