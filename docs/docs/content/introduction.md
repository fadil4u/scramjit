# Introduction

scramjit is a just-in-time compiler for numeric Python, written in pure Python. Put a decorator on a function with a hot loop; on the first call scramjit translates it to C, compiles that with [cimo](https://www.cimo.dev) inside your process, and from then on the call runs native code.

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

> [!PLANNED]
> scramjit is in design. Nothing on these pages is built yet: they describe the design in the [spec](https://github.com/farizrahman4u/scramjit/blob/main/requirements.md), and will turn into usage docs as each [milestone](roadmap.md) lands. Decisions the spec still lists as open are marked **open**.

## When to use it

- **Python in the browser.** Pyodide, JupyterLite, marimo and PyScript have no Python compiler today, so a slow loop has no fix.
- **Plugins in embedded Pythons.** QGIS, Blender, Houdini, Maya, Nuke, FreeCAD and KiCad make compiled dependencies banned or painful.
- **Platforms other JITs haven't reached yet:** new CPython releases, free-threaded builds, Windows on arm64.
- **Fast hot paths in a pure-Python package,** with no wheel matrix to build and release.

## When not to

- Code that is already vectorized numpy or pandas: it is already running native loops.
- Machine learning: use PyTorch or JAX.
- If an existing JIT already works for you on a standard desktop or server, it has a head start, notably SIMD vectorization.

## It is still Python

The decorated function is plain Python. Without scramjit it still runs, slowly, in CPython, and stays debuggable and testable as Python. scramjit's first rule is that it never changes results: a compiled function returns what CPython returns for the same function undecorated, within the documented [semantics](semantics.md).

## How these docs are organized

The first three parts are a walkthrough; read them in order the first time. **Next** at the bottom of each page takes you on. The **Reference** part has the roadmap and the FAQ. Press <kbd>/</kbd> anywhere to search.
