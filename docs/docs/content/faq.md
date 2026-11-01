# FAQ

## Why not use an existing JIT?

If one works for you, it has a head start. LLVM-based JITs are mature and vectorize via LLVM, but need LLVM wheels, don't run in Pyodide and take time to support new CPython releases. Pythran, Cython's pure-Python mode and Shed Skin need a C or C++ toolchain on the target. scramjit needs neither, because it and its compiler are pure Python.

## How do I know my function was compiled?

If the call returned, it ran native code: there is no fallback that runs on Python objects. `f.signatures` lists what was compiled, and `python -m scramjit app.py` shows how much of a run was spent in compiled code. See [Profiling and visibility](profiling.md).

## Is it only for numeric code?

The supported types are numeric: scalars, arrays, tuples and fixed-size lists. But the target is any loop the interpreter is slow at, not only math: searches, flood fills, dynamic programming and scans over byte buffers fit as long as they work on those types.

## Does it help with the GIL?

Yes: compiled code releases the GIL while it runs, so threads calling it run in parallel.

## Is it a new language?

No. Decorated functions are ordinary Python and run unchanged without scramjit. scramjit compiles a subset of Python; outside the subset you get a compile error, not a dialect.

## What does it not compile?

Anything that needs Python objects at runtime: classes, arbitrary objects, strings, dicts, sets and growable lists. See [Arrays, lists and tuples](containers.md#not-supported).

## Will it vectorize my loop?

scramjit emits scalar C. It is built for loops that don't vectorize; for simple array arithmetic in an environment where numpy runs, numpy is already fast.

## Is the generated code sandboxed?

No. Like cimo, compiled code runs with the full privileges of your Python process.
