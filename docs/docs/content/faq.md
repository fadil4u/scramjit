# FAQ

## Is it usable yet?

No. scramjit is in design; see the [roadmap](roadmap.md). The PyPI package is a placeholder that reserves the name.

## Why not use an existing JIT?

If one works for you, it has a head start. LLVM-based JITs are mature and vectorize via LLVM, but need LLVM wheels, don't run in Pyodide and take time to support new CPython releases. Pythran, Cython's pure-Python mode and Shed Skin need a C or C++ toolchain on the target. scramjit needs neither, because it and its compiler are pure Python.

## Is it a new language?

No. Decorated functions are ordinary Python and run unchanged without scramjit. scramjit compiles a subset of Python; outside the subset you get a compile error, not a dialect.

## What does it not compile?

Anything that needs Python objects at runtime: classes, arbitrary objects, strings, dicts, sets and growable lists. Typed containers may come later; see [Arrays, lists and tuples](containers.md#not-yet).

## Will it vectorize my loop?

Not at first. That depends on cimo's planned auto-vectorizer. Until then, expect C speed for scalar loops.

## Is the generated code sandboxed?

No. Like cimo, compiled code runs with the full privileges of your Python process.
