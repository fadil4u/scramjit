# Compilation and caching

> [!PLANNED]
> The in-memory cache is part of M1; the on-disk cache is part of M3.

## When compilation happens

On each call the decorator reads the argument types: Python type, dtype, number of dimensions and layout. If it has no compiled version for that signature yet, it generates C specialized to it, compiles it with cimo and caches it.

## Two caches

- **In memory,** for the life of the process.
- **On disk,** so a second process reuses the compiled code instead of compiling again.

## First-call latency

Compiling happens in pure Python, so the first call to each specialization takes noticeably longer than the ones after it. How long is a measurement scramjit's design depends on: it decides whether library authors can compile on import. It will be recorded in the spec and here once cimo's compile time on generated C has been measured.
