# Errors

## Compile errors point at your Python

Anything outside the supported subset is a compile error raised on the first call. It names the Python file and line and says why:

```text
scramjit.CompileError: ema.py:7: 'y' is a float64 array here and a str on line 9; a variable must keep one type
```

scramjit emits `#line` directives in the C it generates, so errors that cimo finds also map back to the Python line.

## No silent fallback

There is no object mode. A construct that can't be compiled is an error, never a slow path that quietly runs on Python objects. To keep a function in Python, don't decorate it.

## Errors at the call boundary

A call whose arguments contradict an annotation raises `TypeError` before any native code runs, as described in [Types and annotations](types.md#annotations-are-contracts).

## Before running

`python -m scramjit check src/` finds constructs outside the subset in every `@scram` function without running anything. See [Profiling and visibility](profiling.md#check-before-you-run).
