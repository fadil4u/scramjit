# Fallback and errors

Code scramjit can't compile runs as plain Python instead. Nothing breaks; that part is just not faster. Because the fallback is CPython itself, it can't change results.

## Three states

Each decorated function is, for each signature it is called with, in one of three states:

| State | Means |
| --- | --- |
| **Compiled** (green) | the whole function runs as native code |
| **Partly compiled** (amber) | some loops were lifted out and run native; the rest runs as Python |
| **Python** (red) | nothing compiled: the function runs as the original Python |

[Preflight](profiling.md) shows these states with the same colors, and show which lines run as Python and why.

## Loops are lifted out

A function often has a numeric loop surrounded by code scramjit doesn't compile:

```python
@scram
def tally(px, hist):
    labels = {}                          # a dict: Python
    for i in range(len(px)):             # compiled on its own
        hist[px[i]] += 1
    for k in range(len(hist)):           # writes the dict: Python
        if hist[k]:
            labels[k] = f"{k}: {hist[k]}"
    return labels
```

scramjit lifts the first loop out and compiles it on its own. The interpreter runs the rest and calls the compiled loop, passing in the variables it reads (`px`, `hist`) and taking back the ones it writes. A loop can be lifted when everything it reads and writes is a number, an array, a tuple or a fixed-size list.

## The whole function falls back

A function falls back to Python entirely when nothing in it compiles, or for a call whose argument types can't be passed to native code, such as a `str`. That call runs as the original function, `f.__wrapped__`; calls with other types still use their compiled versions.

## FallbackWarning

Each fallback emits `scramjit.FallbackWarning` once per function and signature, naming the line and the reason:

```text
tally.py:9: FallbackWarning: tally(uint8[:], int64[:]) is partly compiled: the loop on line 9 writes labels, a dict
```

It is a standard Python warning, so the usual filters apply:

```console
$ python -W ignore::scramjit.FallbackWarning app.py    # silence it
$ python -W error::scramjit.FallbackWarning app.py     # make every fallback an error
```

## Require compiled code

```python
@scram(fallback=False)
def step(grid, out, dt: float):
    ...
```

With `fallback=False`, anything less than compiled raises `scramjit.CompileError` at the Python line instead of falling back:

```text
scramjit.CompileError: sim.py:24: step(float64[:, :], float64[:, :], float) is partly compiled: line 24 calls print()
```

Use it for functions that must be fast, and in tests. [Preflight](profiling.md#in-ci) checks the same thing for a whole codebase before anything runs.

## Errors at the call boundary

A call whose arguments contradict an annotation raises `TypeError` before anything runs, as the annotation says it should; see [Types and annotations](types.md#annotations-are-contracts). That is a mistake in the call, not a fallback.

## Line numbers

scramjit emits `#line` directives in the C it generates, so every warning, error and report points at your Python file and line, not at generated code.
