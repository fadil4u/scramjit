# Preflight and explain

Code scramjit can't compile runs as Python, so nothing breaks, but speed can go missing quietly. Preflight and `explain()` show the state of every decorated function, which lines run as Python and why, what compiling cost, and what to change.

Every function is in one of three states, shown with the same colors everywhere:

| | State | Means |
| --- | --- | --- |
| 🟢 | **compiled** | the whole function runs as native code |
| 🟡 | **partly compiled** | some loops run native, the rest as Python |
| 🔴 | **Python** | nothing compiled |

`scramjit` is a command installed with the package; `python -m scramjit` works the same.

## Preflight, before you run

```console
$ scramjit preflight src/
```

reads every `@scram` function under the path, without importing or running anything, and reports its state and, line by line, what stays in Python and why:

```text
src/sim.py
  🟢 flood        compiled
  🟢 ema          compiled
  🟡 step         partly compiled
       8  loop stays in Python: line 11 grows a list
src/report.py
  🔴 summarize    Python
      18  iterates a dict

2 compiled · 1 partly · 1 Python
```

Annotations are used. Unannotated arguments are assumed to be numbers or arrays, so a fallback caused by the types of an actual call (a `str`, say) needs `--run`.

## Preflight while it runs

```console
$ scramjit preflight --run simulate.py --steps 2000
$ scramjit preflight --run -m pytest tests/
```

runs the script, or a module such as your test suite, and reports what actually happened, with the real argument types:

```text
scramjit · 4.21 s run · 61% native
   function   calls  compile  native  python
🟢 flood        120  0.14 s   0.22 s       –
🟢 ema ×2         2  0.07 s   0.02 s       –
🟡 step       2,000  disk     2.35 s  1.52 s

hints
  step, line 11: events.append keeps the loop on line 8 in Python;
    collect the events after it with np.flatnonzero(y > 1.0)
  ema: a arrives as int and float; annotate a: float

failed: step is partly compiled
```

- **native** and **python** are the time each function spent in compiled code and in code that fell back. The native share at the top is native time over the whole run; the rest is your own Python and fallbacks.
- **compile** is the time spent compiling, or `disk` when the code came from the [disk cache](caching.md). **×2** means two signatures.
- Functions that weren't called don't appear; run the static mode for those.

Arguments after the script or module are passed to it.

## In CI

Both modes exit non-zero when any function is less than compiled; `--allow-partial` accepts partly compiled ones. The strongest check runs your tests, so every function is checked with the types it really gets:

```console
$ scramjit preflight --run -m pytest
```

## Explain one function

`f.explain()` prints the function with each line marked native or Python, and what the compiler decided. It works in any REPL or notebook:

```text
>>> step.explain()
step(float64[:], float64[:], list)
partly compiled · sim.py · compiled in 38 ms
─────────────────────────────────────────────
 6 🟢 for i in range(len(x)):         native, lifted
 7 🟢     x[i] = x[i] * 0.99 + 0.01   bounds proven
 8 🔴 for i in range(1, len(x)):      Python: line 11
 9 🔴     y[i] = 0.9 * y[i - 1] + 0.1 * x[i]
10 🔴     if y[i] > 1.0:
11 🔴         events.append(i)        grows a list
─────────────────────────────────────────────
hint  collect the events after the loop with np.flatnonzero(y > 1.0)
```

It reports:

- **state per line:** native or Python, and for Python lines, the construct that kept them there;
- **types:** what each variable was inferred as, and where a type was narrowed;
- **bounds checks:** proven unnecessary, hoisted out of the loop, or kept, and what would let a kept one go;
- **loops:** which ones cimo's optimizer unrolled or otherwise improved, and why the others weren't;
- **the call boundary:** arrays passed in place, lists copied, strided arrays.

Pass a signature to explain only that one: `f.explain(f.signatures[0])`.

### From the command line

Give preflight a function instead of a directory:

```console
$ scramjit preflight src/sim.py:step                   # static: one assumed signature
$ scramjit preflight src/sim.py:step --run app.py      # every signature compiled during the run
$ scramjit preflight pkg.sim:step                      # a dotted module path works too
```

It prints the same report as `step.explain()`. Without `--run`, argument types come from annotations, and unannotated arguments are assumed to be numbers or arrays. With `--run`, the report covers each signature the function was actually compiled for. Put the function before `--run`: everything after `--run` is the program and its arguments.

## Specializations and the generated C

```python
>>> ema.signatures
[(float64[:], float64[:], float), (float64[:], float64[:], int)]
>>> print(ema.c(ema.signatures[0]))
```

`f.c()` returns the C scramjit generated, with `#line` directives pointing back at your Python. Without an argument it uses the most recent signature.

## Log compiles and fallbacks

```console
$ SCRAMJIT_LOG=1 python simulate.py
scramjit: compiled step(float64[:, :], float64[:, :], float) in 0.18 s
scramjit: flood(uint8[:, :], int64[:, :], int, int) from disk cache
scramjit: tally(uint8[:], int64[:]) partly compiled: line 9 writes labels, a dict
```

One line per compile, cache hit or fallback. A function that keeps compiling new signatures is being called with arguments of changing types; annotate them, or convert once before the loop.

## Acting on hints

| Hint | Why it costs | Fix |
| --- | --- | --- |
| a loop stays in Python | it runs at interpreter speed | move the dict, string or object work out of the loop, and keep the loop on numbers and arrays |
| an argument arrives as several types | one compile and one specialization per type | annotate it, such as `a: float` |
| a list is copied in and out | O(n) work on every call | pass a numpy array |
| an array is strided | every index multiplies by a stride | `np.ascontiguousarray` once, outside the loop |
| a bounds check stays in a hot loop | a compare and branch per access | index with the loop variable over `range(len(a))`, or `@scram(boundscheck=False)` once verified |
