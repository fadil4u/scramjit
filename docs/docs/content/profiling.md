# Profiling and visibility

Code scramjit can't compile runs as Python, so nothing breaks, but speed can go missing quietly. These tools show the state of every decorated function, which lines run as Python and why, what compiling cost, and what to change.

Every function is in one of three states, shown with the same colors everywhere:

| | State | Means |
| --- | --- | --- |
| 🟢 | **compiled** | the whole function runs as native code |
| 🟡 | **partly compiled** | some loops run native, the rest as Python |
| 🔴 | **Python** | nothing compiled |

The tools are subcommands of the `scramjit` command, which is installed with the package. `python -m scramjit` works the same.

## Preflight

```console
$ scramjit preflight src/
```

reads every `@scram` function under the path, without importing or running anything, and reports its state and, line by line, what stays in Python and why:

```text
src/sim.py
  🟢 step         compiled
  🟢 flood        compiled
src/stats.py
  🟡 tally        partly compiled
      9   loop stays in Python: writes labels, a dict
  🔴 summarize    Python
      18  for k, v in d.items(): iterates a dict

2 compiled · 1 partly compiled · 1 Python
```

- Annotations are used. Unannotated arguments are assumed to be numbers or arrays, so a fallback caused by the argument types of an actual call (a `str`, say) shows up at runtime instead, as a `FallbackWarning`.
- It exits non-zero when any function is less than compiled, so it fits in CI. `--allow-partial` accepts partly compiled functions too.

## Profile a whole program

```console
$ scramjit profile simulate.py --steps 2000
```

runs the script like `python -m cProfile` would, then prints a report:

```text
scramjit · 4.21 s run · 96% in native code
   function       calls   compile   native   share
🟢 step           2,000   disk      3.70 s     88%
🟢 flood            120   0.14 s    0.22 s      5%
🟡 tally             40   0.03 s    0.09 s      2%
🟢 ema ×2             2   0.07 s    0.02 s     <1%

hints
  tally, line 9: loop stays in Python (writes a dict); keep the loop numeric and build the dict after it
  ema: a arrives as int and float (2 specializations); annotate a: float
```

- **native share** is the part of the whole run spent in compiled code; the rest is Python, your own and the parts of decorated functions that fell back. If a hot loop isn't in the table, it isn't decorated.
- **compile** is the time spent compiling, or `disk` when the code came from the [disk cache](caching.md).
- **×2** means the function was compiled for two signatures.

Arguments after the script name are passed to it. `scramjit profile -m package.module` profiles a module.

## Explain one function

`f.explain()` prints the function's source with each line marked native or Python, and what the compiler decided:

```text
>>> tally.explain()
tally(uint8[:], int64[:])  partly compiled · 38 ms

 6 🔴 labels = {}                      Python: a dict
 7 🟢 for i in range(len(px)):         native: lifted loop
 8 🟢     hist[px[i]] += 1             hist[...]: bounds checked
 9 🔴 for k in range(len(hist)):       Python: writes labels
10 🔴     if hist[k]:
11 🔴         labels[k] = f"{k}: {hist[k]}"
12 🔴 return labels

hints
  line 9: keep the loop numeric and build the dict after it
```

It reports:

- **state per line:** native or Python, and for Python lines, the construct that kept them there;
- **types:** what each variable was inferred as, and where a type was narrowed;
- **bounds checks:** proven unnecessary, hoisted out of the loop, or kept, and what would let a kept one go;
- **loops:** which ones cimo's optimizer unrolled or otherwise improved, and why the others weren't;
- **the call boundary:** arrays passed in place, lists copied, strided arrays.

Pass a signature to explain only that one: `f.explain(f.signatures[0])`.

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
